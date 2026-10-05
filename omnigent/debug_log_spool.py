"""Local spool for debug-log rows the sink could not deliver.

Rows land here when shutdown runs out of time to send them, and after a
definite delivery failure. A later process replays them in the background.
Delivery is at-most-once: one process per machine replays (an exclusive file
lock), and a file is renamed to ``*.sending`` only as its POST starts, so a file
left in that state by a dead uploader is dropped rather than sent again.

Each line is a self-contained JSON record (``{"v", "dest", "row"}``), so a torn
or corrupt line costs only that row, never the rest of its file.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import logging
import os
import secrets
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

if sys.platform == "win32":
    import msvcrt

    def _lock_nonblocking(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock_fd(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_nonblocking(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_fd(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


_logger = logging.getLogger(__name__)

SPOOL_DIR_NAME = "debug-log-spool"
_LOCK_NAME = "upload.lock"
# Serializes capacity accounting + publication across writers (processes and
# spool instances), so concurrent writes can't jointly exceed the cap.
_WRITE_LOCK_NAME = "write.lock"
_WRITE_LOCK_WAIT_S = 5.0
_READY_SUFFIX = ".jsonl"
_SENDING_SUFFIX = ".sending"
_TMP_SUFFIX = ".tmp"
_FORMAT_VERSION = 1

ROWS_PER_FILE = 100
# A temp file older than this is debris from a writer that died mid-write.
_STALE_TMP_S = 60.0
MAX_TOTAL_BYTES = 50 * 1024 * 1024
MAX_FILES = 1000
MAX_AGE_S = 7 * 24 * 3600.0
MAX_ROW_BYTES = 1024 * 1024

DebugLogRow = dict[str, object]
# ``failed``: definitely not accepted, safe to keep for a retry. ``unknown``: the
# request may have been accepted (e.g. a read timeout), so it must not be resent.
# ``rejected``: the endpoint permanently refused it (e.g. a 400). ``withdrawn``:
# not sent at all, because ``begin_send()`` declined.
DeliveryResult = Literal["delivered", "failed", "unknown", "rejected", "withdrawn"]
# Called by a sender immediately before its request goes out; ``False`` means
# don't send. Replay uses it to mark the file ``*.sending`` only at that point.
BeginSend = Callable[[], bool]
ReplayResult = Literal["done", "paused", "failed", "busy"]
Diag = Callable[..., None]

# Lock fds held by this process. A forked child closes its copies so it never
# pins the parent's lock; closing (not unlocking) leaves the parent's lock held.
_held_lock_fds: set[int] = set()


def _close_inherited_lock_fds() -> None:
    for fd in list(_held_lock_fds):
        with contextlib.suppress(OSError):
            os.close(fd)
    _held_lock_fds.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_close_inherited_lock_fds)


def _default_diag(key: str, msg: str, *args: object) -> None:  # noqa: ARG001
    _logger.warning("debug-log spool: " + msg, *args)


def destination_key(destination: str) -> str:
    """Return the stable, non-secret key that pins a spool file to its endpoint."""
    return hashlib.sha256(destination.encode("utf-8")).hexdigest()[:16]


class DebugLogSpool:
    """A directory of spooled debug-log batches for one delivery destination."""

    def __init__(self, directory: Path, destination: str, *, diag: Diag | None = None) -> None:
        self._dir = directory
        self._dest = destination_key(destination)
        self._diag = diag or _default_diag
        self._seq = itertools.count()

    @classmethod
    def for_destination(cls, destination: str, *, diag: Diag | None = None) -> DebugLogSpool:
        """Spool under ``<data-dir>/debug-log-spool`` for *destination*."""
        from omnigent.process_logging import data_dir

        return cls(data_dir() / SPOOL_DIR_NAME, destination, diag=diag)

    @property
    def directory(self) -> Path:
        return self._dir

    # ── writing ─────────────────────────────────────────────────────────────

    def write(self, rows: Iterable[DebugLogRow], *, deadline: float | None = None) -> int:
        """Persist *rows* as spool files of at most :data:`ROWS_PER_FILE` rows.

        Local disk only; never touches the network. Stops before a further
        file once the monotonic *deadline* passes; the first file is always
        attempted, so a spent deadline still saves the oldest rows. Callers
        that must not block (``close()``) bound the call from another thread.

        :returns: Number of rows written.
        """
        encoded: list[str] = []
        oversize = 0
        for row in rows:
            try:
                row_json = json.dumps(row, default=str)
            except (TypeError, ValueError):
                oversize += 1
                continue
            if len(row_json) > MAX_ROW_BYTES:
                oversize += 1
                continue
            # ``dest`` is hex, so plain interpolation is valid JSON.
            encoded.append(f'{{"v":{_FORMAT_VERSION},"dest":"{self._dest}","row":{row_json}}}')
        if oversize:
            self._diag("spool_oversize", "dropped %d unserializable/oversize row(s)", oversize)
        if not encoded:
            return 0
        # A batch alone may exceed the cap (100 rows of up to 1 MiB): keep its
        # newest rows that fit, consistent with dropping the oldest first.
        incoming = sum(len(line) + 1 for line in encoded)
        trimmed = 0
        while encoded and incoming > MAX_TOTAL_BYTES:
            incoming -= len(encoded.pop(0)) + 1
            trimmed += 1
        if trimmed:
            self._diag("spool_cap", "spool full; dropped %d oldest row(s)", trimmed)
        if not encoded:
            return 0
        chunks = [encoded[i : i + ROWS_PER_FILE] for i in range(0, len(encoded), ROWS_PER_FILE)]
        written = 0
        write_lock: int | None = None
        try:
            self._dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            write_lock = self._acquire_write_lock(deadline)
            if write_lock is None:
                # Never publish without serialized accounting (the cap must
                # hold); the deadline wins over these rows.
                self._diag(
                    "spool_write_lock", "spool busy; dropped %d row(s) not spooled", len(encoded)
                )
                return 0
            self._prune(
                incoming_files=len(chunks),
                incoming_bytes=sum(len(line) + 1 for line in encoded),
            )
            for index, chunk in enumerate(chunks):
                if index and deadline is not None and time.monotonic() > deadline:
                    break
                self._write_file(chunk)
                written += len(chunk)
        except OSError as exc:
            self._diag("spool_write", "could not write spool file: %s", exc)
        finally:
            if write_lock is not None:
                self._unlock(write_lock)
        if written < len(encoded):
            self._diag(
                "spool_incomplete", "dropped %d row(s) not spooled in time", len(encoded) - written
            )
        return written

    def _write_file(self, lines: list[str]) -> None:
        # Zero-padded so lexical order (used for oldest-first) matches numeric;
        # the random suffix keeps two spool instances (e.g. a replaced handler
        # in the same process) from publishing over each other's file.
        name = (
            f"{int(time.time() * 1000):013d}-{os.getpid():010d}-{next(self._seq):09d}"
            f"-{secrets.token_hex(4)}"
        )
        final = self._dir / f"{name}{_READY_SUFFIX}"
        tmp = self._dir / f"{name}{_TMP_SUFFIX}"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except BaseException:
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        # Atomic: a replayer never sees a partial file.
        os.replace(tmp, final)

    def _prune(self, *, incoming_files: int, incoming_bytes: int) -> None:
        """Delete expired files, then the oldest ones, to fit the size caps.

        Counts every payload state, not just ready files: abandoned temp files
        are reclaimed here, and dead uploaders' claims too when no uploader
        holds the lock, so crash debris can't grow past the cap even in
        processes too short-lived to replay.
        """
        other_bytes = self._reclaim_debris()
        entries: list[tuple[Path, int]] = []
        now_ms = time.time() * 1000
        for path in sorted(self._dir.glob(f"*{_READY_SUFFIX}")):
            if now_ms - _created_ms(path) > MAX_AGE_S * 1000:
                self._drop_file(path, "spool_expired", "dropped %d expired spooled row(s)")
                continue
            with contextlib.suppress(OSError):
                entries.append((path, path.stat().st_size))
        total = other_bytes + sum(size for _, size in entries)
        while entries and (
            len(entries) + incoming_files > MAX_FILES or total + incoming_bytes > MAX_TOTAL_BYTES
        ):
            path, size = entries.pop(0)
            self._drop_file(path, "spool_cap", "spool full; dropped %d oldest row(s)")
            total -= size

    # ── replay ──────────────────────────────────────────────────────────────

    def replay(
        self,
        deliver: Callable[[list[DebugLogRow], BeginSend], DeliveryResult],
        *,
        should_continue: Callable[[], bool],
    ) -> ReplayResult:
        """Deliver spooled files for this destination, oldest first.

        :param deliver: Sends one batch and reports its outcome. It must call
            the given ``begin_send()`` right before its request goes out and
            skip sending when that returns ``False``.
        :param should_continue: Checked between files; ``False`` pauses
            (e.g. live rows are waiting, or the sink is closing).
        :returns: ``"done"`` when nothing is left, ``"paused"`` when stopped
            early, ``"failed"`` on a retryable failure (the file is kept), or
            ``"busy"`` when another process holds the upload lock.
        """
        if not self._dir.is_dir():
            return "done"
        lock_fd = self._try_lock()
        if lock_fd is None:
            return "busy"
        try:
            self._drop_unknown_outcomes()
            now_ms = time.time() * 1000
            for path in sorted(self._dir.glob(f"*{_READY_SUFFIX}")):
                if not should_continue():
                    return "paused"
                if now_ms - _created_ms(path) > MAX_AGE_S * 1000:
                    self._drop_file(path, "spool_expired", "dropped %d expired spooled row(s)")
                    continue
                parsed = self._read(path)
                if parsed is None:
                    continue
                dest, rows = parsed
                if dest is not None and dest != self._dest:
                    continue  # another endpoint's file; left for its own sink
                if not rows:
                    with contextlib.suppress(OSError):
                        path.unlink()
                    continue
                claim = _FileClaim(path)
                result = deliver(_tag_replayed(rows), claim)
                if result in ("failed", "withdrawn"):
                    # Definitely not accepted: keep the file for a later attempt.
                    claim.unsent()
                    return "failed" if result == "failed" else "paused"
                if result == "unknown":
                    self._diag(
                        "spool_unknown", "dropped %d replayed row(s): outcome unknown", len(rows)
                    )
                elif result == "rejected":
                    self._diag("spool_rejected", "dropped %d replayed row(s): rejected", len(rows))
                claim.remove()
            return "done"
        finally:
            self._unlock(lock_fd)

    def _reclaim_debris(self) -> int:
        """Drop stale temp files and dead claims; return bytes still in use."""
        cutoff = time.time() - _STALE_TMP_S
        in_use = 0
        for path in self._dir.glob(f"*{_TMP_SUFFIX}"):
            with contextlib.suppress(OSError):
                stat = path.stat()
                if stat.st_mtime < cutoff:
                    path.unlink()
                else:
                    in_use += stat.st_size  # a writer is mid-write
        lock_fd = self._try_lock()
        if lock_fd is not None:
            # No uploader is active, so every claim belongs to a dead one.
            try:
                self._drop_unknown_outcomes()
            finally:
                self._unlock(lock_fd)
        else:
            for path in self._dir.glob(f"*{_SENDING_SUFFIX}"):
                with contextlib.suppress(OSError):
                    in_use += path.stat().st_size
        return in_use

    def _drop_unknown_outcomes(self) -> None:
        """Drop files a dead uploader left mid-POST; resending could duplicate."""
        for path in self._dir.glob(f"*{_SENDING_SUFFIX}"):
            self._drop_file(
                path, "spool_unknown", "dropped %d spooled row(s): previous upload outcome unknown"
            )
        # Temp files are only ever live inside a writer's _write_file; anything
        # older than a minute is debris from a writer that died mid-write.
        cutoff = time.time() - _STALE_TMP_S
        for path in self._dir.glob(f"*{_TMP_SUFFIX}"):
            with contextlib.suppress(OSError):
                if path.stat().st_mtime < cutoff:
                    path.unlink()

    def _read(self, path: Path) -> tuple[str | None, list[DebugLogRow]] | None:
        """Parse a spool file line by line, skipping lines that don't parse.

        :returns: ``(dest, rows)``, where *dest* is ``None`` when no line was
            readable, or ``None`` when the file is gone.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return None
        except OSError as exc:
            self._diag("spool_corrupt", "could not read spool file %s: %s", path.name, exc)
            return None
        dest: str | None = None
        rows: list[DebugLogRow] = []
        skipped = 0
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                skipped += 1  # e.g. a line torn by a crash or power loss
                continue
            row = record.get("row") if isinstance(record, dict) else None
            line_dest = record.get("dest") if isinstance(record, dict) else None
            if (
                not isinstance(row, dict)
                or not isinstance(line_dest, str)
                or record.get("v") != _FORMAT_VERSION
            ):
                skipped += 1
                continue
            if dest is None:
                dest = line_dest
            elif line_dest != dest:
                skipped += 1
                continue
            rows.append(row)
        if skipped:
            self._diag("spool_partial", "skipped %d unreadable line(s) in %s", skipped, path.name)
        return dest, rows

    def _drop_file(self, path: Path, key: str, msg: str) -> None:
        count = _line_count(path)
        with contextlib.suppress(OSError):
            path.unlink()
            self._diag(key, msg, count)

    # ── locking ─────────────────────────────────────────────────────────────

    def _acquire_write_lock(self, deadline: float | None) -> int | None:
        """Take the writer lock, waiting briefly and never past *deadline*."""
        limit = time.monotonic() + _WRITE_LOCK_WAIT_S
        if deadline is not None:
            limit = min(limit, deadline)
        while True:
            fd = self._try_lock(_WRITE_LOCK_NAME)
            if fd is not None or time.monotonic() >= limit:
                return fd
            time.sleep(0.01)

    def _try_lock(self, name: str = _LOCK_NAME) -> int | None:
        try:
            fd = os.open(self._dir / name, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError:
            return None
        try:
            _lock_nonblocking(fd)
        except OSError:
            os.close(fd)
            return None
        _held_lock_fds.add(fd)
        return fd

    def _unlock(self, fd: int) -> None:
        _held_lock_fds.discard(fd)
        with contextlib.suppress(OSError):
            _unlock_fd(fd)
        with contextlib.suppress(OSError):
            os.close(fd)


class _FileClaim:
    """A replay file's send claim: ``*.sending`` only while a request is out.

    Callable as the :data:`BeginSend` hook. ``unsent()`` reverts the claim
    after an attempt that definitely didn't reach the endpoint (e.g. a connect
    error before a retry), so dying during the retry backoff leaves a normal,
    retryable file rather than one the next uploader must drop.
    """

    def __init__(self, path: Path) -> None:
        self._ready = path
        self._sending = path.with_suffix(_SENDING_SUFFIX)
        self._claimed = False

    def __call__(self) -> bool:
        if not self._claimed:
            try:
                os.replace(self._ready, self._sending)
            except OSError:
                return False
            self._claimed = True
        return True

    def unsent(self) -> None:
        if self._claimed:
            with contextlib.suppress(OSError):
                os.replace(self._sending, self._ready)
            self._claimed = False

    def remove(self) -> None:
        with contextlib.suppress(OSError):
            (self._sending if self._claimed else self._ready).unlink()


def _created_ms(path: Path) -> float:
    try:
        return float(path.name.split("-", 1)[0])
    except ValueError:
        return 0.0


def _line_count(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _tag_replayed(rows: list[DebugLogRow]) -> list[DebugLogRow]:
    """Mark replayed rows so late arrivals are distinguishable in queries."""
    now_us = time.time() * 1_000_000
    tagged: list[DebugLogRow] = []
    for row in rows:
        attrs = row.get("attributes")
        merged = dict(attrs) if isinstance(attrs, dict) else {}
        merged["spooled"] = "true"
        client_time = row.get("client_time")
        if isinstance(client_time, (int, float)):
            merged["spool_delay_s"] = f"{max(0.0, (now_us - client_time) / 1_000_000):.3f}"
        tagged.append({**row, "attributes": merged})
    return tagged
