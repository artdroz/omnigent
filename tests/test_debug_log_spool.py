"""Tests for the debug-log spool (local durability + at-most-once replay)."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from omnigent import debug_log_spool as sp


def _rows(n: int, prefix: str = "row") -> list[dict[str, object]]:
    return [
        {"message": f"{prefix} {i}", "client_time": int(time.time() * 1_000_000), "attributes": {}}
        for i in range(n)
    ]


def _spool(tmp_path: Path, dest: str = "https://zerobus.example/insert") -> sp.DebugLogSpool:
    return sp.DebugLogSpool(tmp_path / "spool", dest)


def _always_continue() -> bool:
    return True


def _files(spool: sp.DebugLogSpool, suffix: str = ".jsonl") -> list[Path]:
    return sorted(spool.directory.glob(f"*{suffix}"))


def test_write_then_replay_delivers_tagged_rows_and_deletes(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    assert spool.write(_rows(250)) == 250
    assert len(_files(spool)) == 3  # 100 + 100 + 50
    assert all(oct(p.stat().st_mode & 0o777) == "0o600" for p in _files(spool))

    delivered: list[dict[str, object]] = []

    def deliver(batch: list[dict[str, object]], begin: sp.BeginSend) -> sp.DeliveryResult:

        assert begin()
        delivered.extend(batch)
        return "delivered"

    assert spool.replay(deliver, should_continue=_always_continue) == "done"
    assert [r["message"] for r in delivered] == [f"row {i}" for i in range(250)]
    attrs = delivered[0]["attributes"]
    assert isinstance(attrs, dict)
    assert attrs["spooled"] == "true"
    assert float(attrs["spool_delay_s"]) >= 0
    assert _files(spool) == []


def test_failed_replay_keeps_file_for_retry(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    spool.write(_rows(150))

    assert spool.replay(lambda _b, _begin: "failed", should_continue=_always_continue) == "failed"
    assert len(_files(spool)) == 2
    assert _files(spool, ".sending") == []


@pytest.mark.parametrize("result", ["unknown", "rejected"])
def test_ambiguous_or_rejected_replay_is_dropped_not_retried(
    tmp_path: Path, result: sp.DeliveryResult
) -> None:
    spool = _spool(tmp_path)
    spool.write(_rows(10))
    calls: list[int] = []

    def deliver(batch: list[dict[str, object]], begin: sp.BeginSend) -> sp.DeliveryResult:

        assert begin()
        calls.append(len(batch))
        return result

    assert spool.replay(deliver, should_continue=_always_continue) == "done"
    assert spool.replay(deliver, should_continue=_always_continue) == "done"
    assert calls == [10]
    assert list(spool.directory.glob("*.json*")) == []


def test_leftover_sending_file_is_dropped_unsent(tmp_path: Path) -> None:
    """A file a dead uploader left mid-POST may already be in ZeroBus."""
    spool = _spool(tmp_path)
    spool.write(_rows(5, prefix="maybe-sent"))
    spool.write(_rows(5, prefix="fresh"))
    first = _files(spool)[0]
    os.replace(first, first.with_suffix(".sending"))
    delivered: list[dict[str, object]] = []

    def deliver(batch: list[dict[str, object]], begin: sp.BeginSend) -> sp.DeliveryResult:

        assert begin()
        delivered.extend(batch)
        return "delivered"

    assert spool.replay(deliver, should_continue=_always_continue) == "done"
    assert {str(r["message"]).split()[0] for r in delivered} == {"fresh"}
    assert list(spool.directory.glob("*.sending")) == []


def test_other_destination_files_are_left_alone(tmp_path: Path) -> None:
    prod = _spool(tmp_path, "https://prod/insert")
    dev = _spool(tmp_path, "https://dev/insert")
    prod.write(_rows(3))
    calls: list[int] = []

    def deliver(batch: list[dict[str, object]], begin: sp.BeginSend) -> sp.DeliveryResult:

        assert begin()
        calls.append(len(batch))
        return "delivered"

    assert dev.replay(deliver, should_continue=_always_continue) == "done"
    assert calls == []
    assert len(_files(prod)) == 1


def test_concurrent_replay_is_busy(tmp_path: Path) -> None:
    first = _spool(tmp_path)
    second = _spool(tmp_path)
    first.write(_rows(3))
    nested: list[sp.ReplayResult] = []

    def deliver(batch: list[dict[str, object]], begin: sp.BeginSend) -> sp.DeliveryResult:

        assert begin()
        nested.append(
            second.replay(lambda _b, _begin: "delivered", should_continue=_always_continue)
        )
        return "delivered"

    assert first.replay(deliver, should_continue=_always_continue) == "done"
    assert nested == ["busy"]


def test_should_continue_pauses_between_files(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    spool.write(_rows(300))
    sent: list[int] = []
    gate = iter([True, False])

    def deliver(batch: list[dict[str, object]], begin: sp.BeginSend) -> sp.DeliveryResult:

        assert begin()
        sent.append(len(batch))
        return "delivered"

    assert spool.replay(deliver, should_continue=lambda: next(gate)) == "paused"
    assert sent == [100]
    assert len(_files(spool)) == 2


def test_size_cap_drops_oldest_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sp, "MAX_FILES", 3)
    spool = _spool(tmp_path)
    for i in range(5):
        spool.write(_rows(1, prefix=f"batch{i}"))

    remaining = [
        json.loads(p.read_text().splitlines()[0])["row"]["message"] for p in _files(spool)
    ]
    assert remaining == ["batch2 0", "batch3 0", "batch4 0"]


def test_expired_files_are_dropped(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    spool.write(_rows(2))
    (path,) = _files(spool)
    old_ms = int((time.time() - sp.MAX_AGE_S - 60) * 1000)
    os.replace(path, path.with_name(f"{old_ms:013d}-{path.name.split('-', 1)[1]}"))

    calls: list[int] = []
    spool.replay(
        lambda b, _begin: calls.append(len(b)) or "delivered", should_continue=_always_continue
    )
    assert calls == []
    assert _files(spool) == []


def test_write_respects_deadline_but_always_saves_the_first_file(tmp_path: Path) -> None:
    """A spent deadline stops further files, not the first: the oldest rows survive."""
    spool = _spool(tmp_path)
    assert spool.write(_rows(500), deadline=time.monotonic() - 1) == sp.ROWS_PER_FILE
    assert len(_files(spool)) == 1
    assert spool.write(_rows(500), deadline=time.monotonic() + 10) == 500


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX fork / signals")
def test_killed_uploader_releases_lock_and_its_batch_is_not_resent(tmp_path: Path) -> None:
    """SIGKILL mid-POST: the lock frees with the process; the batch is dropped."""
    spool = _spool(tmp_path)
    spool.write(_rows(4, prefix="in-flight"))
    script = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from omnigent.debug_log_spool import DebugLogSpool
        spool = DebugLogSpool(Path({str(spool.directory)!r}), "https://zerobus.example/insert")
        def deliver(batch, begin):
            assert begin()
            print("sending", flush=True)
            time.sleep(60)
        spool.replay(deliver, should_continue=lambda: True)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "sending"
        assert (
            spool.replay(lambda _b, _begin: "delivered", should_continue=_always_continue)
            == "busy"
        )
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()

    calls: list[int] = []
    spool.replay(
        lambda b, _begin: calls.append(len(b)) or "delivered", should_continue=_always_continue
    )
    assert calls == []
    assert list(spool.directory.glob("*.json*")) == []
    assert list(spool.directory.glob("*.sending")) == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_forked_child_does_not_pin_a_dead_parents_upload_lock(tmp_path: Path) -> None:
    """A runner forked while the lock is held must not keep it after the parent dies."""
    spool = _spool(tmp_path)
    spool.directory.mkdir(parents=True)
    fd = spool._try_lock()
    assert fd is not None
    pid = os.fork()
    if pid == 0:  # child: outlive the parent's hold
        time.sleep(3)
        os._exit(0)
    try:
        # Simulate the parent dying while holding the lock: its fd closes
        # without an explicit unlock.
        sp._held_lock_fds.discard(fd)
        os.close(fd)
        other = _spool(tmp_path)
        # The child closes its copy in an after-fork hook; give it a moment to run.
        deadline = time.monotonic() + 1.5
        reacquired = other._try_lock()
        while reacquired is None and time.monotonic() < deadline:
            time.sleep(0.05)
            reacquired = other._try_lock()
        assert reacquired is not None, "the forked child kept the upload lock held"
        other._unlock(reacquired)
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)


def _collecting(delivered: list[dict[str, object]]):  # type: ignore[no-untyped-def]
    def deliver(batch: list[dict[str, object]], begin: sp.BeginSend) -> sp.DeliveryResult:
        assert begin()
        delivered.extend(batch)
        return "delivered"

    return deliver


def test_torn_last_line_costs_only_that_row(tmp_path: Path) -> None:
    """A write cut off mid-line (crash, power loss) keeps the complete lines."""
    spool = _spool(tmp_path)
    spool.write(_rows(5))
    (path,) = _files(spool)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) - 20])  # tear the last record

    delivered: list[dict[str, object]] = []
    assert spool.replay(_collecting(delivered), should_continue=_always_continue) == "done"
    assert [r["message"] for r in delivered] == [f"row {i}" for i in range(4)]
    assert _files(spool) == []


def test_corrupt_middle_line_is_skipped(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    spool.write(_rows(3))
    (path,) = _files(spool)
    lines = path.read_text().splitlines()
    lines[1] = '{"v":1,"dest":'  # garbage
    path.write_text("\n".join(lines) + "\n")

    delivered: list[dict[str, object]] = []
    spool.replay(_collecting(delivered), should_continue=_always_continue)
    assert [r["message"] for r in delivered] == ["row 0", "row 2"]


def test_unreadable_file_is_deleted_without_delivery(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    spool.write(_rows(2))
    (path,) = _files(spool)
    path.write_text("not json\n\x00\x01garbage\n")

    delivered: list[dict[str, object]] = []
    assert spool.replay(_collecting(delivered), should_continue=_always_continue) == "done"
    assert delivered == []
    assert _files(spool) == []


def test_file_is_marked_sending_only_when_the_request_goes_out(tmp_path: Path) -> None:
    """Failing or withdrawing before begin_send() leaves a normal, retryable file."""
    spool = _spool(tmp_path)
    spool.write(_rows(2))
    seen_sending: list[bool] = []

    def failing_before_send(
        batch: list[dict[str, object]], begin: sp.BeginSend
    ) -> sp.DeliveryResult:
        seen_sending.append(bool(list(spool.directory.glob("*.sending"))))
        return "failed"  # e.g. the token mint failed: nothing went out

    assert spool.replay(failing_before_send, should_continue=_always_continue) == "failed"
    assert seen_sending == [False]
    assert (
        spool.replay(lambda _b, _begin: "withdrawn", should_continue=_always_continue) == "paused"
    )
    assert len(_files(spool)) == 1
    assert list(spool.directory.glob("*.sending")) == []

    def sends(batch: list[dict[str, object]], begin: sp.BeginSend) -> sp.DeliveryResult:
        assert begin()
        seen_sending.append(bool(list(spool.directory.glob("*.sending"))))
        return "delivered"

    assert spool.replay(sends, should_continue=_always_continue) == "done"
    assert seen_sending == [False, True]
    assert {p.name for p in spool.directory.glob("*")} == {"upload.lock", "write.lock"}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_uploader_killed_before_its_request_leaves_the_file_for_retry(tmp_path: Path) -> None:
    """Dying while still getting a token must not cost the batch."""
    spool = _spool(tmp_path)
    spool.write(_rows(4, prefix="never-sent"))
    script = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from omnigent.debug_log_spool import DebugLogSpool
        spool = DebugLogSpool(Path({str(spool.directory)!r}), "https://zerobus.example/insert")
        def deliver(batch, begin):
            print("minting", flush=True)
            time.sleep(60)  # stuck before the request: begin() never called
        spool.replay(deliver, should_continue=lambda: True)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "minting"
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()

    delivered: list[dict[str, object]] = []
    assert spool.replay(_collecting(delivered), should_continue=_always_continue) == "done"
    assert [r["message"] for r in delivered] == [f"never-sent {i}" for i in range(4)]


def test_a_batch_larger_than_the_cap_keeps_its_newest_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sp, "MAX_TOTAL_BYTES", 20_000)
    spool = _spool(tmp_path)
    rows = [{"message": f"{i:03d}" + "x" * 1_000, "attributes": {}} for i in range(100)]

    written = spool.write(rows)

    total = sum(p.stat().st_size for p in spool.directory.glob("*.jsonl"))
    assert 0 < written < 100
    assert total <= sp.MAX_TOTAL_BYTES
    delivered: list[dict[str, object]] = []
    spool.replay(_collecting(delivered), should_continue=_always_continue)
    kept = [f"{i:03d}" for i in range(100 - written, 100)]
    assert [str(r["message"])[:3] for r in delivered] == kept


def test_files_written_in_the_same_millisecond_replay_in_write_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spool = _spool(tmp_path)
    frozen = time.time()
    monkeypatch.setattr(sp.time, "time", lambda: frozen)
    for i in range(12):  # sequence numbers past 9 must still sort after 2
        spool.write([{"message": f"batch {i}", "attributes": {}}])
    monkeypatch.undo()

    delivered: list[dict[str, object]] = []
    spool.replay(_collecting(delivered), should_continue=_always_continue)
    assert [r["message"] for r in delivered] == [f"batch {i}" for i in range(12)]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_uploader_killed_in_backoff_after_a_definite_failure_leaves_the_file(
    tmp_path: Path,
) -> None:
    """A connect error proved nothing was sent; dying in the backoff must not drop it."""
    spool = _spool(tmp_path)
    spool.write(_rows(3, prefix="not-sent"))
    script = textwrap.dedent(
        f"""
        import time
        from pathlib import Path
        from omnigent.debug_log_spool import DebugLogSpool
        spool = DebugLogSpool(Path({str(spool.directory)!r}), "https://zerobus.example/insert")
        def deliver(batch, claim):
            assert claim()      # request goes out...
            claim.unsent()      # ...and fails before reaching ZeroBus (ConnectError)
            print("backing off", flush=True)
            time.sleep(60)      # retry backoff
        spool.replay(deliver, should_continue=lambda: True)
        """
    )
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "backing off"
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()

    delivered: list[dict[str, object]] = []
    assert spool.replay(_collecting(delivered), should_continue=_always_continue) == "done"
    assert [r["message"] for r in delivered] == [f"not-sent {i}" for i in range(3)]


def test_two_spool_instances_in_the_same_millisecond_keep_both_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E.g. a replaced handler's new spool writing while the old one still does."""
    first, second = _spool(tmp_path), _spool(tmp_path)
    frozen = time.time()
    monkeypatch.setattr(sp.time, "time", lambda: frozen)
    first.write([{"message": "from first", "attributes": {}}])
    second.write([{"message": "from second", "attributes": {}}])
    monkeypatch.undo()

    delivered: list[dict[str, object]] = []
    first.replay(_collecting(delivered), should_continue=_always_continue)
    assert sorted(str(r["message"]) for r in delivered) == ["from first", "from second"]


def test_writes_reclaim_crash_debris_without_replay(tmp_path: Path) -> None:
    """Stale temp files and dead claims don't escape the cap in short-lived processes."""
    spool = _spool(tmp_path)
    spool.write(_rows(1))
    stale_tmp = spool.directory / "0000000000001-0000000001-000000000-dead.tmp"
    stale_tmp.write_text("x" * 1000)
    old = time.time() - 3600
    os.utime(stale_tmp, (old, old))
    dead_claim = spool.directory / "0000000000002-0000000002-000000000-dead.sending"
    dead_claim.write_text("x" * 1000)

    spool.write(_rows(1))  # a write alone, no replay

    assert not stale_tmp.exists()
    assert not dead_claim.exists()  # no uploader holds the lock: its claim is stale


def test_a_live_claim_is_kept_and_counted_toward_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spool = _spool(tmp_path)
    spool.write(_rows(1, prefix="oldest"))
    uploader = _spool(tmp_path)
    lock_fd = uploader._try_lock()  # an active uploader elsewhere
    assert lock_fd is not None
    try:
        live_claim = spool.directory / "0000000000002-0000000002-000000000-live.sending"
        live_claim.write_text("x" * 5000)
        ready_bytes = sum(p.stat().st_size for p in spool.directory.glob("*.jsonl"))
        monkeypatch.setattr(sp, "MAX_TOTAL_BYTES", 5000 + ready_bytes + 50)

        spool.write(_rows(1, prefix="newest"))

        assert live_claim.exists()  # never touch an active uploader's claim
        remaining = [json.loads(p.read_text())["row"]["message"] for p in _files(spool)]
        assert remaining == ["newest 0"]  # the claim's bytes forced out the oldest file
    finally:
        uploader._unlock(lock_fd)


def test_concurrent_writers_cannot_jointly_exceed_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writers that both see room must not both publish past the cap."""
    import threading

    monkeypatch.setattr(sp, "MAX_TOTAL_BYTES", 30_000)
    first, second = _spool(tmp_path), _spool(tmp_path)
    first.write(_rows(1))  # create the directory up front
    for path in _files(first):
        path.unlink()
    both_pruned = threading.Barrier(2)
    real_prune = sp.DebugLogSpool._prune

    def prune_then_wait(self: sp.DebugLogSpool, **kwargs: int) -> None:
        real_prune(self, **kwargs)
        # Without serialization both writers reach here before either publishes.
        with contextlib.suppress(threading.BrokenBarrierError):
            both_pruned.wait(timeout=0.5)

    monkeypatch.setattr(sp.DebugLogSpool, "_prune", prune_then_wait)
    batch = [{"message": f"{i:03d}" + "x" * 180, "attributes": {}} for i in range(100)]
    writers = [threading.Thread(target=s.write, args=(batch,)) for s in (first, second)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(timeout=10)

    total = sum(p.stat().st_size for p in spool_files(tmp_path))
    assert total <= sp.MAX_TOTAL_BYTES, f"{total} bytes > cap"


def spool_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "spool").glob("*.jsonl"))


def test_a_writer_that_cannot_get_the_lock_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No unlocked fallback: the cap holds even when a writer stalls holding the lock."""
    import threading

    monkeypatch.setattr(sp, "MAX_TOTAL_BYTES", 30_000)
    monkeypatch.setattr(sp, "_WRITE_LOCK_WAIT_S", 0.05)
    first, second = _spool(tmp_path), _spool(tmp_path)
    first.write(_rows(1))
    for path in _files(first):
        path.unlink()
    pruned = threading.Event()
    real_prune = sp.DebugLogSpool._prune

    def prune_then_stall(self: sp.DebugLogSpool, **kwargs: int) -> None:
        real_prune(self, **kwargs)
        if self is first:
            pruned.set()
            time.sleep(0.3)  # paused between accounting and publication

    monkeypatch.setattr(sp.DebugLogSpool, "_prune", prune_then_stall)
    batch = [{"message": f"{i:03d}" + "x" * 180, "attributes": {}} for i in range(100)]
    stalled = threading.Thread(target=first.write, args=(batch,))
    stalled.start()
    assert pruned.wait(timeout=5)
    assert second.write(batch) == 0  # the lock wait expired: nothing published
    stalled.join(timeout=10)

    total = sum(p.stat().st_size for p in spool_files(tmp_path))
    assert 0 < total <= sp.MAX_TOTAL_BYTES
