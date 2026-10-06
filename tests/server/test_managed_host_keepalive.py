"""Tests for the managed-path sandbox keepalive.

Covers the resolution chain (runner -> session -> host -> provider), the
per-runner rate limit, the two skip paths (provider can't extend, host has no
sandbox), and the worker pool: one runner's stalled provider call must not starve
other runners. Stubs stand in for the stores/deployment: the module only reads a
few attributes off each, so a real store would add setup without adding cover.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast

import pytest

from omnigent.onboarding.sandboxes.base import SandboxCapabilityError
from omnigent.server import managed_host_keepalive


class _Launcher:
    def __init__(self, raises: BaseException | None = None, returns: object = None) -> None:
        self.calls: list[str] = []
        self._raises = raises
        self._returns = returns

    def keep_alive(self, sandbox_id: str) -> object:
        self.calls.append(sandbox_id)
        if self._raises is not None:
            raise self._raises
        return self._returns


def _record_submission(submitted: list[str], *args: object) -> Future[None]:
    """Record a real executor submission and return its concrete Future type."""
    submitted.append(cast(str, args[-2]))
    return Future()


def _outcomes(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The managed_keepalive outcomes logged so far, in order."""
    return [
        record.attributes["outcome"]
        for record in caplog.records
        if isinstance(getattr(record, "attributes", None), dict) and "outcome" in record.attributes
    ]


class _FakeClock:
    """time module stand-in: a settable monotonic clock, everything else real."""

    def __init__(self, now: float) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> object:
        return getattr(time, name)


class _DeferredExecutor:
    """Holds submitted jobs so a test can run them inline at a chosen clock time."""

    def __init__(self) -> None:
        self.jobs: list[tuple[Callable[..., None], tuple[object, ...], Future[None]]] = []

    def submit(self, fn: Callable[..., None], *args: object) -> Future[None]:
        future: Future[None] = Future()
        self.jobs.append((fn, args, future))
        return future

    def run_next(self) -> None:
        """Run the oldest job the way a worker would, then complete its future."""
        fn, args, future = self.jobs.pop(0)
        try:
            fn(*args)
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(None)


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    *,
    launcher: _Launcher,
    host: object | None,
    host_id: str | None = "host1",
) -> None:
    """Point the module at stub stores returning one session on *host_id*."""
    conversations = SimpleNamespace(
        list_conversations_by_runner_id=lambda _rid: [SimpleNamespace(host_id=host_id)]
    )
    hosts = SimpleNamespace(get_host=lambda _hid: host)
    deployment = SimpleNamespace(
        for_provider=lambda _provider: SimpleNamespace(launcher_factory=lambda: launcher)
    )
    monkeypatch.setattr(managed_host_keepalive, "_conversation_store", conversations)
    monkeypatch.setattr(managed_host_keepalive, "_host_store", hosts)
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", deployment)


def test_extends_the_hosts_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _Launcher()
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="modal"),
    )
    managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == ["sbx1"]


def test_provider_without_keep_alive_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # kubernetes today: the base class raises, and that must not propagate.
    launcher = _Launcher(raises=SandboxCapabilityError("nope"))
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="kubernetes"),
    )
    with caplog.at_level(logging.DEBUG, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == ["sbx1"]  # attempted, error swallowed
    assert _outcomes(caplog) == ["unsupported"]


def test_store_failure_never_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_rid: str) -> list[object]:
        raise RuntimeError("db down")

    monkeypatch.setattr(
        managed_host_keepalive,
        "_conversation_store",
        SimpleNamespace(list_conversations_by_runner_id=_boom),
    )
    monkeypatch.setattr(
        managed_host_keepalive, "_host_store", SimpleNamespace(get_host=lambda _h: None)
    )
    monkeypatch.setattr(
        managed_host_keepalive, "_sandbox_config", SimpleNamespace(for_provider=lambda _p: None)
    )
    managed_host_keepalive._keep_alive_for_runner("r1")  # must not raise


def test_cli_host_without_a_sandbox_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    launcher = _Launcher()
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id=None, sandbox_provider=None),
    )
    with caplog.at_level(logging.DEBUG, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == []
    assert _outcomes(caplog) == ["no_sandbox"]


def test_a_host_row_that_no_longer_exists_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    launcher = _Launcher()
    _wire(monkeypatch, launcher=launcher, host=None)
    with caplog.at_level(logging.DEBUG, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == []
    assert _outcomes(caplog) == ["no_host"]


def test_touch_is_rate_limited_per_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[str] = []
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    # touch() submits ctx.run(job, runner_id, tick), so the runner id is args[-2].
    monkeypatch.setattr(
        managed_host_keepalive,
        "_executor",
        SimpleNamespace(submit=lambda *args: _record_submission(submitted, *args)),
    )
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())

    managed_host_keepalive.touch("r1")
    managed_host_keepalive._inflight.discard("r1")  # first attempt finished
    managed_host_keepalive.touch("r1")  # inside the window: dropped by the throttle
    managed_host_keepalive.touch("r2")  # different runner: allowed
    assert submitted == ["r1", "r2"]


def test_touch_is_a_noop_without_a_sandbox_config(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[str] = []
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", None)
    monkeypatch.setattr(
        managed_host_keepalive,
        "_executor",
        SimpleNamespace(submit=lambda *args: _record_submission(submitted, *args)),
    )
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())
    managed_host_keepalive.touch("r1")
    assert submitted == []


def test_worker_runs_inside_the_callers_workspace_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The regression test for the real bug: the resolution chain reads stores that
    filter on `current_workspace_id()`, so the worker MUST inherit the caller's
    workspace ContextVar. A bare `submit` resolves it to the default workspace
    (0), matching no rows, and the sandbox is never extended.
    """
    from concurrent.futures import ThreadPoolExecutor

    from omnigent.db.db_models import current_workspace_id, workspace_scope

    seen: list[int] = []

    def _record(_rid: str) -> None:
        seen.append(current_workspace_id())

    monkeypatch.setattr(managed_host_keepalive, "_keep_alive_for_runner", _record)
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())
    with ThreadPoolExecutor(max_workers=1) as pool:
        monkeypatch.setattr(managed_host_keepalive, "_executor", pool)
        with workspace_scope(4242):
            managed_host_keepalive.touch("r1")
        pool.shutdown(wait=True)

    assert seen == [4242], "worker did not inherit the caller's workspace scope"


def test_a_host_on_an_unoffered_provider_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    Never extend through another provider's launcher. `recorded()` would fall
    back to the deployment default here, pushing a deadline on the wrong backend
    with a foreign sandbox id; `for_provider()` returns None and we skip.
    """
    launcher = _Launcher()
    conversations = SimpleNamespace(
        list_conversations_by_runner_id=lambda _rid: [SimpleNamespace(host_id="host1")]
    )
    hosts = SimpleNamespace(
        get_host=lambda _hid: SimpleNamespace(sandbox_id="sbx1", sandbox_provider="modal")
    )
    # Deployment no longer offers 'modal'. A default-returning resolver would
    # hand back some other provider's config; for_provider says None.
    deployment = SimpleNamespace(for_provider=lambda provider: None)
    monkeypatch.setattr(managed_host_keepalive, "_conversation_store", conversations)
    monkeypatch.setattr(managed_host_keepalive, "_host_store", hosts)
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", deployment)

    with caplog.at_level(logging.DEBUG, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == []
    assert _outcomes(caplog) == ["provider_unavailable"]


@pytest.mark.parametrize(
    ("failing_step", "outcome"),
    [("get_host", "resolution_error"), ("for_provider", "provider_error")],
)
def test_a_failing_host_does_not_block_the_runners_other_hosts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failing_step: str,
    outcome: str,
) -> None:
    """A failing host is recorded and the runner's other host still refreshes."""
    launcher = _Launcher()
    conversations = SimpleNamespace(
        list_conversations_by_runner_id=lambda _rid: [
            SimpleNamespace(host_id="host-bad"),
            SimpleNamespace(host_id="host-ok"),
        ]
    )

    def _get_host(host_id: str) -> object:
        if host_id == "host-bad" and failing_step == "get_host":
            raise RuntimeError("store down")
        provider = "broken" if host_id == "host-bad" else "modal"
        return SimpleNamespace(
            sandbox_id=host_id.replace("host-", "sbx-"), sandbox_provider=provider
        )

    def _for_provider(provider: str) -> object:
        if provider == "broken":
            raise RuntimeError("provider config exploded")
        return SimpleNamespace(launcher_factory=lambda: launcher)

    monkeypatch.setattr(managed_host_keepalive, "_conversation_store", conversations)
    monkeypatch.setattr(managed_host_keepalive, "_host_store", SimpleNamespace(get_host=_get_host))
    monkeypatch.setattr(
        managed_host_keepalive, "_sandbox_config", SimpleNamespace(for_provider=_for_provider)
    )
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})

    with caplog.at_level(logging.WARNING, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == ["sbx-ok"]
    event = next(
        r for r in caplog.records if getattr(r, "attributes", {}).get("outcome") == outcome
    )
    assert event.attributes["host_id"] == "host-bad"
    assert event.attributes["error_type"] == "RuntimeError"
    assert event.exc_info is not None


def test_a_runner_already_in_flight_is_not_queued_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled provider must not stack a second job for the same runner."""
    submitted: list[str] = []
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    monkeypatch.setattr(
        managed_host_keepalive,
        "_executor",
        SimpleNamespace(submit=lambda *args: _record_submission(submitted, *args)),
    )
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())

    managed_host_keepalive.touch("r1")
    # Past the throttle window, but the first attempt has not finished.
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    managed_host_keepalive.touch("r1")
    assert submitted == ["r1"]

    # Once it clears, the next tick submits again.
    managed_host_keepalive._inflight.discard("r1")
    managed_host_keepalive.touch("r1")
    assert submitted == ["r1", "r1"]


def test_helper_cannot_release_a_new_reservation_at_worker_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reservation remains held until the worker wrapper's cleanup completes."""
    helper_returned = threading.Event()
    allow_worker_return = threading.Event()
    helper_calls = 0
    worker_waits: list[bool] = []
    launcher = _Launcher()
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="modal"),
    )
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-managed-keepalive")
    monkeypatch.setattr(managed_host_keepalive, "_executor", pool)
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())
    original_helper = managed_host_keepalive._keep_alive_for_runner

    def _pause_after_helper(runner_id: str) -> None:
        nonlocal helper_calls
        helper_calls += 1
        original_helper(runner_id)
        helper_returned.set()
        worker_waits.append(allow_worker_return.wait(timeout=2.0))

    monkeypatch.setattr(managed_host_keepalive, "_keep_alive_for_runner", _pause_after_helper)
    try:
        managed_host_keepalive.touch("r1")
        assert helper_returned.wait(timeout=1.0)
        with managed_host_keepalive._state_lock:
            assert "r1" in managed_host_keepalive._inflight
            managed_host_keepalive._last_kept.clear()

        # A second tick is suppressed while the first worker still owns the
        # reservation.
        managed_host_keepalive.touch("r1")
        assert helper_calls == 1
    finally:
        allow_worker_return.set()
        pool.shutdown(wait=True)
    assert worker_waits == [True], "the worker timed out waiting for the test to release it"


def test_configure_builds_the_bounded_worker_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure() builds the production pool at the fixed bound, not a single worker."""
    pool_sizes: list[int] = []
    shutdowns: list[bool] = []

    class _RecordingPool:
        def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
            pool_sizes.append(max_workers)

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            shutdowns.append(cancel_futures)

    monkeypatch.setattr(managed_host_keepalive, "ThreadPoolExecutor", _RecordingPool)
    for name in ("_conversation_store", "_host_store", "_sandbox_config", "_executor"):
        monkeypatch.setattr(managed_host_keepalive, name, None)
    managed_host_keepalive.configure(object(), object(), object())
    assert pool_sizes == [managed_host_keepalive._KEEPALIVE_MAX_WORKERS] == [8]

    # Switching managed sandboxes off releases the pool instead of leaving idle workers.
    managed_host_keepalive.configure(object(), object(), None)
    assert managed_host_keepalive._executor is None
    assert shutdowns == [True]


def test_configure_without_sandboxes_shuts_a_busy_pool_down(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """configure(None) releases the real pool without deadlocking on its cancellation callbacks."""
    started = threading.Event()
    release = threading.Event()
    waits: list[bool] = []

    def _block(_rid: str) -> None:
        started.set()
        waits.append(release.wait(timeout=10))

    monkeypatch.setattr(managed_host_keepalive, "_KEEPALIVE_MAX_WORKERS", 1)
    monkeypatch.setattr(managed_host_keepalive, "_keep_alive_for_runner", _block)
    for name in ("_conversation_store", "_host_store", "_sandbox_config", "_executor"):
        monkeypatch.setattr(managed_host_keepalive, name, None)
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())
    managed_host_keepalive.configure(object(), object(), object())
    pool = managed_host_keepalive._executor
    assert pool is not None
    try:
        managed_host_keepalive.touch("busy")
        assert started.wait(timeout=2.0)
        managed_host_keepalive.touch("queued")  # waits behind the only worker
        assert {"busy", "queued"} <= managed_host_keepalive._inflight

        done = threading.Event()

        def _disable() -> None:
            managed_host_keepalive.configure(object(), None, None)
            done.set()

        with caplog.at_level(logging.DEBUG, logger="omnigent.server.managed_host_keepalive"):
            threading.Thread(target=_disable, daemon=True).start()
            assert done.wait(timeout=5.0), "configure() deadlocked while cancelling the queued job"
        assert managed_host_keepalive._executor is None
        assert "queued" not in managed_host_keepalive._inflight
        assert _outcomes(caplog) == ["cancelled"]
    finally:
        release.set()
        pool.shutdown(wait=True)
    assert waits == [True]


def test_a_tick_one_interval_after_touch_is_not_throttled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The tunnel loop sleeps one interval after touch(), so the throttle must count
    from that tick. Stamping the worker's later start instead drops the next tick
    and doubles the refresh gap, past the 2x-interval shutdown-window floor.
    """
    clock = _FakeClock(1000.0)
    executor = _DeferredExecutor()
    monkeypatch.setattr(managed_host_keepalive, "time", clock)
    monkeypatch.setattr(managed_host_keepalive, "_executor", executor)
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    monkeypatch.setattr(managed_host_keepalive, "_keep_alive_for_runner", lambda _rid: None)
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {"r1": 60.0})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())

    managed_host_keepalive.touch("r1")
    assert len(executor.jobs) == 1
    clock.now += 0.005  # a worker picks the job up a few ms after the tick
    executor.run_next()
    assert "r1" not in managed_host_keepalive._inflight

    clock.now = 1060.001  # the loop wakes one interval (plus overshoot) later
    managed_host_keepalive.touch("r1")
    assert len(executor.jobs) == 1, (
        "the throttle counted from the worker start and dropped a due tick"
    )


def test_submission_failure_releases_the_runner_reservation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed submit must not permanently suppress later refresh attempts."""
    submitted: list[object] = []

    class _RejectingExecutor:
        def submit(self, *_args: object) -> None:
            raise RuntimeError("test submission failure")

    monkeypatch.setattr(managed_host_keepalive, "_executor", _RejectingExecutor())
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())

    with caplog.at_level(logging.WARNING, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive.touch("r1")
    assert "r1" not in managed_host_keepalive._inflight
    assert "r1" not in managed_host_keepalive._last_kept
    assert any(
        getattr(record, "attributes", {}).get("outcome") == "submission_failed"
        for record in caplog.records
    )
    submission_event = next(
        record
        for record in caplog.records
        if getattr(record, "attributes", {}).get("outcome") == "submission_failed"
    )
    assert submission_event.attributes["error_type"] == "RuntimeError"
    assert submission_event.exc_info is not None

    monkeypatch.setattr(
        managed_host_keepalive,
        "_executor",
        SimpleNamespace(submit=lambda *args: _record_submission(submitted, *args)),
    )
    managed_host_keepalive.touch("r1")
    assert submitted == ["r1"]


def _wire_scheduler(
    monkeypatch: pytest.MonkeyPatch, *, interval_s: dict[str, float] | None = None
) -> tuple[_FakeClock, _DeferredExecutor]:
    """Point touch() at a controllable clock and executor with empty scheduling state."""
    clock = _FakeClock(1000.0)
    executor = _DeferredExecutor()
    monkeypatch.setattr(managed_host_keepalive, "time", clock)
    monkeypatch.setattr(managed_host_keepalive, "_executor", executor)
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", dict(interval_s or {}))
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())
    return clock, executor


def test_cancelled_job_releases_the_runner_reservation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Cancelling an accepted but unstarted job frees the runner for the next tick."""
    clock, executor = _wire_scheduler(monkeypatch, interval_s={"r1": 600.0})

    managed_host_keepalive.touch("r1")
    assert "r1" in managed_host_keepalive._inflight
    clock.now += 2.5  # the job sat in the queue until the pool was shut down
    with caplog.at_level(logging.DEBUG, logger="omnigent.server.managed_host_keepalive"):
        assert executor.jobs[0][2].cancel()
    assert "r1" not in managed_host_keepalive._inflight
    assert "r1" not in managed_host_keepalive._last_kept
    assert "r1" not in managed_host_keepalive._runner_interval_s  # no orphaned cadence entry
    event = next(
        r for r in caplog.records if getattr(r, "attributes", {}).get("outcome") == "cancelled"
    )
    assert event.attributes["queue_delay_s"] == 2.5  # the real wait, not a hardcoded zero

    managed_host_keepalive.touch("r1")  # retry is eligible immediately
    assert len(executor.jobs) == 2


def test_a_crashed_job_releases_once_and_keeps_a_newer_reservation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A crashed job releases its own reservation; its late callback leaves a newer one alone."""
    clock, executor = _wire_scheduler(monkeypatch, interval_s={"r1": 60.0})

    def _crash(_rid: str) -> None:
        raise RuntimeError("worker crashed")

    monkeypatch.setattr(managed_host_keepalive, "_keep_alive_for_runner", _crash)
    managed_host_keepalive.touch("r1")
    fn, args, stale_future = executor.jobs.pop(0)
    with pytest.raises(RuntimeError):
        fn(*args)
    assert "r1" not in managed_host_keepalive._inflight  # the job's own cleanup ran
    assert "r1" in managed_host_keepalive._last_kept  # the tick still counts for the throttle

    clock.now += 60.0  # the next tick is due and reserves again
    managed_host_keepalive.touch("r1")
    assert "r1" in managed_host_keepalive._inflight
    with caplog.at_level(logging.WARNING, logger="omnigent.server.managed_host_keepalive"):
        stale_future.set_exception(RuntimeError("worker crashed"))
    assert "r1" in managed_host_keepalive._inflight, (
        "a stale callback released a newer reservation"
    )
    event = next(
        r
        for r in caplog.records
        if getattr(r, "attributes", {}).get("outcome") == "worker_crashed"
    )
    assert event.attributes["error_type"] == "RuntimeError"
    assert event.exc_info is not None


def test_a_refresh_still_in_flight_at_the_next_tick_is_logged_as_stalled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A hung provider call shows up as a warning on each later tick instead of silently."""
    clock, executor = _wire_scheduler(monkeypatch, interval_s={"r1": 60.0})

    managed_host_keepalive.touch("r1")  # the job is accepted but never finishes
    clock.now += 60.0
    with caplog.at_level(logging.WARNING, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive.touch("r1")
    assert len(executor.jobs) == 1  # no duplicate job for the same runner
    assert "r1" in managed_host_keepalive._last_kept  # the next tick will retry
    assert _outcomes(caplog) == ["stalled"]
    assert "60s after its tick" in caplog.records[0].getMessage()


def test_worker_releases_reservation_when_the_provider_raises(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The wrapper releases a failed provider reservation and keeps the traceback."""
    launcher = _Launcher(raises=RuntimeError("boom"))
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="modal"),
    )
    monkeypatch.setattr(managed_host_keepalive, "_inflight", {"r1"})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    with caplog.at_level(logging.WARNING, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive._run_keepalive_job("r1", time.monotonic())
    assert "r1" not in managed_host_keepalive._inflight
    error_events = [
        record
        for record in caplog.records
        if getattr(record, "attributes", {}).get("outcome") == "provider_error"
    ]
    assert error_events
    assert all("boom" not in record.getMessage() for record in error_events)
    assert error_events[0].attributes["error_type"] == "RuntimeError"
    assert error_events[0].exc_info is not None


def test_keepalive_interval_is_provider_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    agent_sandbox refreshes fast (its window is short); other providers keep the
    cheap default so lowering agent_sandbox's cadence does not multiply their
    write load. An explicit env override wins for both.
    """
    from omnigent.onboarding.sandboxes.base import resolve_managed_keepalive_interval_s

    monkeypatch.delenv("OMNIGENT_MANAGED_KEEPALIVE_INTERVAL_S", raising=False)
    assert resolve_managed_keepalive_interval_s("agent_sandbox") == 60.0
    assert resolve_managed_keepalive_interval_s("modal") == 600.0
    assert resolve_managed_keepalive_interval_s() == 600.0
    monkeypatch.setenv("OMNIGENT_MANAGED_KEEPALIVE_INTERVAL_S", "15")
    assert resolve_managed_keepalive_interval_s("agent_sandbox") == 15.0
    assert resolve_managed_keepalive_interval_s("modal") == 15.0
    # A finite-but-huge override is clamped to the max, not passed through where
    # it would overflow the window-floor math (ceil(2 * interval)).
    monkeypatch.setenv("OMNIGENT_MANAGED_KEEPALIVE_INTERVAL_S", "1e308")
    assert resolve_managed_keepalive_interval_s("agent_sandbox") == 3600.0


def test_successful_keepalive_logs_at_info_on_the_server_logger(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    The keepalive INFO is emitted from the server layer (this module), whose
    logger surfaces in the server log — unlike the onboarding-layer launcher.
    """
    launcher = _Launcher()
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="agent_sandbox"),
    )
    with caplog.at_level(logging.INFO, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive._keep_alive_for_runner("r1")
    assert any("kept managed sandbox sbx1 alive" in r.getMessage() for r in caplog.records)
    event = next(
        r for r in caplog.records if getattr(r, "attributes", {}).get("outcome") == "extended"
    )
    assert event.attributes["runner_id"] == "r1"
    assert event.attributes["host_id"] == "host1"
    assert event.attributes["provider"] == "agent_sandbox"


def test_soft_failed_keepalive_suppresses_the_success_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """
    When keep_alive returns False (attempted but not confirmed; the provider
    logged its own warning), the server loop must NOT log a success line, so the
    observability signal is never self-contradictory.
    """
    launcher = _Launcher(returns=False)
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="agent_sandbox"),
    )
    with caplog.at_level(logging.INFO, logger="omnigent.server.managed_host_keepalive"):
        managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == ["sbx1"]  # attempted
    assert not any("kept managed sandbox" in r.getMessage() for r in caplog.records)
    event = next(
        r for r in caplog.records if getattr(r, "attributes", {}).get("outcome") == "soft_failed"
    )
    assert event.attributes["runner_id"] == "r1"
    assert event.attributes["host_id"] == "host1"
    assert event.attributes["provider"] == "agent_sandbox"
    assert event.attributes["error_type"] == "soft_failure"


def test_worker_evidence_contains_queue_and_provider_duration(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Worker events report how long the job queued and how long the provider call took."""
    clock = _FakeClock(1000.0)
    monkeypatch.setattr(managed_host_keepalive, "time", clock)

    class _SlowLauncher(_Launcher):
        def keep_alive(self, sandbox_id: str) -> object:
            clock.now += 0.5  # the provider call takes half a second
            return super().keep_alive(sandbox_id)

    _wire(
        monkeypatch,
        launcher=_SlowLauncher(),
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="agent_sandbox"),
    )
    executor = _DeferredExecutor()
    monkeypatch.setattr(managed_host_keepalive, "_executor", executor)
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())

    managed_host_keepalive.touch("r1")
    clock.now += 0.25  # the job waits a quarter second for a free worker
    with caplog.at_level(logging.INFO, logger="omnigent.server.managed_host_keepalive"):
        executor.run_next()
    event = next(
        r for r in caplog.records if getattr(r, "attributes", {}).get("outcome") == "extended"
    )
    assert event.attributes["queue_delay_s"] == 0.25
    assert event.attributes["provider_duration_s"] == 0.5


def test_keepalive_interval_caches_the_runners_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Before the runner's provider is known, the loop/throttle use the fast
    agent_sandbox cadence (never under-refresh a short window); once
    _keep_alive_for_runner resolves the provider, the runner's own cadence is
    cached and returned.
    """
    monkeypatch.delenv("OMNIGENT_MANAGED_KEEPALIVE_INTERVAL_S", raising=False)
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    # unknown runner -> fast agent_sandbox default, so a short window is safe
    assert managed_host_keepalive.keepalive_interval_s("r1") == 60.0
    _wire(
        monkeypatch,
        launcher=_Launcher(),
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="modal"),
    )
    managed_host_keepalive._keep_alive_for_runner("r1")
    # now cached at modal's slower cadence
    assert managed_host_keepalive.keepalive_interval_s("r1") == 600.0


class _BlockingLauncher:
    """keep_alive stalls on *blocked* until *release* is set; others return at once."""

    def __init__(self, *, blocked: str, release: threading.Event) -> None:
        self.blocked = blocked
        self.release = release
        self.started: list[str] = []
        self.timed_out = False
        self._lock = threading.Lock()

    def keep_alive(self, sandbox_id: str) -> None:
        with self._lock:
            self.started.append(sandbox_id)
        if sandbox_id == self.blocked and not self.release.wait(timeout=30):
            self.timed_out = True


def _wait_for(predicate: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def test_a_stalled_keepalive_does_not_block_other_runners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Runs touch() against the worker pool configure() really builds. While r1's
    provider call is stalled, r2 (a different runner, host and sandbox) must
    still get its keep_alive started.
    """
    release = threading.Event()
    launcher = _BlockingLauncher(blocked="sbx-r1", release=release)
    conversations = SimpleNamespace(
        list_conversations_by_runner_id=lambda rid: [SimpleNamespace(host_id=f"host-{rid}")]
    )
    hosts = SimpleNamespace(
        get_host=lambda hid: SimpleNamespace(
            sandbox_id=hid.replace("host-", "sbx-"), sandbox_provider="modal"
        )
    )
    deployment = SimpleNamespace(
        for_provider=lambda _provider: SimpleNamespace(launcher_factory=lambda: launcher)
    )
    for name in ("_conversation_store", "_host_store", "_sandbox_config", "_executor"):
        monkeypatch.setattr(managed_host_keepalive, name, None)
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    managed_host_keepalive.configure(conversations, hosts, deployment)
    executor = managed_host_keepalive._executor
    assert executor is not None
    try:
        managed_host_keepalive.touch("r1")
        assert _wait_for(lambda: "sbx-r1" in launcher.started, 5.0)

        managed_host_keepalive.touch("r2")
        assert _wait_for(lambda: "sbx-r2" in launcher.started, 1.0), (
            "r2's keep_alive never started while r1's provider call was stalled"
        )
    finally:
        release.set()
        executor.shutdown(wait=True)
    assert not launcher.timed_out, "the stalled keep_alive gave up before the test released it"
