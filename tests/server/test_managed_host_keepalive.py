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


def test_provider_without_keep_alive_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # kubernetes today: the base class raises, and that must not propagate.
    launcher = _Launcher(raises=SandboxCapabilityError("nope"))
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="kubernetes"),
    )
    managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == ["sbx1"]  # attempted, error swallowed


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


def test_cli_host_without_a_sandbox_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _Launcher()
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id=None, sandbox_provider=None),
    )
    managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == []


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
    monkeypatch: pytest.MonkeyPatch,
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

    managed_host_keepalive._keep_alive_for_runner("r1")
    assert launcher.calls == []


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
        assert allow_worker_return.wait(timeout=2.0)

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


def test_stalled_runner_does_not_block_an_independent_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider stall for one runner must not starve another runner."""
    started: dict[str, threading.Event] = {
        "slow": threading.Event(),
        "healthy": threading.Event(),
    }
    release_slow = threading.Event()

    def _keep_alive(runner_id: str) -> None:
        started[runner_id].set()
        if runner_id == "slow":
            assert release_slow.wait(timeout=2.0)

    pool = ThreadPoolExecutor(
        max_workers=managed_host_keepalive._KEEPALIVE_MAX_WORKERS,
        thread_name_prefix="test-managed-keepalive",
    )
    monkeypatch.setattr(managed_host_keepalive, "_executor", pool)
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    monkeypatch.setattr(managed_host_keepalive, "_keep_alive_for_runner", _keep_alive)
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())
    try:
        managed_host_keepalive.touch("slow")
        assert started["slow"].wait(timeout=1.0)
        managed_host_keepalive.touch("healthy")
        assert started["healthy"].wait(timeout=1.0), (
            "a stalled provider occupied all keepalive capacity"
        )
    finally:
        release_slow.set()
        pool.shutdown(wait=True)


def test_configure_builds_the_bounded_worker_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure() builds the production pool at the fixed bound, not a single worker."""
    for name in ("_conversation_store", "_host_store", "_sandbox_config", "_executor"):
        monkeypatch.setattr(managed_host_keepalive, name, None)
    managed_host_keepalive.configure(object(), object(), object())
    executor = managed_host_keepalive._executor
    assert executor is not None
    try:
        assert executor._max_workers == managed_host_keepalive._KEEPALIVE_MAX_WORKERS == 8
    finally:
        executor.shutdown(wait=True)


def test_a_tick_one_interval_after_touch_is_not_throttled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The tunnel loop sleeps one interval after touch(), so the throttle must count
    from that tick. Stamping the worker's later start instead drops the next tick
    and doubles the refresh gap, past the 2x-interval shutdown-window floor.
    """
    clock = [1000.0]
    jobs: list[tuple[Callable[..., None], tuple[object, ...]]] = []

    class _DeferredExecutor:
        def submit(self, fn: Callable[..., None], *args: object) -> Future[None]:
            jobs.append((fn, args))
            return Future()

    monkeypatch.setattr(
        managed_host_keepalive, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    monkeypatch.setattr(managed_host_keepalive, "_executor", _DeferredExecutor())
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    monkeypatch.setattr(managed_host_keepalive, "_keep_alive_for_runner", lambda _rid: None)
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {"r1": 60.0})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())

    managed_host_keepalive.touch("r1")
    assert len(jobs) == 1
    clock[0] += 0.005  # a worker picks the job up a few ms after the tick
    run, args = jobs[0]
    run(*args)  # ctx.run(_run_keepalive_job, "r1", tick), inline
    assert "r1" not in managed_host_keepalive._inflight

    clock[0] = 1060.001  # the loop wakes one interval (plus overshoot) later
    managed_host_keepalive.touch("r1")
    assert len(jobs) == 2, "the throttle counted from the worker start and dropped a due tick"


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


def test_cancelled_job_releases_the_runner_reservation(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Cancelling an accepted but unstarted job frees the runner for the next tick."""
    futures: list[Future[None]] = []

    def _submit(*_args: object) -> Future[None]:
        future: Future[None] = Future()
        futures.append(future)
        return future

    monkeypatch.setattr(managed_host_keepalive, "_executor", SimpleNamespace(submit=_submit))
    monkeypatch.setattr(managed_host_keepalive, "_sandbox_config", object())
    monkeypatch.setattr(managed_host_keepalive, "_host_store", object())
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())

    managed_host_keepalive.touch("r1")
    assert "r1" in managed_host_keepalive._inflight
    with caplog.at_level(logging.DEBUG, logger="omnigent.server.managed_host_keepalive"):
        assert futures[0].cancel()
    assert "r1" not in managed_host_keepalive._inflight
    assert "r1" not in managed_host_keepalive._last_kept
    assert any(
        getattr(record, "attributes", {}).get("outcome") == "cancelled"
        for record in caplog.records
    )

    managed_host_keepalive.touch("r1")  # retry is eligible immediately
    assert len(futures) == 2


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
    """Worker events carry bounded scheduling and provider timing evidence."""

    class _SignalingLauncher(_Launcher):
        def __init__(self) -> None:
            super().__init__()
            self.started = threading.Event()

        def keep_alive(self, sandbox_id: str) -> object:
            self.started.set()
            return super().keep_alive(sandbox_id)

    launcher = _SignalingLauncher()
    _wire(
        monkeypatch,
        launcher=launcher,
        host=SimpleNamespace(sandbox_id="sbx1", sandbox_provider="agent_sandbox"),
    )
    pool = ThreadPoolExecutor(
        max_workers=managed_host_keepalive._KEEPALIVE_MAX_WORKERS,
        thread_name_prefix="test-managed-keepalive",
    )
    monkeypatch.setattr(managed_host_keepalive, "_executor", pool)
    monkeypatch.setattr(managed_host_keepalive, "_last_kept", {})
    monkeypatch.setattr(managed_host_keepalive, "_runner_interval_s", {})
    monkeypatch.setattr(managed_host_keepalive, "_inflight", set())
    try:
        with caplog.at_level(logging.INFO, logger="omnigent.server.managed_host_keepalive"):
            managed_host_keepalive.touch("r1")
            assert launcher.started.wait(timeout=1.0)
            pool.shutdown(wait=True)
        event = next(
            r for r in caplog.records if getattr(r, "attributes", {}).get("outcome") == "extended"
        )
        assert event.attributes["queue_delay_s"] >= 0
        assert event.attributes["provider_duration_s"] >= 0
    finally:
        pool.shutdown(wait=True)


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
        self._lock = threading.Lock()

    def keep_alive(self, sandbox_id: str) -> None:
        with self._lock:
            self.started.append(sandbox_id)
        if sandbox_id == self.blocked:
            assert self.release.wait(timeout=30), "test never released the stalled keep_alive"


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
