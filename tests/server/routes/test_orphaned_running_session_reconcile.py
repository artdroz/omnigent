"""Lazy-on-read reconciliation of orphaned "running" sessions.

Targeted, fast coverage for the two route-layer fixes that keep a
runner-less session from being stuck as "running" forever:

* ``GET /v1/sessions`` (``routes_core.list_sessions``) settles a row that
  still reads running/waiting but whose runner is confirmed gone.
* ``POST /v1/sessions/{id}/events`` with ``stop_session``
  (``routes_events``) settles the same orphaned row instead of returning a
  false success over a still-"running" row.

Both route through :func:`reconcile_orphaned_running_status`, whose own
``failed``-sticky invariant is unit-tested directly.

The reconciliation only fires when the runner is confirmed gone from every
replica (no live tunnel here AND ``runner_last_seen`` stale past the TTL);
each facet is paired with a fresh-runner control that must be left running,
proving the grace-window guard.

A separate backstop covers the opposite case: a session stuck ``running`` whose
runner is still *alive* but whose terminal ``idle`` edge was lost. The confirmed-
gone path never fires for it, so ``list_sessions`` fires a background runner
status probe (:func:`spawn_live_runner_idle_reconcile`) that reuses the shared,
backed-off ``_probe_runner_live_status``. The runner's native-aware status read
rewrites the cached relay status — settling a lost-edge row to idle, or
confirming a still-running turn — for the next poll to serve.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from typing import cast

import httpx
import pytest

from omnigent.db.utils import generate_agent_id
from omnigent.errors import OmnigentError
from omnigent.runner.routing import RoutedRunner, RunnerRouter
from omnigent.server.routes._sessions.common import (
    _runner_status_probe_backoff,
    _runner_status_probe_inflight,
    _session_status_edge_seq,
)
from omnigent.server.routes._sessions.helpers import (
    _live_runner_probe_cooldown,
    _live_runner_reconcile_tasks,
    _session_active_response_cache,
    _session_status_cache,
    reconcile_orphaned_running_status,
    spawn_live_runner_idle_reconcile,
)
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore
from omnigent.stores.conversation_store.sqlalchemy_store import (
    SqlAlchemyConversationStore,
)


def _seed_running_session(db_uri: str, *, runner_fresh: bool) -> str:
    """Seed a session persisted as ``running`` with a bound runner.

    :param db_uri: SQLite database URI shared with the test app.
    :param runner_fresh: When ``True``, stamp ``runner_last_seen`` now so
        the runner reads reachable (alive on another replica within the
        grace window); when ``False``, leave it unset so the runner reads
        confirmed-gone.
    :returns: The seeded session/conversation id.
    """
    agent_store = SqlAlchemyAgentStore(db_uri)
    conv_store = SqlAlchemyConversationStore(db_uri)
    agent_id = generate_agent_id()
    agent_store.create(
        agent_id,
        name=f"orphan-agent-{agent_id}",
        bundle_location="test:///bundle",
    )
    conv = conv_store.create_conversation(agent_id=agent_id)
    runner_id = f"runner_{conv.id}"
    assert conv_store.set_runner_id(conv.id, runner_id)
    conv_store.set_session_live_status(conv.id, "running")
    if runner_fresh:
        conv_store.touch_runner_liveness([runner_id], int(time.time()))
    # Ensure no stale relay-cache entry: the reconciliation's suspect gate
    # is a cache MISS (the "running" came from the DB mirror, not a runner
    # this replica is actively relaying).
    _session_status_cache.pop(conv.id, None)
    return conv.id


def _cancel_pending_reconcile_tasks() -> None:
    """Cancel in-flight reconcile tasks before clearing the tracking set.

    A fire-and-forget probe outliving its test would otherwise race the next
    test's caches; cancelling first keeps the isolation clean.
    """
    for task in list(_live_runner_reconcile_tasks):
        task.cancel()
    _live_runner_reconcile_tasks.clear()


@pytest.fixture(autouse=True)
def _isolate_status_cache() -> Iterator[None]:
    """Keep the module-level relay status caches from leaking across tests."""
    status_snapshot = dict(_session_status_cache)
    response_snapshot = dict(_session_active_response_cache)
    _live_runner_probe_cooldown.clear()
    _cancel_pending_reconcile_tasks()
    _runner_status_probe_backoff.clear()
    _runner_status_probe_inflight.clear()
    _session_status_edge_seq.clear()
    yield
    _session_status_cache.clear()
    _session_status_cache.update(status_snapshot)
    _session_active_response_cache.clear()
    _session_active_response_cache.update(response_snapshot)
    _live_runner_probe_cooldown.clear()
    _cancel_pending_reconcile_tasks()
    _runner_status_probe_backoff.clear()
    _runner_status_probe_inflight.clear()
    _session_status_edge_seq.clear()


def _seed_live_idle_suspect(db_uri: str) -> tuple[str, str]:
    """Seed a live-runner lost-edge suspect.

    Persisted and relay-cached ``running`` with a fresh bound runner and no
    tracked in-flight response — a real turn's running edge always names one,
    so its absence is the "no turn behind the running" signal the probe
    confirms against the runner.

    :param db_uri: SQLite database URI shared with the test app.
    :returns: ``(session_id, runner_id)`` for the seeded row.
    """
    agent_store = SqlAlchemyAgentStore(db_uri)
    conv_store = SqlAlchemyConversationStore(db_uri)
    agent_id = generate_agent_id()
    agent_store.create(
        agent_id,
        name=f"lost-edge-agent-{agent_id}",
        bundle_location="test:///bundle",
    )
    conv = conv_store.create_conversation(agent_id=agent_id)
    runner_id = f"runner_{conv.id}"
    assert conv_store.set_runner_id(conv.id, runner_id)
    conv_store.set_session_live_status(conv.id, "running")
    conv_store.touch_runner_liveness([runner_id], int(time.time()))
    _session_status_cache[conv.id] = "running"
    _session_active_response_cache.pop(conv.id, None)
    return conv.id, runner_id


# ── reconcile_orphaned_running_status helper ───────────────────────────────


def test_reconcile_is_conditional_and_marks_scheduled_run_incomplete(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a stale running row settles, and a scheduled fire fails."""
    from omnigent.server import session_live_state

    sid = _seed_running_session(db_uri, runner_fresh=False)
    store = SqlAlchemyConversationStore(db_uri)
    completions: list[tuple[str, str, str | None]] = []

    def _record_completion(
        conversation_id: str,
        status: str,
        *,
        error_code: str | None = None,
        error: str | None = None,
    ) -> None:
        del error
        completions.append((conversation_id, status, error_code))

    monkeypatch.setattr(session_live_state, "persist_scheduled_run_completion", _record_completion)

    assert reconcile_orphaned_running_status(sid, store, int(time.time()) - 90)
    assert store.get_conversation(sid).live_status == "idle"  # type: ignore[union-attr]
    assert _session_status_cache[sid] == "idle"
    assert completions == [(sid, "failed", "incomplete")]
    assert not reconcile_orphaned_running_status(sid, store, int(time.time()) - 90)

    fresh_sid = _seed_running_session(db_uri, runner_fresh=True)
    assert not reconcile_orphaned_running_status(fresh_sid, store, int(time.time()) - 90)
    fresh = store.get_conversation(fresh_sid)
    assert fresh is not None
    assert fresh.live_status == "running"


# ── Facet 1: GET /v1/sessions settles orphaned "running" rows ───────────────


async def test_list_reconciles_orphaned_running_session(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """A persisted-running session whose runner is confirmed gone reads
    "idle" in the list, not a phantom "running"."""
    session_id = _seed_running_session(db_uri, runner_fresh=False)

    resp = await client.get("/v1/sessions")
    assert resp.status_code == 200
    item = next(s for s in resp.json()["data"] if s["id"] == session_id)
    assert item["status"] == "idle"


async def test_list_leaves_running_session_with_fresh_runner(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """A running session whose runner is fresh (alive on another replica
    within the grace window) is left running — the reconciliation must not
    fire while the runner could still be executing the turn."""
    session_id = _seed_running_session(db_uri, runner_fresh=True)

    resp = await client.get("/v1/sessions")
    assert resp.status_code == 200
    item = next(s for s in resp.json()["data"] if s["id"] == session_id)
    assert item["status"] == "running"


# ── Facet 2: stop_session settles instead of false-succeeding ───────────────


async def test_stop_reconciles_orphaned_running_session(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """Stopping a runner-less session that still reads "running" settles it
    to idle so the 2xx success is honest, not a phantom stop over a
    still-"running" row."""
    session_id = _seed_running_session(db_uri, runner_fresh=False)

    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "stop_session", "data": {}},
    )
    assert 200 <= resp.status_code < 300
    # The stop handler settles the status synchronously via the publish
    # chokepoint; assert the cache directly so this isolates the stop-path
    # fix from the list-path fix.
    assert _session_status_cache.get(session_id) == "idle"


async def test_stop_leaves_running_session_with_fresh_runner(
    client: httpx.AsyncClient,
    db_uri: str,
) -> None:
    """A stop that can't reach a still-fresh runner does NOT force the
    session idle — it might be executing on another replica within the
    grace window."""
    session_id = _seed_running_session(db_uri, runner_fresh=True)

    resp = await client.post(
        f"/v1/sessions/{session_id}/events",
        json={"type": "stop_session", "data": {}},
    )
    assert 200 <= resp.status_code < 300
    # No reconciliation fired: the relay cache was never written to idle.
    assert _session_status_cache.get(session_id) != "idle"


# ── spawn_live_runner_idle_reconcile fires the shared runner status probe ───


class _StubRunnerRouter:
    """Minimal runner router: returns a preset routed runner, or raises."""

    def __init__(self, routed: RoutedRunner | None | BaseException) -> None:
        self._routed = routed

    def rebind(self, routed: RoutedRunner) -> None:
        """Model the session being re-pinned to a different runner."""
        self._routed = routed

    def client_for_existing_conversation(self, conversation_id: str) -> RoutedRunner | None:
        del conversation_id
        if isinstance(self._routed, BaseException):
            raise self._routed
        return self._routed


def _runner_client(runner_status: str, *, counts_native_turns: bool = True) -> httpx.AsyncClient:
    """An httpx client whose GET /v1/sessions/{id} returns a fixed status.

    :param counts_native_turns: Whether the stub advertises that its status
        folds in native-pane turns. Omitted from the payload when ``False`` to
        model a runner that predates that capability.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        payload: dict[str, object] = {"status": runner_status}
        if counts_native_turns:
            payload["counts_native_turns"] = True
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(base_url="http://runner", transport=httpx.MockTransport(handler))


async def _drain_reconcile_tasks() -> None:
    """Await every spawned fire-and-forget reconcile task."""
    await asyncio.gather(*list(_live_runner_reconcile_tasks))


@pytest.mark.parametrize(
    ("runner_status", "expected_cache"),
    [
        ("idle", "idle"),  # the lost idle edge → settle the stuck row
        ("waiting", "waiting"),
        ("failed", "failed"),  # the runner owns a terminal failure → relay it
        ("running", "running"),  # a real in-flight turn → leave running
    ],
)
async def test_spawn_reconcile_relays_runner_status_to_cache(
    db_uri: str,
    runner_status: str,
    expected_cache: str,
) -> None:
    """The background probe rewrites the cached relay status from the runner's
    own native-aware status read, so a lost-edge row settles to idle while a
    genuinely running turn is confirmed and left running."""
    sid, runner_id = _seed_live_idle_suspect(db_uri)
    client = _runner_client(runner_status)
    router = cast(
        RunnerRouter,
        _StubRunnerRouter(RoutedRunner(runner_id=runner_id, client=client)),
    )
    try:
        spawn_live_runner_idle_reconcile(sid, runner_id, router)
        await _drain_reconcile_tasks()
    finally:
        await client.aclose()

    assert _session_status_cache.get(sid) == expected_cache


async def test_spawn_reconcile_keeps_running_when_runner_lacks_native_capability(
    db_uri: str,
) -> None:
    """A runner that predates native-pane turn tracking (no
    ``counts_native_turns``) must not false-settle a live row: its bare
    ``idle`` can be a false idle for a native turn whose terminal it does not
    count. The row stays running until a native-aware runner answers."""
    sid, runner_id = _seed_live_idle_suspect(db_uri)
    client = _runner_client("idle", counts_native_turns=False)
    router = cast(
        RunnerRouter,
        _StubRunnerRouter(RoutedRunner(runner_id=runner_id, client=client)),
    )
    try:
        spawn_live_runner_idle_reconcile(sid, runner_id, router)
        await _drain_reconcile_tasks()
    finally:
        await client.aclose()

    assert _session_status_cache.get(sid) == "running"


async def test_spawn_reconcile_keeps_running_when_turn_opens_during_probe(
    db_uri: str,
) -> None:
    """A turn that opens and names a response id while the probe is in flight
    is a real in-flight turn: the probe's now-stale ``idle`` must not clobber
    it, even from a native-aware runner."""
    sid, runner_id = _seed_live_idle_suspect(db_uri)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        # A new turn opens mid-probe, naming a response id.
        _session_active_response_cache[sid] = "resp_new"
        return httpx.Response(200, json={"status": "idle", "counts_native_turns": True})

    client = httpx.AsyncClient(base_url="http://runner", transport=httpx.MockTransport(handler))
    router = cast(
        RunnerRouter,
        _StubRunnerRouter(RoutedRunner(runner_id=runner_id, client=client)),
    )
    try:
        spawn_live_runner_idle_reconcile(sid, runner_id, router)
        await _drain_reconcile_tasks()
    finally:
        await client.aclose()

    assert _session_status_cache.get(sid) == "running"


async def test_spawn_reconcile_keeps_running_when_status_edge_lands_during_probe(
    db_uri: str,
) -> None:
    """A status edge during the probe (e.g. a new native ``running`` that
    carries no response id) bumps the status epoch, so the probe's now-stale
    ``idle`` is not written over the fresher running row."""
    sid, runner_id = _seed_live_idle_suspect(db_uri)

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        # A fresh edge lands while the probe awaits the runner.
        _session_status_edge_seq[sid] = _session_status_edge_seq.get(sid, 0) + 1
        return httpx.Response(200, json={"status": "idle", "counts_native_turns": True})

    client = httpx.AsyncClient(base_url="http://runner", transport=httpx.MockTransport(handler))
    router = cast(
        RunnerRouter,
        _StubRunnerRouter(RoutedRunner(runner_id=runner_id, client=client)),
    )
    try:
        spawn_live_runner_idle_reconcile(sid, runner_id, router)
        await _drain_reconcile_tasks()
    finally:
        await client.aclose()

    assert _session_status_cache.get(sid) == "running"


def test_publish_status_bumps_edge_seq_on_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A status transition bumps the per-session epoch (so an in-flight probe
    can tell a fresher edge landed); a no-op repeat of the same value does
    not."""
    from omnigent.server import session_live_state
    from omnigent.server.routes._sessions import helpers

    monkeypatch.setattr(
        session_live_state, "persist_scheduled_run_completion", lambda *a, **k: None
    )
    monkeypatch.setattr(helpers, "_publish_child_status_to_parent", lambda *a, **k: None)

    sid = "conv_edge_seq_probe"
    _session_status_cache.pop(sid, None)
    _session_status_edge_seq.pop(sid, None)

    helpers._publish_status(sid, "running", persist_live_status=False)
    first = _session_status_edge_seq.get(sid, 0)
    assert first >= 1

    # A repeat of the same value is not a transition: no bump.
    helpers._publish_status(sid, "running", persist_live_status=False)
    assert _session_status_edge_seq.get(sid, 0) == first

    # A real transition bumps again.
    helpers._publish_status(sid, "idle", persist_live_status=False)
    assert _session_status_edge_seq.get(sid, 0) == first + 1


async def test_spawn_reconcile_leaves_row_when_runner_unreachable(
    db_uri: str,
) -> None:
    """An offline runner (or one pinned to another replica) and an unpinned
    conversation both leave the cached row running: no confirmation, no
    rewrite, and the fire-and-forget task never raises out."""
    sid, runner_id = _seed_live_idle_suspect(db_uri)

    offline = cast(RunnerRouter, _StubRunnerRouter(OmnigentError("runner offline")))
    spawn_live_runner_idle_reconcile(sid, runner_id, offline)
    await _drain_reconcile_tasks()
    assert _session_status_cache.get(sid) == "running"

    # Clear the cooldown the first spawn recorded so the unpinned control is
    # actually dispatched rather than skipped as a repeat probe.
    _live_runner_probe_cooldown.clear()
    unpinned = cast(RunnerRouter, _StubRunnerRouter(None))
    spawn_live_runner_idle_reconcile(sid, runner_id, unpinned)
    await _drain_reconcile_tasks()
    assert _session_status_cache.get(sid) == "running"


async def test_spawn_reconcile_cooldown_is_keyed_to_probed_runner(
    db_uri: str,
) -> None:
    """A probe sets a per-session cooldown keyed to the probed runner: the next
    hot list poll within the window skips the still-busy runner, while a
    session rebound to a different runner is re-probed at once, on that
    replacement runner."""
    sid, runner_a = _seed_live_idle_suspect(db_uri)
    runner_b = runner_a[:-1] + ("0" if runner_a[-1] != "0" else "1")
    probes: dict[str, int] = {runner_a: 0, runner_b: 0}

    def _counting_client(runner_id: str) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            probes[runner_id] += 1
            return httpx.Response(200, json={"status": "running"})

        return httpx.AsyncClient(
            base_url=f"http://{runner_id}", transport=httpx.MockTransport(handler)
        )

    client_a = _counting_client(runner_a)
    client_b = _counting_client(runner_b)
    stub = _StubRunnerRouter(RoutedRunner(runner_id=runner_a, client=client_a))
    router = cast(RunnerRouter, stub)
    try:
        spawn_live_runner_idle_reconcile(sid, runner_a, router)
        await _drain_reconcile_tasks()
        assert probes == {runner_a: 1, runner_b: 0}
        assert _live_runner_probe_cooldown.get(sid) == runner_a

        # The same runner within the window is skipped ...
        spawn_live_runner_idle_reconcile(sid, runner_a, router)
        await _drain_reconcile_tasks()
        assert probes == {runner_a: 1, runner_b: 0}

        # ... but a rebind to a different runner re-probes that runner at once.
        stub.rebind(RoutedRunner(runner_id=runner_b, client=client_b))
        spawn_live_runner_idle_reconcile(sid, runner_b, router)
        await _drain_reconcile_tasks()
        assert probes == {runner_a: 1, runner_b: 1}
        assert _live_runner_probe_cooldown.get(sid) == runner_b
    finally:
        await client_a.aclose()
        await client_b.aclose()


# ── Facet 3: GET /v1/sessions hands lost-edge suspects to the probe ─────────


async def test_list_schedules_live_runner_reconcile_for_lost_edge(
    client: httpx.AsyncClient,
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hot list path schedules the (fire-and-forget) runner probe for a
    cache-fed running row with a fresh runner and no in-flight response, but
    not for a row that still tracks a real turn."""
    from omnigent.server.routes.sessions import routes_core

    scheduled: list[tuple[str, str]] = []
    monkeypatch.setattr(
        routes_core,
        "spawn_live_runner_idle_reconcile",
        lambda session_id, runner_id, *_: scheduled.append((session_id, runner_id)),
    )

    suspect_id, suspect_runner = _seed_live_idle_suspect(db_uri)
    busy_id, _ = _seed_live_idle_suspect(db_uri)
    # A real in-flight turn names a response id, so it is excluded from the probe.
    _session_active_response_cache[busy_id] = "resp_live"

    resp = await client.get("/v1/sessions")
    assert resp.status_code == 200

    assert (suspect_id, suspect_runner) in scheduled
    assert all(sid != busy_id for sid, _ in scheduled)
