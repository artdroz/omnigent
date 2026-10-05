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
gone path never fires for it, so ``list_sessions`` probes the live runner's turn
snapshot and settles only when it confirms no in-flight turn
(:func:`reconcile_live_runner_idle_status` / ``settle_live_runner_idle_status``).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import cast

import httpx
import pytest

from omnigent.db.utils import generate_agent_id
from omnigent.errors import OmnigentError
from omnigent.runner.routing import RoutedRunner, RunnerRouter
from omnigent.server.routes._sessions.helpers import (
    _session_active_response_cache,
    _session_status_cache,
    reconcile_live_runner_idle_status,
    reconcile_orphaned_running_status,
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


@pytest.fixture(autouse=True)
def _isolate_status_cache() -> Iterator[None]:
    """Keep the module-level relay status caches from leaking across tests."""
    status_snapshot = dict(_session_status_cache)
    response_snapshot = dict(_session_active_response_cache)
    yield
    _session_status_cache.clear()
    _session_status_cache.update(status_snapshot)
    _session_active_response_cache.clear()
    _session_active_response_cache.update(response_snapshot)


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


# ── reconcile_orphaned_running_status helper ─────────────────────────────


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


# ── Facet 1: GET /v1/sessions settles orphaned "running" rows ────────────


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


# ── Facet 2: stop_session settles instead of false-succeeding ────────────


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


# ── settle_live_runner_idle_status store method ─────────────────────────────


def test_settle_live_runner_idle_status_transitions_on_runner_match(db_uri: str) -> None:
    """A running row settles to idle on a matching runner, and the transition
    is idempotent once the row is no longer running."""
    sid, runner_id = _seed_live_idle_suspect(db_uri)
    store = SqlAlchemyConversationStore(db_uri)

    assert store.settle_live_runner_idle_status(sid, runner_id)
    assert store.get_conversation(sid).live_status == "idle"  # type: ignore[union-attr]
    assert not store.settle_live_runner_idle_status(sid, runner_id)


def test_settle_live_runner_idle_status_guards_runner_and_failure(db_uri: str) -> None:
    """A rebind to a different runner, or a terminal failed status, survives:
    neither is overwritten by the probe-driven settle."""
    store = SqlAlchemyConversationStore(db_uri)

    sid, _ = _seed_live_idle_suspect(db_uri)
    assert not store.settle_live_runner_idle_status(sid, "runner_other")
    assert store.get_conversation(sid).live_status == "running"  # type: ignore[union-attr]

    failed_sid, failed_runner = _seed_live_idle_suspect(db_uri)
    store.set_session_live_status(failed_sid, "failed")
    assert not store.settle_live_runner_idle_status(failed_sid, failed_runner)
    assert store.get_conversation(failed_sid).live_status == "failed"  # type: ignore[union-attr]


# ── reconcile_live_runner_idle_status helper (runner probe is the arbiter) ──


class _StubRunnerRouter:
    """Minimal runner router: returns a preset routed runner, or raises."""

    def __init__(self, routed: RoutedRunner | None | BaseException) -> None:
        self._routed = routed

    def client_for_existing_conversation(self, conversation_id: str) -> RoutedRunner | None:
        del conversation_id
        if isinstance(self._routed, BaseException):
            raise self._routed
        return self._routed


def _runner_client(status_code: int, runner_status: str | None = None) -> httpx.AsyncClient:
    """An httpx client whose GET /v1/sessions/{id} returns a fixed snapshot."""

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        if status_code == 404:
            return httpx.Response(404, json={"error": "not_found"})
        return httpx.Response(status_code, json={"status": runner_status})

    return httpx.AsyncClient(base_url="http://runner", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    ("status_code", "runner_status", "expect_settled"),
    [
        (404, None, True),  # never initialized on the runner → no turn → settle
        (200, "idle", True),  # initialized, no active turn → settle
        (200, "running", False),  # a real in-flight turn → leave running
        (200, "failed", False),  # terminal failure owned by the runner → leave
        (500, None, False),  # inconclusive probe → leave running
    ],
)
async def test_reconcile_live_runner_idle_status_probe_outcomes(
    db_uri: str,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    runner_status: str | None,
    expect_settled: bool,
) -> None:
    """The runner's turn snapshot decides the settle: only a confirmed no-turn
    state clears the stuck row; a live turn or any uncertainty leaves it."""
    from omnigent.server import session_live_state

    sid, runner_id = _seed_live_idle_suspect(db_uri)
    store = SqlAlchemyConversationStore(db_uri)
    completions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        session_live_state,
        "persist_scheduled_run_completion",
        lambda conversation_id, status, **_: completions.append((conversation_id, status)),
    )

    client = _runner_client(status_code, runner_status)
    routed = RoutedRunner(runner_id=runner_id, client=client)
    router = cast(RunnerRouter, _StubRunnerRouter(routed))
    try:
        settled = await reconcile_live_runner_idle_status(sid, runner_id, store, router)
    finally:
        await client.aclose()

    assert settled is expect_settled
    assert store.get_conversation(sid).live_status == (  # type: ignore[union-attr]
        "idle" if expect_settled else "running"
    )
    if expect_settled:
        assert _session_status_cache[sid] == "idle"
        # The turn completed (its idle edge was lost), so the scheduled run
        # succeeded — unlike the confirmed-gone path, which marks it incomplete.
        assert completions == [(sid, "succeeded")]
    else:
        assert completions == []


async def test_reconcile_live_runner_idle_status_leaves_unconfirmable_runner(
    db_uri: str,
) -> None:
    """An offline runner (or one pinned to another replica) and an unpinned
    conversation both leave the row running: no confirmation, no settle."""
    sid, runner_id = _seed_live_idle_suspect(db_uri)
    store = SqlAlchemyConversationStore(db_uri)

    offline = cast(RunnerRouter, _StubRunnerRouter(OmnigentError("runner offline")))
    assert not await reconcile_live_runner_idle_status(sid, runner_id, store, offline)
    assert store.get_conversation(sid).live_status == "running"  # type: ignore[union-attr]

    unpinned = cast(RunnerRouter, _StubRunnerRouter(None))
    assert not await reconcile_live_runner_idle_status(sid, runner_id, store, unpinned)
    assert store.get_conversation(sid).live_status == "running"  # type: ignore[union-attr]


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
