"""Regression for OMNI-10125: the sidebar progress spinner spins forever on a
session with no active work after a lost terminal-idle edge.

The sidebar badge renders a spinning ``running-dot`` purely from
``conversation.status == "running"`` in ``GET /v1/sessions``, so the stuck
spinner is exactly a stuck list status. This asserts the status, the lowest
layer that reproduces the bug.

The bug needs a *live* runner: the orphan-reconcile backstop only settles a
stuck "running" row when the runner is confirmed gone. A fresh/live runner with
no in-flight turn is left running, so a DB-seed with a fake runner (see
``tests/server/routes/test_orphaned_running_session_reconcile.py``) cannot
express it. This binds to the e2e lane's real runner instead.

The observation runs past the SPA's connected list-refetch interval (60s) to
show the stuck status does not self-clear within the window the frontend would
re-poll in.
"""

from __future__ import annotations

import time

import httpx

# Exceed the SPA's connected list refetch interval (60s) so a clear would have
# to come from a reconcile, not a stale cached response.
OBSERVE_SECONDS = 80.0


def _list_status(base_url: str, session_id: str) -> str | None:
    resp = httpx.get(f"{base_url}/v1/sessions", timeout=10.0)
    resp.raise_for_status()
    item = next((s for s in resp.json()["data"] if s["id"] == session_id), None)
    return item["status"] if item else None


def _post_lost_idle_running(base_url: str, session_id: str) -> None:
    """Post a lone ``running`` status with no following ``idle`` over the real
    external_session_status wire — a lost terminal-idle edge."""
    resp = httpx.post(
        f"{base_url}/v1/sessions/{session_id}/events",
        json={"type": "external_session_status", "data": {"status": "running"}},
        timeout=15.0,
    )
    resp.raise_for_status()


def test_list_settles_lost_idle_running_session_with_live_runner(
    seeded_session: tuple[str, str],
) -> None:
    base_url, session_id = seeded_session

    assert _list_status(base_url, session_id) in (None, "idle"), (
        "a freshly bound session with no active work should not start running"
    )

    _post_lost_idle_running(base_url, session_id)
    assert _list_status(base_url, session_id) == "running", (
        "the lost terminal-idle edge should land as a running status"
    )

    # The runner is alive and has no in-flight turn, so the list must stop
    # reporting running (the spinner must stop) rather than spin forever.
    start = time.time()
    deadline = start + OBSERVE_SECONDS
    status = "running"
    while time.time() < deadline:
        status = _list_status(base_url, session_id) or "idle"
        elapsed = time.time() - start
        print(f"[OMNI-10125] t={elapsed:5.1f}s GET /v1/sessions status={status!r}")
        if status != "running":
            break
        time.sleep(10.0)

    assert status != "running", (
        f"GET /v1/sessions still reports running after {OBSERVE_SECONDS:.0f}s "
        "(past the 60s list-refetch interval) for a live-runner session with no "
        "active work; the sidebar spinner never stops (OMNI-10125)"
    )
