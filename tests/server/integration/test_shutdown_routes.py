"""Authenticated host evidence and acknowledgment ordering across route boundaries."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

import httpx
import pytest
from asgiref.testing import ApplicationCommunicator
from fastapi import FastAPI

from omnigent.errors import OmnigentError
from omnigent.host.frames import (
    HostHelloFrame,
    HostShutdownFrame,
    decode_host_frame,
    encode_host_frame,
)
from omnigent.host.shutdown import ShutdownIntent
from omnigent.server import shutdown_attribution
from omnigent.server.auth import AuthProvider
from omnigent.server.host_registry import HostRegistry
from omnigent.server.routes.host_tunnel import create_host_tunnel_router
from omnigent.server.routes.hosts import create_hosts_router
from omnigent.stores.conversation_store.sqlalchemy_store import SqlAlchemyConversationStore
from omnigent.stores.host_store import HostStore
from tests.server.integration.test_host_tunnel_route import _websocket_scope

pytestmark = pytest.mark.asyncio
_HOST = "a" * 32
_RUNNER = "shutdown-route-runner"


class _Auth(AuthProvider):
    def get_user_id(self, request):
        return request.headers.get("x-test-user")


def _hello() -> HostHelloFrame:
    return HostHelloFrame(
        version="test",
        frame_protocol_version=1,
        name="test",
        runners=[_RUNNER],
        process_id="process",
        connection_id="host-connection",
    )


def _intent() -> ShutdownIntent:
    return ShutdownIntent(
        reason="user_stopped_host",
        action="host_stop",
        initiator="local_cli",
        initiator_user_id="spoofed-user",
        host_id=_HOST,
        host_process_id="process",
        host_connection_id="host-connection",
    )


@pytest.mark.parametrize("user", ["owner", "different-user", None])
async def test_http_shutdown_requires_owner_and_uses_verified_actor(
    db_uri: str, user: str | None, app: FastAPI
) -> None:
    registry = HostRegistry()
    host_store = HostStore(db_uri)
    store = SqlAlchemyConversationStore(db_uri)
    host_store.upsert_on_connect(host_id=_HOST, name="test", user_id="owner")
    conv = store.create_conversation(host_id=_HOST, workspace="/tmp", runner_id=_RUNNER)
    await shutdown_attribution.begin_connection(conv.id, _RUNNER, "runner-connection", store)
    registry.register(_HOST, AsyncMock(), _hello(), owner="owner")
    route_app = FastAPI(exception_handlers={OmnigentError: app.exception_handlers[OmnigentError]})
    route_app.include_router(
        create_hosts_router(registry, host_store, store, auth_provider=_Auth()), prefix="/v1"
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(route_app), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/v1/hosts/{_HOST}/shutdown",
                json=_intent().model_dump(),
                headers={"x-test-user": user} if user else {},
            )
        assert response.status_code == (200 if user == "owner" else 403 if user else 401)
        evidence = await shutdown_attribution.matching_shutdown(conv.id, store)
        if user == "owner":
            assert evidence is not None
            assert evidence.intent.initiator_user_id == "owner"
        else:
            assert evidence is None
    finally:
        shutdown_attribution.session_scopes.pop(conv.id, None)
        shutdown_attribution.session_shutdowns.pop(conv.id, None)


async def test_host_ack_waits_for_scoped_persistence(
    db_uri: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = HostRegistry()
    host_store = HostStore(db_uri)
    store = SqlAlchemyConversationStore(db_uri)
    conv = store.create_conversation(host_id=_HOST, workspace="/tmp", runner_id=_RUNNER)
    await shutdown_attribution.begin_connection(conv.id, _RUNNER, "runner-connection", store)
    app = FastAPI()
    app.include_router(
        create_host_tunnel_router(
            registry,
            host_store,
            conversation_store=store,
            auth_provider=_Auth(),
            local_single_user=False,
        ),
        prefix="/v1",
    )
    scope = _websocket_scope(f"/v1/hosts/{_HOST}/tunnel")
    scope["headers"] = [(b"x-test-user", b"owner")]
    communicator = ApplicationCommunicator(app, scope)
    entered, release = threading.Event(), threading.Event()
    compare = store.compare_shutdown_state

    def blocked_write(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return compare(*args, **kwargs)

    monkeypatch.setattr(store, "compare_shutdown_state", blocked_write)
    try:
        await communicator.send_input({"type": "websocket.connect"})
        assert (await communicator.receive_output(timeout=5))["type"] == "websocket.accept"
        await communicator.send_input(
            {"type": "websocket.receive", "text": encode_host_frame(_hello())}
        )
        intent = _intent()
        await communicator.send_input(
            {
                "type": "websocket.receive",
                "text": encode_host_frame(HostShutdownFrame(intent, [_RUNNER])),
            }
        )
        ack = asyncio.create_task(communicator.receive_output(timeout=5))
        assert await asyncio.to_thread(entered.wait, 5)
        assert not ack.done(), "acknowledgment must follow evidence persistence"
        release.set()
        output = await ack
        frame = decode_host_frame(output["text"])
        assert frame.shutdown_id == intent.shutdown_id
        evidence = await shutdown_attribution.matching_shutdown(conv.id, store)
        assert evidence is not None
        assert evidence.intent.initiator_user_id is None
    finally:
        release.set()
        await communicator.send_input({"type": "websocket.disconnect", "code": 1000})
        await communicator.wait(timeout=5)
        shutdown_attribution.session_scopes.pop(conv.id, None)
        shutdown_attribution.session_shutdowns.pop(conv.id, None)


@pytest.mark.parametrize("response_id", [None, "stopped-turn", "new-turn"])
async def test_external_running_invalidates_stop_only_for_a_new_response(
    client: httpx.AsyncClient, db_uri: str, response_id: str | None
) -> None:
    from omnigent.runtime import session_stream
    from omnigent.server import session_live_state
    from omnigent.server.routes._sessions.common import _session_status_cache

    store = SqlAlchemyConversationStore(db_uri)
    conv = store.create_conversation(runner_id=_RUNNER)
    await shutdown_attribution.begin_connection(conv.id, _RUNNER, "connection", store)
    scope = await shutdown_attribution.advance_lifecycle(conv.id, store, "stopped-turn")
    store.set_session_live_status(conv.id, "running")
    _session_status_cache[conv.id] = "running"
    evidence = await shutdown_attribution.record_session_shutdown(
        store.get_conversation(conv.id),
        ShutdownIntent(
            reason="user_stopped_session", action="stop_session", initiator="authenticated_user"
        ),
        store,
    )
    assert evidence is not None
    try:
        data = {"status": "running"}
        if response_id is not None:
            data["response_id"] = response_id
        response = await client.post(
            f"/v1/sessions/{conv.id}/events",
            json={"type": "external_session_status", "data": data},
        )
        assert response.status_code == 202, response.text
        assert response.json() == {"queued": False}
        current = await shutdown_attribution.matching_shutdown(conv.id, store)
        if response_id == "new-turn":
            assert current is None
            assert shutdown_attribution.session_scopes[conv.id] != scope
        else:
            assert current == evidence
            assert shutdown_attribution.session_scopes[conv.id] == scope
    finally:
        await session_live_state.drain_pending_writes()
        shutdown_attribution.session_scopes.pop(conv.id, None)
        shutdown_attribution.session_shutdowns.pop(conv.id, None)
        _session_status_cache.pop(conv.id, None)
        session_stream.close(conv.id)
