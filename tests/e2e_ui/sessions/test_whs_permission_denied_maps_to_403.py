"""E2E regression: a WHS 403 (gRPC ``PERMISSION_DENIED``) on the session listing
must surface as a handled 403, not an unhandled 500.

In the Databricks embedding the conversation store lists sessions through the
workspace hierarchy service over Barnacle's gRPC channel. When that service
answers HTTP 403, the client raises ``grpc.RpcError`` with
``StatusCode.PERMISSION_DENIED`` / ``"Received http2 header with status: 403"``
from inside ``GET /v1/sessions`` -- the request the SPA sidebar issues on load.
Unhandled, it reached the catch-all: ``500 internal_error`` for the client and
"Failed to load: An internal error occurred." in the sidebar.

The real backend is Databricks-internal, so a real deny-all gRPC server stands
in for it at the same boundary; the server, SPA and HTTP path are all real.

Run::

    .venv/bin/python -m pytest \\
        tests/e2e_ui/sessions/test_whs_permission_denied_maps_to_403.py --ui-skip-build -v
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent import futures
from pathlib import Path

import httpx
import pytest
import uvicorn
from playwright.sync_api import Page, expect

# Without grpcio there is no real permission-denied RPC to raise.
grpc = pytest.importorskip("grpc", reason="grpcio is required to raise the permission-denied RPC")

from omnigent.runtime import _globals  # noqa: E402
from omnigent.runtime import init as init_runtime  # noqa: E402
from omnigent.runtime.agent_cache import AgentCache  # noqa: E402
from omnigent.server import app as app_module  # noqa: E402
from omnigent.stores.agent_store.sqlalchemy_store import SqlAlchemyAgentStore  # noqa: E402
from omnigent.stores.artifact_store.local import LocalArtifactStore  # noqa: E402
from omnigent.stores.conversation_store.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyConversationStore,
)
from omnigent.stores.file_store.sqlalchemy_store import SqlAlchemyFileStore  # noqa: E402
from tests.e2e_ui.conftest import _find_free_port  # noqa: E402

# The details a proxied HTTP 403 surfaces with on the gRPC channel, and the WHS listing RPC.
_WHS_403_DETAILS = "Received http2 header with status: 403"
_WHS_METHOD = "/whs.WorkspaceHierarchyService/ListTreeNodeChildren"
# Process-wide runtime state that init_runtime rebinds; restored after the test so
# later in-process tests do not inherit the deny-all store.
_RUNTIME_GLOBALS = (
    "_conversation_store",
    "_agent_store",
    "_agent_cache",
    "_file_store",
    "_artifact_store",
    "_comment_store",
    "_policy_store",
    "_caps",
    "_terminal_registry",
)

# The handled 403 is new server behaviour; older pinned servers answer 500.
pytestmark = pytest.mark.min_server_version("0.17.0")


class _DenyAllHandler(grpc.GenericRpcHandler):
    """A gRPC handler that denies every method with ``PERMISSION_DENIED``.

    Stands in for the workspace hierarchy service behind Barnacle answering a
    request with HTTP 403, which the gRPC layer surfaces as
    ``StatusCode.PERMISSION_DENIED`` with that details string.
    """

    def service(self, handler_call_details: object) -> object:
        def _deny(request: bytes, context: grpc.ServicerContext) -> bytes:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, _WHS_403_DETAILS)
            raise AssertionError("unreachable")  # abort() never returns

        return grpc.unary_unary_rpc_method_handler(_deny)


def _start_deny_all_grpc() -> tuple[grpc.Server, grpc.Channel]:
    """Start the deny-all gRPC server and a channel to it.

    :returns: ``(server, channel)`` -- both owned by the caller, who tears
        them down when done.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    server.add_generic_rpc_handlers((_DenyAllHandler(),))
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    return server, channel


def _wait_until_serving(base_url: str, timeout: float = 30.0) -> None:
    """Block until the app answers, so the SPA has something to load.

    Probing ``/`` (an always-served route) proves uvicorn is up without
    tripping the faulted listing funnel.

    :param base_url: Server root, e.g. ``http://127.0.0.1:12345``.
    :param timeout: Seconds to wait before giving up.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(f"{base_url}/", timeout=2.0)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise TimeoutError(f"server at {base_url} did not start within {timeout}s")


@pytest.fixture
def whs_403_server(built_spa: None, tmp_path: Path) -> Iterator[str]:
    """Run the real Omnigent app whose session listing hits a WHS 403.

    A deny-all gRPC backend stands in for the workspace hierarchy service: the
    store's ``list_conversations`` -- the funnel behind ``GET /v1/sessions`` --
    calls it, so a genuine ``grpc._channel._InactiveRpcError``
    (``PERMISSION_DENIED`` / ``"Received http2 header with status: 403"``) is
    raised inside the request.

    :param built_spa: Ensures the web SPA bundle is present on disk.
    :param tmp_path: Per-test scratch dir for the sqlite DB / artifacts.
    :yields: The base URL of the running server.
    """
    saved_globals = {name: getattr(_globals, name) for name in _RUNTIME_GLOBALS}
    grpc_server, channel = _start_deny_all_grpc()
    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None
    try:
        list_children = channel.unary_unary(
            _WHS_METHOD,
            request_serializer=lambda payload: payload,
            response_deserializer=lambda payload: payload,
        )

        class _WhsBackedConversationStore(SqlAlchemyConversationStore):
            """Conversation store whose listing funnel goes through the WHS."""

            def list_conversations(self, *args: object, **kwargs: object) -> object:
                # Raises grpc._channel._InactiveRpcError(PERMISSION_DENIED, ...),
                # the same shape whs_client.list_children raises in production.
                list_children(b"")
                raise AssertionError("unreachable")  # the RPC always raises

        db_uri = f"sqlite:///{tmp_path / 'whs403.db'}"
        artifact_store = LocalArtifactStore(str(tmp_path / "artifacts"))
        agent_store = SqlAlchemyAgentStore(db_uri)
        file_store = SqlAlchemyFileStore(db_uri)
        conversation_store = _WhsBackedConversationStore(db_uri)
        agent_cache = AgentCache(artifact_store=artifact_store, cache_dir=tmp_path / "cache")
        init_runtime(
            conversation_store=conversation_store,
            agent_store=agent_store,
            agent_cache=agent_cache,
            file_store=file_store,
            artifact_store=artifact_store,
        )
        app = app_module.create_app(
            agent_store=agent_store,
            file_store=file_store,
            conversation_store=conversation_store,
            artifact_store=artifact_store,
            agent_cache=agent_cache,
        )

        port = _find_free_port()
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        base_url = f"http://127.0.0.1:{port}"
        deadline = time.time() + 30.0
        while not server.started:
            if time.time() > deadline:
                raise TimeoutError("uvicorn did not start")
            time.sleep(0.1)
        _wait_until_serving(base_url)
        yield base_url
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10.0)
        channel.close()
        grpc_server.stop(grace=None)
        for name, value in saved_globals.items():
            setattr(_globals, name, value)


def test_whs_permission_denied_is_handled_not_500(
    request: pytest.FixtureRequest,
    whs_403_server: str,
) -> None:
    """A WHS gRPC ``PERMISSION_DENIED`` must surface as a handled 403, not a 500.

    Opens the real SPA, waits for the sidebar's failure line, checks that it
    names the denied resource and a remedy, then pins the HTTP contract:
    ``GET /v1/sessions`` answers ``403 upstream_permission_denied`` rather than
    ``500 internal_error``.

    :param request: Opens the ``page`` fixture only once the server is up, so a
        recording starts at the first navigation rather than on a blank page.
    :param whs_403_server: Base URL of the app whose listing hits the WHS 403.
    """
    page: Page = request.getfixturevalue("page")

    # 1. Drive the real user journey: open the app; the sidebar issues its
    #    session-list query on load, which hits the WHS 403.
    page.goto(f"{whs_403_server}/", wait_until="domcontentloaded")

    # The sidebar surfaces a load failure once the query settles. This renders
    # for the reproduction footage regardless of the eventual status code.
    failure = page.get_by_text("Failed to load", exact=False).first
    expect(failure).to_be_visible(timeout=30_000)
    # The user must see what was denied and what to do about it, not the
    # generic "An internal error occurred." the unhandled 500 produced.
    expect(failure).to_contain_text("/v1/sessions")
    expect(failure).to_contain_text("denied by a backing service")
    expect(failure).to_contain_text("ask an administrator")
    # Hold the failed state on screen so the recording shows what the user sees.
    page.wait_for_timeout(1_500)

    # 2. Server contract: the gRPC PERMISSION_DENIED maps to a coded 403 naming
    #    the resource, never a raw 500 internal_error.
    resp = httpx.get(
        f"{whs_403_server}/v1/sessions", params={"limit": 30, "visibility": "all"}, timeout=15.0
    )
    body = resp.json()
    error_code = body.get("error", {}).get("code")

    assert resp.status_code != 500, (
        "WHS gRPC PERMISSION_DENIED escaped as an unhandled 500 instead of "
        f"a handled 403. Body: {body!r}"
    )
    assert resp.status_code == 403, (
        f"expected a handled 403 for a workspace-hierarchy permission denial, "
        f"got {resp.status_code}. Body: {body!r}"
    )
    assert error_code == "upstream_permission_denied", (
        f"expected error code 'upstream_permission_denied', got {error_code!r}. Body: {body!r}"
    )
