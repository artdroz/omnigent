"""The app maps backend gRPC permission denials to a handled 403.

A store backend reached over gRPC (e.g. a workspace hierarchy service behind
a channel proxy) can answer a request with ``PERMISSION_DENIED`` whose details
read ``"Received http2 header with status: 403"``. Unhandled, that RpcError
lands in the catch-all and the user gets a 500 ``internal_error`` (with a full
traceback logged per client retry) instead of an actionable 403. These tests
pin the mapping at the app layer: a permission denial becomes a handled 403
naming the resource — for pypi grpcio's ``RpcError`` and for a vendored copy
of grpc (a different class identity, matched structurally) alike — and every
other gRPC status keeps the unhandled-500 contract.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable

import httpx
import pytest
from fastapi import FastAPI

# grpcio is not a base dependency; without it there is no backend RpcError to raise.
grpc = pytest.importorskip("grpc", reason="grpcio is required to build the backend RpcError")

from omnigent.errors import ErrorCategory  # noqa: E402
from omnigent.stores.conversation_store.sqlalchemy_store import (  # noqa: E402
    SqlAlchemyConversationStore,
)

# The exact details string a proxied HTTP 403 surfaces as on the gRPC channel.
_DENIAL_DETAILS = "Received http2 header with status: 403"


@pytest.fixture
async def catchall_client(
    app: FastAPI,
    client: httpx.AsyncClient,
) -> AsyncIterator[httpx.AsyncClient]:
    """A client that keeps the response the bare-``Exception`` handler sent.

    Starlette re-raises an exception up the ASGI stack after that handler's
    response is sent (a real server has already answered the client and only
    logs it), so this transport must not turn the re-raise back into a test
    error. Depends on ``client`` for the app's setup/teardown.

    :param app: The FastAPI application under test.
    :param client: The shared client fixture, for its lifecycle handling.
    :yields: The non-raising client.
    """
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class _BackendRpcError(grpc.RpcError):
    """A gRPC error exposing the ``code()``/``details()`` of a failed call."""

    def __init__(self, status: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._status = status
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._status

    def details(self) -> str:
        return self._details


def _make_vendored_rpc_error() -> Exception:
    """A permission-denied RPC error from a *vendored* grpc.

    The deployed build vendors grpc, so the exception's class identity differs
    from pypi grpcio's ``grpc.RpcError`` — only its shape (an ancestor named
    ``RpcError`` plus the ``code()`` accessor) is shared. The mapping must
    match that shape, not the pypi class.

    :returns: The exception instance.
    """

    class _Status:
        name = "PERMISSION_DENIED"

    class RpcError(Exception):
        def code(self) -> object:
            return _Status()

        def details(self) -> str:
            return _DENIAL_DETAILS

    return RpcError("RPC terminated with PERMISSION_DENIED")


def _fail_listing_with(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    """Make every conversation listing raise *exc*.

    Stands in for a workspace-hierarchy-backed store whose backend fails the
    listing RPC: the error is raised from inside ``GET /v1/sessions``, the
    funnel every sidebar session-list query goes through.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param exc: The exception the backend call raises.
    """

    def _raise(self: SqlAlchemyConversationStore, *args: object, **kwargs: object) -> object:
        raise exc

    monkeypatch.setattr(SqlAlchemyConversationStore, "list_conversations", _raise)


def _app_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    """
    :param caplog: The capture fixture.
    :returns: The records the server's app logger emitted during the request.
    """
    return [r for r in caplog.records if r.name == "omnigent.server.app"]


@pytest.mark.parametrize(
    "make_error",
    [
        pytest.param(
            lambda: _BackendRpcError(grpc.StatusCode.PERMISSION_DENIED, _DENIAL_DETAILS),
            id="grpcio",
        ),
        pytest.param(_make_vendored_rpc_error, id="vendored"),
    ],
)
async def test_grpc_permission_denied_maps_to_403(
    catchall_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    make_error: Callable[[], Exception],
) -> None:
    """A backend ``PERMISSION_DENIED`` answers a handled 403, not a raw 500.

    The denial is an expected upstream access outcome: it answers the coded
    403 naming the resource and is logged exactly once at WARNING, with the
    traceback and audit attributes, instead of an ERROR-level ``Unhandled
    exception`` per client retry. The vendored shape must match as well: an
    isinstance check against pypi grpcio's ``grpc.RpcError`` would never fire
    for the vendored copy the deployed build ships.
    """
    _fail_listing_with(monkeypatch, make_error())
    with caplog.at_level(logging.WARNING, logger="omnigent.server.app"):
        resp = await catchall_client.get("/v1/sessions", params={"limit": 30})
    assert resp.status_code == 403
    error = resp.json()["error"]
    assert error["code"] == "upstream_permission_denied"
    # The message names the denied resource so the user has something to act on.
    assert "/v1/sessions" in error["message"]
    records = _app_records(caplog)
    assert len(records) == 1, [r.getMessage() for r in records]
    (record,) = records
    assert record.levelno == logging.WARNING
    assert record.getMessage().startswith("Upstream call denied by a backing service:")
    assert record.exc_info is not None
    assert record.attributes["code"] == "upstream_permission_denied"
    assert record.attributes["http_status"] == "403"
    assert record.attributes["error_category"] == ErrorCategory.UPSTREAM.value


@pytest.mark.parametrize(
    ("status", "details"),
    [
        pytest.param(grpc.StatusCode.UNAVAILABLE, "connection refused", id="unavailable"),
        pytest.param(grpc.StatusCode.UNAUTHENTICATED, "bad credentials", id="unauthenticated"),
    ],
)
async def test_other_grpc_errors_keep_the_500_contract(
    catchall_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    status: grpc.StatusCode,
    details: str,
) -> None:
    """A non-denial gRPC failure still surfaces as the standard 500 shape.

    Only ``PERMISSION_DENIED`` (and, elsewhere, ``CANCELLED``) are expected
    upstream outcomes. ``UNAUTHENTICATED`` is deliberately not mapped: broken
    backend credentials would 401 every user, which is an operational fault,
    not a caller access outcome. Both keep the ERROR-level unhandled signal
    with its traceback.
    """
    _fail_listing_with(monkeypatch, _BackendRpcError(status, details))
    with caplog.at_level(logging.WARNING, logger="omnigent.server.app"):
        resp = await catchall_client.get("/v1/sessions", params={"limit": 30})
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "internal_error"
    records = _app_records(caplog)
    assert len(records) == 1, [r.getMessage() for r in records]
    (record,) = records
    assert record.levelno == logging.ERROR
    assert record.getMessage().startswith("Unhandled exception:")
    assert record.exc_info is not None
    assert record.attributes["http_status"] == "500"
