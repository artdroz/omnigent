"""
Tests for the standalone LLM client retry logic (llms/client.py).

Covers the public ``Client().responses.create(retry=...)`` interface,
verifying that transient failures are retried with backoff and permanent
failures surface immediately. All tests are async because the client
methods are async.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from omnigent.llms.adapters.base import BaseAdapter
from omnigent.llms.adapters.openai import OpenAIAdapter, OpenAICompatibleAdapter
from omnigent.llms.client import Client
from omnigent.llms.errors import (
    ContextWindowExceededError,
    LLMErrorDetail,
    PermanentLLMError,
    RetryableLLMError,
)
from omnigent.llms.types import (
    MessageOutput,
    OutputText,
    Response,
    ResponseCompletedEvent,
    ResponseTextDeltaEvent,
)
from omnigent.spec.types import RetryPolicy

# ── Helpers ──────────────────────────────────────────────────


@dataclass
class _SleepTracker:
    """
    Tracks calls to ``asyncio.sleep`` during retry backoff.

    :param calls: List of sleep durations passed to each call.
    """

    calls: list[float]


def _make_response() -> Response:
    """
    Build a minimal ``Response`` for successful-call assertions.

    :returns: A ``Response`` with a single text output.
    """
    return Response(
        output=[MessageOutput(content=[OutputText(text="Hello")])],
        model="test-model",
    )


class _MockAdapter:
    """
    Fake adapter whose ``chat_completions`` is async and returns
    a preconfigured value or raises a preconfigured exception.

    Use ``return_value`` for a single fixed return, or
    ``side_effect`` for a list of values / exception to cycle
    through (list items are consumed in order; a bare exception is
    raised on every call).

    :param return_value: Value returned by every call when
        ``side_effect`` is ``None``.
    :param side_effect: A list of return values / exceptions, or
        a single exception raised on every call. When a list, each
        call pops the first item.
    """

    def __init__(
        self,
        *,
        return_value: Any = None,
        side_effect: list[Any] | Exception | None = None,
    ) -> None:
        """
        Initialize the mock adapter.

        :param return_value: Fixed return value for all calls.
        :param side_effect: List of return-values/exceptions or a
            single exception.
        """
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_count = 0

    async def chat_completions(self, *args: Any, **kwargs: Any) -> Any:
        """
        Async mock of ``BaseAdapter.chat_completions()``.

        :param args: Positional arguments (ignored).
        :param kwargs: Keyword arguments (ignored).
        :returns: The configured return value.
        :raises: The configured side-effect exception.
        """
        self.call_count += 1
        if self.side_effect is not None:
            if isinstance(self.side_effect, list):
                item = self.side_effect.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item
            # Single exception — raise on every call
            raise self.side_effect
        return self.return_value


def _patch_client_deps(
    monkeypatch: pytest.MonkeyPatch,
    mock_adapter: _MockAdapter | BaseAdapter,
) -> _SleepTracker:
    """
    Patch all external dependencies of ``Client().responses.create()``
    so that calls route through ``mock_adapter.chat_completions``.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param mock_adapter: A :class:`_MockAdapter` (or a real adapter)
        whose ``chat_completions`` controls call success/failure.
    :returns: A :class:`_SleepTracker` recording backoff sleep calls.
    """
    # Route model parsing to a fake routed model
    routed = MagicMock(provider="test", model="test-model")
    monkeypatch.setattr(
        "omnigent.llms.client.parse_model_string",
        lambda model: routed,
    )

    # Return the mock adapter (not OpenAIAdapter, so we hit
    # the chat_completions path instead of responses_create)
    monkeypatch.setattr(
        "omnigent.llms.client.get_adapter",
        lambda provider: mock_adapter,
    )

    # Stub the responses-to-chat conversion helpers
    monkeypatch.setattr(
        "omnigent.llms.client.responses_input_to_chat_messages",
        lambda input, instructions: [{"role": "user", "content": "test"}],
    )
    monkeypatch.setattr(
        "omnigent.llms.client.chat_response_to_response",
        lambda result: _make_response(),
    )

    # Capture retry-backoff sleep calls via the _sleep indirection
    # so the real asyncio.sleep is not patched globally.
    tracker = _SleepTracker(calls=[])

    async def _fake_sleep(duration: float) -> None:
        """
        Record the sleep duration without blocking.

        :param duration: The sleep duration in seconds.
        """
        tracker.calls.append(duration)

    monkeypatch.setattr(
        "omnigent.llms.client._sleep",
        _fake_sleep,
    )

    return tracker


def _default_create_kwargs() -> dict[str, Any]:
    """
    Minimal kwargs for ``Client().responses.create()``.

    :returns: Dict with required ``input`` and ``model`` keys.
    """
    return {
        "input": [{"role": "user", "content": "hi"}],
        "model": "test/test-model",
    }


@dataclass
class _HttpSequence:
    """
    What a mocked ``httpx`` transport observed, appended to live.

    :param requests: Every request the transport received.
    :param clients: Every ``httpx.AsyncClient`` the adapter created.
    """

    requests: list[httpx.Request] = field(default_factory=list)
    clients: list[httpx.AsyncClient] = field(default_factory=list)


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]],
) -> list[httpx.AsyncClient]:
    """
    Patch ``httpx.AsyncClient`` so every new client sends through
    ``handler`` and return the list of clients created so far.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param handler: Sync or async ``httpx.MockTransport`` handler.
    :returns: The created clients, appended to live.
    """
    clients: list[httpx.AsyncClient] = []
    real_async_client = httpx.AsyncClient

    def _factory(**kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        client = real_async_client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return clients


def _serve_http_sequence(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[httpx.Response | Exception],
) -> _HttpSequence:
    """
    Patch ``httpx.AsyncClient`` so consecutive requests receive
    ``outcomes`` in order (a response to return or an exception to
    raise).

    :param monkeypatch: Pytest monkeypatch fixture.
    :param outcomes: One entry per expected request, consumed in order.
    :returns: The requests and clients observed so far.
    """
    observed = _HttpSequence()
    pending = list(outcomes)

    def _handler(request: httpx.Request) -> httpx.Response:
        observed.requests.append(request)
        outcome = pending.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    observed.clients = _install_mock_transport(monkeypatch, _handler)
    return observed


class _InterruptedBody(httpx.AsyncByteStream):
    """
    Response body that delivers ``chunks`` and then raises ``error``,
    like a connection dropped mid-stream.

    :param chunks: Byte chunks delivered before the failure.
    :param error: Exception raised once the chunks are consumed.
    """

    def __init__(self, chunks: list[bytes], error: Exception) -> None:
        self._chunks = chunks
        self._error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        raise self._error


_SSE_EVENT_STREAM_HEADERS = {"content-type": "text/event-stream"}

_LOOPBACK_SSE_RESPONSE_HEAD = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/event-stream\r\n"
    b"Transfer-Encoding: chunked\r\n"
    b"Connection: close\r\n\r\n"
)


def _chunked(body: bytes) -> bytes:
    """
    Encode ``body`` as one complete HTTP/1.1 chunked message body.

    :param body: The payload bytes.
    :returns: The chunk, its terminator, and the final zero chunk.
    """
    return f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"


async def _read_http_request(reader: asyncio.StreamReader) -> bytes:
    """
    Read one HTTP/1.1 request (head plus ``Content-Length`` body).

    :param reader: The connection's stream reader.
    :returns: The raw request bytes.
    """
    head = await reader.readuntil(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
    return head + (await reader.readexactly(length) if length else b"")


_CHAT_COMPLETIONS_SSE_HELLO_DELTA = (
    b'data: {"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
)

_CHAT_COMPLETIONS_SSE_HELLO = (
    _CHAT_COMPLETIONS_SSE_HELLO_DELTA
    + b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    + b"data: [DONE]\n\n"
)

_RESPONSES_SSE_HELLO = (
    b'event: response.output_text.delta\ndata: {"delta":"Hello"}\n\n'
    b'event: response.completed\ndata: {"response":{"model":"test-model","output":'
    b'[{"type":"message","content":[{"type":"output_text","text":"Hello"}]}]}}\n\n'
)


def _streaming_adapter() -> OpenAICompatibleAdapter:
    """
    Build the real Chat Completions adapter the streaming tests route
    through, so the request is opened lazily like in production.

    :returns: An adapter pointed at a fake host.
    """
    return OpenAICompatibleAdapter(base_url="https://fake-host/v1")


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def retry_config() -> RetryPolicy:
    """
    A retry config with 3 total attempts (1 initial + 2 retries)
    and fast backoff for testing. ``max_retries`` is the
    "retries beyond the first attempt" — see RetryPolicy.
    """
    return RetryPolicy(
        max_retries=2,
        backoff_base_s=2.0,
        backoff_max_s=30.0,
    )


# ── Tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_without_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When ``retry=None``, the call succeeds normally without retry
    wrapping.
    """
    mock_adapter = _MockAdapter(return_value={"id": "test"})
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    result = await Client().responses.create(
        **_default_create_kwargs(),
    )

    # The response should contain the expected text from the mock
    # conversion; failure means the non-retry path is broken.
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "Hello"

    # No backoff sleeps should occur when retry is disabled.
    assert tracker.calls == []

    # Adapter should be called exactly once.
    assert mock_adapter.call_count == 1


@pytest.mark.asyncio
async def test_create_with_retry_success_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    With retry config, first-attempt success works and no backoff
    sleep occurs.
    """
    mock_adapter = _MockAdapter(return_value={"id": "test"})
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    result = await Client().responses.create(
        **_default_create_kwargs(),
        retry=retry_config,
    )

    # Successful first attempt returns the converted response.
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "Hello"

    # No backoff sleep when the first attempt succeeds.
    assert tracker.calls == []

    # Only one call to the adapter — no retries needed.
    assert mock_adapter.call_count == 1


@pytest.mark.asyncio
async def test_create_with_retry_timeout_then_success(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    Timeout on first attempt triggers a retry; second attempt
    succeeds.
    """
    # First call times out, second call succeeds
    mock_adapter = _MockAdapter(
        side_effect=[
            httpx.TimeoutException("timeout"),
            {"id": "test"},
        ],
    )
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    result = await Client().responses.create(
        **_default_create_kwargs(),
        retry=retry_config,
    )

    # The retry should recover and return a valid response.
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "Hello"

    # Exactly one backoff sleep between the failed first attempt
    # and the successful second attempt.
    assert len(tracker.calls) == 1

    # Two adapter calls total: one timeout, one success.
    assert mock_adapter.call_count == 2


@pytest.mark.asyncio
async def test_create_with_retry_http_429_then_success(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    Rate limit (429) on first attempt triggers retry; second
    attempt succeeds.
    """
    http_429 = httpx.HTTPStatusError(
        "rate limited",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(429),
    )
    mock_adapter = _MockAdapter(
        side_effect=[http_429, {"id": "test"}],
    )
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    result = await Client().responses.create(
        **_default_create_kwargs(),
        retry=retry_config,
    )

    # Recovery after 429 should produce a valid response.
    assert isinstance(result, Response)
    assert result.output[0].content[0].text == "Hello"

    # One backoff sleep between the 429 and the successful retry.
    assert len(tracker.calls) == 1

    # Two adapter calls: one 429, one success.
    assert mock_adapter.call_count == 2


@pytest.mark.parametrize(
    "first_attempt_fault",
    ["http_503", "connect_error"],
)
@pytest.mark.asyncio
async def test_create_streaming_retries_failure_at_stream_open(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
    first_attempt_fault: str,
) -> None:
    """
    A transient failure on the first streaming HTTP attempt consumes
    the configured retry budget just like the non-streaming path.

    The real adapter opens the connection only when the returned
    iterator is first consumed, so the failure surfaces during
    iteration rather than inside ``create()``; the retry policy must
    still cover it when no event has reached the caller yet.
    """
    tracker = _patch_client_deps(monkeypatch, _streaming_adapter())
    first_attempt: httpx.Response | Exception
    if first_attempt_fault == "http_503":
        first_attempt = httpx.Response(
            503,
            content=b'{"error": {"message": "upstream unavailable"}}',
        )
    else:
        first_attempt = httpx.ConnectError("connection refused")
    http = _serve_http_sequence(
        monkeypatch,
        [
            first_attempt,
            httpx.Response(
                200,
                headers=_SSE_EVENT_STREAM_HEADERS,
                content=_CHAT_COMPLETIONS_SSE_HELLO,
            ),
        ],
    )

    stream = await Client().responses.create(
        **_default_create_kwargs(),
        stream=True,
        retry=retry_config,
    )
    events = [event async for event in stream]

    assert [req.url.path for req in http.requests] == ["/v1/chat/completions"] * 2
    assert len(tracker.calls) == 1
    assert [e.delta for e in events if isinstance(e, ResponseTextDeltaEvent)] == ["Hello"]
    completed = events[-1]
    assert isinstance(completed, ResponseCompletedEvent)
    assert completed.response.output[0].content[0].text == "Hello"
    # The failed attempt's connection must not linger behind the retry.
    assert all(client.is_closed for client in http.clients)


@pytest.mark.asyncio
async def test_create_streaming_retries_body_closed_before_output_over_loopback(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    A provider that accepts the request but closes the chunked body
    before any event is a transient transport loss and is retried.

    Runs over a real loopback socket so the test pins the exception
    httpx raises for that close (``RemoteProtocolError``), which a
    mock transport cannot reproduce.
    """
    requests: list[bytes] = []
    # ``None`` closes the connection right after the headers.
    bodies: list[bytes | None] = [None, _chunked(_CHAT_COMPLETIONS_SSE_HELLO)]

    async def _serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        requests.append(await _read_http_request(reader))
        body = bodies.pop(0)
        writer.write(_LOOPBACK_SSE_RESPONSE_HEAD)
        if body is not None:
            writer.write(body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(_serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        tracker = _patch_client_deps(
            monkeypatch,
            OpenAICompatibleAdapter(base_url=f"http://127.0.0.1:{port}/v1"),
        )
        stream = await Client().responses.create(
            **_default_create_kwargs(),
            stream=True,
            retry=retry_config,
        )
        events = [event async for event in stream]
    finally:
        server.close()
        await server.wait_closed()

    assert len(requests) == 2
    assert len(tracker.calls) == 1
    assert [e.delta for e in events if isinstance(e, ResponseTextDeltaEvent)] == ["Hello"]
    assert isinstance(events[-1], ResponseCompletedEvent)


@pytest.mark.asyncio
async def test_create_streaming_retry_covers_openai_responses_path(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    The native OpenAI Responses stream is opened lazily too, so a
    transient failure at its open shares the same retry budget.
    """
    tracker = _patch_client_deps(monkeypatch, OpenAIAdapter(base_url="https://fake-host/v1"))
    http = _serve_http_sequence(
        monkeypatch,
        [
            httpx.Response(503, content=b"upstream unavailable"),
            httpx.Response(
                200,
                headers=_SSE_EVENT_STREAM_HEADERS,
                content=_RESPONSES_SSE_HELLO,
            ),
        ],
    )

    stream = await Client().responses.create(
        **_default_create_kwargs(),
        stream=True,
        retry=retry_config,
    )
    events = [event async for event in stream]

    assert [req.url.path for req in http.requests] == ["/v1/responses"] * 2
    assert len(tracker.calls) == 1
    assert [e.delta for e in events if isinstance(e, ResponseTextDeltaEvent)] == ["Hello"]
    assert isinstance(events[-1], ResponseCompletedEvent)


@pytest.mark.asyncio
async def test_create_streaming_exhausted_retries_raise_classified_error(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    When every streaming attempt fails at open, iteration raises the
    classified ``RetryableLLMError`` only after the whole budget is
    spent, and each failed connection is released.
    """
    tracker = _patch_client_deps(monkeypatch, _streaming_adapter())
    total_attempts = retry_config.max_retries + 1
    http = _serve_http_sequence(
        monkeypatch,
        [httpx.Response(503, content=b"upstream unavailable") for _ in range(total_attempts)],
    )

    stream = await Client().responses.create(
        **_default_create_kwargs(),
        stream=True,
        retry=retry_config,
    )
    with pytest.raises(RetryableLLMError) as exc_info:
        _ = [event async for event in stream]

    assert exc_info.value.code == "503"
    assert len(http.requests) == total_attempts
    assert len(tracker.calls) == retry_config.max_retries
    assert all(client.is_closed for client in http.clients)


@pytest.mark.asyncio
async def test_create_streaming_permanent_error_at_stream_open_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    A non-retryable status at stream open surfaces as
    ``PermanentLLMError`` from iteration without a second request.
    """
    tracker = _patch_client_deps(monkeypatch, _streaming_adapter())
    http = _serve_http_sequence(monkeypatch, [httpx.Response(401, content=b"bad key")])

    stream = await Client().responses.create(
        **_default_create_kwargs(),
        stream=True,
        retry=retry_config,
    )
    with pytest.raises(PermanentLLMError) as exc_info:
        _ = [event async for event in stream]

    assert exc_info.value.code == "401"
    assert len(http.requests) == 1
    assert tracker.calls == []


@pytest.mark.asyncio
async def test_create_streaming_does_not_replay_after_first_event(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    Once an event has reached the caller, a mid-stream failure
    propagates unchanged: the request is never replayed, so the
    caller cannot receive duplicated output.
    """
    tracker = _patch_client_deps(monkeypatch, _streaming_adapter())
    http = _serve_http_sequence(
        monkeypatch,
        [
            httpx.Response(
                200,
                headers=_SSE_EVENT_STREAM_HEADERS,
                stream=_InterruptedBody(
                    [_CHAT_COMPLETIONS_SSE_HELLO_DELTA],
                    httpx.ReadError("connection reset"),
                ),
            ),
        ],
    )

    stream = await Client().responses.create(
        **_default_create_kwargs(),
        stream=True,
        retry=retry_config,
    )
    delivered: list[Any] = []
    with pytest.raises(httpx.ReadError):
        async for event in stream:
            delivered.append(event)

    assert [e.delta for e in delivered if isinstance(e, ResponseTextDeltaEvent)] == ["Hello"]
    assert len(http.requests) == 1
    assert tracker.calls == []


@pytest.mark.asyncio
async def test_create_streaming_cancellation_while_opening_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    Cancelling the consumer while the first request is in flight
    propagates ``CancelledError`` instead of being treated as a
    transient failure to retry.
    """
    tracker = _patch_client_deps(monkeypatch, _streaming_adapter())
    requests: list[httpx.Request] = []
    request_started = asyncio.Event()

    async def _hang(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        request_started.set()
        await asyncio.Event().wait()
        return httpx.Response(503)

    _install_mock_transport(monkeypatch, _hang)

    stream = await Client().responses.create(
        **_default_create_kwargs(),
        stream=True,
        retry=retry_config,
    )

    async def _first_event() -> Any:
        return await anext(stream)

    consumer = asyncio.create_task(_first_event())
    await request_started.wait()
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert len(requests) == 1
    assert tracker.calls == []


@pytest.mark.asyncio
async def test_create_with_retry_permanent_error_no_retry(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    HTTP 401 raises PermanentLLMError immediately with no retry.
    """
    http_401 = httpx.HTTPStatusError(
        "unauthorized",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(401),
    )
    mock_adapter = _MockAdapter(side_effect=http_401)
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(PermanentLLMError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # Error code should reflect the HTTP status; failure means
    # _classify_error mapped to the wrong category.
    assert exc_info.value.code == "401"

    # Detail should carry the status code for diagnostics.
    assert exc_info.value.detail is not None
    assert exc_info.value.detail.status_code == 401

    # No backoff sleeps — permanent errors abort immediately.
    assert tracker.calls == []

    # Only one adapter call — no retry attempted.
    assert mock_adapter.call_count == 1


@pytest.mark.asyncio
async def test_create_with_retry_exhausted_raises(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    All attempts timeout, raising RetryableLLMError after
    exhaustion.
    """
    # All 3 attempts time out
    mock_adapter = _MockAdapter(
        side_effect=httpx.TimeoutException("timeout"),
    )
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(RetryableLLMError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # Code should be "timeout" since all failures were timeouts.
    assert exc_info.value.code == "timeout"

    # Two backoff sleeps (between attempt 1->2 and 2->3; no sleep
    # after the final failed attempt).
    assert len(tracker.calls) == 2

    # All 3 attempts should have been made before giving up.
    assert mock_adapter.call_count == 3


@pytest.mark.asyncio
async def test_create_with_retry_already_classified_reraise(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    If the adapter raises PermanentLLMError directly, it is
    re-raised without reclassification.
    """
    # Adapter raises an already-classified error
    original_error = PermanentLLMError(
        "auth failed",
        code="auth_error",
        detail=LLMErrorDetail(provider="test"),
    )
    mock_adapter = _MockAdapter(side_effect=original_error)
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(PermanentLLMError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # The exact same error object should be re-raised, not
    # wrapped in a new PermanentLLMError. Failure means
    # _execute_with_retry reclassified an already-classified
    # error.
    assert exc_info.value is original_error
    assert exc_info.value.code == "auth_error"

    # No backoff sleeps — already-classified errors bypass retry.
    assert tracker.calls == []

    # Only one adapter call — no retry for pre-classified errors.
    assert mock_adapter.call_count == 1


@pytest.mark.asyncio
async def test_create_with_retry_already_classified_retryable_reraise(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    If the adapter raises RetryableLLMError directly, it is
    re-raised without reclassification or further retries.
    """
    original_error = RetryableLLMError(
        "rate limited upstream",
        code="429",
        detail=LLMErrorDetail(status_code=429),
    )
    mock_adapter = _MockAdapter(side_effect=original_error)
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(RetryableLLMError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # Same error object re-raised, not reclassified or wrapped.
    assert exc_info.value is original_error

    # No backoff -- pre-classified RetryableLLMError is
    # immediately re-raised by the
    # ``except (PermanentLLMError, RetryableLLMError)`` clause
    # in _execute_with_retry.
    assert tracker.calls == []

    # Only one call -- no further retries for pre-classified
    # errors.
    assert mock_adapter.call_count == 1


@pytest.mark.asyncio
async def test_create_with_retry_connection_error_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    ``ConnectionError`` is a transient network failure (tunnel
    disconnect, socket reset) and must be retried.
    """
    mock_adapter = _MockAdapter(
        side_effect=ConnectionError("connection refused"),
    )
    tracker = _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(RetryableLLMError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    assert exc_info.value.code == "connection_error"
    assert "connection refused" in str(exc_info.value)

    # Retryable errors are retried — backoff sleeps must have fired.
    assert len(tracker.calls) == retry_config.max_retries

    # 1 initial + max_retries retries.
    assert mock_adapter.call_count == retry_config.max_retries + 1


@pytest.mark.parametrize(
    ("status_code", "expected_error_type"),
    [
        # 429 is in default retryable retryable_status_codes — should retry
        (429, RetryableLLMError),
        # 500 is in default retryable retryable_status_codes — should retry
        (500, RetryableLLMError),
        # 502 is in default retryable retryable_status_codes — should retry
        (502, RetryableLLMError),
        # 503 is in default retryable retryable_status_codes — should retry
        (503, RetryableLLMError),
        # 400 is NOT retryable — should be permanent
        (400, PermanentLLMError),
        # 401 is NOT retryable — should be permanent
        (401, PermanentLLMError),
        # 403 is NOT retryable — should be permanent
        (403, PermanentLLMError),
        # 404 is NOT retryable — should be permanent
        (404, PermanentLLMError),
    ],
    ids=[
        "429-retryable",
        "500-retryable",
        "502-retryable",
        "503-retryable",
        "400-permanent",
        "401-permanent",
        "403-permanent",
        "404-permanent",
    ],
)
@pytest.mark.asyncio
async def test_create_with_retry_http_status_classification(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
    status_code: int,
    expected_error_type: type,
) -> None:
    """
    HTTP status codes are classified correctly as retryable or
    permanent based on the retry config's ``retryable_status_codes`` list.
    """
    http_error = httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(status_code),
    )
    # Always fail so we can check classification
    mock_adapter = _MockAdapter(side_effect=http_error)
    _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(expected_error_type) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # The error code should match the HTTP status string.
    assert exc_info.value.code == str(status_code)

    # Detail must carry the status code for downstream
    # diagnostics.
    assert exc_info.value.detail is not None
    assert exc_info.value.detail.status_code == status_code


# ── ContextWindowExceededError tests ──────────────────


@pytest.mark.asyncio
async def test_context_overflow_openai_error_body(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    OpenAI context overflow (HTTP 400 with
    context_length_exceeded code) raises
    ContextWindowExceededError with correct token counts.
    """
    openai_body = json.dumps(
        {
            "error": {
                "code": "context_length_exceeded",
                "message": (
                    "This model's maximum context length is"
                    " 128000 tokens. However, you requested"
                    " 142000 tokens (10000 in the messages,"
                    " 132000 in the completion). Please"
                    " reduce the length of the messages or"
                    " completion."
                ),
            }
        }
    )
    http_400 = httpx.HTTPStatusError(
        "context window exceeded",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(
            400,
            content=openai_body.encode(),
            headers={"content-type": "application/json"},
        ),
    )
    mock_adapter = _MockAdapter(side_effect=http_400)
    _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(ContextWindowExceededError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # max_context_tokens must match the limit in the OpenAI
    # error message.
    assert exc_info.value.max_context_tokens == 128000, (
        "Expected max_context_tokens=128000, got"
        f" {exc_info.value.max_context_tokens}. Failure means"
        " the OpenAI error pattern regex did not match."
    )
    # actual_tokens must match the reported count from the
    # error message.
    assert exc_info.value.actual_tokens == 142000, (
        f"Expected actual_tokens=142000, got {exc_info.value.actual_tokens}."
    )
    # Code must be context_length_exceeded for downstream
    # detection.
    assert exc_info.value.code == "context_length_exceeded"


@pytest.mark.asyncio
async def test_context_overflow_anthropic_sum_pattern(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    Anthropic overflow error (N + M > limit) raises
    ContextWindowExceededError with correct token counts.
    """
    # Anthropic's "{input} + {max_tokens} > {limit}" format
    anthropic_body = "197202 + 21333 > 200000"
    http_400 = httpx.HTTPStatusError(
        "overflow",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(
            400,
            content=anthropic_body.encode(),
        ),
    )
    mock_adapter = _MockAdapter(side_effect=http_400)
    _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(ContextWindowExceededError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # In Anthropic's "a + b > limit" pattern the full request size is
    # a + b (197202 + 21333 = 218535), not just a (the prompt alone).
    assert exc_info.value.max_context_tokens == 200000, (
        f"Expected max_context_tokens=200000, got {exc_info.value.max_context_tokens}."
    )
    assert exc_info.value.actual_tokens == 218535, (
        f"Expected actual_tokens=218535 (197202+21333), got {exc_info.value.actual_tokens}."
    )


@pytest.mark.asyncio
async def test_context_overflow_anthropic_long_pattern(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    Anthropic overflow (prompt is too long: N tokens > M maximum)
    raises ContextWindowExceededError with correct token counts.
    """
    anthropic_body = "prompt is too long: 210000 tokens > 200000 maximum"
    http_400 = httpx.HTTPStatusError(
        "overflow",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(
            400,
            content=anthropic_body.encode(),
        ),
    )
    mock_adapter = _MockAdapter(side_effect=http_400)
    _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(ContextWindowExceededError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    assert exc_info.value.max_context_tokens == 200000, (
        f"Expected max=200000, got {exc_info.value.max_context_tokens}."
    )
    assert exc_info.value.actual_tokens == 210000, (
        f"Expected actual=210000, got {exc_info.value.actual_tokens}."
    )


@pytest.mark.asyncio
async def test_context_overflow_gemini_pattern(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    Gemini overflow error raises ContextWindowExceededError with
    correct token counts extracted.
    """
    gemini_body = (
        "input token count (1100000) exceeds the maximum number of tokens allowed (1048576)"
    )
    http_400 = httpx.HTTPStatusError(
        "overflow",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(
            400,
            content=gemini_body.encode(),
        ),
    )
    mock_adapter = _MockAdapter(side_effect=http_400)
    _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(ContextWindowExceededError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    assert exc_info.value.max_context_tokens == 1048576, (
        f"Expected max=1048576, got {exc_info.value.max_context_tokens}."
    )
    assert exc_info.value.actual_tokens == 1100000, (
        f"Expected actual=1100000, got {exc_info.value.actual_tokens}."
    )


@pytest.mark.asyncio
async def test_unrecognized_400_not_context_overflow(
    monkeypatch: pytest.MonkeyPatch,
    retry_config: RetryPolicy,
) -> None:
    """
    A generic HTTP 400 that doesn't match any overflow pattern
    raises PermanentLLMError (not ContextWindowExceededError).
    """
    generic_body = "invalid request: missing required 'model' field"
    http_400 = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("POST", "http://test"),
        response=httpx.Response(
            400,
            content=generic_body.encode(),
        ),
    )
    mock_adapter = _MockAdapter(side_effect=http_400)
    _patch_client_deps(monkeypatch, mock_adapter)

    with pytest.raises(PermanentLLMError) as exc_info:
        await Client().responses.create(
            **_default_create_kwargs(),
            retry=retry_config,
        )

    # Must be PermanentLLMError, NOT ContextWindowExceededError.
    # Failure means an unrelated 400 would enter the
    # compact-retry loop.
    assert not isinstance(exc_info.value, ContextWindowExceededError), (
        "Unrecognized 400 must not be classified as"
        " ContextWindowExceededError -- it would incorrectly"
        " trigger the compaction-retry loop."
    )
    assert exc_info.value.code == "400"


# ── Structured output translation (text → response_format) ──────────


class _CapturingAdapter:
    """Adapter stub that captures the ``extra`` dict passed to chat_completions.

    :param captured_extra: List to append the extra dict into on each call.
    """

    def __init__(self, captured_extra: list[dict[str, Any]]) -> None:
        self._captured = captured_extra

    async def chat_completions(
        self,
        messages: Any,
        model: str,
        tools: Any,
        stream: bool,
        extra: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Record *extra* and return a minimal chat response.

        :param messages: Chat messages (ignored).
        :param model: Model id (ignored).
        :param tools: Tool schemas (ignored).
        :param stream: Streaming flag (ignored).
        :param extra: The extra kwargs dict — this is what we capture.
        :param kwargs: Additional kwargs (ignored).
        :returns: Minimal chat completion response.
        """
        self._captured.append(dict(extra))
        return {
            "choices": [{"message": {"content": "ok"}}],
            "model": model,
        }


@pytest.mark.asyncio
async def test_text_json_schema_translated_to_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Responses API ``text`` param with ``json_schema`` format is translated
    to Chat Completions ``response_format`` for non-OpenAI adapters.

    Without this translation, the ``text`` kwarg is sent as-is in the
    Chat Completions body and rejected with 400 by providers that don't
    recognise it (e.g. Databricks). A failure here means the structured
    output schema is lost or malformed in the Chat Completions path.
    """
    from omnigent.llms.routing import RoutedModel

    captured: list[dict[str, Any]] = []
    adapter = _CapturingAdapter(captured)
    routed = RoutedModel(provider="databricks", model="test-model")

    monkeypatch.setattr("omnigent.llms.client.parse_model_string", lambda model: routed)
    monkeypatch.setattr("omnigent.llms.client.get_adapter", lambda provider: adapter)
    monkeypatch.setattr(
        "omnigent.llms.client.responses_input_to_chat_messages",
        lambda input, instructions: [{"role": "user", "content": "test"}],
    )
    monkeypatch.setattr(
        "omnigent.llms.client.chat_response_to_response",
        lambda result: _make_response(),
    )

    client = Client()
    await client.responses.create(
        input=[{"role": "user", "content": "test"}],
        model="databricks/test-model",
        text={
            "format": {
                "type": "json_schema",
                "name": "my_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"tier": {"type": "string"}},
                    "required": ["tier"],
                },
            },
        },
    )

    # The adapter should have received response_format, not text.
    assert len(captured) == 1, f"Expected 1 call, got {len(captured)}"
    extra = captured[0]
    assert "text" not in extra, (
        f"'text' should have been removed from extra, but got {extra!r}. "
        "The raw Responses API param leaked into the Chat Completions body."
    )
    assert "response_format" in extra, (
        f"Expected 'response_format' in extra, got keys: {list(extra.keys())}. "
        "The text→response_format translation did not fire."
    )
    rf = extra["response_format"]
    assert rf["type"] == "json_schema", (
        f"response_format.type should be 'json_schema', got {rf['type']!r}"
    )
    assert rf["json_schema"]["name"] == "my_schema", (
        f"Schema name not preserved: {rf['json_schema']!r}"
    )
    assert rf["json_schema"]["strict"] is True
    assert "schema" in rf["json_schema"]


@pytest.mark.asyncio
async def test_text_without_json_schema_not_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``text`` param without ``json_schema`` format passes through unchanged.

    Only ``json_schema``-typed text should be translated. Other shapes
    (e.g. ``{"format": {"type": "text"}}``) must not be mangled.
    """
    from omnigent.llms.routing import RoutedModel

    captured: list[dict[str, Any]] = []
    adapter = _CapturingAdapter(captured)
    routed = RoutedModel(provider="databricks", model="test-model")

    monkeypatch.setattr("omnigent.llms.client.parse_model_string", lambda model: routed)
    monkeypatch.setattr("omnigent.llms.client.get_adapter", lambda provider: adapter)
    monkeypatch.setattr(
        "omnigent.llms.client.responses_input_to_chat_messages",
        lambda input, instructions: [{"role": "user", "content": "test"}],
    )
    monkeypatch.setattr(
        "omnigent.llms.client.chat_response_to_response",
        lambda result: _make_response(),
    )

    client = Client()
    await client.responses.create(
        input=[{"role": "user", "content": "test"}],
        model="databricks/test-model",
        text={"format": {"type": "text"}},
    )

    assert len(captured) == 1
    extra = captured[0]
    # Non-json_schema text should not be translated — it stays absent
    # (popped from extra but no response_format injected).
    assert "response_format" not in extra
    assert "text" not in extra
