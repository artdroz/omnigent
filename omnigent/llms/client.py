"""
Main LLM client — presents the OpenAI Responses API interface and
routes to provider adapters. All methods are async for non-blocking
I/O.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from omnigent.llms._responses_to_chat import (
    chat_response_to_response,
    chat_stream_to_response_events,
    responses_input_to_chat_messages,
)
from omnigent.llms._usage_observer import notify as _notify_usage
from omnigent.llms.adapters import get_adapter
from omnigent.llms.adapters.openai import OpenAIAdapter
from omnigent.llms.errors import (
    PermanentLLMError,
    RetryableLLMError,
)
from omnigent.llms.routing import parse_model_string
from omnigent.llms.types import (
    Response,
    ResponseCompletedEvent,
    ResponseStreamEvent,
)
from omnigent.runtime.llm_retry import classify_llm_error
from omnigent.spec.types import RetryPolicy
from omnigent.util.reasoning_effort import OPENAI_EFFORTS, validate_effort_or_llm_error

_logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _emit_usage_from_response(response: Response) -> None:
    usage = response.usage
    if usage is None:
        return
    _notify_usage(
        model=response.model,
        input_tokens=int(usage.input_tokens or 0),
        output_tokens=int(usage.output_tokens or 0),
        total_tokens=int(usage.total_tokens or 0),
    )


async def _aclose(stream: AsyncIterator[Any] | None) -> None:
    """
    Close ``stream`` when it supports ``aclose``, releasing the
    provider connection an adapter generator may still hold.

    Close-time errors are logged, not raised: cleanup must not
    replace the failure being handled or abort a retry.

    :param stream: The iterator to close, or ``None``.
    """
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await aclose()
    except Exception:
        _logger.debug("Ignoring error while closing LLM stream", exc_info=True)


async def _tee_stream_for_usage(
    stream: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[ResponseStreamEvent]:
    try:
        async for event in stream:
            if isinstance(event, ResponseCompletedEvent):
                _emit_usage_from_response(event.response)
            yield event
    finally:
        await _aclose(stream)


class _ResponsesNamespace:
    """
    Namespace providing ``client.responses.create()`` to mirror
    the OpenAI SDK interface.

    :param client: The parent :class:`Client` instance.
    """

    def __init__(self, client: Client) -> None:
        self._client = client

    async def create(
        self,
        *,
        input: list[dict[str, Any]],
        instructions: str | None = None,
        model: str,
        tools: list[dict[str, Any]] | None = None,
        reasoning: dict[str, str] | None = None,
        stream: bool = False,
        connection_params: dict[str, str] | None = None,
        timeout: int | None = None,
        retry: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """
        Create a response from the LLM, routing to the
        appropriate provider based on the model string.

        :param input: Responses API input items, e.g.
            ``[{"role": "user", "content": "Hello"}]``.
        :param instructions: System instructions string.
        :param model: Provider-prefixed model string, e.g.
            ``"anthropic/claude-sonnet-4-20250514"`` or
            ``"gpt-5.4"``.
        :param tools: OpenAI-format tool schemas, or ``None``.
        :param reasoning: Reasoning configuration dict, e.g.
            ``{"effort": "high", "summary": "concise"}``.
        :param stream: If ``True``, return an async iterator of
            streaming events. If ``False``, return a
            :class:`Response`.
        :param connection_params: Per-call connection overrides.
            Keys are provider-specific, e.g.
            ``{"api_key": "...", "base_url": "..."}`` for
            OpenAI-compatible providers, or
            ``{"aws_region": "us-west-2"}`` for Bedrock.
            ``None`` uses the adapter's default credentials.
        :param timeout: Request timeout in seconds. ``None``
            uses the adapter's default (120s non-streaming, 300s
            streaming).
        :param retry: Retry policy for transient failures
            (timeouts, rate limits). ``None`` disables
            client-level retries. Useful for standalone calls
            outside the workflow engine. With ``stream=True`` the
            same budget also covers failures while opening the
            stream, until the first event reaches the caller.
        :param kwargs: Additional provider-specific kwargs (e.g.
            ``temperature``, ``max_tokens``).
        :returns: A :class:`Response` when ``stream=False``, or
            an async iterator of :data:`ResponseStreamEvent`
            when ``stream=True``.
        :raises PermanentLLMError: On non-retryable errors.
        :raises RetryableLLMError: When all retry attempts are
            exhausted.
        """

        async def call_fn() -> Response | AsyncIterator[ResponseStreamEvent]:
            """
            Dispatch to the adapter.

            :returns: Response or streaming event iterator.
            """
            return await self._do_create(
                input=input,
                instructions=instructions,
                model=model,
                tools=tools,
                reasoning=reasoning,
                stream=stream,
                connection_params=connection_params,
                timeout=timeout,
                **kwargs,
            )

        if retry is None:
            result = await call_fn()
        else:
            attempts = _RetryAttempts(retry)
            result = await _execute_with_retry(call_fn, attempts)
            if not isinstance(result, Response):
                result = _retry_stream_open(result, call_fn, attempts)
        if isinstance(result, Response):
            _emit_usage_from_response(result)
            return result
        return _tee_stream_for_usage(result)

    async def _do_create(
        self,
        *,
        input: list[dict[str, Any]],
        instructions: str | None,
        model: str,
        tools: list[dict[str, Any]] | None,
        reasoning: dict[str, str] | None,
        stream: bool,
        connection_params: dict[str, str] | None,
        timeout: int | None,
        **kwargs: Any,
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        """
        Route the LLM call to the appropriate provider adapter.

        :param input: Responses API input items.
        :param instructions: System instructions string.
        :param model: Provider-prefixed model string.
        :param tools: Tool schemas or ``None``.
        :param reasoning: Reasoning config or ``None``.
        :param stream: Enable streaming.
        :param connection_params: Connection overrides or
            ``None``.
        :param timeout: Timeout in seconds or ``None``.
        :param kwargs: Additional provider-specific kwargs.
        :returns: Response or async streaming event iterator.
        """
        routed = parse_model_string(model)
        adapter = get_adapter(routed.provider)

        # OpenAI supports the Responses API natively — use it
        # directly so reasoning token events flow through
        # unmodified.
        if isinstance(adapter, OpenAIAdapter):
            if reasoning and reasoning.get("effort"):
                effort = validate_effort_or_llm_error(
                    reasoning.get("effort"), "OpenAI Responses", OPENAI_EFFORTS
                )
                if effort == "none":
                    reasoning = {"effort": effort}
            return await adapter.responses_create(
                input=input,
                instructions=instructions,
                model=routed.model,
                tools=tools,
                reasoning=reasoning,
                stream=stream,
                connection_params=connection_params,
                timeout=timeout,
                **kwargs,
            )

        messages = responses_input_to_chat_messages(
            input,
            instructions,
        )

        extra: dict[str, Any] = dict(kwargs)
        # Translate Responses API ``text`` (structured output) to the
        # Chat Completions ``response_format`` parameter. The Responses
        # API shape is ``text={"format": {"type": "json_schema", ...}}``;
        # the Chat Completions equivalent is
        # ``response_format={"type": "json_schema", "json_schema": ...}``.
        # Without this, the ``text`` kwarg is sent as-is in the Chat
        # Completions body and rejected with 400 by providers that don't
        # recognise it (e.g. Databricks).
        text_param = extra.pop("text", None)
        if isinstance(text_param, dict):
            fmt = text_param.get("format")
            if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
                extra["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {k: v for k, v in fmt.items() if k != "type"},
                }
        if reasoning:
            extra["reasoning_effort"] = reasoning.get("effort")

        if stream:
            chunks = await adapter.chat_completions(
                messages,
                routed.model,
                tools,
                True,
                extra,
                connection_params=connection_params,
                timeout=timeout,
            )
            assert not isinstance(chunks, dict)
            return chat_stream_to_response_events(
                chunks,
                model=routed.model,
            )

        result = await adapter.chat_completions(
            messages,
            routed.model,
            tools,
            False,
            extra,
            connection_params=connection_params,
            timeout=timeout,
        )
        assert isinstance(result, dict)
        return chat_response_to_response(result)


@dataclass
class _RetryAttempts:
    """
    Retry budget shared by every attempt of one ``create()`` call.

    :param config: Retry policy (max_retries, backoff, etc.).
    :param failures: Attempts that have failed so far.
    """

    config: RetryPolicy
    failures: int = 0

    async def absorb(self, exc: Exception) -> None:
        """
        Account for a failed attempt: re-raise it when it is not
        retryable or the budget is spent, otherwise sleep the backoff.

        :param exc: The exception the attempt raised.
        :raises PermanentLLMError: On non-retryable errors.
        :raises RetryableLLMError: When all retry attempts are
            exhausted.
        """
        if isinstance(exc, (PermanentLLMError, RetryableLLMError)):
            raise exc
        classified = classify_llm_error(
            exc,
            self.config.retryable_status_codes,
        )
        if isinstance(classified, PermanentLLMError):
            raise classified from exc
        self.failures += 1
        if self.failures > self.config.max_retries:
            raise classified from exc
        await _backoff_sleep(self.failures - 1, self.config)


async def _execute_with_retry(
    call_fn: Callable[[], Awaitable[_T]],
    attempts: _RetryAttempts,
) -> _T:
    """
    Execute ``call_fn`` with retry on transient failures.

    Standalone retry logic for the LLM client. Uses
    ``asyncio.sleep`` for backoff so the event loop stays free.

    :param call_fn: Zero-argument async callable that performs
        the LLM call.
    :param attempts: The retry budget to charge failures to.
    :returns: The successful result from ``call_fn``.
    :raises PermanentLLMError: On non-retryable errors.
    :raises RetryableLLMError: When all retry attempts are
        exhausted.
    """
    while True:
        try:
            return await call_fn()
        except Exception as exc:
            await attempts.absorb(exc)


async def _retry_stream_open(
    stream: AsyncIterator[ResponseStreamEvent],
    call_fn: Callable[[], Awaitable[Response | AsyncIterator[ResponseStreamEvent]]],
    attempts: _RetryAttempts,
) -> AsyncIterator[ResponseStreamEvent]:
    """
    Yield ``stream``, retrying the request while no event has
    reached the caller.

    Adapters send the HTTP request on the first ``__anext__``, so a
    failure there is charged to the same budget as iterator creation.
    After the first event nothing is replayed; later failures
    propagate unchanged.

    :param stream: The iterator returned by the first attempt.
    :param call_fn: Creates a fresh iterator for a retry.
    :param attempts: The retry budget shared with creation.
    :returns: Async iterator of streaming events.
    """
    current: AsyncIterator[ResponseStreamEvent] | None = stream
    try:
        while True:
            if current is None:
                reopened = await _execute_with_retry(call_fn, attempts)
                if isinstance(reopened, Response):
                    raise TypeError("stream=True call returned a non-streaming Response")
                current = reopened
            try:
                first = await anext(current)
            except StopAsyncIteration:
                return
            except Exception as exc:
                await _aclose(current)
                current = None
                await attempts.absorb(exc)
            else:
                yield first
                async for event in current:
                    yield event
                return
    finally:
        await _aclose(current)


async def _backoff_sleep(
    attempt: int,
    config: RetryPolicy,
) -> None:
    """
    Sleep with exponential backoff and jitter.

    Uses ``asyncio.sleep`` for non-blocking backoff.

    :param attempt: Zero-based attempt index (0 = first
        attempt).
    :param config: Retry policy with backoff parameters.
    """
    delay = config.compute_backoff_delay(retry_index=attempt + 1)
    total_tries = config.max_retries + 1
    _logger.info(
        "LLM retry %d/%d after %.1fs",
        attempt + 2,
        total_tries,
        delay,
    )
    await _sleep(delay)


async def _sleep(seconds: float) -> None:
    """
    Indirection point for the LLM retry backoff sleep.

    Exists so tests can stub the retry delay without patching
    ``asyncio.sleep`` globally (patching ``omnigent.llms.client.asyncio.sleep``
    walks the dotted path into the real ``asyncio`` module singleton
    and leaks the mock into every other test in the process).

    :param seconds: Delay in seconds.
    """
    await asyncio.sleep(seconds)


class Client:
    """
    Multi-provider async LLM client.

    Provides ``await client.responses.create()`` matching the
    OpenAI SDK interface, routing to any supported provider based
    on the model string prefix.

    Usage::

        client = Client()
        resp = await client.responses.create(
            input=[{"role": "user", "content": "Hello"}],
            instructions="You are helpful.",
            model="anthropic/claude-sonnet-4-20250514",
        )
    """

    def __init__(self) -> None:
        """Initialize the client with a responses namespace."""
        self.responses = _ResponsesNamespace(self)
