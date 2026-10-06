"""Multiplex host-owned skill catalogs across authorized UI streams."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any

from fastapi import Request, WebSocket

from omnigent.host.frames import CAP_SKILL_SUBSCRIPTIONS, HostSkillsResultFrame, encode_host_frame
from omnigent.server.host_registry import HostRegistry

if TYPE_CHECKING:
    from omnigent.server.routes.skills import ResolvedSkillsTarget, SkillsResponse

_logger = logging.getLogger(__name__)


@dataclass
class _Subscription:
    target: ResolvedSkillsTarget
    listeners: set[asyncio.Queue[HostSkillsResultFrame]] = field(default_factory=set)
    latest: HostSkillsResultFrame | None = None
    task: asyncio.Task[None] | None = None


class SkillCatalogs:
    """One host watch per target, independent of the number of browser streams."""

    def __init__(
        self,
        registry: HostRegistry,
        resolve: Callable[..., Awaitable[ResolvedSkillsTarget]],
    ) -> None:
        self.registry = registry
        self.resolve = resolve
        self.subscriptions: dict[str, _Subscription] = {}
        self.revision = 0

    def next_revision(self) -> int:
        self.revision = max(self.revision + 1, time.time_ns() // 1_000_000)
        return self.revision

    @staticmethod
    def _key(target: ResolvedSkillsTarget) -> str:
        return f"{id(target.connection)}:{json.dumps(asdict(target.frame), sort_keys=True)}"

    async def read(self, target: ResolvedSkillsTarget) -> SkillsResponse:
        from omnigent.server.routes.skills import request_host_skills

        frame = target.frame
        result = await request_host_skills(
            host_registry=self.registry,
            host_conn=target.connection,
            harness=frame.harness,
            path=frame.path,
            session_id=frame.session_id,
            agent_id=frame.agent_id,
            agent_version=frame.agent_version,
            sub_agent_name=frame.sub_agent_name,
            skills_filter=frame.skills_filter,
        )
        return target.response(result)

    def invalidate(self, target: ResolvedSkillsTarget) -> None:
        if CAP_SKILL_SUBSCRIPTIONS in target.connection.hello.capabilities:
            self.registry.send_text(
                target.connection,
                encode_host_frame(
                    replace(target.frame, request_id=secrets.token_hex(8), action="invalidate")
                ),
            )

    async def _pump(self, entry: _Subscription) -> None:
        target = entry.target
        conn = target.connection
        frame = replace(target.frame, request_id=secrets.token_hex(8), action="watch")
        queue: asyncio.Queue[HostSkillsResultFrame] = asyncio.Queue(maxsize=1)
        conn.skill_subscriptions[frame.request_id] = queue
        try:
            self.registry.send_text(conn, encode_host_frame(frame))
            while True:
                result = await queue.get()
                entry.latest = result
                for listener in entry.listeners:
                    if listener.full():
                        listener.get_nowait()
                    listener.put_nowait(entry.latest)
        finally:
            conn.skill_subscriptions.pop(frame.request_id, None)
            with contextlib.suppress(ConnectionError):
                self.registry.send_text(conn, encode_host_frame(replace(frame, action="unwatch")))

    async def subscribe(
        self, request: Request | WebSocket, **scope: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Reauthorize and rebind on scope changes, host reconnects, and agent updates."""
        entry = None
        key = None
        listener: asyncio.Queue[HostSkillsResultFrame] = asyncio.Queue(maxsize=1)
        previous = None
        try:
            while True:
                try:
                    target = await self.resolve(request, **scope)
                    current_key = self._key(target)
                    if key != current_key:
                        await self._release(key, entry, listener)
                        entry = None
                        key = current_key
                        previous = None
                        while not listener.empty():
                            listener.get_nowait()
                        if CAP_SKILL_SUBSCRIPTIONS in target.connection.hello.capabilities:
                            entry = self.subscriptions.get(key)
                            if entry is None:
                                entry = _Subscription(target)
                                self.subscriptions[key] = entry
                                entry.task = asyncio.create_task(self._pump(entry))
                            entry.listeners.add(listener)
                            if entry.latest is not None:
                                listener.put_nowait(entry.latest)
                    if entry is None:
                        # Older hosts can still serve catalogs through the compatibility API.
                        response = await self.read(target)
                    else:
                        try:
                            result = await asyncio.wait_for(listener.get(), 15.0)
                        except TimeoutError:
                            if entry.task is not None and entry.task.done():
                                await entry.task
                            continue
                        authorized = await self.resolve(request, **scope)
                        if self._key(authorized) != key:
                            continue
                        response = target.response(entry.latest or result)
                    payload = {
                        "status": "ready",
                        "skills": [skill.model_dump() for skill in response.skills],
                        "host_id": target.connection.host_id,
                        "workspace": target.frame.path,
                        "agent_id": target.frame.agent_id,
                        "sub_agent_name": target.frame.sub_agent_name,
                    }
                    if payload != previous:
                        yield {**payload, "revision": self.next_revision()}
                        previous = payload
                    if entry is None:
                        await asyncio.sleep(60)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 — keep the stream alive across discovery failures
                    _logger.debug("Skill subscription unavailable", exc_info=True)
                    status = getattr(exc, "status_code", getattr(exc, "http_status", 503))
                    payload = {"status": "unavailable" if status in (401, 403, 404) else "error"}
                    if payload != previous:
                        yield {**payload, "revision": self.next_revision()}
                        previous = payload
                    await self._release(key, entry, listener)
                    entry = None
                    key = None
                    await asyncio.sleep(15)
        finally:
            await self._release(key, entry, listener)

    async def _release(
        self,
        key: str | None,
        entry: _Subscription | None,
        listener: asyncio.Queue[HostSkillsResultFrame],
    ) -> None:
        if entry is None:
            return
        entry.listeners.discard(listener)
        if not entry.listeners:
            if key is not None:
                self.subscriptions.pop(key, None)
            if entry.task is not None:
                entry.task.cancel()
                with contextlib.suppress(asyncio.CancelledError, ConnectionError):
                    await entry.task


async def with_session_skills(
    source: AsyncIterator[str], request: Request, session_id: str
) -> AsyncIterator[str]:
    """Attach catalog delivery to the existing session stream's lifetime."""
    catalogs = getattr(request.app.state, "skill_catalogs", None)
    if catalogs is None:
        async for item in source:
            yield item
        return
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=64)

    async def forward_source() -> None:
        try:
            async for item in source:
                await queue.put(item)
        except Exception:
            _logger.exception("Session stream failed")
        await queue.put(None)

    async def forward_skills() -> None:
        async for catalog in catalogs.subscribe(request, session_id=session_id):
            event = {"type": "session.skills", "conversation_id": session_id, **catalog}
            await queue.put(f"event: session.skills\ndata: {json.dumps(event)}\n\n")

    tasks = [asyncio.create_task(forward_source()), asyncio.create_task(forward_skills())]
    try:
        while (item := await queue.get()) is not None:
            yield item
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class SkillSocketSubscriptions:
    """Pre-session targets multiplexed over the existing app updates socket."""

    def __init__(
        self, request: Request | WebSocket, send: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None:
        self.request = request
        self.send = send
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.scopes: dict[str, dict[str, str]] = {}

    async def handle(self, message: dict[str, Any]) -> None:
        catalogs = getattr(self.request.app.state, "skill_catalogs", None)
        if catalogs is not None and message.get("type") == "refresh_skills":
            identity = message.get("target_id")
            scope = self.scopes.get(identity) if isinstance(identity, str) else None
            if scope is not None:
                target = await catalogs.resolve(self.request, **scope)
                catalogs.invalidate(target)
            return
        targets = message.get("targets")
        if catalogs is None or not isinstance(targets, list) or len(targets) > 8:
            return
        desired: dict[str, dict[str, str]] = {}
        for target in targets:
            if not isinstance(target, dict):
                continue
            identity = target.get("id")
            if not isinstance(identity, str) or not identity or len(identity) > 8192:
                continue
            scope = {
                key: target[key]
                for key in ("host_id", "harness", "path", "agent_id")
                if key in target
            }
            if not all(
                isinstance(value, str) and 0 < len(value) <= 4096 for value in scope.values()
            ):
                continue
            if not all(key in scope for key in ("host_id", "harness", "path")):
                continue
            desired[identity] = scope
        # The identity includes the entire target, so changed scope gets a new task.
        for identity in list(self.tasks):
            if identity not in desired or self.scopes.get(identity) != desired[identity]:
                task = self.tasks.pop(identity)
                self.scopes.pop(identity, None)
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        for identity, scope in desired.items():
            if identity not in self.tasks:
                self.scopes[identity] = scope
                self.tasks[identity] = asyncio.create_task(
                    self._forward(catalogs, identity, scope)
                )

    async def _forward(
        self, catalogs: SkillCatalogs, identity: str, scope: dict[str, str]
    ) -> None:
        async for catalog in catalogs.subscribe(self.request, **scope):
            await self.send({"type": "skills", "target_id": identity, **catalog})

    async def close(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()
        self.scopes.clear()
