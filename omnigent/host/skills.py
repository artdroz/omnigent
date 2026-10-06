"""Host-owned menu discovery for directories and session agent bundles."""

from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

from omnigent.host.frames import HostSkillsFrame, HostSkillsResultFrame
from omnigent.spec import load
from omnigent.spec.skill_sources import (
    resolve_harness_skills,
    resolve_session_skills,
    skill_source_context_from_env,
)
from omnigent.spec.types import AgentSpec

SKILLS_CACHE_TTL_SECONDS = 60.0
_MAX_CACHED_CATALOGS = 128


def _subagent_bundle(spec: AgentSpec, directory: Path, name: str) -> tuple[AgentSpec, Path]:
    """Find a child's actual bundle directory, including nested sub-agents."""
    if spec.name == name:
        return spec, directory
    for child in spec.sub_agents:
        segment = child.source_rel_dir
        if not segment or segment in (".", "..") or Path(segment).parts != (segment,):
            continue
        try:
            return _subagent_bundle(child, directory / "agents" / segment, name)
        except LookupError:
            continue
    raise LookupError(f"Sub-agent {name!r} is absent from the session bundle")


class HostSkillDiscovery:
    """Cache successful catalogs and serialize scans off the host's event loop."""

    def __init__(self, fetch_bundle: Callable[[HostSkillsFrame], httpx.Response]) -> None:
        self._fetch_bundle = fetch_bundle
        self._lock = threading.Lock()
        self._cache: OrderedDict[
            tuple[str | tuple[str, ...] | None, ...], tuple[float, list[dict[str, str]]]
        ] = OrderedDict()

    def invalidate(self) -> None:
        """Invalidate catalogs sharing user or plugin skill roots."""
        with self._lock:
            self._cache.clear()

    def discover(self, frame: HostSkillsFrame, root: Path) -> list[dict[str, str]]:
        """Return metadata for the exact launch target or effective session bundle."""
        key = (
            str(root),
            frame.harness,
            frame.session_id,
            frame.agent_id,
            frame.agent_version,
            frame.sub_agent_name,
            tuple(frame.skills_filter)
            if isinstance(frame.skills_filter, list)
            else frame.skills_filter,
        )
        with self._lock:
            cached = self._cache.get(key)
            if frame.action != "watch" and cached is not None and cached[0] > time.monotonic():
                self._cache.move_to_end(key)
                return list(cached[1])
            if frame.session_id is None:
                ctx = skill_source_context_from_env(
                    roots=(root,), harness=frame.harness, skills_filter=frame.skills_filter
                )
                skills = resolve_harness_skills(ctx, frame.harness)
            else:
                response = self._fetch_bundle(frame)
                response.raise_for_status()
                if (
                    response.headers.get("X-Agent-Id") != frame.agent_id
                    or response.headers.get("X-Agent-Version") != frame.agent_version
                ):
                    raise ValueError("The session's agent changed during discovery; retry")
                # Only metadata escapes this scope; invocation reads its own bundle.
                with tempfile.TemporaryDirectory(prefix="host-skill-bundle-") as tmp:
                    directory = Path(tmp)
                    spec = load(
                        response.content,
                        dest=directory,
                        expand_env=response.headers.get("X-Agent-Session-Scoped") == "false",
                        prune_invalid_sub_agents=True,
                    )
                    if frame.sub_agent_name:
                        spec, directory = _subagent_bundle(spec, directory, frame.sub_agent_name)
                    skills = resolve_session_skills(spec, (root, directory), directory)
            result = [{"name": skill.name, "description": skill.description} for skill in skills]
            self._cache[key] = (time.monotonic() + SKILLS_CACHE_TTL_SECONDS, result)
            self._cache.move_to_end(key)
            while len(self._cache) > _MAX_CACHED_CATALOGS:
                self._cache.popitem(last=False)
            return list(result)


class HostSkillSubscriptions:
    """Refresh watched catalogs on signals, with reconciliation for unobserved changes."""

    def __init__(
        self,
        discovery: HostSkillDiscovery,
        discover: Callable[[HostSkillsFrame], HostSkillsResultFrame],
        *,
        interval: float = 30.0,
    ) -> None:
        self.discovery = discovery
        self.discover = discover
        self.interval = interval
        self.tasks: dict[str, tuple[asyncio.Task[None], asyncio.Event]] = {}

    async def handle(
        self, frame: HostSkillsFrame, send: Callable[[HostSkillsResultFrame], Awaitable[None]]
    ) -> None:
        if frame.action == "invalidate":
            await asyncio.to_thread(self.discovery.invalidate)
            for _, changed in self.tasks.values():
                changed.set()
            return
        previous = self.tasks.pop(frame.request_id, None)
        if previous is not None:
            previous[0].cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await previous[0]
        if frame.action != "watch":
            return
        changed = asyncio.Event()
        self.tasks[frame.request_id] = (
            asyncio.create_task(self._watch(frame, changed, send)),
            changed,
        )

    async def _watch(
        self,
        frame: HostSkillsFrame,
        changed: asyncio.Event,
        send: Callable[[HostSkillsResultFrame], Awaitable[None]],
    ) -> None:
        previous = None
        while True:
            changed.clear()
            result = await asyncio.to_thread(self.discover, frame)
            if result != previous:
                await send(result)
                previous = result
            try:
                await asyncio.wait_for(changed.wait(), self.interval)
                await asyncio.sleep(0.2)
            except TimeoutError:
                pass

    async def close(self) -> None:
        tasks = [task for task, _ in self.tasks.values()]
        self.tasks.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
