"""Catalog streams share host watches while retaining scope and authorization."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import HTTPException

from omnigent.host.frames import (
    CAP_SKILL_SUBSCRIPTIONS,
    HostSkillsFrame,
    HostSkillsResultFrame,
    decode_host_frame,
)
from omnigent.server.routes.skills import ResolvedSkillsTarget
from omnigent.server.skill_catalogs import SkillCatalogs, SkillSocketSubscriptions


def setup_catalogs():
    conn = SimpleNamespace(
        host_id="host",
        hello=SimpleNamespace(capabilities=[CAP_SKILL_SUBSCRIPTIONS]),
        skill_subscriptions={},
    )
    target = ResolvedSkillsTarget(
        conn, HostSkillsFrame("", "session", "/repo", session_id="a"), None
    )
    resolver = AsyncMock(return_value=target)
    registry = Mock()
    return SkillCatalogs(registry, resolver), conn, resolver, registry


async def host_queue(conn):
    async with asyncio.timeout(2):
        while not conn.skill_subscriptions:
            await asyncio.sleep(0)
    return next(iter(conn.skill_subscriptions.values()))


async def test_two_streams_share_watch_and_last_close_unsubscribes():
    catalogs, conn, _, registry = setup_catalogs()
    a = catalogs.subscribe(None, session_id="a")
    b = catalogs.subscribe(None, session_id="a")
    first = asyncio.create_task(anext(a))
    second = asyncio.create_task(anext(b))
    queue = await host_queue(conn)
    queue.put_nowait(
        HostSkillsResultFrame(
            "watch", "ok", [{"name": "review", "description": "Review"}], session_id="a"
        )
    )
    x, y = await asyncio.wait_for(asyncio.gather(first, second), 2)
    assert x["skills"] == y["skills"]
    assert x["status"] == y["status"]
    assert registry.send_text.call_count == 1
    await a.aclose()
    assert registry.send_text.call_count == 1
    await b.aclose()
    assert not catalogs.subscriptions
    assert not conn.skill_subscriptions
    assert decode_host_frame(registry.send_text.call_args.args[1]).action == "unwatch"


async def test_late_stream_gets_host_catalog_without_another_scan():
    catalogs, conn, _, registry = setup_catalogs()
    a = catalogs.subscribe(None, session_id="a")
    first = asyncio.create_task(anext(a))
    queue = await host_queue(conn)
    queue.put_nowait(HostSkillsResultFrame("watch", "ok", [], session_id="a"))
    initial = await asyncio.wait_for(first, 2)
    b = catalogs.subscribe(None, session_id="a")
    latest = await asyncio.wait_for(anext(b), 2)
    assert latest["skills"] == initial["skills"]
    assert latest["revision"] > initial["revision"]
    assert registry.send_text.call_count == 1
    await a.aclose()
    await b.aclose()


async def test_authorization_is_rechecked_before_delivery():
    catalogs, conn, resolver, _ = setup_catalogs()
    stream = catalogs.subscribe(None, session_id="a")
    read = asyncio.create_task(anext(stream))
    queue = await host_queue(conn)
    resolver.side_effect = HTTPException(403, "revoked")
    queue.put_nowait(HostSkillsResultFrame("watch", "ok", [], session_id="a"))
    assert (await asyncio.wait_for(read, 2))["status"] == "unavailable"
    await stream.aclose()
    assert not catalogs.subscriptions


async def test_socket_target_replacement_cancels_old_stream():
    catalogs, _, _, _ = setup_catalogs()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(skill_catalogs=catalogs)))
    socket = SkillSocketSubscriptions(request, AsyncMock())
    await socket.handle(
        {
            "targets": [
                {"id": "target", "host_id": "host", "harness": "claude-native", "path": "/repo"}
            ]
        }
    )
    original = socket.tasks["target"]
    await socket.handle(
        {
            "targets": [
                {"id": "target", "host_id": "host", "harness": "claude-native", "path": "/new"}
            ]
        }
    )
    assert original.cancelled()
    assert socket.tasks["target"] is not original
    await socket.close()


async def test_rebinds_to_reconnected_host_without_delivering_old_results():
    catalogs, conn, resolver, _ = setup_catalogs()
    stream = catalogs.subscribe(None, session_id="a")
    read = asyncio.create_task(anext(stream))
    old_queue = await host_queue(conn)
    new_conn = SimpleNamespace(
        host_id="host",
        hello=conn.hello,
        skill_subscriptions={},
    )
    resolver.return_value = ResolvedSkillsTarget(
        new_conn, HostSkillsFrame("", "session", "/repo", session_id="a"), None
    )
    old_queue.put_nowait(
        HostSkillsResultFrame(
            "old", "ok", [{"name": "stale", "description": "old"}], session_id="a"
        )
    )
    new_queue = await host_queue(new_conn)
    new_queue.put_nowait(HostSkillsResultFrame("new", "ok", [], session_id="a"))
    assert (await asyncio.wait_for(read, 2))["skills"] == []
    assert not conn.skill_subscriptions
    await stream.aclose()
