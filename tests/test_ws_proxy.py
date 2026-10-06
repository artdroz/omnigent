"""Proxy environment selection and real CONNECT socket round-trips."""

from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager

import pytest

from omnigent.util.ws_proxy import (
    open_proxy_connect_socket,
    redact_proxy_url,
    ws_env_proxy_url,
)

_TUNNEL_URL = "ws://server.sandbox.test:8000/v1/hosts/h/tunnel"
_OK = b"HTTP/1.1 200 Connection Established\r\n\r\n"


@pytest.mark.parametrize(
    ("url", "env", "expected"),
    [
        (_TUNNEL_URL, {"http_proxy": "http://p:1"}, "http://p:1"),
        ("wss://s.test/t", {"https_proxy": "http://p:1"}, "http://p:1"),
        ("wss://s.test/t", {"http_proxy": "http://p:1"}, None),
        ("ws://s.test/t", {"ALL_PROXY": "http://p:1"}, "http://p:1"),
        ("wss://s.test/t", {"ALL_PROXY": "http://p:1"}, "http://p:1"),
        (_TUNNEL_URL, {"http_proxy": "http://p:1", "HTTP_PROXY": "http://p:2"}, "http://p:1"),
        (_TUNNEL_URL, {}, None),
        (_TUNNEL_URL, {"http_proxy": "p:1"}, "http://p:1"),
        (_TUNNEL_URL, {"http_proxy": "socks5://p:1"}, None),
        ("unix:///tmp/sock", {"all_proxy": "http://p:1"}, None),
        ("not a url", {"all_proxy": "http://p:1"}, None),
    ],
)
def test_proxy_selection(url, env, expected):
    assert ws_env_proxy_url(url, env) == expected


@pytest.mark.parametrize(
    ("host", "no_proxy", "bypass"),
    [
        ("example.com", "example.com", True),
        ("sub.example.com", "example.com", True),
        ("notexample.com", "example.com", False),
        ("sub.example.com", "*.example.com", True),
        ("sub.example.com", ".example.com", True),
        ("anything.test", "*", True),
        ("example.com:8443", "example.com:8443", True),
        ("example.com:8000", "example.com:8443", False),
        ("127.0.0.1:8000", "localhost,127.0.0.1,::1", True),
        ("localhost:8000", "localhost,127.0.0.1,::1", True),
        ("[::1]:8000", "localhost,127.0.0.1,::1", True),
        ("server.sandbox.test:8000", "localhost,127.0.0.1,::1", False),
    ],
)
def test_no_proxy(host, no_proxy, bypass):
    env = {"http_proxy": "http://p:1", "no_proxy": no_proxy}
    assert ws_env_proxy_url(f"ws://{host}/t", env) == (None if bypass else "http://p:1")


@pytest.mark.parametrize("userinfo", ["", "user:secret@"])
def test_redact_proxy_url(userinfo):
    assert redact_proxy_url(f"http://{userinfo}proxy:3128") == "http://proxy:3128"


@asynccontextmanager
async def _connect_proxy(reply):
    requests = []

    async def respond(reader, writer):
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(reply)
            await writer.drain()
            if reply == _OK:
                writer.write(await reader.readexactly(4))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(respond, "127.0.0.1", 0) as server:
        yield server.sockets[0].getsockname()[1], requests


@pytest.mark.parametrize("userinfo", ["", "alice:open%20sesame@"])
async def test_connect_tunnel(userinfo):
    async with _connect_proxy(_OK) as (port, requests):
        url = f"http://{userinfo}127.0.0.1:{port}"
        with await open_proxy_connect_socket(url, _TUNNEL_URL, timeout=5) as sock:
            expected = (
                b"CONNECT server.sandbox.test:8000 HTTP/1.1\r\nHost: server.sandbox.test:8000\r\n"
            )
            if userinfo:
                expected += (
                    b"Proxy-Authorization: Basic "
                    + base64.b64encode(b"alice:open sesame")
                    + b"\r\n"
                )
            assert requests == [expected + b"\r\n"]
            assert sock.gettimeout() is None
            reader, writer = await asyncio.open_connection(sock=sock)
            try:
                writer.write(b"ping")
                await writer.drain()
                assert await asyncio.wait_for(reader.readexactly(4), timeout=5) == b"ping"
            finally:
                writer.close()
                await writer.wait_closed()


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n", "refused CONNECT"),
        (_OK + b"GARBAGE", "unexpected bytes"),
        (b"", "closed the connection"),
    ],
)
async def test_connect_error(reply, error):
    async with _connect_proxy(reply) as (port, _requests):
        with pytest.raises(OSError, match=error):
            await open_proxy_connect_socket(f"http://127.0.0.1:{port}", _TUNNEL_URL, timeout=5)
