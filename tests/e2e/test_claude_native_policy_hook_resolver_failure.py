"""E2E regression test: the claude-native policy hook under a resolver failure.

Claude Code's ``UserPromptSubmit`` hook is a curl POST to the in-runner relay's
``/hook/claude/evaluate-policy``, which proxies the evaluate call upstream. A
hostname lookup that fails briefly (macOS ``[Errno 8] nodename nor servname
provided, or not known``) must not drop the prompt, and a sustained outage must
block with a reason that names the condition instead of the raw errno.

The test drives a real ``omnigent server`` subprocess and the production relay
started as the runner starts it, and POSTs the same request curl makes. The
resolver failure is injected by wrapping ``socket.getaddrinfo`` for the server
hostname (``localhost``, so each fresh connection performs a real lookup).
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import yaml

from tests._helpers.session import bundle_files, post_session_bundle

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Every HTTP call the test itself makes targets 127.0.0.1 (an IP literal, no
# getaddrinfo); CI shells often carry an egress proxy, so bypass autodetection.
_http = httpx.Client(trust_env=False)
atexit.register(_http.close)

_PYTHONPATH = os.pathsep.join(
    [
        str(_REPO_ROOT),
        str(_REPO_ROOT / "sdks" / "python-client"),
        str(_REPO_ROOT / "sdks" / "ui"),
        os.environ.get("PYTHONPATH", ""),
    ]
)

_HEALTH_TIMEOUT_S = 120.0
_POLL_S = 0.5
# The relay retries over its transient budget before failing closed.
_HOOK_TIMEOUT_S = 120.0

# httpx keepalive_expiry / uvicorn keep-alive are both 5 s; wait past that so the
# hook's POST opens a fresh connection that must re-resolve the server hostname.
_KEEPALIVE_WAIT_S = 6.0
# A brief resolver blip, well inside the relay's transient retry budget.
_BLIP_CLEAR_S = 4.0
# Sustained outage: a short budget keeps the fail-closed leg quick.
_SUSTAINED_BUDGET_S = 3.0

# The prior Claude session id the hook stamps onto its evaluation request.
_EXTERNAL_SID = "11111111-2222-4333-8444-555566667777"

_PROMPT = "summarize the open issues in this repository"


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _localhost_env(extra: dict[str, str]) -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONPATH": _PYTHONPATH,
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(name, None)
    env.update(extra)
    return env


def _terminate(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _wait_http_ok(url: str, deadline: float) -> None:
    last = "not polled"
    while time.monotonic() < deadline:
        try:
            if _http.get(url, timeout=2.0).status_code == 200:
                return
        except httpx.HTTPError as exc:
            last = repr(exc)
        time.sleep(_POLL_S)
    raise AssertionError(f"server never became healthy at {url}; last: {last}")


def _start_server(tmp_path: Path) -> tuple[subprocess.Popen[bytes], int]:
    """Spawn a real ``omnigent server`` on a free loopback port; return it + port."""
    port = _find_free_port()
    # The child dups the fd at spawn, so the parent closes its handle at once
    # and leaks no descriptor while the server keeps writing its own log.
    with (tmp_path / "server.log").open("w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from omnigent.cli import main; main()",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--database-uri",
                f"sqlite:///{tmp_path / 'db.sqlite'}",
                "--artifact-location",
                str(tmp_path / "artifacts"),
            ],
            env=_localhost_env({}),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    _wait_http_ok(f"http://127.0.0.1:{port}/health", time.monotonic() + _HEALTH_TIMEOUT_S)
    return proc, port


def _create_session(port: int) -> str:
    """Register a minimal inline agent and return its session id."""
    cfg = {
        "name": "policy-hook-dns-repro",
        "prompt": "You are a test agent.",
        "executor": {"harness": "claude-native", "model": "claude-sonnet-4-20250514"},
    }
    data = yaml.safe_dump(cfg).encode()
    bundle_bytes = bundle_files({"policy-hook-dns-repro.yaml": data})
    resp = post_session_bundle(
        _http.post, f"http://127.0.0.1:{port}/v1/sessions", bundle_bytes, timeout=30.0
    )
    resp.raise_for_status()
    return str(resp.json()["session_id"])


class _ResolverFault:
    """Wraps ``socket.getaddrinfo`` to fail lookups of one host on demand.

    While active, lookups of *host* raise ``gaierror(EAI_NONAME)``. An optional
    clear deadline models a transient blip that recovers on its own.
    """

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._real = socket.getaddrinfo
        # The relay's background loop and the test thread both resolve through
        # this hook, so guard the active/clear/counter state against races.
        self._lock = threading.Lock()
        self.active = False
        self._clear_at: float | None = None
        self.calls_failed = 0

    def _match(self, host: object) -> bool:
        name = host.decode() if isinstance(host, (bytes, bytearray)) else str(host)
        return name == self._host

    def getaddrinfo(self, host, port=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        with self._lock:
            if (
                self.active
                and self._match(host)
                and (port is None or str(port) == str(self._port))
            ):
                if self._clear_at is not None and time.monotonic() >= self._clear_at:
                    self.active = False
                else:
                    self.calls_failed += 1
                    raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
        return self._real(host, port, *args, **kwargs)

    def activate(self, clear_after: float | None = None) -> None:
        with self._lock:
            self._clear_at = None if clear_after is None else time.monotonic() + clear_after
            self.active = True

    def install(self) -> None:
        socket.getaddrinfo = self.getaddrinfo

    def uninstall(self) -> None:
        socket.getaddrinfo = self._real


class _Relay:
    """The production tool relay running in-process on a background loop.

    ``policy_client`` addresses the server by the ``localhost`` hostname so the
    relay's upstream policy POST performs a real ``getaddrinfo`` the fault can
    intercept, as the runner's server client does in production.
    """

    def __init__(self, server_port: int, session_id: str, tmp_path: Path) -> None:
        from omnigent.harnesses.claude_native.bridge import (
            prepare_bridge_dir,
            start_tool_relay,
            write_active_session_id,
        )

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._client = httpx.AsyncClient(
            base_url=httpx.URL(f"http://localhost:{server_port}"),
            trust_env=False,
            timeout=httpx.Timeout(5.0, read=None),
        )
        self.bridge_dir = prepare_bridge_dir(session_id, workspace=tmp_path)

        async def _noop_tool(name: str, arguments: dict[str, object]) -> dict[str, object]:
            del name, arguments
            return {}

        self._relay = start_tool_relay(
            bridge_dir=self.bridge_dir,
            tools=[],
            tool_executor=_noop_tool,
            loop=self._loop,
            policy_client=self._client,
            session_id=session_id,
        )
        write_active_session_id(self.bridge_dir, session_id)
        info = json.loads((self.bridge_dir / "tool_relay.json").read_text())
        self.url = str(info["url"])
        self.token = str(info["token"])

    def submit_prompt(self, prompt: str) -> dict[str, object]:
        """POST a ``UserPromptSubmit`` event exactly as the production curl hook does."""
        payload = {
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
            "session_id": _EXTERNAL_SID,
            "cwd": str(self.bridge_dir),
        }
        resp = httpx.post(
            f"{self.url.rstrip('/')}/hook/claude/evaluate-policy",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            content=json.dumps(payload).encode(),
            trust_env=False,
            timeout=_HOOK_TIMEOUT_S,
        )
        resp.raise_for_status()
        text = resp.content.decode().strip()
        return json.loads(text) if text else {}

    def close(self) -> None:
        self._relay.close()
        # Close the async client on the loop it was created and used on, before
        # that loop stops; a fresh asyncio.run() loop cannot close it cleanly.
        asyncio.run_coroutine_threadsafe(self._client.aclose(), self._loop).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()


@pytest.fixture(scope="module")
def _server(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("policy-hook-dns")
    proc, port = _start_server(tmp_path)
    try:
        session_id = _create_session(port)
        yield port, session_id, tmp_path
    finally:
        _terminate(proc)


def test_transient_resolver_blip_does_not_drop_prompt(_server) -> None:
    """A prompt survives a resolver failure that clears within the retry budget.

    The relay's upstream lookup fails for a few seconds and then recovers; the
    recovered lookup yields a real verdict and the prompt proceeds.
    """
    port, session_id, tmp_path = _server
    fault = _ResolverFault("localhost", port)
    relay = _Relay(port, session_id, tmp_path / "relay-transient")
    fault.install()
    try:
        # Control: healthy chain over the localhost hostname -> not blocked.
        control = relay.submit_prompt(_PROMPT)
        assert control.get("decision") != "block", (
            "Control leg (no resolver fault) unexpectedly blocked the prompt; the "
            f"server/relay chain is unhealthy so the fault leg proves nothing: {control!r}"
        )

        # Force a fresh connection (so the next POST must re-resolve), then make
        # the hostname briefly unresolvable, recovering within the budget.
        time.sleep(_KEEPALIVE_WAIT_S)
        fault.activate(clear_after=_BLIP_CLEAR_S)

        decision = relay.submit_prompt(_PROMPT)
        assert decision.get("decision") != "block", (
            f"a hostname-resolution failure that cleared after {_BLIP_CLEAR_S:.0f}s "
            f"still dropped the prompt; the relay failed closed early. decision={decision!r}"
        )
        assert fault.calls_failed > 0, "the fault never intercepted a lookup"
    finally:
        fault.uninstall()
        relay.close()


def test_sustained_resolver_failure_block_reason_is_sanitized(
    _server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sustained outage fails closed with a reason that names the condition.

    The real ``httpx.ConnectError`` → ``socket.gaierror`` chain must be
    recognised as a resolver failure; the block reason must not carry the raw
    ``getaddrinfo`` errno text.
    """
    from omnigent.native import native_policy_hook

    monkeypatch.setattr(native_policy_hook, "_EVALUATE_POLICY_RETRY_BUDGET_S", _SUSTAINED_BUDGET_S)
    port, session_id, tmp_path = _server
    fault = _ResolverFault("localhost", port)
    relay = _Relay(port, session_id, tmp_path / "relay-sustained")
    fault.install()
    try:
        time.sleep(_KEEPALIVE_WAIT_S)
        fault.activate()  # sustained; never clears

        decision = relay.submit_prompt(_PROMPT)
        reason = str(decision.get("reason", ""))

        # Confirm we hit the fail-closed path at all.
        assert (
            decision.get("decision") == "block" and "failing closed for this request" in reason
        ), f"expected the fail-closed request block; got {decision!r}"
        assert "could not resolve the Omnigent server hostname" in reason, reason
        assert "[Errno" not in reason and "getaddrinfo" not in reason.lower(), (
            f"the fail-closed block reason leaks a raw OS resolver errno. reason={reason!r}"
        )
    finally:
        fault.uninstall()
        relay.close()
