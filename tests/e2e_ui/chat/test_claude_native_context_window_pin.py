"""E2E: a claude-native session on a 1M-capable gateway Opus runs with a 1M context window."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pyte
import pytest
import yaml
from playwright.sync_api import Page, expect
from websockets.sync.client import connect as ws_connect

from tests.e2e_ui.conftest import (
    _create_native_claude_session,
    _ensure_runner_online,
    _server_state,
    set_fallback_mock_llm,
)

_OPUS = "system.ai.claude-opus-5"
_SONNET = "system.ai.claude-sonnet-5"
_REPLY = "context-window pin check: acknowledged."
_ONE_MILLION = 1_000_000

_TERMINAL = '[data-testid="terminal-view"]'
_XTERM_INPUT = ".xterm-helper-textarea"
_ASSISTANT = '[data-testid="message-bubble"][data-role="assistant"]'
_WORKING = '[data-testid="working-indicator"]'
# claude-native auto-launch + first-run pre-accept + WS attach.
_TERMINAL_READY_TIMEOUT_MS = 240_000
_MOCK_TURN_TIMEOUT_MS = 120_000
_CONTEXT_READOUT_TIMEOUT_S = 60.0
_SNAPSHOT_WINDOW_TIMEOUT_S = 45.0
# ``53.1k/200k tokens (27%)`` as Claude Code's /context prints it.
_TOKENS_RE = re.compile(r"([\d.]+)\s*([km]?)/([\d.]+)\s*([km]?)\s+tokens\s*\(", re.IGNORECASE)
_UNITS = {"": 1, "k": 1_000, "m": 1_000_000}
_PREPARED_ENV = Path(".omnigent/repro-env/environment.json")


def _provider_config_path() -> Path:
    """The provider config the runner launching Claude Code reads."""
    if _server_state.get("workflow_owned") and _PREPARED_ENV.is_file():
        prepared = json.loads(_PREPARED_ENV.read_text(encoding="utf-8"))
        return Path(prepared["config_home"]) / "config.yaml"
    from omnigent.config import global_config_path

    return global_config_path().resolve()


@pytest.fixture
def gateway_1m_claude_provider(live_server: str, mock_llm_server_url: str) -> Iterator[None]:
    """Route Claude Code to the mock with opus/sonnet pinned to bare 1M-capable gateway ids."""
    config_path = _provider_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    backup = config_path.with_name(config_path.name + ".context-window-pin-backup")
    if backup.exists():
        raise RuntimeError(f"Unrestored provider-config backup at {backup}")
    original = config_path.read_bytes() if config_path.exists() else None
    if original is not None:
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(original)
    config = {
        "runner": {"idle_timeout_s": 0},
        "providers": {
            "gateway-1m-claude": {
                "kind": "key",
                "default": ["anthropic"],
                "anthropic": {
                    "base_url": mock_llm_server_url,
                    "api_key": "mock-key",
                    "models": {"default": _OPUS, "opus": _OPUS, "sonnet": _SONNET},
                },
            }
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    try:
        yield
    finally:
        if original is not None:
            backup.replace(config_path)
        else:
            config_path.unlink(missing_ok=True)


def _terminal_id(base_url: str, session_id: str) -> str | None:
    response = httpx.get(f"{base_url}/v1/sessions/{session_id}/resources", timeout=10.0)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else payload
    for row in rows or []:
        if isinstance(row, dict) and row.get("type") == "terminal":
            return str(row["id"])
    return None


def _pane_text(base_url: str, session_id: str, *, seconds: float = 2.0) -> str:
    """Render the terminal screen from a read-only attach, seeded by ``capture-pane``."""
    terminal_id = _terminal_id(base_url, session_id)
    if terminal_id is None:
        return ""
    url = (
        base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        + f"/v1/sessions/{session_id}/resources/terminals/{terminal_id}/attach?read_only=true"
    )
    raw = bytearray()
    deadline = time.monotonic() + seconds
    with ws_connect(url, open_timeout=15, max_size=None) as ws:
        while time.monotonic() < deadline:
            try:
                frame = ws.recv(timeout=max(0.1, deadline - time.monotonic()))
            except TimeoutError:
                break
            if isinstance(frame, bytes):
                raw.extend(frame)
    screen = pyte.Screen(220, 200)
    pyte.ByteStream(screen).feed(bytes(raw))
    return "\n".join(line.rstrip() for line in screen.display)


def _wait_pane(base_url: str, session_id: str, needle: str, *, timeout_s: float) -> str:
    deadline = time.monotonic() + timeout_s
    text = ""
    while time.monotonic() < deadline:
        text = _pane_text(base_url, session_id)
        if needle.lower() in text.lower():
            return text
        time.sleep(1.0)
    return text


def _context_window_from_readout(pane: str) -> tuple[int | None, str]:
    """Parse the window out of Claude Code's ``/context`` usage line."""
    for line in pane.splitlines():
        match = _TOKENS_RE.search(line)
        if match:
            window = float(match.group(3)) * _UNITS[match.group(4).lower()]
            return int(window), line.strip()
    return None, ""


def _open_terminal(page: Page, base_url: str, session_id: str) -> None:
    page.goto(f"{base_url}/c/{session_id}")
    expect(page.get_by_test_id("view-mode-toggle")).to_be_visible(
        timeout=_TERMINAL_READY_TIMEOUT_MS
    )
    page.get_by_test_id("view-mode-terminal").click()
    terminal = page.locator(_TERMINAL).last
    expect(terminal).to_have_attribute(
        "data-state", "connected", timeout=_TERMINAL_READY_TIMEOUT_MS
    )
    _wait_pane(base_url, session_id, "manual mode", timeout_s=90.0)


def _type_slash_command(page: Page, command: str) -> None:
    xterm_input = page.locator(_TERMINAL).last.locator(_XTERM_INPUT)
    expect(xterm_input).to_be_attached(timeout=30_000)
    xterm_input.focus()
    page.keyboard.type(command, delay=40)
    page.wait_for_timeout(1000)
    page.keyboard.press("Enter")


def _send_turn(page: Page) -> None:
    page.get_by_test_id("view-mode-chat").click()
    composer = page.get_by_role("textbox", name="Message the agent")
    expect(composer).to_be_visible(timeout=30_000)
    composer.fill("Reply with one short sentence.")
    page.get_by_role("button", name="Send", exact=True).click()
    expect(page.locator(_ASSISTANT, has_text=_REPLY).first).to_be_visible(
        timeout=_MOCK_TURN_TIMEOUT_MS
    )
    expect(page.locator(_WORKING)).to_have_count(0, timeout=_MOCK_TURN_TIMEOUT_MS)


def _snapshot_context_window(base_url: str, session_id: str) -> int | None:
    deadline = time.monotonic() + _SNAPSHOT_WINDOW_TIMEOUT_S
    while time.monotonic() < deadline:
        snapshot = httpx.get(f"{base_url}/v1/sessions/{session_id}", timeout=10.0).json()
        window = snapshot.get("context_window")
        if isinstance(window, int) and snapshot.get("last_total_tokens"):
            return window
        time.sleep(2.0)
    return None


@pytest.mark.timeout(600)
def test_claude_native_1m_capable_default_model_gets_1m_window(
    request: pytest.FixtureRequest,
    live_server: str,
    mock_llm_server_url: str,
    gateway_1m_claude_provider: None,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Claude Code reports a 1M window for the pinned 1M-capable Opus, and so does Omnigent."""
    for key in ("default", _OPUS, _SONNET):
        set_fallback_mock_llm(mock_llm_server_url, key, _REPLY)
    respawned = _ensure_runner_online(live_server, tmp_path_factory)
    runner_id = str(_server_state["runner_id"])
    session_id = _create_native_claude_session(live_server, runner_id)
    try:
        page: Page = request.getfixturevalue("page")
        _open_terminal(page, live_server, session_id)
        _type_slash_command(page, "/context")
        pane = _wait_pane(
            live_server, session_id, "tokens (", timeout_s=_CONTEXT_READOUT_TIMEOUT_S
        )
        assert _OPUS in pane, f"/context did not name the pinned model {_OPUS}:\n{pane[-2000:]}"
        claude_window, tokens_line = _context_window_from_readout(pane)
        assert claude_window is not None, f"/context printed no usage line:\n{pane[-2000:]}"

        # Primary check: Claude Code sizes the window from the pinned id, so the bug
        # surfaces here before any composer turn. Fails at 200K on the buggy build.
        assert claude_window >= _ONE_MILLION, (
            f"{_OPUS} is a 1M-capable model, but Claude Code sized the session at "
            f"{claude_window:,} tokens ({tokens_line!r})"
        )

        _send_turn(page)
        omnigent_window = _snapshot_context_window(live_server, session_id)
        assert (omnigent_window or 0) >= _ONE_MILLION, (
            f"Claude Code reported {claude_window:,} tokens for {_OPUS}, but Omnigent's "
            f"session snapshot sizes the composer ring at context_window={omnigent_window}"
        )
    finally:
        httpx.delete(f"{live_server}/v1/sessions/{session_id}", timeout=10.0)
        if respawned is not None:
            respawned.terminate()
            try:
                respawned.wait(timeout=5)
            except subprocess.TimeoutExpired:
                respawned.kill()
                respawned.wait(timeout=5)
