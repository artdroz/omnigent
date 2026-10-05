"""A native Claude Opus session must report Opus's 1M context window, not a 200K cap."""

from __future__ import annotations

import logging
import re
import time

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import reset_mock_llm, set_fallback_mock_llm

_log = logging.getLogger(__name__)

_OPUS_1M_WINDOW = 1_000_000
_TERMINAL_READY_TIMEOUT_MS = 120_000
_MOCK_TURN_TIMEOUT_MS = 90_000
_TOKEN = "OPUS_CONTEXT_OK"


def _wait_terminal_connected(page: Page) -> None:
    expect(page.get_by_test_id("view-mode-toggle")).to_be_visible(
        timeout=_TERMINAL_READY_TIMEOUT_MS
    )
    page.get_by_test_id("view-mode-terminal").click()
    terminal = page.locator('[data-testid="terminal-view"]').last
    expect(terminal).to_have_attribute(
        "data-state", "connected", timeout=_TERMINAL_READY_TIMEOUT_MS
    )
    page.get_by_test_id("view-mode-chat").click()


@pytest.mark.nightly
@pytest.mark.timeout(300)
def test_native_claude_opus_reports_1m_context_window(
    page: Page,
    native_claude_opus_mock_session: tuple[str, str],
    mock_llm_server_url: str,
) -> None:
    base_url, session_id = native_claude_opus_mock_session
    _log.info("opus native session: base_url=%s session_id=%s", base_url, session_id)

    reset_mock_llm(mock_llm_server_url)
    for key in ("default", "databricks-claude-opus-5"):
        set_fallback_mock_llm(mock_llm_server_url, key, _TOKEN)

    page.goto(f"{base_url}/c/{session_id}")
    _wait_terminal_connected(page)

    composer = page.get_by_label("Message the agent")
    expect(composer).to_be_visible(timeout=30_000)
    composer.fill(f"Reply with exactly this token and nothing else: {_TOKEN}")
    page.get_by_role("button", name="Send", exact=True).click()

    expect(
        page.locator(
            '[data-testid="message-bubble"][data-role="assistant"]', has_text=_TOKEN
        ).first
    ).to_be_visible(timeout=_MOCK_TURN_TIMEOUT_MS)
    expect(page.locator('[data-testid="working-indicator"]')).to_have_count(
        0, timeout=_MOCK_TURN_TIMEOUT_MS
    )

    # The forwarder posts the statusLine window after the turn settles; poll for it.
    context_window: int | None = None
    for _ in range(40):
        snapshot = httpx.get(f"{base_url}/v1/sessions/{session_id}", timeout=15).json()
        context_window = snapshot.get("context_window")
        if context_window:
            break
        time.sleep(3)

    ring = page.get_by_test_id("composer-context-ring")
    expect(ring.first).to_be_visible(timeout=60_000)

    assert context_window == _OPUS_1M_WINDOW, (
        f"Opus native session reported a {context_window}-token context "
        f"window; expected Opus's {_OPUS_1M_WINDOW}-token (1M) window."
    )

    # The window figure the user reads lives in the ring's hover tooltip:
    # "<used> / 1M tokens" for Opus, or "/ 200K tokens" when capped at the default.
    ring.first.hover()
    tooltip = page.locator('[data-slot="tooltip-content"]')
    expect(tooltip).to_contain_text(re.compile(r"/\s*1M tokens"), timeout=15_000)
    expect(tooltip).not_to_contain_text("200K")
