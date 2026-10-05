"""E2E: choosing a running sub-agent from the composer's sub-agent indicator.

The composer workspace bar shows a bot-glyph + count pill
(``SubagentTaskIndicator``) while sub-agents are busy. Choosing one from its
popover navigates the page to that child's session; inside the child the pill
must keep tallying the still-running sub-agent and lead with a row back to the
parent, so the climb-out is the same control the user came in through.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests._helpers.session import bind_session_runner, bundle_files, post_session_bundle
from tests.e2e_ui.conftest import (
    _ensure_runner_online,
    _server_state,
    configure_mock_llm,
    open_right_rail,
)

_LEAD_NAME = "delegating_lead"
_RESEARCHER = "researcher"


@dataclass(frozen=True)
class HeldSubagentSession:
    """A runner-bound parent whose single sub-agent turn parks on the mock gate.

    :param base_url: Product server base URL.
    :param mock_url: Mock LLM server base URL (``POST {mock_url}/gate/release``
        lets the parked sub-agent finish).
    :param session_id: Parent session id.
    :param routing_token: Substring that selects the parent's mock queue; the
        user's message must contain it.
    :param report_code: Per-run nonce only the sub-agent's reply carries.
    """

    base_url: str
    mock_url: str
    session_id: str
    routing_token: str
    report_code: str


def _lead_yaml(code: str) -> str:
    return f"""\
name: {_LEAD_NAME}
prompt: |
  You are a research lead. You never research anything yourself. When the
  user asks for research, call `sys_session_send` to delegate it to your
  `{_RESEARCHER}` sub-agent, then end your turn and wait. When the report
  arrives in your inbox, relay it to the user verbatim.

executor:
  model: gpt-4o-mini
  harness: openai-agents

tools:
  {_RESEARCHER}:
    type: agent
    description: Researches one topic and reports back.
    executor:
      model: gpt-4o-mini
      harness: openai-agents
    prompt: |
      You are a researcher. When asked to research anything, reply with
      exactly: Research complete. Report code: {code}.
"""


def create_held_subagent_session(
    base_url: str, mock_url: str, runner_id: str
) -> HeldSubagentSession:
    """Register the lead bundle, bind it to *runner_id*, and script the mock queues.

    The sub-agent's only response is gate-held (``block``), so after the parent
    dispatches it the child stays ``busy`` until ``release_held_subagent``.
    """
    suffix = uuid.uuid4().hex[:10]
    routing_token = f"lead-parent-{suffix}"
    child_token = f"lead-child-{suffix}"
    code = f"beacon-{suffix}"

    configure_mock_llm(
        mock_url,
        [
            {
                "tool_calls": [
                    {
                        "call_id": "call_researcher",
                        "name": "sys_session_send",
                        "arguments": json.dumps(
                            {
                                "agent": _RESEARCHER,
                                "title": _RESEARCHER,
                                "args": (
                                    "Research the composer sub-agent indicator. "
                                    f"Routing marker: {child_token}"
                                ),
                            }
                        ),
                    }
                ]
            },
            {"text": "Dispatched the researcher; waiting for the report."},
            {"text": f"The researcher reported: Report code: {code}."},
        ],
        key=routing_token,
        match=routing_token,
    )
    configure_mock_llm(
        mock_url,
        [{"text": f"Research complete. Report code: {code}.", "block": True}],
        key=child_token,
        match=child_token,
    )

    bundle = bundle_files({f"{_LEAD_NAME}.yaml": _lead_yaml(code).encode()})
    create = post_session_bundle(httpx.post, f"{base_url}/v1/sessions", bundle, timeout=30.0)
    create.raise_for_status()
    session_id = str(create.json()["session_id"])
    bind_session_runner(httpx.patch, base_url, session_id, runner_id, timeout=10.0)
    return HeldSubagentSession(
        base_url=base_url,
        mock_url=mock_url,
        session_id=session_id,
        routing_token=routing_token,
        report_code=code,
    )


def child_sessions(chat: HeldSubagentSession) -> list[dict]:
    resp = httpx.get(f"{chat.base_url}/v1/sessions/{chat.session_id}/child_sessions", timeout=10.0)
    resp.raise_for_status()
    return list(resp.json()["data"])


def wait_for_held_gate(chat: HeldSubagentSession, *, timeout_s: float = 60.0) -> None:
    """Wait until the child's model request is parked on the mock gate."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if httpx.get(f"{chat.mock_url}/gate/pending", timeout=5.0).json().get("pending"):
            return
        time.sleep(0.5)
    raise AssertionError("the sub-agent never reached the gate-held mock response")


def release_held_subagent(chat: HeldSubagentSession, *, settle_s: float = 60.0) -> None:
    """Release the mock gate and wait for the child to leave ``busy``."""
    with contextlib.suppress(httpx.HTTPError):
        httpx.post(f"{chat.mock_url}/gate/release", timeout=5.0)
    deadline = time.monotonic() + settle_s
    while time.monotonic() < deadline:
        with contextlib.suppress(httpx.HTTPError):
            if all(not child["busy"] for child in child_sessions(chat)):
                return
        time.sleep(1.0)


def delete_session_tree(chat: HeldSubagentSession) -> None:
    with contextlib.suppress(httpx.HTTPError):
        for child in child_sessions(chat):
            httpx.delete(f"{chat.base_url}/v1/sessions/{child['id']}", timeout=10.0)
    with contextlib.suppress(httpx.HTTPError):
        httpx.delete(f"{chat.base_url}/v1/sessions/{chat.session_id}", timeout=10.0)


@pytest.fixture
def held_subagent_session(
    live_server: str,
    mock_llm_server_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[HeldSubagentSession]:
    respawned = _ensure_runner_online(live_server, tmp_path_factory)
    chat = create_held_subagent_session(
        live_server, mock_llm_server_url, str(_server_state["runner_id"])
    )
    try:
        yield chat
    finally:
        # Never leave the shared runner parked on the gate.
        release_held_subagent(chat)
        delete_session_tree(chat)
        if respawned is not None:
            respawned.terminate()
            try:
                respawned.wait(timeout=5)
            except subprocess.TimeoutExpired:
                respawned.kill()
                respawned.wait(timeout=5)


def dispatch_until_pill_shows(page: Page, chat: HeldSubagentSession) -> str:
    """Open the parent, ask for research, and wait for the composer pill to show 1.

    :returns: The busy child's session id.
    """
    page.goto(f"{chat.base_url}/c/{chat.session_id}")
    composer = page.get_by_role("textbox", name="Message the agent")
    expect(composer).to_be_visible(timeout=30_000)
    composer.fill(
        "Please research the composer sub-agent indicator and report back. "
        f"Routing marker: {chat.routing_token}"
    )
    page.get_by_role("button", name="Send", exact=True).click()

    pill = page.get_by_test_id("subagent-task-pill")
    expect(pill).to_be_visible(timeout=60_000)
    expect(pill).to_have_text("1")
    wait_for_held_gate(chat)
    children = child_sessions(chat)
    assert len(children) == 1 and children[0]["busy"], children
    return str(children[0]["id"])


@pytest.mark.timeout(300)
def test_subagent_indicator_stays_visible_after_opening_running_subagent(
    request: pytest.FixtureRequest,
    held_subagent_session: HeldSubagentSession,
) -> None:
    chat = held_subagent_session
    page: Page = request.getfixturevalue("page")
    child_id = dispatch_until_pill_shows(page, chat)

    page.get_by_test_id("subagent-task-pill").click()
    popover = page.get_by_role("dialog", name=re.compile("sub-agent"))
    row = popover.get_by_role("link").filter(has_text="Working")
    expect(row).to_be_visible()
    row.click()
    page.wait_for_url(re.compile(re.escape(f"/c/{child_id}")))

    # The whole page swapped to the child; the breadcrumb still links back.
    back_link = page.get_by_role("link", name="Back to parent session")
    expect(back_link).to_be_visible(timeout=30_000)
    expect(back_link).to_have_attribute("href", re.compile(re.escape(f"/c/{chat.session_id}")))
    breadcrumb = page.get_by_role("navigation", name="Conversation")
    expect(breadcrumb.get_by_text(_RESEARCHER, exact=True)).to_be_visible()

    expect(page.get_by_test_id("composer-workspace-controls")).to_be_visible(timeout=30_000)
    assert child_sessions(chat)[0]["busy"], "the sub-agent must still be running here"
    open_right_rail(page)
    rail = page.get_by_role("complementary", name="Workspace")
    rail.get_by_role("tab", name=re.compile("^Agents")).click()
    child_row = rail.locator(f'[data-testid="subagent-row"][data-child-session-id="{child_id}"]')
    expect(child_row.locator('[data-testid="subagent-status-avatar"]')).to_have_attribute(
        "data-activity", "working"
    )

    # The sub-agent is still running, so its indicator must not vanish; from
    # here it tallies the viewed sub-agent itself.
    pill = page.get_by_test_id("subagent-task-pill")
    expect(pill).to_be_visible(timeout=15_000)
    expect(pill).to_have_text("1")
    expect(pill).to_have_accessible_name("1 sub-agent: 1 active")

    # Its popover leads with the way back and marks the viewed sub-agent current.
    pill.click()
    popover = page.get_by_role("dialog", name=re.compile("sub-agent"))
    parent_row = popover.get_by_test_id("subagent-indicator-parent-row")
    expect(parent_row).to_be_visible()
    expect(parent_row).to_contain_text("Back to parent")
    expect(parent_row).to_have_attribute("href", re.compile(re.escape(f"/c/{chat.session_id}")))
    current_row = popover.get_by_role("link").filter(has_text="Working")
    expect(current_row).to_have_attribute("aria-current", "page")
    expect(current_row).to_have_attribute("href", re.compile(re.escape(f"/c/{child_id}")))

    # Climbing out through that row lands on the parent with its tally intact.
    parent_row.click()
    page.wait_for_url(re.compile(re.escape(f"/c/{chat.session_id}")))
    expect(page.get_by_test_id("subagent-task-pill")).to_have_text("1", timeout=30_000)
    expect(page.get_by_role("link", name="Back to parent session")).to_have_count(0)
