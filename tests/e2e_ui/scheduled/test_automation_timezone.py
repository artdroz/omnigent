"""UI journeys: an automation's schedule is evaluated in the user's local timezone.

Both tests run the browser in ``America/Los_Angeles`` so the SPA's ``Intl``
zone differs from the server process zone (set ``TZ`` on the pytest process to
make the spawned server run in another zone, e.g. ``TZ=America/Chicago``).

* ``test_dialog_automation_fires_at_browser_local_time`` creates a daily
  automation through the New automation dialog for a wall-clock time a few
  minutes ahead in the browser's zone and waits for the scheduler to fire it.
* ``test_chat_created_automation_uses_user_local_timezone`` asks the agent in
  chat to create a daily 9:00 AM automation without naming a zone; the mock
  model answers with a ``sys_scheduled_task_create`` call that omits
  ``timezone`` (the tool gives the agent no user-zone information).

The ``online_host`` fixture starts a real ``omnigent host`` daemon against the
live e2e server so a due automation has a host to run on; without one the fire
path records a ``no_online_host`` failure instead of launching a session.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest
from playwright.sync_api import Page, expect

from tests.e2e_ui.conftest import _server_state, configure_mock_llm
from tests.e2e_ui.scheduled.test_scheduled_tasks_page import _builtin_agent_id, _row_by_name

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HOST_ONLINE_TIMEOUT_S = 180.0
_FIRE_GRACE_S = 90.0
_BROWSER_TZ = "America/Los_Angeles"
_LA = ZoneInfo(_BROWSER_TZ)
# Optional evidence screenshots; unset means none are written.
_SHOTS_DIR = (
    Path(os.environ["OMNIGENT_E2E_SHOTS_DIR"])
    if os.environ.get("OMNIGENT_E2E_SHOTS_DIR")
    else None
)


def _online_host_row(base_url: str, host_name: str) -> dict[str, Any] | None:
    hosts = httpx.get(f"{base_url}/v1/hosts", timeout=10.0).json().get("hosts", [])
    return next(
        (h for h in hosts if h.get("name") == host_name and h.get("status") == "online"),
        None,
    )


@pytest.fixture(scope="module")
def online_host(
    live_server: str,
    mock_llm_server_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Any]]:
    """A real ``omnigent host`` daemon, online on ``live_server`` for the module."""
    tmp = tmp_path_factory.mktemp("automation_host")
    home = tmp / "home"
    home.mkdir()
    host_name = f"automation-host-{uuid.uuid4().hex[:8]}"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "OMNIGENT_CONFIG_HOME": str(home / ".config" / "omnigent"),
        "PYTHONPATH": os.pathsep.join(
            [
                str(_REPO_ROOT),
                str(_REPO_ROOT / "sdks" / "python-client"),
                str(_REPO_ROOT / "sdks" / "ui"),
            ]
        ),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "OMNIGENT_HOST_NAME": host_name,
        "OMNIGENT_HOST_ID": uuid.uuid4().hex,
        "OPENAI_BASE_URL": f"{mock_llm_server_url}/v1",
        "OPENAI_API_KEY": "mock-key",
    }
    if "TZ" in os.environ:
        env["TZ"] = os.environ["TZ"]
    log_path = tmp / "host.log"
    with log_path.open("w") as log_handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "omnigent",
                "host",
                "--server",
                live_server,
                "--non-interactive",
                "--no-open",
            ],
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + _HOST_ONLINE_TIMEOUT_S
        row: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            row = _online_host_row(live_server, host_name)
            if row is not None:
                break
            if proc.poll() is not None:
                raise RuntimeError(
                    f"omnigent host exited early ({proc.returncode}):\n"
                    f"{log_path.read_text()[-2000:]}"
                )
            time.sleep(1.0)
        if row is None:
            raise RuntimeError(f"host never came online:\n{log_path.read_text()[-2000:]}")
        yield {**row, "log_path": str(log_path)}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _server_process_tz() -> list[str]:
    pid = _server_state.get("pid")
    if pid is None:
        return ["<attached server: TZ unknown>"]
    environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    return [e.decode() for e in environ if e.startswith(b"TZ=")] or ["TZ unset (process default)"]


def _task_by_name(base_url: str, name: str) -> dict[str, Any] | None:
    resp = httpx.get(f"{base_url}/v1/scheduled-tasks", timeout=10.0)
    resp.raise_for_status()
    return next((t for t in resp.json()["scheduled_tasks"] if t["name"] == name), None)


def _runs(base_url: str, task_id: str) -> list[dict[str, Any]]:
    resp = httpx.get(f"{base_url}/v1/scheduled-tasks/{task_id}/runs", timeout=10.0)
    resp.raise_for_status()
    return resp.json()["runs"]


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _clock_12h(hour24: int, minute: int) -> tuple[int, str]:
    period = "PM" if hour24 >= 12 else "AM"
    hour12 = hour24 % 12 or 12
    return hour12, period


def _shot(page: Page, name: str) -> None:
    if _SHOTS_DIR is None:
        return
    _SHOTS_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(_SHOTS_DIR / f"{name}.png"))


def _pick_agent(page: Page, agent_id: str, label: str) -> None:
    trigger = page.get_by_test_id("task-agent-picker").get_by_test_id(
        "new-chat-landing-agent-select"
    )
    trigger.click()
    item = page.get_by_test_id(f"new-chat-landing-agent-{agent_id}")
    if not item.is_visible():
        page.get_by_test_id("new-chat-landing-custom-agents").hover()
    item.click()
    expect(trigger).to_contain_text(re.compile(label, re.IGNORECASE), timeout=10_000)


def _type_time(page: Page, hour24: int, minute: int) -> None:
    hour12, period = _clock_12h(hour24, minute)
    time_input = page.get_by_test_id("schedule-time")
    time_input.fill("")
    time_input.click()
    page.keyboard.type(f"{hour12}:{minute:02d} {period}")
    # Focusing the time input opened its picker popover; a forced click on the
    # name input blurs and closes it without hitting the dialog overlay.
    page.get_by_test_id("task-name-input").click(force=True)
    expect(time_input).to_have_value(f"{hour12:02d}:{minute:02d} {period}")


def _send(page: Page, text: str) -> None:
    page.get_by_label("Message the agent").fill(text)
    page.get_by_role("button", name="Send", exact=True).click()


@pytest.mark.browser_context_args(timezone_id=_BROWSER_TZ)
def test_dialog_automation_fires_at_browser_local_time(
    request: pytest.FixtureRequest,
    live_server: str,
    online_host: dict[str, Any],
) -> None:
    """A daily automation created in the dialog fires at the browser's local wall-clock time."""
    agent_id = _builtin_agent_id(live_server, "hello_world")
    name = f"Local-time digest {uuid.uuid4().hex[:6]}"
    prompt = "Summarize what changed today."
    print(f"\nserver TZ env={_server_process_tz()} host={online_host['host_id']}")

    page: Page = request.getfixturevalue("page")
    page.goto(f"{live_server}/tasks")
    browser_tz = page.evaluate("Intl.DateTimeFormat().resolvedOptions().timeZone")
    assert browser_tz == _BROWSER_TZ, browser_tz
    expect(page.get_by_test_id("new-task-button")).to_be_visible(timeout=30_000)

    # Pick the due time only now so the dialog steps below start well ahead of it.
    due_local = (datetime.now(_LA) + timedelta(minutes=3)).replace(second=0, microsecond=0)
    hour12, period = _clock_12h(due_local.hour, due_local.minute)
    print(f"browser now={datetime.now(_LA).isoformat()} due={due_local.isoformat()}")

    page.get_by_test_id("new-task-button").click()
    dialog = page.get_by_test_id("create-scheduled-task-dialog")
    expect(dialog).to_be_visible(timeout=30_000)
    page.get_by_test_id("task-name-input").fill(name)
    page.get_by_test_id("task-prompt-input").fill(prompt)
    _pick_agent(page, agent_id, "hello_world")
    expect(page.get_by_test_id("schedule-preset-trigger")).to_contain_text("Daily")
    _type_time(page, due_local.hour, due_local.minute)
    timezone_controls = dialog.get_by_test_id("task-timezone-trigger").count()
    print(f"dialog timezone controls={timezone_controls} dialog text={dialog.inner_text()!r}")
    _shot(page, "dialog-before-create")
    page.get_by_test_id("create-scheduled-task-submit").click()

    row = _row_by_name(page, name)
    expect(row).to_be_visible(timeout=30_000)
    schedule_line = row.get_by_test_id("task-schedule-line")
    expect(schedule_line).to_contain_text(f"Every day at {hour12}:{due_local.minute:02d} {period}")
    next_run_label = row.get_by_test_id("task-next-run")
    expect(next_run_label).to_contain_text("Next run", timeout=30_000)
    page.wait_for_timeout(3_000)
    _shot(page, "row-after-create")

    task = _task_by_name(live_server, name)
    assert task is not None, "created automation is not listed by the API"
    task_id = task["id"]
    try:
        label_text = next_run_label.inner_text()
        line_text = schedule_line.inner_text()
        expected_instant = due_local.astimezone(UTC)
        actual_instant = _parse_iso(task["next_run_at"])
        print(
            f"stored timezone={task['timezone']} rrule={task['rrule']} "
            f"next_run_at={task['next_run_at']} expected={expected_instant.isoformat()} "
            f"row={line_text!r}"
        )
        assert task["timezone"] == _BROWSER_TZ, task
        assert actual_instant == expected_instant, (
            f"next_run_at {actual_instant.isoformat()} != chosen "
            f"{due_local.isoformat()} ({expected_instant.isoformat()} UTC)"
        )
        assert re.search(r"Next run in [1-4] mins?", label_text), label_text

        # Let the scheduler fire it for real on the online host.
        deadline = time.monotonic() + (expected_instant - datetime.now(UTC)).total_seconds()
        deadline += _FIRE_GRACE_S
        runs: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            runs = _runs(live_server, task_id)
            if runs and runs[0]["status"] in ("succeeded", "failed"):
                break
            page.wait_for_timeout(2_000)
        print(f"runs={runs}")
        assert runs, f"no run recorded by {_FIRE_GRACE_S:.0f}s after {due_local.isoformat()}"
        fired_at = datetime.fromtimestamp(runs[0]["fired_at"], tz=UTC)
        assert -5 <= (fired_at - expected_instant).total_seconds() <= _FIRE_GRACE_S, (
            f"fired at {fired_at.isoformat()} but the chosen local time was "
            f"{due_local.isoformat()} ({expected_instant.isoformat()} UTC)"
        )
        assert runs[0]["status"] == "succeeded", runs[0]
        _shot(page, "row-after-fire")

        page.goto(f"{live_server}/c/{runs[0]['conversation_id']}")
        expect(page.get_by_text(prompt).first).to_be_visible(timeout=30_000)
        page.wait_for_timeout(3_000)
        _shot(page, "fired-session")
    finally:
        httpx.delete(f"{live_server}/v1/scheduled-tasks/{task_id}", timeout=10.0)


def _next_wall_clock(hour: int, minute: int, tz: ZoneInfo, now: datetime) -> datetime:
    local_now = now.astimezone(tz)
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    return candidate.astimezone(UTC)


@pytest.mark.browser_context_args(timezone_id=_BROWSER_TZ)
def test_chat_created_automation_uses_user_local_timezone(
    request: pytest.FixtureRequest,
    live_server: str,
    seeded_session: tuple[str, str],
    mock_llm_server_url: str,
) -> None:
    """An automation the agent creates from chat for "9:00 AM" uses the user's local zone."""
    base_url, session_id = seeded_session
    agent_id = _builtin_agent_id(live_server, "hello_world")
    name = f"Daily inbox summary {uuid.uuid4().hex[:6]}"
    nonce = uuid.uuid4().hex[:8]
    user_text = (
        f"Create an automation that runs every day at 9:00 AM and summarizes my inbox. [{nonce}]"
    )
    tool_arguments = {
        "name": name,
        "prompt": "Summarize my inbox.",
        "rrule": "FREQ=DAILY;BYHOUR=9;BYMINUTE=0",
        "agent_id": agent_id,
    }
    configure_mock_llm(
        mock_llm_server_url,
        [
            {
                "tool_calls": [
                    {
                        "call_id": f"call_{nonce}",
                        "name": "sys_scheduled_task_create",
                        "arguments": json.dumps(tool_arguments),
                    }
                ]
            },
            {"text": f"Done. I created the automation “{name}” to run every day at 9:00 AM."},
        ],
        match=nonce,
        required_tools=["sys_scheduled_task_create"],
    )
    print(f"\nserver TZ env={_server_process_tz()} session={session_id}")

    page: Page = request.getfixturevalue("page")
    page.goto(f"{base_url}/c/{session_id}")
    browser_tz = page.evaluate("Intl.DateTimeFormat().resolvedOptions().timeZone")
    assert browser_tz == _BROWSER_TZ, browser_tz
    expect(page.get_by_label("Message the agent")).to_be_visible(timeout=30_000)
    sent_at = datetime.now(UTC)
    _send(page, user_text)
    expect(page.get_by_text("I created the automation")).to_be_visible(timeout=120_000)
    page.wait_for_timeout(2_000)
    _shot(page, "chat-reply")

    task = _task_by_name(live_server, name)
    assert task is not None, "the agent's sys_scheduled_task_create did not create a task"
    task_id = task["id"]
    try:
        page.goto(f"{live_server}/tasks")
        row = _row_by_name(page, name)
        expect(row).to_be_visible(timeout=30_000)
        expect(row.get_by_test_id("task-schedule-line")).to_contain_text("Every day at 9:00 AM")
        expect(row.get_by_test_id("task-next-run")).to_contain_text("Next run", timeout=30_000)
        page.wait_for_timeout(3_000)
        _shot(page, "chat-created-row")
        line_text = row.get_by_test_id("task-schedule-line").inner_text()

        captured = httpx.get(f"{mock_llm_server_url}/mock/requests", timeout=10.0).json()
        if _SHOTS_DIR is not None:
            _SHOTS_DIR.mkdir(parents=True, exist_ok=True)
            (_SHOTS_DIR / "mock-requests.json").write_text(json.dumps(captured, indent=2))
        print(f"mock captured {len(captured.get('requests', []))} request(s)")
        actual_instant = _parse_iso(task["next_run_at"])
        expected_local = _next_wall_clock(9, 0, _LA, sent_at)
        utc_nine = _next_wall_clock(9, 0, ZoneInfo("UTC"), sent_at)
        print(
            f"stored timezone={task['timezone']} next_run_at={task['next_run_at']} "
            f"9:00 {_BROWSER_TZ}={expected_local.isoformat()} 9:00 UTC={utc_nine.isoformat()} "
            f"row={line_text!r}"
        )
        assert task["timezone"] == _BROWSER_TZ and actual_instant == expected_local, (
            f"chat-created automation is evaluated in {task['timezone']!r} "
            f"(next_run_at {actual_instant.isoformat()}, i.e. "
            f"{actual_instant.astimezone(_LA).strftime('%-I:%M %p')} {_BROWSER_TZ}) while the "
            f"row reads {line_text!r}; expected 9:00 AM {_BROWSER_TZ} = "
            f"{expected_local.isoformat()}"
        )
    finally:
        httpx.delete(f"{live_server}/v1/scheduled-tasks/{task_id}", timeout=10.0)
