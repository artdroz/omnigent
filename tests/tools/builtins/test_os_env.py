"""``sys_os_shell`` rejects argument names its schema does not declare."""

from __future__ import annotations

import json
from typing import Any

from omnigent.tools.base import ToolContext
from omnigent.tools.builtins.os_env import SysOsShellTool


class _RecordingOSEnv:
    """Stands in for :class:`OSEnvironment`; records ``shell`` calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def shell(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        self.calls.append({"command": command, "timeout": timeout})
        return {"stdout": "", "stderr": "", "exit_code": 0, "timed_out": False}


def _ctx() -> ToolContext:
    return ToolContext(task_id="task", agent_id="agent", conversation_id="conv")


def _invoke(tool: SysOsShellTool, arguments: dict[str, Any]) -> dict[str, Any]:
    return json.loads(tool.invoke(json.dumps(arguments), _ctx()))


def test_sys_os_shell_rejects_unknown_argument_names() -> None:
    os_env = _RecordingOSEnv()
    tool = SysOsShellTool(os_env)

    result = _invoke(tool, {"command": "sleep 300", "timeout_seconds": 500})

    assert "error" in result, result
    assert "timeout_seconds" in result["error"]
    assert "command, timeout" in result["error"]
    assert os_env.calls == []


def test_sys_os_shell_forwards_declared_arguments() -> None:
    os_env = _RecordingOSEnv()
    tool = SysOsShellTool(os_env)

    result = _invoke(tool, {"command": "sleep 1", "timeout": 500})

    assert result["exit_code"] == 0
    assert os_env.calls == [{"command": "sleep 1", "timeout": 500}]


def test_sys_os_shell_schema_declares_no_additional_properties() -> None:
    parameters = SysOsShellTool.get_schema()["function"]["parameters"]

    assert parameters.get("additionalProperties") is False, parameters
    assert sorted(parameters["properties"]) == ["command", "timeout"]
