"""Gateway selection survives the host and runner process boundaries."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from omnigent.cli import _build_host_daemon_env
from omnigent.host.connect import _build_runner_env


@pytest.mark.parametrize("server_url", [None, "https://example.databricksapps.com"])
@pytest.mark.parametrize("override", [None, "/config/gateway.cfg"])
def test_gateway_config_survives_host_and_runner(
    tmp_path: Path, server_url: str | None, override: str | None
) -> None:
    source = {
        "OMNIGENT_CONFIG_HOME": str(tmp_path),
        "UNRELATED_SECRET": "synthetic-secret",
        "ISAAC_OTHER_SECRET": "synthetic-secret",
    }
    if override is not None:
        source["ISAAC_GATEWAY_CFG"] = override
    with patch.dict(os.environ, source, clear=True):
        daemon_env = _build_host_daemon_env(server_url=server_url)
        runner_env = _build_runner_env(
            daemon_env,
            server_url=server_url or "http://localhost:6767",
            runner_id="runner_gateway_test",
            binding_token="synthetic-binding",
            workspace=str(tmp_path),
            parent_pid=12345,
        )
    for env in (daemon_env, runner_env):
        if override is None:
            assert "ISAAC_GATEWAY_CFG" not in env
        else:
            assert env["ISAAC_GATEWAY_CFG"] == override
        assert "UNRELATED_SECRET" not in env
        assert "ISAAC_OTHER_SECRET" not in env
