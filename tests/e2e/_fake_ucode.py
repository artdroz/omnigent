"""Stand-in for the ``ucode`` / ``ug`` CLI used by the setup-workspace e2e test.

The real CLI needs a private git source and Databricks OAuth, so this writes only
the files Omnigent reads after ``ug configure``: ``~/.ucode/state.json`` (shape
per :mod:`omnigent.onboarding.ucode_state`) and the ``~/.databrickscfg`` profile.
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

_AGENT_DISPLAY = {"claude": "Claude Code", "codex": "Codex", "pi": "Pi"}
_CLAUDE_MODELS = {
    "opus": "databricks-claude-opus-4-7",
    "sonnet": "databricks-claude-sonnet-4-6",
    "haiku": "databricks-claude-haiku-4-5",
}
_CODEX_MODELS = ["databricks-gpt-5-5"]


def _state_path() -> Path:
    return Path.home() / ".ucode" / "state.json"


def _cfg_path() -> Path:
    return Path.home() / ".databrickscfg"


def _read_cfg() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(_cfg_path())
    return cfg


def _profile_for_host(cfg: configparser.ConfigParser, url: str) -> str | None:
    for section in cfg.sections():
        if cfg.get(section, "host", fallback="").rstrip("/") == url:
            return section
    return None


def _save_profile(url: str, requested: str | None) -> str:
    cfg = _read_cfg()
    existing = _profile_for_host(cfg, url)
    if existing is not None and requested in (None, existing):
        return existing
    name = requested or urlparse(url).netloc.split(".")[0]
    cfg[name] = {"host": url, "auth_type": "databricks-cli"}
    _cfg_path().parent.mkdir(parents=True, exist_ok=True)
    with _cfg_path().open("w") as handle:
        cfg.write(handle)
    return name


def _agent_entry(url: str, agent: str, profile: str) -> dict[str, object]:
    auth_command = f"databricks auth token --host {url} --profile {profile}"
    if agent == "claude":
        base_url = f"{url}/ai-gateway/anthropic"
        return {
            "model": _CLAUDE_MODELS["opus"],
            "base_url": base_url,
            "auth_command": auth_command,
            "auth_refresh_interval_ms": 900000,
            "env": {"ANTHROPIC_BASE_URL": base_url},
        }
    if agent == "codex":
        return {
            "model": _CODEX_MODELS[0],
            "base_url": f"{url}/ai-gateway/codex/v1",
            "auth_command": auth_command,
            "auth_refresh_interval_ms": 900000,
        }
    return {
        "model": _CLAUDE_MODELS["opus"],
        "base_urls": {
            "claude": f"{url}/ai-gateway/anthropic",
            "codex": f"{url}/ai-gateway/codex/v1",
        },
        "auth_command": auth_command,
        "auth_refresh_interval_ms": 900000,
    }


def _load_state() -> dict[str, object]:
    path = _state_path()
    if not path.exists():
        return {"state_version": 3, "workspaces": {}}
    return json.loads(path.read_text())


def configure(targets: list[tuple[str, str | None]], agents: list[str]) -> int:
    """Configure each ``(workspace url, requested profile name)`` target."""
    state = _load_state()
    workspaces = state.get("workspaces")
    if not isinstance(workspaces, dict):
        workspaces = {}
        state["workspaces"] = workspaces
    for url, requested in targets:
        url = url.rstrip("/")
        name = _save_profile(url, requested)
        print(f"Select workspace: {name}  {url}")
        print("Databricks Login")
        print(f"  Workspace: {url}")
        print(f"Profile {name} was successfully saved")
        print("Databricks authentication complete")
        print("Unity Gateway connected")
        entry = workspaces.get(url)
        existing_agents = entry.get("agents") if isinstance(entry, dict) else None
        if not isinstance(existing_agents, dict):
            existing_agents = {}
        for agent in agents:
            existing_agents[agent] = _agent_entry(url, agent, name)
            print(f"Settings configured for {_AGENT_DISPLAY.get(agent, agent)}")
        workspaces[url] = {
            "workspace": url,
            "fable_enabled": True,
            "claude_models": dict(_CLAUDE_MODELS),
            "codex_models": list(_CODEX_MODELS),
            "base_urls": {
                "claude": f"{url}/ai-gateway/anthropic",
                "codex": f"{url}/ai-gateway/codex/v1",
            },
            "available_tools": sorted(existing_agents),
            "agents": existing_agents,
        }
        state["current_workspace"] = url
        print("Configuration")
        print(f"  Workspace: {url}")
        print(
            "  Coding Agents: "
            + ", ".join(_AGENT_DISPLAY.get(a, a) for a in sorted(existing_agents))
        )
    _state_path().parent.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=2) + "\n")
    print("Configuration complete - launch with ug.")
    return 0


def status() -> int:
    state = _load_state()
    current = state.get("current_workspace")
    workspaces = state.get("workspaces")
    if (
        not isinstance(current, str)
        or not isinstance(workspaces, dict)
        or current not in workspaces
    ):
        print("Not configured - run `ug configure`.")
        return 1
    profile = _profile_for_host(_read_cfg(), current) or "-"
    entry = workspaces[current]
    agents = entry.get("agents", {}) if isinstance(entry, dict) else {}
    print("Configured")
    print("Provider")
    print(f"  Workspace URL:  {current}")
    print(f"  CLI profile:    {profile}")
    print("Coding Agents")
    for agent, data in sorted(agents.items()):
        print(f"  {_AGENT_DISPLAY.get(agent, agent)}")
        print("    Model provider:  Databricks AI Gateway")
        endpoint = data.get("base_url") or ", ".join(data.get("base_urls", {}).values())
        print(f"    Endpoint:  {endpoint}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ug", description="fake ucode for tests")
    sub = parser.add_subparsers(dest="command", required=True)
    conf = sub.add_parser("configure")
    # The two real invocations: ``--workspaces <urls>`` from interactive setup and
    # ``--profiles <names>`` from the headless sandbox boot.
    conf.add_argument("--workspaces", default=None)
    conf.add_argument("--profiles", default=None)
    conf.add_argument("--agents", default="claude,codex,pi")
    conf.add_argument("--profile", default=None)
    for flag in (
        "--enable-fable",
        "--skip-validate",
        "--skip-upgrade",
        "--skip-unavailable",
        "--use-pat",
    ):
        conf.add_argument(flag, action="store_true")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "configure":
        agents = [a for a in args.agents.split(",") if a]
        targets: list[tuple[str, str | None]] = []
        if args.profiles:
            cfg = _read_cfg()
            for name in args.profiles.split(","):
                host = cfg.get(name, "host", fallback=None) if name else None
                if not host:
                    parser.error(f"profile {name!r} has no host in ~/.databrickscfg")
                targets.append((host, name))
        elif args.workspaces:
            targets = [(u, args.profile) for u in args.workspaces.split(",") if u]
        else:
            parser.error("configure needs --workspaces or --profiles")
        return configure(targets, agents)
    return status()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
