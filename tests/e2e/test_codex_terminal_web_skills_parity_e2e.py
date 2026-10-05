"""E2E regression: the Codex web skills menu must list shared ``~/.agents/skills``.
It drives the real ``GET /v1/skills`` resolver in-process (no browser or live host)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from omnigent.host.frames import HostSkillsFrame
from omnigent.host.skills import HostSkillDiscovery

_CODEX_HOST_SKILL = "codex-host-skill"
_AGENTS_SHARED_SKILL = "agents-shared-skill"
_CUSTOM_HOME_SKILL = "custom-codex-skill"


def _seed_skill(skills_dir: Path, name: str, description: str) -> None:
    """Write a minimal ``<skills_dir>/<name>/SKILL.md`` with valid frontmatter."""
    skill = skills_dir / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody of {name}.\n"
    )


def _menu_names(harness: str, workspace: Path) -> list[str]:
    """Resolve the skill names the web composer menu would list for ``harness``."""

    def unexpected_bundle(_: HostSkillsFrame) -> httpx.Response:
        raise AssertionError("Directory discovery must not fetch a session bundle")

    discovery = HostSkillDiscovery(unexpected_bundle)
    return [
        s["name"]
        for s in discovery.discover(HostSkillsFrame("menu", harness, str(workspace)), workspace)
    ]


def test_codex_web_menu_lists_shared_agents_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``codex-native`` web menu lists both the ``~/.codex/skills`` host tier and a
    skill under the shared ``~/.agents/skills``; the broken build drops the shared one."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    _seed_skill(home / ".codex" / "skills", _CODEX_HOST_SKILL, "Codex host-dir skill (both)")
    _seed_skill(
        home / ".agents" / "skills", _AGENTS_SHARED_SKILL, "shared .agents skill (dropped today)"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    names = _menu_names("codex-native", workspace)

    # Precondition (passes on the broken build too): the Codex host-dir tier
    # the menu already surfaced is still listed.
    assert _CODEX_HOST_SKILL in names, (
        f"precondition: ~/.codex/skills skill missing from the Codex menu; got {names}"
    )

    # THE BUG: the Codex CLI loads ~/.agents/skills, but the menu's source list
    # omitted it, so the menu dropped a command the terminal can invoke.
    assert _AGENTS_SHARED_SKILL in names, (
        f"the Codex web menu omits {_AGENTS_SHARED_SKILL!r} from ~/.agents/skills, "
        f"which the Codex terminal loads — the menu drops skills the terminal shows. "
        f"Menu: {names}"
    )


def test_codex_web_menu_shared_skills_survive_custom_codex_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom ``$CODEX_HOME`` relocates the host tier but not the shared
    ``~/.agents/skills`` tier, which lives under the user home and must still be listed."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    custom_codex_home = tmp_path / "custom-codex"
    monkeypatch.setenv("CODEX_HOME", str(custom_codex_home))
    _seed_skill(custom_codex_home / "skills", _CUSTOM_HOME_SKILL, "custom $CODEX_HOME host skill")
    _seed_skill(
        home / ".agents" / "skills", _AGENTS_SHARED_SKILL, "shared .agents skill (dropped today)"
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    names = _menu_names("codex-native", workspace)

    # Precondition (passes on the broken build too): the custom-home host tier
    # the menu already honored is still listed from $CODEX_HOME.
    assert _CUSTOM_HOME_SKILL in names, (
        f"precondition: $CODEX_HOME/skills skill missing from the Codex menu; got {names}"
    )

    # THE BUG (custom-home dimension): the shared ~/.agents/skills tier must
    # survive a relocated Codex home, since it lives under the user home.
    assert _AGENTS_SHARED_SKILL in names, (
        f"the Codex web menu omits {_AGENTS_SHARED_SKILL!r} from ~/.agents/skills "
        f"under a custom $CODEX_HOME — the shared tier must not depend on the Codex "
        f"home location. Menu: {names}"
    )
