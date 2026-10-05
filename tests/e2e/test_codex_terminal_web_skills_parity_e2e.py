"""E2E regression: the Codex web skills menu must list shared ``~/.agents/skills``.

A codex-family session's web composer menu is fed by ``GET /v1/skills``
(``resolve_harness_skills`` → the Codex ``codex_host_skills`` provider), which
draws from the same ``codex_skill_sources`` list the executor symlinks into
``$CODEX_HOME/skills/``. The Codex CLI loads skills from ``~/.agents/skills``,
but that list omitted it, so the menu dropped commands the terminal can run.

The tests drive the real resolver (``HostSkillDiscovery`` →
``resolve_harness_skills``) that feeds the composer menu, so they need no
browser or live host: they fail on the broken build and pass once the shared
dir joins the source list, including under a custom ``$CODEX_HOME``.
"""

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
    """Resolve the skill names the web composer menu would list for ``harness``.

    Uses the in-process ``HostSkillDiscovery`` path behind ``GET /v1/skills``,
    with a bundle fetch that must never fire for directory discovery.
    """

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
    """The codex-family web menu must list the shared ``~/.agents/skills`` tier.

    The user journey: the user home carries a skill under the Codex host dir
    (``~/.codex/skills``) and another under the shared ``~/.agents/skills``; the
    user opens the web composer's slash menu for a ``codex-native`` session. The
    Codex CLI loads both, so the menu must list both. On the broken build the
    menu omits the ``~/.agents/skills`` skill — exactly the reported "Codex
    skills menu omits skills from ~/.agents/skills".
    """
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
    """A custom ``$CODEX_HOME`` moves the host dir but keeps the shared tier.

    A native ``codex-native`` session honors ``$CODEX_HOME`` for its host dir,
    so the menu reads the host skill from there instead of ``~/.codex``. The
    shared ``~/.agents/skills`` dir lives under the user home, not the Codex
    home, so moving ``$CODEX_HOME`` must not drop it. On the broken build the
    menu still omits the shared skill regardless of ``$CODEX_HOME``.
    """
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
