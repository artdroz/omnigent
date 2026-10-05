"""``omni setup`` surfaces the drift after ``ug configure`` switches ucode's
current workspace away from the configured profile, while the Databricks label
keeps naming the profile sessions route through (not ucode's current_workspace)."""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

pexpect = pytest.importorskip("pexpect")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FAKE_UCODE = Path(__file__).with_name("_fake_ucode.py")

WORKSPACE_A = "https://workspace-a.cloud.databricks.com"
WORKSPACE_B = "https://workspace-b.cloud.databricks.com"
PROFILE_A = "ai_devtools"
PROFILE_B = "fevm2"

_ANSI_RE = re.compile(rb"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\x1b[=>]")
# ``omni setup`` resolves ucode through ``uvx --from <git source> ucode``.
_UVX_SHIM = (
    "#!/usr/bin/env bash\n"
    'if [ "${{1:-}}" = "--from" ] && [ "${{3:-}}" = "ucode" ]; then\n'
    "  shift 3\n"
    '  exec {python} {script} "$@"\n'
    "fi\n"
    'echo "fake uvx: unsupported invocation: $*" >&2\n'
    "exit 2\n"
)
_UG_SHIM = '#!/usr/bin/env bash\nexec {python} {script} "$@"\n'


def _write_shims(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name, template in (("uvx", _UVX_SHIM), ("ug", _UG_SHIM)):
        path = bin_dir / name
        path.write_text(template.format(python=sys.executable, script=_FAKE_UCODE))
        path.chmod(0o755)


def _env(home: Path, bin_dir: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("DATABRICKS_", "OMNIGENT_"))
        and not k.endswith("_API_KEY")
        and k not in ("GH_TOKEN", "GITHUB_TOKEN")
    }
    env.update(
        HOME=str(home),
        PATH=f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
        PYTHONPATH=str(_REPO_ROOT),
        NO_COLOR="1",
        TERM="xterm",
        OMNIGENT_CONFIG_HOME=str(home / ".omnigent"),
    )
    return env


def _spawn_setup(env: dict[str, str]) -> pexpect.spawn:
    child = pexpect.spawn(
        sys.executable,
        ["-m", "omnigent", "setup"],
        env=env,
        encoding=None,
        dimensions=(40, 120),
        timeout=120,
    )
    child.expect(rb"Configure harnesses", timeout=120)
    return child


def _frame(child: pexpect.spawn, settle: float = 1.5) -> str:
    """Return the ANSI-stripped output already buffered plus anything arriving within *settle*."""
    buf = bytes(child.buffer)
    child.buffer = b""
    deadline = time.time() + settle
    while time.time() < deadline:
        try:
            buf += child.read_nonblocking(4096, timeout=0.3)
        except pexpect.TIMEOUT:
            continue
        except pexpect.EOF:
            break
    return _ANSI_RE.sub(b"", buf).decode("utf-8", "replace")


def _status_row(overview: str, harness: str) -> str:
    for line in reversed(overview.splitlines()):
        stripped = line.replace("❯", "").strip()
        if stripped.startswith(harness + " "):
            return " ".join(stripped.split())
    raise AssertionError(f"no {harness!r} row in the harness overview:\n{overview}")


def _quit(child: pexpect.spawn) -> None:
    with contextlib.suppress(Exception):
        child.send(b"\x1b")
        child.expect(pexpect.EOF, timeout=30)
    with contextlib.suppress(Exception):
        child.close(force=True)


def _add_databricks_workspace(child: pexpect.spawn, url: str) -> str:
    """Add a Databricks credential for *url* from the Claude page; return the overview after."""
    _frame(child, 2.0)
    child.send(b"\r")  # Claude is the first harness row
    child.expect(rb"select or add a credential", timeout=60)
    _frame(child, 1.0)
    child.send(b"\r")  # "+ Add a credential" is the only action with no credentials yet
    child.expect(rb"What do you want to add", timeout=60)
    _frame(child, 1.0)
    for _ in range(3):  # Anthropic key, Claude subscription, Gateway, then Databricks
        child.send(b"j")
        time.sleep(0.3)
    menu = _frame(child, 1.0)
    # Each Down redraws the menu; the last frame carries the current pointer.
    pointer_rows = [line for line in menu.splitlines() if "❯" in line]
    pointer = pointer_rows[-1] if pointer_rows else ""
    assert "Databricks" in pointer, menu
    child.send(b"\r")
    child.expect(rb"Databricks workspace URL", timeout=60)
    time.sleep(1.0)
    child.send(url.encode() + b"\r")
    child.expect(rb"Added databricks", timeout=120)
    _frame(child, 2.0)
    child.send(b"\x1b")
    child.expect(rb"Configure harnesses", timeout=60)
    return _frame(child, 2.5)


def _switch_ucode_workspace(env: dict[str, str], url: str, profile: str) -> None:
    subprocess.run(
        ["ug", "configure", "--workspaces", url, "--profile", profile, "--agents", "claude,codex"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.timeout(300)
def test_setup_surfaces_ucode_workspace_drift(tmp_path: Path) -> None:
    """After ``ug configure`` moves ucode to workspace B, the Claude row keeps
    naming ``Databricks (ai_devtools)`` while the overview gains a banner
    naming the workspace ucode switched to."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".databrickscfg").write_text(
        f"[{PROFILE_A}]\nhost = {WORKSPACE_A}\nauth_type = databricks-cli\n"
    )
    bin_dir = tmp_path / "bin"
    _write_shims(bin_dir)
    env = _env(home, bin_dir)

    child = _spawn_setup(env)
    try:
        overview = _add_databricks_workspace(child, WORKSPACE_A)
    finally:
        _quit(child)
    assert f"Databricks ({PROFILE_A})" in _status_row(overview, "Claude"), overview

    _switch_ucode_workspace(env, WORKSPACE_B, PROFILE_B)
    state = json.loads((home / ".ucode" / "state.json").read_text())
    assert state["current_workspace"] == WORKSPACE_B

    child = _spawn_setup(env)
    try:
        # The banner renders above the "Configure harnesses" title _spawn_setup
        # consumed, so it lands in child.before; the rows arrive in the buffer.
        banner = _ANSI_RE.sub(b"", bytes(child.before or b"")).decode("utf-8", "replace")
        overview = _frame(child, 2.5)
    finally:
        _quit(child)
    full = banner + overview

    # The credential keeps naming the profile sessions actually route through
    # (providers.databricks.profile), not ucode's current_workspace.
    claude_row = _status_row(overview, "Claude")
    assert f"Databricks ({PROFILE_A})" in claude_row, overview

    # ...but the overview no longer hides the divergence: it names the workspace
    # ucode switched to. That host appears nowhere on the unfixed base.
    host_b = WORKSPACE_B.split("://", 1)[-1]
    assert host_b in full, (
        f"ucode's current workspace is now {WORKSPACE_B} (profile {PROFILE_B}), but the "
        f"`omni setup` overview never surfaces it — shown output was {full!r}; the "
        "configured profile and ucode's current_workspace drift apart silently"
    )
