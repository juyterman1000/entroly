"""`entroly ravs hook uninstall --claude-code` removes what `install` added.

`entroly ravs hook install --claude-code` registers a PostToolUse hook in
~/.claude/settings.json, but the CLI offered no way back: `ravs hook` had only
`install`, so users had to hand-edit their settings to stop capture.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _run(args, home, cwd):
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "ENTROLY_DIR": str(home / ".entroly-state"),
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
        "NO_COLOR": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "entroly", "ravs", "hook", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_uninstall_removes_only_the_capture_hook(tmp_path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".claude").mkdir(parents=True)
    project.mkdir()
    settings = home / ".claude" / "settings.json"
    user_hook = {"matcher": "Edit", "hooks": [{"type": "command", "command": "fmt"}]}
    settings.write_text(
        json.dumps({"model": "opus", "hooks": {"PostToolUse": [user_hook]}}),
        encoding="utf-8",
    )
    assert _run(["install", "--claude-code"], home, project).returncode == 0
    assert "ravs capture" in settings.read_text(encoding="utf-8")

    removed = _run(["uninstall", "--claude-code"], home, project)

    assert removed.returncode == 0, removed.stdout + removed.stderr
    assert "Removed RAVS PostToolUse hook" in removed.stdout
    document = json.loads(settings.read_text(encoding="utf-8"))
    assert document["model"] == "opus"
    assert document["hooks"]["PostToolUse"] == [user_hook]

    again = _run(["uninstall", "--claude-code"], home, project)
    assert again.returncode == 0
    assert "No RAVS hook found" in again.stdout
