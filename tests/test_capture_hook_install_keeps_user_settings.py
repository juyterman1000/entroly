"""`entroly ravs hook install --claude-code` must never discard user settings.

The installer merges one PostToolUse entry into ``~/.claude/settings.json``.
When that file could not be parsed it was treated as ``{}`` and overwritten
with only the capture hook: every permission, deny rule, other hook and env
entry was deleted, no backup was written, and the CLI printed "Installed".
A UTF-8 BOM (Windows PowerShell 5.1 ``Out-File -Encoding utf8``, Notepad), a
trailing comma or a ``//`` comment was enough to trigger it.

These tests drive the real installer against a real file in a temporary home.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from entroly.ravs.hooks import install_claude_code

USER_SETTINGS = {
    "permissions": {
        "allow": ["Bash(npm test)"],
        "deny": ["Bash(rm -rf:*)", "Read(~/.ssh/**)"],
    },
    "env": {"ANTHROPIC_MODEL": "claude-opus"},
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": "/usr/local/bin/guard"}],
            }
        ]
    },
}


@pytest.fixture()
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".claude").mkdir(parents=True)
    project.mkdir()
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.chdir(project)
    return home


def _settings(home):
    return home / ".claude" / "settings.json"


def test_a_bom_prefixed_settings_file_is_merged_not_replaced(home) -> None:
    path = _settings(home)
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(USER_SETTINGS, indent=2).encode())

    result = install_claude_code()

    assert result["action"] == "installed"
    merged = json.loads(path.read_text(encoding="utf-8"))
    assert merged["permissions"] == USER_SETTINGS["permissions"]
    assert merged["env"] == USER_SETTINGS["env"]
    assert merged["hooks"]["PreToolUse"] == USER_SETTINGS["hooks"]["PreToolUse"]
    assert "ravs capture" in json.dumps(merged["hooks"]["PostToolUse"])


UNMERGEABLE = {
    "trailing_comma": '{\n  "permissions": {"deny": ["Bash(rm -rf:*)"]},\n}\n',
    "line_comment": '{\n  // keep me\n  "permissions": {"deny": ["Bash(rm -rf:*)"]}\n}\n',
    "truncated": '{"permissions": {"deny": ["Bash(rm -rf:*)"]',
    "top_level_array": '[{"permissions": {}}]\n',
    "hooks_not_object": '{"permissions": {"deny": ["x"]}, "hooks": null}\n',
    "post_tool_use_not_list": '{"hooks": {"PostToolUse": {"matcher": "Bash"}}}\n',
}


@pytest.mark.parametrize("dry_run", [False, True], ids=["install", "dry_run"])
@pytest.mark.parametrize("content", UNMERGEABLE.values(), ids=UNMERGEABLE.keys())
def test_unmergeable_settings_are_refused_and_left_byte_identical(
    home, content, dry_run
) -> None:
    path = _settings(home)
    path.write_text(content, encoding="utf-8")
    before = path.read_bytes()

    result = install_claude_code(dry_run=dry_run)

    assert result["action"] == "unreadable"
    assert result["detail"]
    assert path.read_bytes() == before
    assert sorted(p.name for p in path.parent.iterdir()) == ["settings.json"]


def test_the_cli_reports_the_refusal_and_fails(tmp_path) -> None:
    """The customer path: the command must not claim it installed anything."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    (home / ".claude").mkdir(parents=True)
    project.mkdir()
    path = home / ".claude" / "settings.json"
    path.write_text(UNMERGEABLE["trailing_comma"], encoding="utf-8")
    before = path.read_bytes()
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "ENTROLY_DIR": str(tmp_path / "state"),
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
        "NO_COLOR": "1",
    }

    completed = subprocess.run(
        [sys.executable, "-m", "entroly", "ravs", "hook", "install", "--claude-code"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "Not installed" in completed.stdout
    assert "Installed RAVS" not in completed.stdout
    assert path.read_bytes() == before
