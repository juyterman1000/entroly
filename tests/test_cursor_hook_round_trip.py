"""`entroly activation install|uninstall --host cursor` must round-trip the file.

The installer merges one hook into `.claude/settings.local.json`, a file the
user (and Claude Code) also edits. It used to re-serialize the whole file:
indentation changed, non-ASCII became \\u escapes, CRLF became LF, the mode
became 0600, a symlinked file was replaced by a regular copy, and uninstall
left `"UserPromptSubmit": []` behind (or a file where there had been none).
These drive the real installer on real files.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from entroly.agent_activation import CURSOR_HOOK_COMMAND, configure_cursor_hook

ORIGINALS = {
    "crlf_unicode_4space": (
        '{\r\n    "env": {"GREETING": "héllo ✓"},\r\n    "hooks": {\r\n'
        '        "UserPromptSubmit": [\r\n            {"hooks": [{"type": "command", '
        '"command": "echo mine"}]}\r\n        ]\r\n    }\r\n}\r\n'
    ),
    "compact_no_newline": '{"permissions":{"allow":["Read"]},"model":"opus"}',
    "empty_prompt_list": '{\n  "hooks": {\n    "UserPromptSubmit": []\n  }\n}\n',
    "hooks_without_prompt": '{\n  "hooks": {"Stop": []}\n}\n',
    "empty_object": "{}\n",
}


def _settings(tmp_path):
    project = tmp_path / "project"
    target = project / ".claude" / "settings.local.json"
    target.parent.mkdir(parents=True)
    return project, target


def _commands(target):
    document = json.loads(target.read_bytes().decode("utf-8"))
    return [
        hook["command"]
        for entry in document["hooks"]["UserPromptSubmit"]
        for hook in entry["hooks"]
    ]


@pytest.mark.parametrize("original", ORIGINALS.values(), ids=ORIGINALS.keys())
def test_install_then_uninstall_restores_the_exact_bytes(tmp_path, original) -> None:
    project, target = _settings(tmp_path)
    target.write_bytes(original.encode("utf-8"))

    assert configure_cursor_hook(project)["status"] == "installed"
    assert CURSOR_HOOK_COMMAND in _commands(target)
    installed = target.read_bytes().decode("utf-8")
    assert installed.count("\r\n") == original.count("\r\n")
    assert "\\u00e9" not in installed
    assert configure_cursor_hook(project, uninstall=True)["status"] == "disabled"

    assert target.read_bytes() == original.encode("utf-8")


def test_user_edits_made_while_installed_survive_uninstall_byte_for_byte(tmp_path) -> None:
    project, target = _settings(tmp_path)
    original = ORIGINALS["crlf_unicode_4space"]
    target.write_bytes(original.encode("utf-8"))
    configure_cursor_hook(project)

    installed = target.read_bytes().decode("utf-8")
    target.write_bytes(installed.replace("echo mine", "echo edited").encode("utf-8"))
    configure_cursor_hook(project, uninstall=True)

    assert target.read_bytes() == original.replace("echo mine", "echo edited").encode("utf-8")


def test_file_mode_is_kept(tmp_path) -> None:
    project, target = _settings(tmp_path)
    target.write_text('{"model": "opus"}\n', encoding="utf-8")
    os.chmod(target, 0o700)

    configure_cursor_hook(project)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    configure_cursor_hook(project, uninstall=True)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_a_symlinked_settings_file_stays_linked(tmp_path) -> None:
    project, target = _settings(tmp_path)
    dotfile = tmp_path / "dotfiles" / "settings.local.json"
    dotfile.parent.mkdir()
    dotfile.write_text('{"model": "opus"}\n', encoding="utf-8")
    target.symlink_to(dotfile)

    configure_cursor_hook(project)
    assert target.is_symlink()
    assert CURSOR_HOOK_COMMAND in _commands(dotfile)

    configure_cursor_hook(project, uninstall=True)
    assert target.is_symlink()
    assert dotfile.read_text(encoding="utf-8") == '{"model": "opus"}\n'


def test_a_file_the_installer_created_is_removed_on_uninstall(tmp_path) -> None:
    project, target = _settings(tmp_path)

    configure_cursor_hook(project)
    assert target.is_file()
    report = configure_cursor_hook(project, uninstall=True)

    assert not target.exists()
    assert report["recoverable_at"]
