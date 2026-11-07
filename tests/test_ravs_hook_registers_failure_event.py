"""The RAVS hook must also run on the event Claude Code fires for failures.

`entroly ravs hook install --claude-code` registered only ``PostToolUse``.
Claude Code fires that event after a Bash command exits 0, and fires
``PostToolUseFailure`` after a non-zero exit, so no failing command ever
reached the capture hook. The installer now registers both, with
``--host claude-code`` on the command so capture may read a ``PostToolUse``
as exit 0 (see ``entroly.hook_outcome``), and re-running it upgrades an
older registration in place instead of adding a second one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly.ravs.hooks import install_claude_code, uninstall_claude_code

EVENTS = ("PostToolUse", "PostToolUseFailure")


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.chdir(project)
    return home


def _path(home: Path) -> Path:
    return home / ".claude" / "settings.json"


def _settings(home: Path) -> dict:
    return json.loads(_path(home).read_text(encoding="utf-8"))


def _capture_commands(settings: dict, event: str) -> list[str]:
    return [
        hook["command"]
        for entry in settings.get("hooks", {}).get(event, [])
        for hook in entry.get("hooks", [])
        if "ravs capture" in hook.get("command", "")
    ]


def test_a_fresh_install_registers_the_success_and_the_failure_event(home) -> None:
    assert install_claude_code()["action"] == "installed"

    settings = _settings(home)
    for event in EVENTS:
        [command] = _capture_commands(settings, event)
        assert command.endswith("ravs capture --stdin --host claude-code --quiet")
        [entry] = settings["hooks"][event]
        assert entry["matcher"] == "Bash"


def test_reinstalling_upgrades_an_old_registration_in_place(home) -> None:
    old_capture = {"type": "command", "command": "entroly ravs capture --stdin --quiet"}
    sibling = {"type": "command", "command": "my-own-logger"}
    formatter = {"matcher": "Edit", "hooks": [{"type": "command", "command": "fmt"}]}
    original = {
        "permissions": {"deny": ["Bash(rm -rf:*)"]},
        "hooks": {
            "PostToolUse": [
                {"matcher": "Bash|Read", "hooks": [old_capture, sibling]},
                formatter,
            ]
        },
    }
    _path(home).write_text(json.dumps(original), encoding="utf-8")

    assert install_claude_code()["action"] == "updated"

    settings = _settings(home)
    assert settings["permissions"] == original["permissions"]
    post = settings["hooks"]["PostToolUse"]
    assert post[0]["matcher"] == "Bash|Read"
    assert post[0]["hooks"] == [
        {"type": "command", "command": "entroly ravs capture --stdin --host claude-code --quiet"},
        sibling,
    ]
    assert post[1] == formatter
    [failure] = settings["hooks"]["PostToolUseFailure"]
    assert failure["matcher"] == "Bash|Read", "mirrors the registration it pairs with"
    assert _capture_commands(settings, "PostToolUseFailure") == [
        "entroly ravs capture --stdin --host claude-code --quiet"
    ]


def test_the_flag_stays_an_argument_of_capture_before_a_shell_tail(home) -> None:
    hand_written = "entroly ravs capture --stdin --quiet 2>/dev/null || true"
    _path(home).write_text(json.dumps({"hooks": {"PostToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": hand_written}]}]}}),
        encoding="utf-8")

    install_claude_code()

    settings = _settings(home)
    for event in EVENTS:
        assert _capture_commands(settings, event) == [
            "entroly ravs capture --stdin --host claude-code --quiet 2>/dev/null || true"
        ]


def test_an_up_to_date_install_is_left_byte_identical(home) -> None:
    install_claude_code()
    before = _path(home).read_bytes()

    assert install_claude_code()["action"] == "already_present"
    assert _path(home).read_bytes() == before


def test_dry_run_reports_an_upgrade_without_writing(home) -> None:
    original = {"hooks": {"PostToolUse": [{"matcher": "Bash", "hooks": [
        {"type": "command", "command": "entroly ravs capture --stdin --quiet"}]}]}}
    _path(home).write_text(json.dumps(original), encoding="utf-8")
    before = _path(home).read_bytes()

    result = install_claude_code(dry_run=True)

    assert result["action"] == "dry_run"
    assert _capture_commands(result["settings"], "PostToolUseFailure")
    assert _path(home).read_bytes() == before


def test_uninstall_removes_both_registrations_and_nothing_else(home) -> None:
    notifier = {"matcher": "Bash", "hooks": [{"type": "command", "command": "notify failed"}]}
    _path(home).write_text(
        json.dumps({"hooks": {"PostToolUseFailure": [notifier]}}), encoding="utf-8"
    )
    install_claude_code()

    assert uninstall_claude_code()["action"] == "uninstalled"

    settings = _settings(home)
    assert settings["hooks"] == {"PostToolUseFailure": [notifier]}


def test_a_failure_section_that_is_not_a_list_is_refused_untouched(home) -> None:
    _path(home).write_text(
        json.dumps({"hooks": {"PostToolUseFailure": {"matcher": "Bash"}}}), encoding="utf-8"
    )
    before = _path(home).read_bytes()

    assert install_claude_code()["action"] == "unreadable"
    assert _path(home).read_bytes() == before
