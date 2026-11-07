"""`entroly uninstall` must not leave hooks that run the deleted executable.

`entroly ravs hook install --claude-code` writes the absolute path of this
installation's `entroly` into ~/.claude/settings.json. `entroly uninstall`
ran `pip uninstall entroly`, which deletes that file, and left both hooks in
place: from then on every Bash call in Claude Code ran a missing command, and
Claude Code showed a hook error for each one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import entroly.cli as cli
from entroly.ravs.hooks import install_claude_code

USER_HOOK = {"matcher": "Edit", "hooks": [{"type": "command", "command": "fmt"}]}


@pytest.fixture()
def home(tmp_path, monkeypatch) -> Path:
    import entroly.product_telemetry as product_telemetry

    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: home))
    monkeypatch.chdir(project)
    monkeypatch.setattr(product_telemetry, "disable_and_purge", lambda **_: {})
    (home / ".claude" / "settings.json").write_text(
        json.dumps({"model": "opus", "hooks": {"PostToolUse": [USER_HOOK]}}),
        encoding="utf-8",
    )
    assert install_claude_code()["action"] == "installed"
    return home


def _args(**overrides):
    values = {
        "reason": None, "benefit": None, "surface": None, "duration": None,
        "send_feedback": False, "endpoint": None, "skip_feedback": True,
        "delete_remote_telemetry": False, "dry_run": False,
        "feedback_only": False, "yes": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _settings(home: Path) -> dict:
    return json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))


def test_uninstall_takes_the_capture_hooks_with_it(home, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli.subprocess, "run", lambda command, check: SimpleNamespace(returncode=0)
    )

    assert cli.cmd_uninstall(_args()) == 0

    settings = _settings(home)
    assert "ravs capture" not in json.dumps(settings)
    assert settings == {"model": "opus", "hooks": {"PostToolUse": [USER_HOOK]}}
    output = capsys.readouterr().out
    assert "Removed the RAVS capture hooks" in output
    assert "entroly ravs hook install --claude-code" in output, "how to undo it"


def test_dry_run_names_the_hooks_and_changes_nothing(home, monkeypatch, capsys) -> None:
    before = (home / ".claude" / "settings.json").read_bytes()
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda *_a, **_k: pytest.fail("dry run must not invoke pip"),
    )

    assert cli.cmd_uninstall(_args(dry_run=True)) == 0

    assert (home / ".claude" / "settings.json").read_bytes() == before
    output = capsys.readouterr().out
    assert "Would also remove the RAVS capture hooks" in output
    assert str(home / ".claude" / "settings.json") in output


def test_feedback_only_leaves_the_hooks_alone(home, monkeypatch) -> None:
    before = (home / ".claude" / "settings.json").read_bytes()
    monkeypatch.setattr(
        cli.subprocess, "run",
        lambda *_a, **_k: pytest.fail("feedback-only must not invoke pip"),
    )

    assert cli.cmd_uninstall(_args(feedback_only=True)) == 0

    assert (home / ".claude" / "settings.json").read_bytes() == before


def test_no_hooks_means_no_hook_message(tmp_path, monkeypatch, capsys) -> None:
    import entroly.product_telemetry as product_telemetry

    monkeypatch.setattr("pathlib.Path.home", staticmethod(lambda: tmp_path / "empty-home"))
    monkeypatch.setattr(product_telemetry, "disable_and_purge", lambda **_: {})
    monkeypatch.setattr(
        cli.subprocess, "run", lambda command, check: SimpleNamespace(returncode=0)
    )

    assert cli.cmd_uninstall(_args()) == 0
    assert "RAVS capture hooks" not in capsys.readouterr().out
