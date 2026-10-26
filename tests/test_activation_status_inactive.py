"""`entroly activation status` must not call a hook "active" when nothing runs it.

README.md tells users to check `entroly activation status --json` instead of
assuming an installed integration is active. Status was built from receipts
alone with a 7-day freshness window, so it kept saying "active":

* after `entroly activation uninstall` removed the hook,
* after the interpreter that ran the hook was moved or deleted, and
* when the only runs were manual terminal tests that no agent host made.

These drive the real hook writer, the real CLI installer and the real status.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from entroly.agent_activation import ActivationSelection, activation_status, run_hook


def _selected(*_):
    return ActivationSelection("activated", ("app.py",), "x", 1, True)


@pytest.fixture()
def no_host_environment(monkeypatch):
    for name in ("CURSOR_PROJECT_DIR", "CODEX_HOME", "USER_PROMPT", "VSCODE_PID", "TERM_PROGRAM"):
        monkeypatch.delenv(name, raising=False)


def _cli(args, state_root: Path, cwd: Path):
    env = {
        **os.environ,
        "ENTROLY_DIR": str(state_root),
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "entroly", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_a_hook_removed_with_uninstall_is_not_active(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state_root = tmp_path / "entroly"
    state = state_root / "activation"

    installed = _cli(["activation", "install", "--host", "cursor", "--project", "."], state_root, project)
    assert installed.returncode == 0, installed.stderr
    run_hook(
        {"hook_event_name": "UserPromptSubmit", "prompt": "inspect", "cwd": str(project)},
        host="cursor",
        state_dir=state,
        selector=_selected,
    )
    assert activation_status(project, state_dir=state)["state"] == "active"

    removed = _cli(["activation", "uninstall", "--host", "cursor", "--project", "."], state_root, project)
    assert removed.returncode == 0, removed.stderr

    report = activation_status(project, state_dir=state)
    assert report["state"] == "inactive"
    assert "uninstall" in report["inactive_reason"]

    _cli(["activation", "install", "--host", "cursor", "--project", "."], state_root, project)
    run_hook(
        {"hook_event_name": "UserPromptSubmit", "prompt": "inspect", "cwd": str(project)},
        host="cursor",
        state_dir=state,
        selector=_selected,
    )
    assert activation_status(project, state_dir=state)["state"] == "active"


def test_a_run_by_an_interpreter_that_no_longer_exists_is_not_active(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    run_hook(
        {"hook_event_name": "UserPromptSubmit", "prompt": "inspect", "cwd": str(project)},
        host="claude-code",
        state_dir=state,
        selector=_selected,
    )
    receipt_path = next(state.rglob("events/*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    moved = tmp_path / "moved-venv" / "bin" / "python"
    receipt["runtime"] = {"python": str(moved), "package": str(tmp_path)}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    report = activation_status(project, state_dir=state)

    assert report["state"] == "inactive"
    assert str(moved) in report["inactive_reason"]


def test_manual_runs_without_a_host_are_not_active(tmp_path, no_host_environment) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    run_hook({"prompt": "inspect", "cwd": str(project)}, state_dir=state, selector=_selected)

    report = activation_status(project, state_dir=state)

    assert report["by_host"] == {"unknown": 1}
    assert report["state"] == "inactive"
    assert "without an agent host" in report["inactive_reason"]


def test_a_manual_test_does_not_hide_host_runs(tmp_path, no_host_environment) -> None:
    project = tmp_path / "project"
    project.mkdir()
    state = tmp_path / "state"
    run_hook(
        {"hook_event_name": "UserPromptSubmit", "prompt": "inspect", "cwd": str(project)},
        host="claude-code",
        state_dir=state,
        selector=_selected,
    )
    run_hook({"prompt": "inspect", "cwd": str(project)}, state_dir=state, selector=_selected)

    report = activation_status(project, state_dir=state)

    assert report["state"] == "active"
    assert report["inactive_reason"] is None
    assert report["latest"]["host"] == "claude-code"
