"""A new CLI process must recover an interrupted claim and enforce its gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from entroly.work_graph import WorkGraph, WorkGraphUnavailableError


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _cli(repo: Path, env: dict[str, str], *args: str) -> tuple[int, dict]:
    result = subprocess.run(
        [sys.executable, "-m", "entroly.work_graph_cli", "--project", str(repo), "--json", *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.stdout, result.stderr
    return result.returncode, json.loads(result.stdout)


def test_new_process_recovers_claim_and_requires_acknowledgement(tmp_path: Path) -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "feature/interrupted-auth")
    env = dict(os.environ, ENTROLY_DIR=str(tmp_path / "state"))

    code, claimed = _cli(
        repo, env, "claim", "--agent", "claude", "--task", "Fix auth", "--path", "app.py"
    )
    assert code == 0, claimed
    code, state = _cli(repo, env, "state")
    assert code == 0, state
    claimed_workstream = state["unfinished"][0]["node_id"]

    # The first agent stops without leaving a handoff or a remaining-work note.
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    code, recovered = _cli(repo, env, "resume", "--to-agent", "codex")
    assert code == 0, recovered
    assert recovered["resume"]["selected_workstream"]["node_id"] == claimed_workstream
    assert "app.py" in recovered["resume"]["current_worktree_changed_paths"]
    assert "app.py" in recovered["continuation_proof"]["outstanding_work_refs"]
    assert recovered["acknowledgement"]["required"] is True
    token = recovered["acknowledgement"]["token"]

    code, blocked = _cli(repo, env, "claim", "--agent", "codex", "--task", "Finish auth")
    assert code == 1
    assert blocked["error"] == "recovery_acknowledgement_required"
    (repo / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    code, stale = _cli(repo, env, "acknowledge", "--token", token)
    assert code == 1
    assert "repository changed after recovery" in stale["detail"]
    (repo / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    code, acknowledged = _cli(repo, env, "acknowledge", "--token", token)
    assert code == 0, acknowledged
    assert acknowledged["acknowledged"] is True
    code, successor = _cli(repo, env, "claim", "--agent", "codex", "--task", "Finish auth")
    assert code == 0, successor
