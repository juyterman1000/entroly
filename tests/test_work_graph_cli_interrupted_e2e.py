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


def test_new_process_exposes_handoff_and_source_bound_review(tmp_path: Path) -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "app.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "feature/review")
    env = dict(os.environ, ENTROLY_DIR=str(tmp_path / "state"))

    code, claimed = _cli(repo, env, "claim", "--agent", "claude", "--task", "Update value")
    assert code == 0, claimed
    code, state = _cli(repo, env, "state")
    assert code == 0, state
    workstream = state["unfinished"][0]["node_id"]
    source.write_text("VALUE = 2\n", encoding="utf-8")

    code, sent = _cli(
        repo, env, "handoff", "--workstream", workstream,
        "--from-agent", "claude", "--to-agent", "codex",
    )
    assert code == 0, sent
    ref = sent["handoff_ref"]
    assert ref.startswith("handoff:")
    code, inbox = _cli(repo, env, "handoffs", "--to-agent", "codex")
    assert code == 0, inbox
    assert inbox["handoffs"][0]["ref"] == ref
    assert inbox["handoffs"][0]["current"] is True

    code, recovered = _cli(repo, env, "resume", "--workstream", workstream, "--to-agent", "codex")
    assert code == 0, recovered
    code, accepted = _cli(repo, env, "acknowledge", "--token", recovered["acknowledgement"]["token"])
    assert code == 0, accepted

    code, recorded = _cli(
        repo, env, "review", "--handoff-ref", ref, "--reviewer", "codex",
        "--verdict", "changes_requested", "--summary", "Check the value",
        "--finding", "app.py", "1", "high", "Value changed",
        "--check", "focused test", "0",
    )
    assert code == 0, recorded
    assert recorded["review"]["trust_boundary"].startswith("agent_statement")
    assert recorded["review"]["findings"][0]["source_digest"].startswith("sha256:")
    code, listed = _cli(repo, env, "reviews", "--workstream", workstream)
    assert code == 0, listed
    assert listed["reviews"][0]["current"] is True

    source.write_text("VALUE = 3\n", encoding="utf-8")
    code, stale = _cli(repo, env, "reviews", "--workstream", workstream)
    assert code == 0, stale
    assert stale["reviews"][0]["current"] is False
    code, rejected = _cli(
        repo, env, "review", "--handoff-ref", ref, "--reviewer", "codex",
        "--verdict", "approve", "--summary", "Looks good",
    )
    assert code == 1
    assert "stale" in rejected["detail"]
