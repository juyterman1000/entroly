from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from entroly.work_graph import WorkGraph, WorkGraphUnavailableError
from entroly.work_graph_content_digest import enrich_worktree_content_digests
from entroly.work_graph_inbox import pending_handoffs, save_handoff
from entroly.work_graph_mcp import work_resume
from entroly.work_graph_repo import discover_repository_observation
from entroly.work_graph_review import record_review, reviews
from entroly.work_graph_store import WorkGraphStore


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_cross_agent_review_is_source_bound_and_stale_after_edit(tmp_path, monkeypatch):
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
    source.write_text("VALUE = 2\n", encoding="utf-8")
    monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))

    store = WorkGraphStore.for_repository(repo)
    observed = discover_repository_observation(
        repo, agent_id="claude", session_id="s1",
        task_hint={
            "task_id": "change-value", "title": "Change value", "trust": "observed",
            "explicit_status": "in_progress", "remaining_work": ["review"],
            "source_kind": "user_statement", "source_ref": "user:change-value",
        },
    )
    enrich_worktree_content_digests(repo, observed)
    graph = store.submit_repository_observation(observed, repository_path=repo)
    workstream_id = graph.unfinished()[0]["node_id"]
    handoff = store.handoff(workstream_id, "claude", "codex")
    proof = store.continuation_proof(
        handoff, outstanding_work_refs=["review app.py"], created_at_ms=1_000,
    )
    handoff_ref = save_handoff(store, handoff, proof)
    assert handoff_ref.startswith("handoff:")
    pending = pending_handoffs(root=repo, store=store, to_agent="codex")
    assert pending["handoffs"][0]["current"] is True
    assert pending["handoffs"][0]["handoff"] == handoff
    assert work_resume(workstream_id=workstream_id)["status"] == "ok"
    assert pending_handoffs(root=repo, store=store, to_agent="codex")["handoffs"][0]["current"] is True

    result = record_review(
        handoff=handoff, reviewer_id="codex", verdict="changes_requested",
        findings=[{"path": "app.py", "line": 1, "severity": "high", "message": "Value changed"}],
        summary="Check callers", project="", checks=[{"name": "pytest focused", "exit_code": 0}],
    )
    assert result["status"] == "ok"
    assert result["review"]["findings"][0]["source_digest"].startswith("sha256:")
    assert result["review"]["checks"] == [{"name": "pytest focused", "exit_code": 0}]
    assert reviews()["reviews"][0]["current"] is True

    source.write_text("VALUE = 3\n", encoding="utf-8")
    assert reviews()["reviews"][0]["current"] is False
    assert pending_handoffs(root=repo, store=store, to_agent="codex")["handoffs"][0]["current"] is False
    with pytest.raises(ValueError, match="stale"):
        record_review(
            handoff=handoff, reviewer_id="codex", verdict="approve",
            findings=[], summary="Looks good",
        )


def test_review_rejects_same_agent_before_repo_access():
    with pytest.raises(ValueError, match="writer and reviewer"):
        record_review(
            handoff={"from_agent": "claude", "to_agent": "claude"},
            reviewer_id="claude", verdict="comment", findings=[], summary="Self review",
        )
