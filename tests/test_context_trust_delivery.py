"""Production delivery tests for the Rust-owned Context/Trust seams."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from entroly.trust import TrustEngine
from entroly.work_graph import WorkGraph


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _dirty_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Entroly Test")
    (repo / "src").mkdir()
    (repo / "src" / "auth.py").write_text("def auth():\n    return True\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    (repo / "src" / "auth.py").write_text("def auth():\n    return False\n", encoding="utf-8")
    return repo


def test_work_graph_context_scope_is_bounded_and_text_light() -> None:
    graph = WorkGraph("repo:context-trust-test")
    graph.observe_repository(
        {
            "repo_id": "repo:context-trust-test",
            "observed_at_ms": 1234,
            "repository_label": "demo",
            "agent_id": "agent:test",
            "session_id": "session:test",
            "task_hint": {
                "task_id": "task:auth",
                "title": "repair auth",
                "trust": "observed",
                "explicit_status": "in_progress",
                "remaining_work": ["run tests"],
                "source_kind": "user_statement",
                "source_ref": "test://task",
            },
            "branch": {
                "name": "main",
                "head_sha": "abc",
                "base_ref": "refs/heads/main",
                "default_branch": "main",
                "ahead_by": 0,
                "behind_by": 0,
                "merge_in_progress": False,
                "rebase_in_progress": False,
                "detached": False,
            },
            "changes": [
                {
                    "path": "src/auth.py",
                    "kind": "modified",
                    "staged": False,
                    "conflicted": False,
                    "old_path": "",
                    "content_digest": "git-blob:0123456789012345678901234567890123456789",
                }
            ],
            "commits": [],
            "verifications": [],
            "decisions": [],
            "claims": [],
            "leases": [],
            "model_executions": [],
        }
    )
    scope = graph.context_scope()
    assert scope["repo_id"] == graph.repo_id
    assert scope["graph_revision"] == graph.revision
    assert scope["graph_commitment"] == graph.graph_commitment
    assert scope["workstream_id"]
    assert "src/auth.py" in scope["changed_paths"]
    assert scope["task_ids"] == sorted(set(scope["task_ids"]))
    assert scope["agent_ids"] == sorted(set(scope["agent_ids"]))
    payload = json.dumps(scope, sort_keys=True)
    assert "repair auth" not in payload
    assert "run tests" not in payload
    assert "selected_context" not in payload


def test_trust_engine_is_evidence_bounded_and_fail_closed() -> None:
    evidence = "The service retries a request three times before returning an error."
    claim = "The service retries a request three times."
    engine = TrustEngine("rag")
    assessment = engine.assess_claim(evidence, claim)
    assert assessment["status"] in {"supported", "unsupported", "unknown"}
    assert assessment["evidence_commitment"] == "sha256:" + hashlib.sha256(
        evidence.encode("utf-8")
    ).hexdigest()
    assert 0.0 <= assessment["support_density"] <= 1.0
    assert engine.file_criticality("file:SECURITY.md") == "safety"
    assert engine.has_safety_signal("AWS_SECRET_ACCESS_KEY=example")
    with pytest.raises(ValueError):
        TrustEngine("rga")
