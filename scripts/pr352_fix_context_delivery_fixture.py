#!/usr/bin/env python3
"""Fix the staged Context delivery regression fixture without changing semantics.

The delivery test should exercise Rust WorkGraph -> WorkScope. Repository/Git
inference has separate coverage, so use a canonical observation with explicit
in-progress work rather than depending on temp-repo inference behavior.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/test_context_trust_delivery.py"

OLD = r'''def test_work_graph_context_scope_is_bounded_and_text_light(tmp_path: Path) -> None:
    repo = _dirty_repo(tmp_path)
    graph = WorkGraph.from_repository(
        str(repo),
        agent_id="agent:test",
        session_id="session:test",
        task_hint={
            "task_id": "task:auth",
            "title": "repair auth",
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": ["run tests"],
            "source_kind": "user_statement",
            "source_ref": "test://task",
        },
        include_checkpoint=False,
        observed_at_ms=1234,
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
'''

NEW = r'''def test_work_graph_context_scope_is_bounded_and_text_light() -> None:
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
'''


def main() -> int:
    text = TEST.read_text(encoding="utf-8")
    if NEW in text:
        print("Context delivery fixture already canonical")
        return 0
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(f"expected exactly one old fixture, found {count}")
    TEST.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("replaced Context delivery fixture with canonical observation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
