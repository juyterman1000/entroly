from __future__ import annotations

import copy

import pytest

from entroly.work_graph import WorkGraph, WorkGraphUnavailableError


def _observation() -> dict:
    return {
        "repo_id": "repo:test",
        "observed_at_ms": 1_000,
        "repository_label": "test repo",
        "agent_id": "claude",
        "session_id": "session-1",
        "task_hint": {
            "task_id": "task-stream",
            "title": "Fix streaming parity",
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": ["finish Rust parity"],
            "source_kind": "user_statement",
            "source_ref": "user:task",
        },
        "branch": {
            "name": "feature/streaming",
            "head_sha": "abc123",
            "default_branch": "main",
            "ahead_by": 1,
        },
        "changes": [
            {
                "path": "src/stream.rs",
                "kind": "modified",
                "staged": False,
                "conflicted": False,
            }
        ],
        "decisions": [
            {
                "decision_id": "decision-1",
                "text": "Preserve provider event ordering",
                "source_ref": "checkpoint:1",
                "source_kind": "checkpoint",
                "trust": "observed",
            }
        ],
    }


def _graph() -> WorkGraph:
    try:
        return WorkGraph("repo:test")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))


def test_native_work_graph_is_deterministic_and_resumable() -> None:
    a = _graph()
    b = _graph()
    observation = _observation()

    a.observe_repository(observation)
    b.observe_repository(copy.deepcopy(observation))

    assert a.graph_commitment == b.graph_commitment
    assert a.unfinished() == b.unfinished()
    assert a.summary()["unfinished_count"] == 1

    work = a.unfinished()
    assert len(work) == 1
    resume = a.resume(work[0]["node_id"])
    assert "Fix streaming parity" in resume["task_labels"]
    assert "src/stream.rs" in resume["changed_paths"]


def test_native_work_graph_roundtrip_and_handoff_integrity() -> None:
    graph = _graph()
    graph.observe_repository(_observation())
    workstream_id = graph.unfinished()[0]["node_id"]

    restored = WorkGraph.from_json(graph.export_json())
    assert restored.graph_commitment == graph.graph_commitment
    assert restored.snapshot() == graph.snapshot()

    receipt = graph.handoff(workstream_id, "claude", "codex", 2_000)
    assert WorkGraph.verify_handoff_integrity(receipt)
    assert graph.verify_handoff(receipt)

    receipt["to_agent"] = "tampered"
    assert not WorkGraph.verify_handoff_integrity(receipt)
    assert not graph.verify_handoff(receipt)


def test_missing_native_binding_has_actionable_install_guidance(monkeypatch) -> None:
    from entroly import work_graph as module

    monkeypatch.setattr(module, "_RustWorkGraph", None)
    monkeypatch.setattr(module, "_NATIVE_IMPORT_ERROR", ImportError("missing test binding"))
    with pytest.raises(WorkGraphUnavailableError) as exc_info:
        module.WorkGraph("repo:test")
    message = str(exc_info.value)
    assert 'pip install "entroly[native]"' in message
    assert "missing test binding" in message
