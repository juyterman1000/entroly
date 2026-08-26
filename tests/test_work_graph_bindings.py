from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from entroly.work_graph import (
    WorkGraph,
    WorkGraphUnavailableError,
    create_model_execution_outcome,
    create_recovery_handle,
    create_routing_decision,
    create_verification_record,
    create_work_context_receipt,
    verification_freshness,
    verify_recovered_bytes,
    verify_recovery_handle,
    verify_work_context_receipt,
)


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


def test_continuity_contracts_are_exported_from_python_package_root() -> None:
    import entroly

    for name in (
        "WorkGraph",
        "create_routing_decision",
        "create_model_execution_outcome",
        "create_verification_record",
        "create_work_context_receipt",
        "verify_work_context_receipt",
        "create_recovery_handle",
        "verify_recovery_handle",
        "verify_recovered_bytes",
        "verification_freshness",
    ):
        assert callable(getattr(entroly, name, None)), name


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


def test_public_work_graph_records_execution_chain_and_seals_continuation(
    tmp_path: Path,
) -> None:
    entroly_core = pytest.importorskip("entroly_core")
    from entroly.work_graph_store import WorkGraphStore

    graph = _graph()
    graph.observe_repository(_observation())
    work = graph.unfinished()[0]
    workstream_id = work["node_id"]
    task_id = work["task_ids"][0]

    context_receipt = json.loads(
        entroly_core.context_receipt_build_json(
            "repo:test",
            "abc123",
            graph.graph_commitment,
            workstream_id,
            "sha256:sources",
            ["src/stream.rs#0:20"],
            ["src/stream.rs#20:40"],
            ["evidence:test"],
            ["src/stream.rs#20:40"],
            ["rh_example"],
            ["evidence:test"],
            512,
            "work-scope/v1",
            "execution:pending",
            1_050,
        )
    )
    store = WorkGraphStore("repo:test", root=str(tmp_path / "state"))
    store.save(graph)
    graph, _receipt_event = store.record_context_receipt(
        context_receipt, agent_id="claude", session_id="session-1"
    )
    stale_receipt = json.loads(
        entroly_core.context_receipt_build_json(
            "repo:test",
            "older-head",
            graph.graph_commitment,
            workstream_id,
            "sha256:sources",
            ["src/stream.rs#0:20"],
            [],
            [],
            [],
            [],
            [],
            512,
            "work-scope/v1",
            "execution:pending",
            1_055,
        )
    )
    with pytest.raises(ValueError, match="integrity mismatch"):
        store.record_context_receipt(stale_receipt)
    memory = json.loads(
        entroly_core.memory_record_build_json(
            "repo:test",
            "vault/streaming-decision",
            "observed",
            task_id=task_id,
            workstream_id=workstream_id,
            source_agent="claude",
            source_session="session-1",
            source_execution="execution:pending",
            content_commitment="sha256:memory",
            evidence_ids=["evidence:test"],
            created_at_ms=1_060,
            observed_at_ms=1_060,
        )
    )
    graph, _memory_event = store.record_memory(memory, now_ms=1_070)

    route = create_routing_decision(
        repository_id="repo:test",
        task_id=task_id,
        workstream_id=workstream_id,
        provider="openai",
        model="gpt-5",
        runtime="responses-api",
        context_budget_tokens=4096,
        policy_version="policy:v1",
        reason_codes=["capability_match"],
        feature_commitments=["sha256:features"],
        receipt_id=context_receipt["receipt_id"],
        evidence_ids=["evidence:route"],
        decided_at_ms=1_100,
    )
    outcome = create_model_execution_outcome(
        routing_id=route["routing_id"],
        repository_id="repo:test",
        task_id=task_id,
        workstream_id=workstream_id,
        provider="openai",
        model="gpt-5",
        runtime="responses-api",
        receipt_id=context_receipt["receipt_id"],
        request_commitment="sha256:request",
        response_commitment="sha256:response",
        state="succeeded",
        verification_state="passed",
        latency_ms=25,
        input_tokens=100,
        output_tokens=20,
        cost_micro_usd=250,
        evidence_ids=["evidence:outcome"],
        completed_at_ms=1_200,
    )
    verification = create_verification_record(
        repository_id="repo:test",
        subject_id=outcome["outcome_id"],
        subject_commitment=outcome["outcome_commitment"],
        verified_repository_commitment="abc123",
        verdict="passed",
        evidence_ids=["evidence:test"],
        dependency_commitments=["sha256:source"],
        observed_at_ms=1_300,
    )
    assert (
        verification_freshness(
            verification,
            current_repository_commitment="abc123",
            now_ms=1_300,
        )
        == "current"
    )

    graph, _execution_event = store.record_execution_chain(
        route, outcome, verification
    )
    assert store.load().graph_commitment == graph.graph_commitment
    snapshot = graph.snapshot()
    assert any(
        node.get("attributes", {}).get("routing_id") == route["routing_id"]
        for node in snapshot["nodes"]
    )
    assert any(
        node.get("attributes", {}).get("freshness") == "current"
        for node in snapshot["nodes"]
    )

    handoff = graph.handoff(workstream_id, "claude", "codex", 1_400)
    proof = graph.continuation_proof(
        handoff,
        context_receipt_commitments=[context_receipt["receipt_commitment"]],
        routing_commitments=[route["decision_commitment"]],
        execution_outcome_commitments=[outcome["outcome_commitment"]],
        verification_commitments=[verification["record_commitment"]],
        memory_commitments=[memory["record_commitment"]],
        outstanding_work_refs=["run Linux CI"],
        recovery_handle_ids=["rh_example"],
        created_at_ms=1_500,
    )
    assert proof["graph_commitment"] == graph.graph_commitment
    assert proof["workstream_id"] == workstream_id

    reconstructed = graph.reconstructed_continuation_proof(
        workstream_id,
        "codex",
        context_receipt_commitments=[context_receipt["receipt_commitment"]],
        verification_commitments=[verification["record_commitment"]],
        outstanding_work_refs=["run Windows CI"],
        created_at_ms=1_501,
    )
    assert reconstructed["from_agent"] == ""
    assert reconstructed["handoff_commitment"] == ""
    assert "unknown:previous-agent-intent" in reconstructed["outstanding_work_refs"]

    stored_reconstruction = store.reconstructed_continuation_proof(
        workstream_id,
        "codex",
        verification_commitments=[verification["record_commitment"]],
        outstanding_work_refs=["run package tests"],
        created_at_ms=1_502,
    )
    assert stored_reconstruction["graph_commitment"] == graph.graph_commitment
