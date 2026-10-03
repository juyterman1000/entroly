"""What can the Entroly continuity arm (B3) actually contain?

This is the cheapest possible falsification of the continuity wedge, and it runs
before any paid agent call. If Entroly's production-reachable handoff state
carries no more decision-relevant information than an agent-written paragraph
(baseline B1), continuity is dead and a two-vendor pilot would spend real money
measuring noise.

Rules this probe obeys, matching the pilot's rules:

  * Only production-reachable surfaces -- the exact calls a shipped entry point
    makes, with the canonical constructors the package exports. No hand
    enrichment of the Entroly arm.
  * No hidden chain-of-thought and no unrecorded intent. Everything below is a
    file state, a tool result, a test verdict or an explicit recorded decision.
  * The interesting content is a FAILED verification -- a rejected hypothesis the
    next agent must not retry. That is the single most valuable thing a handoff
    can carry and the thing a summary is most likely to omit or soften.

Output is the raw payload plus a leaf census, because the honest comparison
against B1 is about which facts transfer, not about who writes better prose.

Run: python research/experiments/b3_handoff_content.py
Writes research/ledger/b3_handoff_content.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "ledger" / "b3_handoff_content.json"

REPO_ID = "repo:continuity-pilot"


def census(obj, prefix="") -> dict[str, dict]:
    """Flatten to leaf paths with type and emptiness.

    An empty list is exactly what this probe is hunting: a field that exists in
    the schema and carries nothing is not transferred knowledge.
    """
    out: dict[str, dict] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(census(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = {"type": "list", "len": len(obj), "empty": not obj}
        for i, item in enumerate(obj[:3]):
            out.update(census(item, f"{prefix}[{i}]"))
    else:
        out[prefix] = {
            "type": type(obj).__name__,
            "empty": obj is None or obj == "" or obj == 0,
            "preview": str(obj)[:160],
        }
    return out


def observation() -> dict:
    """A real interrupted task, recorded only as observable artefacts."""
    return {
        "repo_id": REPO_ID,
        "observed_at_ms": 1_000,
        "repository_label": "entroly-codebase-intelligence",
        "agent_id": "agent:claude",
        "session_id": "session-a",
        "task_hint": {
            "task_id": "task-budget",
            "title": "make the per-request token budget configurable",
            "trust": "observed",
            "explicit_status": "in_progress",
            # The two things a handoff must carry and a summary tends to lose.
            "remaining_work": [
                "wire the new config field through the Rust engine",
                "re-run tests/test_engine_budget.py",
            ],
            "source_kind": "user_statement",
            "source_ref": "user:task",
        },
        "branch": {
            "name": "feat/configurable-budget",
            "head_sha": "abc123",
            "default_branch": "main",
            "ahead_by": 1,
        },
        "changes": [
            {"path": "entroly/config.py", "kind": "modified",
             "staged": False, "conflicted": False},
            {"path": "entroly/engine.py", "kind": "modified",
             "staged": False, "conflicted": False},
        ],
        "decisions": [
            {
                "decision_id": "decision-1",
                "text": "budget must stay per-request; a global default broke cache alignment",
                "source_ref": "checkpoint:1",
                "source_kind": "checkpoint",
                "trust": "observed",
            },
        ],
    }


def main() -> int:
    os.environ["ENTROLY_NO_SELF_HEAL"] = "1"
    work_dir = tempfile.mkdtemp(prefix="entroly_b3_")

    report: dict[str, object] = {
        "baseline_commit": "9ba8d410",
        "rules": "production-reachable calls only; no hand enrichment",
    }

    try:
        import entroly_core
    except ImportError as exc:
        report["status"] = f"native engine unavailable: {exc}"
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["status"])
        return 1

    from entroly.work_graph import (
        WorkGraph,
        create_model_execution_outcome,
        create_routing_decision,
        create_verification_record,
    )
    from entroly.work_graph_store import WorkGraphStore

    graph = WorkGraph(REPO_ID)
    graph.observe_repository(observation())
    work = graph.unfinished()[0]
    workstream_id = work["node_id"]
    task_id = work["task_ids"][0]

    store = WorkGraphStore(REPO_ID, root=str(pathlib.Path(work_dir) / "state"))
    store.save(graph)

    # A context receipt: what evidence agent A actually received, including what
    # was OMITTED and how to recover it. A prose summary has no equivalent.
    receipt = json.loads(entroly_core.context_receipt_build_json(
        REPO_ID, "abc123", graph.graph_commitment, workstream_id,
        "sha256:sources",
        ["entroly/config.py#0:40", "entroly/engine.py#1750:1890"],
        ["entroly-core/src/lib.rs#700:780"],          # omitted
        ["evidence:pytest"],
        ["entroly-core/src/lib.rs#700:780"],          # recoverable
        ["rh_budget_01"],                             # recovery handles
        ["evidence:pytest"],
        512, "work-scope/v1", "execution:pending", 1_050,
    ))
    graph, _ = store.record_context_receipt(
        receipt, agent_id="agent:claude", session_id="session-a"
    )

    route = create_routing_decision(
        repository_id=REPO_ID, task_id=task_id, workstream_id=workstream_id,
        provider="anthropic", model="claude-opus", runtime="messages-api",
        context_budget_tokens=8192, policy_version="policy:v1",
        reason_codes=["capability_match"],
        feature_commitments=["sha256:features"],
        receipt_id=receipt["receipt_id"],
        evidence_ids=["evidence:route"], decided_at_ms=1_100,
    )
    outcome = create_model_execution_outcome(
        routing_id=route["routing_id"], repository_id=REPO_ID,
        task_id=task_id, workstream_id=workstream_id,
        provider="anthropic", model="claude-opus", runtime="messages-api",
        receipt_id=receipt["receipt_id"],
        request_commitment="sha256:request",
        response_commitment="sha256:response",
        # The failure is the point.
        state="failed", verification_state="failed",
        latency_ms=25, input_tokens=4100, output_tokens=380,
        cost_micro_usd=9_900,
        evidence_ids=["evidence:pytest"], completed_at_ms=1_200,
    )
    verification = create_verification_record(
        repository_id=REPO_ID, subject_id=outcome["outcome_id"],
        subject_commitment=outcome["outcome_commitment"],
        verified_repository_commitment="abc123",
        verdict="failed", evidence_ids=["evidence:pytest"],
        dependency_commitments=["sha256:source"], observed_at_ms=1_300,
    )
    graph, _ = store.record_execution_chain(route, outcome, verification)

    # ── What the next agent can actually be handed ──────────────────────
    handoff = store.handoff(
        workstream_id, "agent:claude", "agent:codex", generated_at_ms=1_400,
    )
    resume = store.resume()
    scope = store.load().context_scope(workstream_id)
    proof = store.continuation_proof(
        handoff,
        context_receipt_commitments=[receipt["receipt_commitment"]],
        routing_commitments=[route["decision_commitment"]],
        execution_outcome_commitments=[outcome["outcome_commitment"]],
        verification_commitments=[verification["record_commitment"]],
        memory_commitments=[],
        outstanding_work_refs=list(
            work.get("remaining_work", []) or []
        ),
        recovery_handle_ids=["rh_budget_01"],
        created_at_ms=1_500,
    )

    payloads = {
        "handoff": handoff, "resume": resume,
        "context_scope": scope, "continuation_proof": proof,
    }
    summary = {}
    for name, payload in payloads.items():
        leaves = census(payload)
        empty = sorted(k for k, v in leaves.items() if v.get("empty"))
        summary[name] = {
            "leaf_fields": len(leaves),
            "empty_fields": len(empty),
            "fill_rate": round(1 - len(empty) / max(len(leaves), 1), 3),
            "empty_field_names": empty,
        }

    report.update({
        "workstream_id": workstream_id,
        "handoff_verifies": store.load().verify_handoff(handoff),
        "payloads": payloads,
        "census_summary": summary,
        # The decisive question for B1 vs B3.
        "facts_a_prose_summary_cannot_carry": {
            "omitted_evidence_spans": receipt.get("omitted_source_refs")
            or receipt.get("omitted"),
            "recovery_handles": receipt.get("recovery_handle_ids"),
            "verification_verdict": verification["verdict"],
            "verified_against_repo_commitment":
                verification.get("verified_repository_commitment"),
            "handoff_integrity_verifiable": store.load().verify_handoff(handoff),
            "outstanding_work_refs": proof.get("outstanding_work_refs"),
        },
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "handoff_verifies": report["handoff_verifies"],
        "census_summary": summary,
        "facts_a_prose_summary_cannot_carry":
            report["facts_a_prose_summary_cannot_carry"],
    }, indent=2, default=str)[:6000])
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
