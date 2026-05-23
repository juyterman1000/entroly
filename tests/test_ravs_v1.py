"""
Tests for RAVS v1 (instrumentation + shadow framework).

Coverage:
  R-01  EVENTS SCHEMA INVARIANTS    enums closed; fields stable
  R-02  APPEND-ONLY LOG ROUNDTRIP   trace+outcome round-trips through JSONL
  R-03  ORPHAN OUTCOMES SKIPPED     outcomes without parent trace are not yielded
  R-04  REDUCER DEFAULT             strong > medium; weak excluded
  R-05  REDUCER STRICT              requires strong; otherwise unknown
  R-06  REDUCER LEGACY              includes weak (for backwards-compat eval only)
  R-07  REDUCER CODE_CHANGE         requires code-touching trace AND strong signal
  R-08  REDUCER CI_ONLY             only ci_result events count
  R-09  RETRY COLLECTOR REPHRASE    same-text re-issue → retry_event for prior id
  R-10  RETRY COLLECTOR TOPIC       different text → topic_change
  R-11  RETRY COLLECTOR WINDOW      ttl-expired prior produces no event
  R-12  ESCALATION STRONGER         same query, weaker→stronger model → escalation_event
  R-13  ESCALATION DOWNGRADE        same query, stronger→weaker → no event (no signal)
  R-14  ESCALATION SAME MODEL       same query, same model → no event (RetryCollector's job)
  R-15  ESCALATION UNKNOWN MODELS   unknown ordering → conservative no event
  R-16  SHADOW V1 STUBS             all 4 default policies return insufficient_data
  R-17  SHADOW AGREEMENT METRIC     stub abstentions counted; agreed=False
  R-18  NO COUNTERFACTUAL REGRET    aggregator never returns 'regret' field
  R-19  DECOMPOSITION EVIDENCE      kind/source/executor/verifier enums respected
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.ravs.events import (  # noqa: E402
    DECOMPOSITION_EXECUTORS,
    DECOMPOSITION_KINDS,
    DECOMPOSITION_SOURCES,
    DECOMPOSITION_VERIFIERS,
    EVENT_STRENGTHS,
    HONEST_OUTCOME_TYPES,
    AppendOnlyEventLog,
    DecompositionEvidence,
    OutcomeEvent,
    TraceEvent,
    derive_label,
)
from entroly.ravs.shadow import (  # noqa: E402
    ShadowEvaluator,
    aggregate_shadow_agreement,
    make_default_policies,
    shadow_agreement,
)
from entroly.ravs.collectors.escalation import EscalationCollector  # noqa: E402
from entroly.ravs.collectors.retry import RetryCollector  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────


@pytest.fixture
def log(tmp_path):
    return AppendOnlyEventLog(tmp_path / "events.jsonl")


def _outcome(rid, etype, value, strength, ts=None, train=True):
    return OutcomeEvent(
        request_id=rid, event_type=etype, value=value, strength=strength,
        source="test", include_in_default_training=train,
        timestamp=ts if ts is not None else time.time(),
    )


# ══════════════════════════════════════════════════════════════════════
# R-01..R-03: Schema + log invariants
# ══════════════════════════════════════════════════════════════════════


def test_r01_schema_enum_closure():
    # Enums are closed sets — adding a value is an explicit code change.
    assert "computation" in DECOMPOSITION_KINDS
    assert "judgment" in DECOMPOSITION_KINDS
    assert "synthesis" in DECOMPOSITION_KINDS
    assert "tool_call" in DECOMPOSITION_SOURCES
    assert "test_result" in DECOMPOSITION_SOURCES
    assert "sympy" in DECOMPOSITION_EXECUTORS
    assert "none" in DECOMPOSITION_EXECUTORS
    assert "exact" in DECOMPOSITION_VERIFIERS
    assert "none" in DECOMPOSITION_VERIFIERS
    assert "test_result" in HONEST_OUTCOME_TYPES
    assert EVENT_STRENGTHS == ("strong", "medium", "weak")


def test_r02_append_only_roundtrip(log):
    trace = TraceEvent(request_id="r1", query_text="hello",
                       retrieved_fragments=["src/a.py"])
    log.write_trace(trace)
    log.write_outcome(_outcome("r1", "test_result", "passed", "strong"))
    pairs = list(log.traces_with_outcomes())
    assert len(pairs) == 1
    t, outs = pairs[0]
    assert t["request_id"] == "r1"
    assert t["retrieved_fragments"] == ["src/a.py"]
    assert len(outs) == 1
    assert outs[0]["event_type"] == "test_result"


def test_r03_orphan_outcomes_skipped(log):
    log.write_outcome(_outcome("orphan-no-trace", "test_result", "passed", "strong"))
    log.write_trace(TraceEvent(request_id="r1"))
    pairs = list(log.traces_with_outcomes())
    assert len(pairs) == 1
    assert pairs[0][0]["request_id"] == "r1"
    assert pairs[0][1] == []  # no outcomes for r1


# ══════════════════════════════════════════════════════════════════════
# R-04..R-08: Reducer rules
# ══════════════════════════════════════════════════════════════════════


def test_r04_reducer_default_excludes_weak():
    trace = {"retrieved_fragments": ["src/a.py"]}
    outs = [
        _outcome("r", "agent_self_report", "success", "weak", ts=2.0, train=False),
        _outcome("r", "test_result", "passed", "strong", ts=1.0),
    ]
    # Default: strong wins even when weaker is more recent
    assert derive_label(trace, outs, rule="default") == "passed"
    # No strong → falls back to medium
    outs_no_strong = [_outcome("r", "retry_event", "failure", "medium", ts=1.0)]
    assert derive_label(trace, outs_no_strong, rule="default") == "failure"


def test_r05_reducer_strict_requires_strong():
    trace = {}
    medium_only = [_outcome("r", "retry_event", "failure", "medium")]
    assert derive_label(trace, medium_only, rule="strict") == "unknown"
    with_strong = medium_only + [_outcome("r", "ci_result", "passed", "strong")]
    assert derive_label(trace, with_strong, rule="strict") == "passed"


def test_r06_reducer_legacy_includes_weak():
    trace = {}
    weak_only = [_outcome("r", "agent_self_report", "success", "weak", train=False)]
    assert derive_label(trace, weak_only, rule="legacy") == "success"
    # Default rule still ignores it
    assert derive_label(trace, weak_only, rule="default") == "unknown"


def test_r07_reducer_code_change_requires_code_AND_strong():
    code_trace = {"retrieved_fragments": ["src/auth.py", "src/util.rs"]}
    nocode_trace = {"retrieved_fragments": ["docs/README.md"]}
    strong = [_outcome("r", "test_result", "passed", "strong")]
    weak = [_outcome("r", "agent_self_report", "success", "weak", train=False)]

    assert derive_label(code_trace, strong, rule="code_change") == "passed"
    assert derive_label(code_trace, weak, rule="code_change") == "unknown"  # weak excluded
    assert derive_label(nocode_trace, strong, rule="code_change") == "unknown"  # not code
    assert derive_label({"retrieved_fragments": []}, strong, rule="code_change") == "unknown"


def test_r08_reducer_ci_only():
    trace = {}
    test = [_outcome("r", "test_result", "passed", "strong")]
    ci = [_outcome("r", "ci_result", "failed", "strong")]
    # ci_only ignores test_result events
    assert derive_label(trace, test, rule="ci_only") == "unknown"
    assert derive_label(trace, ci, rule="ci_only") == "failed"


# ══════════════════════════════════════════════════════════════════════
# R-09..R-11: RetryCollector
# ══════════════════════════════════════════════════════════════════════


def test_r09_retry_collector_emits_for_rephrase(log):
    rc = RetryCollector(log, time_window_s=120.0, similarity_threshold=0.5)
    # Same client, near-identical query (rephrase) → retry_event for prior id
    rc.observe(client_key="c1", request_id="req-1",
               query_text="how do I fix the auth login bug", timestamp=100.0)
    emitted = rc.observe(client_key="c1", request_id="req-2",
                         query_text="how do I fix the auth login bug exactly",
                         timestamp=110.0)
    assert emitted == "retry_event"
    pairs = list(log.traces_with_outcomes())
    # No traces written (collector only writes outcomes); read raw events
    events = list(log.read_all())
    outcome_events = [e for e in events if e.get("kind") == "outcome"]
    assert len(outcome_events) == 1
    assert outcome_events[0]["request_id"] == "req-1"  # attached to PRIOR request
    assert outcome_events[0]["event_type"] == "retry_event"
    assert outcome_events[0]["strength"] == "medium"
    assert outcome_events[0]["value"] == "failure"


def test_r10_retry_collector_topic_change(log):
    rc = RetryCollector(log, time_window_s=120.0, similarity_threshold=0.5)
    rc.observe(client_key="c1", request_id="req-1",
               query_text="how do I authenticate users", timestamp=100.0)
    emitted = rc.observe(client_key="c1", request_id="req-2",
                         query_text="compile the rust binary in release mode",
                         timestamp=110.0)
    assert emitted == "topic_change"
    events = [e for e in log.read_all() if e.get("kind") == "outcome"]
    assert len(events) == 1
    assert events[0]["event_type"] == "topic_change"
    assert events[0]["value"] == "success"


def test_r11_retry_collector_outside_window(log):
    rc = RetryCollector(log, time_window_s=10.0)  # tight window
    rc.observe(client_key="c1", request_id="req-1",
               query_text="anything", timestamp=100.0)
    # 30 seconds later — outside the 10s window
    emitted = rc.observe(client_key="c1", request_id="req-2",
                         query_text="anything else", timestamp=130.0)
    assert emitted is None
    assert [e for e in log.read_all() if e.get("kind") == "outcome"] == []


# ══════════════════════════════════════════════════════════════════════
# R-12..R-15: EscalationCollector
# ══════════════════════════════════════════════════════════════════════


def test_r12_escalation_emits_on_haiku_to_opus(log):
    ec = EscalationCollector(log, time_window_s=300.0, similarity_threshold=0.5)
    q = "is this database migration safe under concurrent writes"
    ec.observe(client_key="u1", request_id="r1", query_text=q,
               model="claude-haiku-4", timestamp=100.0)
    emitted = ec.observe(client_key="u1", request_id="r2", query_text=q,
                          model="claude-opus-4", timestamp=130.0)
    assert emitted == "escalation_event"
    events = [e for e in log.read_all() if e.get("kind") == "outcome"]
    assert len(events) == 1
    assert events[0]["request_id"] == "r1"  # attached to ORIGINAL
    assert events[0]["event_type"] == "escalation_event"
    assert events[0]["metadata"]["original_model"] == "claude-haiku-4"
    assert events[0]["metadata"]["escalated_to"] == "claude-opus-4"


def test_r13_escalation_no_event_on_downgrade(log):
    ec = EscalationCollector(log, similarity_threshold=0.5)
    q = "what's the capital of france"
    ec.observe(client_key="u1", request_id="r1", query_text=q,
               model="claude-opus-4", timestamp=100.0)
    # User then re-asks via Haiku — that's a downgrade, not escalation
    emitted = ec.observe(client_key="u1", request_id="r2", query_text=q,
                         model="claude-haiku-4", timestamp=110.0)
    assert emitted is None
    assert [e for e in log.read_all() if e.get("kind") == "outcome"] == []


def test_r14_escalation_no_event_same_model(log):
    ec = EscalationCollector(log, similarity_threshold=0.5)
    q = "fix this bug"
    ec.observe(client_key="u1", request_id="r1", query_text=q,
               model="gpt-4o-mini", timestamp=100.0)
    emitted = ec.observe(client_key="u1", request_id="r2", query_text=q,
                         model="gpt-4o-mini", timestamp=110.0)
    # Same model = retry's job, not escalation's
    assert emitted is None


def test_r15_escalation_unknown_models_safe(log):
    ec = EscalationCollector(log, similarity_threshold=0.5)
    q = "do this thing"
    # Unknown models — collector cannot determine ordering, conservatively skips
    ec.observe(client_key="u1", request_id="r1", query_text=q,
               model="some-future-model-X", timestamp=100.0)
    emitted = ec.observe(client_key="u1", request_id="r2", query_text=q,
                         model="other-future-model-Y", timestamp=110.0)
    assert emitted is None
    assert [e for e in log.read_all() if e.get("kind") == "outcome"] == []


# ══════════════════════════════════════════════════════════════════════
# R-16..R-18: Shadow framework
# ══════════════════════════════════════════════════════════════════════


def test_r16_shadow_v1_all_stubs_insufficient_data():
    se = ShadowEvaluator()
    recs = se.evaluate(
        query_text="anything", query_features={},
        current_model="gpt-4o-mini",
        candidates=["gpt-4o-mini", "gpt-4o"],
    )
    # 4 default policies
    assert len(recs) == 4

    # current_heuristic is deterministic — it always makes a recommendation
    # (never returns insufficient_data because it needs no training data)
    assert not recs["current_heuristic"]["insufficient_data"]
    assert recs["current_heuristic"]["model"] is not None

    # Learned policies (bucket_beta, embedding_knn, logistic_failure_predictor)
    # return insufficient_data when cold (no training observations yet)
    learned = ["bucket_beta", "embedding_knn", "logistic_failure_predictor"]
    for name in learned:
        assert recs[name]["insufficient_data"], (
            f"{name} should return insufficient_data when cold, got: {recs[name]}"
        )
        assert recs[name]["model"] is None


def test_r17_shadow_agreement_records_abstention():
    trace = {
        "policy_decision": "current_heuristic",
        "model": "gpt-4o-mini",
        "shadow_recommendations": {
            "p1": {"insufficient_data": True, "model": None,
                  "predicted_p_success": None, "policy_name": "p1", "reason": ""},
            "p2": {"insufficient_data": False, "model": "gpt-4o",
                  "predicted_p_success": 0.8, "policy_name": "p2", "reason": ""},
        },
    }
    out = shadow_agreement(trace)
    assert out["p1"]["abstained"] is True
    assert out["p1"]["agreed"] is False
    assert out["p2"]["abstained"] is False
    assert out["p2"]["agreed"] is False  # recommended gpt-4o, prod used gpt-4o-mini
    assert out["p2"]["recommended_model"] == "gpt-4o"


def test_r18_no_counterfactual_regret_in_aggregator():
    """The aggregator MUST NEVER produce a 'regret' field. We don't know
    what the shadow model would have done — only what it recommended."""
    trace = {
        "policy_decision": "current_heuristic",
        "model": "gpt-4o-mini",
        "shadow_recommendations": {
            "p1": {"insufficient_data": False, "model": "gpt-4o",
                  "predicted_p_success": 0.95, "policy_name": "p1", "reason": ""},
        },
    }
    agg = aggregate_shadow_agreement([trace, trace])
    assert "p1" in agg
    # These are the only honest aggregate metrics
    assert set(agg["p1"].keys()) <= {"agree_rate", "abstain_rate", "n"}
    # And specifically, no field that names or implies counterfactual outcomes
    assert "regret" not in agg["p1"]
    assert "would_have_succeeded" not in agg["p1"]
    assert "counterfactual" not in str(agg).lower()


# ══════════════════════════════════════════════════════════════════════
# R-19: DecompositionEvidence schema
# ══════════════════════════════════════════════════════════════════════


def test_r19_decomposition_evidence_enum_validation(log):
    # All enums in spec must be representable in a real evidence record
    ev = DecompositionEvidence(
        kind="computation", source="tool_call",
        executor_candidate="sympy", verifier_candidate="exact",
        confidence=0.95, span_hash="abc", notes="factor(x^2-1)",
    )
    trace = TraceEvent(
        request_id="r-cmp",
        query_text="factor x^2-1",
        retrieved_fragments=["math.py"],
        decomposition_evidence=[
            {"kind": ev.kind, "source": ev.source,
             "executor_candidate": ev.executor_candidate,
             "verifier_candidate": ev.verifier_candidate,
             "confidence": ev.confidence,
             "span_hash": ev.span_hash, "notes": ev.notes},
        ],
    )
    log.write_trace(trace)

    # Round-trip
    pairs = list(log.traces_with_outcomes())
    assert len(pairs) == 1
    t, _ = pairs[0]
    assert len(t["decomposition_evidence"]) == 1
    e = t["decomposition_evidence"][0]
    assert e["kind"] in DECOMPOSITION_KINDS
    assert e["source"] in DECOMPOSITION_SOURCES
    assert e["executor_candidate"] in DECOMPOSITION_EXECUTORS
    assert e["verifier_candidate"] in DECOMPOSITION_VERIFIERS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
