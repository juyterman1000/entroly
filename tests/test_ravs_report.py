"""
Tests for RAVS v1 offline evaluation report.

Coverage:
  RPT-01  EMPTY LOG              empty/missing log produces valid zero report
  RPT-02  BYTE-STABLE JSON       same input → identical JSON on two runs
  RPT-03  MALFORMED LINES        bad lines counted, skipped, not fatal
  RPT-04  HONEST LABEL COVERAGE  strong signals produce non-zero honest coverage
  RPT-05  WEAK EXCLUDED DEFAULT  weak self-report excluded from headline metrics
  RPT-06  WEAK INCLUDED FLAG     --include-weak makes weak count in headlines
  RPT-07  RETRY RATE             retry_event outcomes counted in retry_rate
  RPT-08  ESCALATION RATE        escalation_event outcomes counted
  RPT-09  COST BY MODEL          per-model cost aggregation correct
  RPT-10  LATENCY BY MODEL       p50/p95 computed correctly
  RPT-11  DECOMP EVIDENCE RATE   traces with decomposition evidence counted
  RPT-12  DECOMP BY KIND         kind counts correct
  RPT-13  SHADOW AGREEMENT       stub policies produce agreement metrics
  RPT-14  SHADOW COST DELTA      labeled as ESTIMATE, no regret field
  RPT-15  SINCE FILTER           --since filters traces by timestamp
  RPT-16  PRECEDENCE ORDER       user_acceptance > ci_result > test > escalation > retry > weak
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
    AppendOnlyEventLog,
    DecompositionEvidence,
    OutcomeEvent,
    TraceEvent,
)
from entroly.ravs.report import generate_report, format_report_text  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────


@pytest.fixture
def log(tmp_path):
    return AppendOnlyEventLog(tmp_path / "events.jsonl")


def _trace(rid, model="gpt-4o-mini", cost=-1.0, latency=-1.0,
           ctx_tokens=0, ts=None, retrieved=None, decomp=None,
           shadow_recs=None):
    return TraceEvent(
        request_id=rid, model=model, cost_usd=cost,
        latency_ms=latency, context_size_tokens=ctx_tokens,
        timestamp=ts if ts is not None else time.time(),
        retrieved_fragments=retrieved or [],
        decomposition_evidence=decomp or [],
        shadow_recommendations=shadow_recs or {},
    )


def _outcome(rid, etype, value, strength, ts=None, train=True):
    return OutcomeEvent(
        request_id=rid, event_type=etype, value=value,
        strength=strength, source="test",
        include_in_default_training=train,
        timestamp=ts if ts is not None else time.time(),
    )


def _report_from_log(log, **kwargs):
    return generate_report(str(log.path), **kwargs)


# ══════════════════════════════════════════════════════════════════════
# RPT-01: Empty / missing log
# ══════════════════════════════════════════════════════════════════════


def test_rpt01_empty_log_valid_zero_report(tmp_path):
    report = generate_report(str(tmp_path / "nonexistent.jsonl"))
    assert report["total_requests"] == 0
    assert report["honest_label_coverage"] == 0.0
    assert report["weak_label_coverage"] == 0.0
    assert report["unknown_label_rate"] == 0.0
    assert report["retry_rate"] == 0.0
    assert report["escalation_rate"] == 0.0
    assert report["parsing"]["malformed_lines_skipped"] == 0
    # Must be valid JSON
    json.dumps(report, sort_keys=True)


def test_rpt01b_empty_file_valid_zero_report(tmp_path):
    (tmp_path / "empty.jsonl").write_text("")
    report = generate_report(str(tmp_path / "empty.jsonl"))
    assert report["total_requests"] == 0


# ══════════════════════════════════════════════════════════════════════
# RPT-02: Byte-stable JSON
# ══════════════════════════════════════════════════════════════════════


def test_rpt02_byte_stable_json(log):
    log.write_trace(_trace("r1", cost=0.001, latency=150.0, ts=1000.0))
    log.write_outcome(_outcome("r1", "test_result", "passed", "strong", ts=1001.0))
    log.write_trace(_trace("r2", cost=0.002, latency=200.0, ts=1002.0))
    log.write_outcome(_outcome("r2", "command_exit", "failure", "strong", ts=1003.0))

    r1 = generate_report(str(log.path))
    r2 = generate_report(str(log.path))

    # Exclude generated_at_utc (varies by wall clock)
    r1.pop("generated_at_utc", None)
    r2.pop("generated_at_utc", None)

    j1 = json.dumps(r1, sort_keys=True, ensure_ascii=False)
    j2 = json.dumps(r2, sort_keys=True, ensure_ascii=False)
    assert j1 == j2


# ══════════════════════════════════════════════════════════════════════
# RPT-03: Malformed lines
# ══════════════════════════════════════════════════════════════════════


def test_rpt03_malformed_lines_counted_and_skipped(tmp_path):
    log_path = tmp_path / "events.jsonl"
    log_path.write_text(
        '{"kind":"trace","request_id":"r1","timestamp":1000}\n'
        'THIS IS NOT JSON\n'
        '{"kind":"outcome","request_id":"r1","event_type":"test_result","value":"passed","strength":"strong","timestamp":1001}\n'
        '{"broken json\n'
        '\n'
    )
    report = generate_report(str(log_path))
    assert report["total_requests"] == 1
    assert report["parsing"]["malformed_lines_skipped"] == 2


# ══════════════════════════════════════════════════════════════════════
# RPT-04: Honest label coverage
# ══════════════════════════════════════════════════════════════════════


def test_rpt04_honest_label_coverage(log):
    log.write_trace(_trace("r1", ts=1000.0))
    log.write_trace(_trace("r2", ts=1001.0))
    log.write_trace(_trace("r3", ts=1002.0))
    log.write_outcome(_outcome("r1", "test_result", "passed", "strong", ts=1003.0))
    log.write_outcome(_outcome("r2", "ci_result", "failed", "strong", ts=1004.0))
    # r3 has no outcome → unknown

    report = _report_from_log(log)
    assert report["total_requests"] == 3
    assert report["honest_label_coverage"] == round(2 / 3, 4)
    assert report["unknown_label_rate"] == round(1 / 3, 4)


# ══════════════════════════════════════════════════════════════════════
# RPT-05: Weak excluded by default
# ══════════════════════════════════════════════════════════════════════


def test_rpt05_weak_excluded_default(log):
    log.write_trace(_trace("r1", ts=1000.0))
    log.write_outcome(_outcome("r1", "agent_self_report", "success", "weak", ts=1001.0, train=False))

    report = _report_from_log(log)
    assert report["honest_label_coverage"] == 0.0  # weak excluded
    assert report["unknown_label_rate"] == 1.0
    assert report["label_breakdown"]["unknown"] == 1


# ══════════════════════════════════════════════════════════════════════
# RPT-06: --include-weak
# ══════════════════════════════════════════════════════════════════════


def test_rpt06_weak_included_with_flag(log):
    log.write_trace(_trace("r1", ts=1000.0))
    log.write_outcome(_outcome("r1", "agent_self_report", "success", "weak", ts=1001.0, train=False))

    report = _report_from_log(log, include_weak=True)
    assert report["honest_label_coverage"] == 1.0
    assert report["unknown_label_rate"] == 0.0
    assert report["label_breakdown"]["successes"] == 1


# ══════════════════════════════════════════════════════════════════════
# RPT-07: Retry rate
# ══════════════════════════════════════════════════════════════════════


def test_rpt07_retry_rate(log):
    log.write_trace(_trace("r1", ts=1000.0))
    log.write_trace(_trace("r2", ts=1001.0))
    log.write_outcome(_outcome("r1", "retry_event", "failure", "medium", ts=1002.0))

    report = _report_from_log(log)
    assert report["retry_rate"] == 0.5
    assert report["retry_count"] == 1


# ══════════════════════════════════════════════════════════════════════
# RPT-08: Escalation rate
# ══════════════════════════════════════════════════════════════════════


def test_rpt08_escalation_rate(log):
    log.write_trace(_trace("r1", ts=1000.0))
    log.write_trace(_trace("r2", ts=1001.0))
    log.write_trace(_trace("r3", ts=1002.0))
    log.write_outcome(_outcome("r1", "escalation_event", "failure", "medium", ts=1003.0))
    log.write_outcome(_outcome("r2", "escalation_event", "failure", "medium", ts=1004.0))

    report = _report_from_log(log)
    assert report["escalation_rate"] == round(2 / 3, 4)
    assert report["escalation_count"] == 2


# ══════════════════════════════════════════════════════════════════════
# RPT-09: Cost by model
# ══════════════════════════════════════════════════════════════════════


def test_rpt09_cost_by_model(log):
    log.write_trace(_trace("r1", model="gpt-4o-mini", cost=0.001, ts=1000.0))
    log.write_trace(_trace("r2", model="gpt-4o-mini", cost=0.003, ts=1001.0))
    log.write_trace(_trace("r3", model="gpt-4o", cost=0.020, ts=1002.0))

    report = _report_from_log(log)
    mini = report["cost_by_model"]["gpt-4o-mini"]
    assert mini["requests"] == 2
    assert mini["total_cost_usd"] == 0.004
    assert mini["avg_cost_usd"] == 0.002

    full = report["cost_by_model"]["gpt-4o"]
    assert full["requests"] == 1
    assert full["total_cost_usd"] == 0.02


# ══════════════════════════════════════════════════════════════════════
# RPT-10: Latency by model
# ══════════════════════════════════════════════════════════════════════


def test_rpt10_latency_by_model(log):
    # 10 requests to build a meaningful p50/p95
    for i in range(10):
        log.write_trace(_trace(f"r{i}", model="gpt-4o-mini",
                               latency=100.0 + i * 10, ts=1000.0 + i))

    report = _report_from_log(log)
    lat = report["latency_by_model"]["gpt-4o-mini"]
    assert lat["n"] == 10
    assert lat["p50_ms"] > 0
    assert lat["p95_ms"] >= lat["p50_ms"]


# ══════════════════════════════════════════════════════════════════════
# RPT-11: Decomposition evidence rate
# ══════════════════════════════════════════════════════════════════════


def test_rpt11_decomp_evidence_rate(log):
    log.write_trace(_trace("r1", ts=1000.0, decomp=[
        {"kind": "computation", "source": "tool_call"},
    ]))
    log.write_trace(_trace("r2", ts=1001.0))  # no decomp
    log.write_trace(_trace("r3", ts=1002.0, decomp=[
        {"kind": "retrieval", "source": "response_pattern"},
        {"kind": "computation", "source": "test_result"},
    ]))

    report = _report_from_log(log)
    assert report["decomposition_evidence_rate"] == round(2 / 3, 4)
    assert report["decomposition_evidence_count"] == 3


# ══════════════════════════════════════════════════════════════════════
# RPT-12: Decomposition by kind
# ══════════════════════════════════════════════════════════════════════


def test_rpt12_decomp_by_kind(log):
    log.write_trace(_trace("r1", ts=1000.0, decomp=[
        {"kind": "computation", "source": "tool_call"},
        {"kind": "computation", "source": "test_result"},
        {"kind": "retrieval", "source": "tool_call"},
    ]))

    report = _report_from_log(log)
    dbk = report["decomposition_by_kind"]
    assert dbk["computation"] == 2
    assert dbk["retrieval"] == 1


# ══════════════════════════════════════════════════════════════════════
# RPT-13: Shadow agreement
# ══════════════════════════════════════════════════════════════════════


def test_rpt13_shadow_agreement(log):
    shadow = {
        "p1": {"insufficient_data": True, "model": None,
               "predicted_p_success": None, "policy_name": "p1", "reason": ""},
        "p2": {"insufficient_data": False, "model": "gpt-4o",
               "predicted_p_success": 0.8, "policy_name": "p2", "reason": ""},
    }
    log.write_trace(_trace("r1", model="gpt-4o-mini", ts=1000.0,
                           shadow_recs=shadow))

    report = _report_from_log(log)
    spa = report["shadow_policy_agreement"]
    assert "p1" in spa
    assert spa["p1"]["abstain_rate"] == 1.0
    assert "p2" in spa
    assert spa["p2"]["agree_rate"] == 0.0  # recommended gpt-4o, prod used gpt-4o-mini


# ══════════════════════════════════════════════════════════════════════
# RPT-14: Shadow cost delta — ESTIMATE, not regret
# ══════════════════════════════════════════════════════════════════════


def test_rpt14_shadow_cost_delta_is_estimate(log):
    # Need at least some cost data for both models
    log.write_trace(_trace("r1", model="gpt-4o-mini", cost=0.001, ts=1000.0))
    log.write_trace(_trace("r2", model="gpt-4o", cost=0.020, ts=1001.0))
    shadow = {
        "p1": {"insufficient_data": False, "model": "gpt-4o",
               "predicted_p_success": 0.9, "policy_name": "p1", "reason": ""},
    }
    log.write_trace(_trace("r3", model="gpt-4o-mini", cost=0.001, ts=1002.0,
                           shadow_recs=shadow))

    report = _report_from_log(log)
    scd = report["shadow_policy_recommendation_cost_delta"]

    # Verify it's labeled as estimate
    if "p1" in scd:
        assert "ESTIMATE" in scd["p1"]["note"]
        assert "counterfactual" in scd["p1"]["note"].lower()

    # Verify no regret field anywhere
    report_str = json.dumps(report)
    assert "regret" not in report_str.lower()
    assert "would_have" not in report_str.lower()


# ══════════════════════════════════════════════════════════════════════
# RPT-15: --since filter
# ══════════════════════════════════════════════════════════════════════


def test_rpt15_since_filter(log):
    log.write_trace(_trace("old", ts=1000.0))
    log.write_trace(_trace("new", ts=2000.0))
    log.write_outcome(_outcome("old", "test_result", "passed", "strong", ts=1001.0))
    log.write_outcome(_outcome("new", "test_result", "failed", "strong", ts=2001.0))

    # Without --since: 2 traces
    report_all = _report_from_log(log)
    assert report_all["total_requests"] == 2

    # With --since=1500: only 'new'
    report_since = _report_from_log(log, since_timestamp=1500.0)
    assert report_since["total_requests"] == 1
    assert report_since["label_breakdown"]["failures"] == 1


# ══════════════════════════════════════════════════════════════════════
# RPT-16: Precedence order
# ══════════════════════════════════════════════════════════════════════


def test_rpt16_precedence_user_acceptance_wins_over_test(log):
    """user_acceptance (strong) > test_result (strong) in precedence."""
    log.write_trace(_trace("r1", ts=1000.0))
    log.write_outcome(_outcome("r1", "test_result", "passed", "strong", ts=1001.0))
    log.write_outcome(_outcome("r1", "user_acceptance", "rejected", "strong", ts=1002.0))

    report = _report_from_log(log)
    # user_acceptance has higher precedence than test_result
    assert report["label_breakdown"]["failures"] == 1
    assert report["label_breakdown"]["successes"] == 0


def test_rpt16_ci_wins_over_retry(log):
    """ci_result > retry_event in precedence."""
    log.write_trace(_trace("r1", ts=1000.0))
    log.write_outcome(_outcome("r1", "retry_event", "failure", "medium", ts=1001.0))
    log.write_outcome(_outcome("r1", "ci_result", "passed", "strong", ts=1002.0))

    report = _report_from_log(log)
    assert report["label_breakdown"]["successes"] == 1


def test_rpt16_text_format_renders(log):
    """Smoke test: text format doesn't crash and contains key sections."""
    log.write_trace(_trace("r1", model="gpt-4o-mini", cost=0.001,
                           latency=150.0, ts=1000.0))
    log.write_outcome(_outcome("r1", "test_result", "passed", "strong", ts=1001.0))

    report = _report_from_log(log)
    text = format_report_text(report)
    assert "RAVS v1" in text
    assert "Label Coverage" in text
    assert "Behavioral Signals" in text
    assert "Cost by Model" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
