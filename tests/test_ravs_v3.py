"""
Tests for RAVS v3 — Guarded Production Router.

Coverage:
  V3-01  DISABLED BY DEFAULT       router starts disabled
  V3-02  FAIL CLOSED               disabled router returns use_original=True
  V3-03  GATE MUST PASS            can't enable without passing gate
  V3-04  ENABLE AFTER GATE         enabling works after gate passes
  V3-05  AUTO DISABLE              auto-disables if gate fails while enabled
  V3-06  HIGH RISK BLOCKED         security/auth queries never downgraded
  V3-07  STANDARD TOLERANCE        coding queries tolerate 2% loss max
  V3-08  LOW TOLERANCE             chat queries tolerate 5% loss max
  V3-09  CHEAPER MODEL SELECTED    routes to cheapest passing model
  V3-10  NO CHEAPER AVAILABLE      returns original if no model is cheaper
  V3-11  UNKNOWN MODEL BLOCKED     blocks if model stats unknown
  V3-12  DECISION UNDER 1MS        routing decision is fast
  V3-13  STATS ACCOUNTING          counters track decisions correctly
  V3-14  GATE FROM REPORT          compute_gate_status from report dict
  V3-15  GATE SAMPLE SIZE          gate fails without enough samples
  V3-16  THREAD SAFE               concurrent access doesn't crash
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.ravs.router import (  # noqa: E402
    DomainRisk,
    GateStatus,
    GuardedRouter,
    RoutingDecision,
    classify_risk,
    compute_gate_status,
)


def _make_ready_router() -> GuardedRouter:
    """Create a router with gate passed and model stats loaded."""
    router = GuardedRouter()

    gate = GateStatus(
        passed=True,
        reason="all gates passed",
        decomposition_evidence_rate=0.55,
        executor_coverage=0.60,
        shadow_success_rate=0.85,
        sample_size=100,
        model_readiness={
            "gpt-4o": True,
            "gpt-4o-mini": True,
            "claude-3-haiku": True,
        },
    )
    router.update_gate(gate)

    router.update_model_stats(
        model_success={"gpt-4o": 0.92, "gpt-4o-mini": 0.90, "claude-3-haiku": 0.88},
        model_cost={"gpt-4o": 0.005, "gpt-4o-mini": 0.0003, "claude-3-haiku": 0.00025},
    )

    router.enable()
    return router


# ══════════════════════════════════════════════════════════════════════
# V3-01: Disabled by default
# ══════════════════════════════════════════════════════════════════════


def test_v3_01_disabled_by_default():
    router = GuardedRouter()
    assert not router.stats()["enabled"]


# ══════════════════════════════════════════════════════════════════════
# V3-02: Fail closed when disabled
# ══════════════════════════════════════════════════════════════════════


def test_v3_02_fail_closed_when_disabled():
    router = GuardedRouter()
    decision = router.route("calculate 2+2", "gpt-4o")
    assert decision.use_original
    assert "disabled" in decision.reason


# ══════════════════════════════════════════════════════════════════════
# V3-03: Can't enable without passing gate
# ══════════════════════════════════════════════════════════════════════


def test_v3_03_gate_must_pass_before_enable():
    router = GuardedRouter()
    router.enable()  # should be a no-op
    assert not router.stats()["enabled"]


# ══════════════════════════════════════════════════════════════════════
# V3-04: Enable after gate passes
# ══════════════════════════════════════════════════════════════════════


def test_v3_04_enable_after_gate():
    router = _make_ready_router()
    assert router.stats()["enabled"]


# ══════════════════════════════════════════════════════════════════════
# V3-05: Auto-disable if gate fails
# ══════════════════════════════════════════════════════════════════════


def test_v3_05_auto_disable_on_gate_fail():
    router = _make_ready_router()
    assert router.stats()["enabled"]

    # Gate fails
    bad_gate = GateStatus(passed=False, reason="sample_size too low")
    router.update_gate(bad_gate)
    assert not router.stats()["enabled"]


# ══════════════════════════════════════════════════════════════════════
# V3-06: High risk blocked (security, auth, payments)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("query", [
    "fix the authentication bypass vulnerability",
    "update the password hashing function",
    "review the payment processing logic",
    "check the encryption key rotation",
])
def test_v3_06_high_risk_blocked(query):
    router = _make_ready_router()
    decision = router.route(query, "gpt-4o")
    assert decision.use_original
    assert "high_risk" in decision.reason


# ══════════════════════════════════════════════════════════════════════
# V3-07: Standard tolerance (2% max)
# ══════════════════════════════════════════════════════════════════════


def test_v3_07_standard_tolerance():
    router = _make_ready_router()

    # gpt-4o success=0.92, gpt-4o-mini success=0.90
    # tolerance for standard = 0.02
    # 0.90 >= 0.92 - 0.02 = 0.90 → PASSES (exactly at boundary)
    decision = router.route("add unit tests for the parser", "gpt-4o")
    assert not decision.use_original
    assert decision.recommended_model in ("gpt-4o-mini", "claude-3-haiku")


# ══════════════════════════════════════════════════════════════════════
# V3-08: Low tolerance (5% max)
# ══════════════════════════════════════════════════════════════════════


def test_v3_08_low_tolerance():
    router = _make_ready_router()

    # "explain" triggers LOW risk (5% tolerance)
    # 0.90 >= 0.92 - 0.05 = 0.87 → PASSES
    decision = router.route("explain how async works in Python", "gpt-4o")
    assert not decision.use_original


# ══════════════════════════════════════════════════════════════════════
# V3-09: Cheapest model selected
# ══════════════════════════════════════════════════════════════════════


def test_v3_09_cheapest_model_selected():
    router = _make_ready_router()
    decision = router.route("add logging to the handler", "gpt-4o")

    # claude-3-haiku ($0.00025) < gpt-4o-mini ($0.0003)
    # but haiku success=0.88 vs standard tolerance=0.02
    # 0.88 >= 0.92 - 0.02 = 0.90 → FAILS (0.88 < 0.90)
    # so gpt-4o-mini should be selected (0.90 >= 0.90 → passes)
    assert decision.recommended_model == "gpt-4o-mini"


# ══════════════════════════════════════════════════════════════════════
# V3-10: No cheaper model available
# ══════════════════════════════════════════════════════════════════════


def test_v3_10_no_cheaper_available():
    router = _make_ready_router()
    # Already using cheapest model (claude-3-haiku at $0.00025)
    decision = router.route("add tests", "claude-3-haiku")
    assert decision.use_original
    assert "no_cheaper" in decision.reason


# ══════════════════════════════════════════════════════════════════════
# V3-11: Unknown model blocked
# ══════════════════════════════════════════════════════════════════════


def test_v3_11_unknown_model_blocked():
    router = _make_ready_router()
    decision = router.route("add tests", "unknown-model-v9")
    assert decision.use_original
    assert "unknown" in decision.reason


# ══════════════════════════════════════════════════════════════════════
# V3-12: Decision under 1ms
# ══════════════════════════════════════════════════════════════════════


def test_v3_12_decision_under_1ms():
    router = _make_ready_router()
    times = []
    for _ in range(100):
        d = router.route("add logging to handler", "gpt-4o")
        times.append(d.decision_time_ms)
    avg = sum(times) / len(times)
    assert avg < 1.0, f"avg decision time {avg:.3f}ms exceeds 1ms"


# ══════════════════════════════════════════════════════════════════════
# V3-13: Stats accounting
# ══════════════════════════════════════════════════════════════════════


def test_v3_13_stats_accounting():
    router = _make_ready_router()

    router.route("add tests", "gpt-4o")                # should route
    router.route("fix auth vulnerability", "gpt-4o")   # should block (high risk)
    router.route("add more tests", "gpt-4o-mini")      # no cheaper

    stats = router.stats()
    assert stats["decisions_made"] == 3
    assert stats["routes_changed"] >= 1
    assert stats["routes_blocked"] >= 1


# ══════════════════════════════════════════════════════════════════════
# V3-14: Gate computed from report
# ══════════════════════════════════════════════════════════════════════


def test_v3_14_gate_from_report():
    report = {
        "total_requests": 100,
        "decomposition_evidence_rate": 0.55,
        "decomposition_evidence_count": 55,
        "success_rate": 0.85,
        "cost_by_model": {
            "gpt-4o": {"requests": 50},
            "gpt-4o-mini": {"requests": 50},
        },
    }
    gate = compute_gate_status(report)
    assert gate.passed
    assert gate.sample_size == 100
    assert gate.model_readiness["gpt-4o"]
    assert gate.model_readiness["gpt-4o-mini"]


# ══════════════════════════════════════════════════════════════════════
# V3-15: Gate fails without enough samples
# ══════════════════════════════════════════════════════════════════════


def test_v3_15_gate_sample_size():
    report = {
        "total_requests": 10,  # too few
        "decomposition_evidence_rate": 0.55,
        "success_rate": 0.85,
        "cost_by_model": {},
    }
    gate = compute_gate_status(report)
    assert not gate.passed
    assert "sample_size" in gate.reason


# ══════════════════════════════════════════════════════════════════════
# V3-16: Thread safety
# ══════════════════════════════════════════════════════════════════════


def test_v3_16_thread_safe():
    router = _make_ready_router()
    errors: list[str] = []

    def hammer():
        try:
            for _ in range(50):
                d = router.route("add tests", "gpt-4o")
                assert isinstance(d, RoutingDecision)
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    assert router.stats()["decisions_made"] == 200  # 4 threads × 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
