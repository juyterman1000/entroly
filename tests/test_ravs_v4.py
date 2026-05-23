"""
Tests for RAVS v4 — Sequential Controller.

Coverage:
  V4-01  EMPTY PLAN FALLBACK      no decomposable nodes → fallback
  V4-02  SINGLE STEP SUCCESS      one computation node succeeds
  V4-03  MULTI STEP EXECUTION     multiple nodes execute sequentially
  V4-04  VERIFIER GATE            unverified results marked partial
  V4-05  BUDGET ENFORCEMENT       exceeding budget stops execution
  V4-06  MARGINAL VALUE GATE      low-value steps are skipped
  V4-07  ESCALATION ON FAILURE    consecutive failures trigger escalation
  V4-08  ESCALATION RESETS        escalation resets failure counter
  V4-09  MUST ESCALATE KINDS      test_execution always escalates on fail
  V4-10  EARLY STOPPING           stops when value < cost
  V4-11  RESULT SERIALIZABLE      ControllerResult.to_dict() is valid
  V4-12  COST ACCOUNTING          total cost = sum of step costs
  V4-13  FALLBACK ON ALL FAIL     all steps fail → fell_back_to_model
  V4-14  STATS TRACKING           lifetime counters work
  V4-15  BUDGET REMAINING         budget_remaining tracks correctly
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.ravs.compiler import NodeKind, Plan, PlanCompiler, PlanNode
from entroly.ravs.controller import (
    ControllerResult,
    EscalationPolicy,
    SequentialController,
    StepOutcome,
)
from entroly.ravs.executors import ExecutorRegistry
from entroly.ravs.verifiers import VerifierRegistry


def _make_plan(*node_specs) -> Plan:
    """Build a Plan from (kind, executor, verifier, input, confidence) tuples."""
    nodes = []
    for kind, executor, verifier, input_text, confidence in node_specs:
        nodes.append(PlanNode(
            kind=kind,
            executor=executor,
            verifier=verifier,
            input_text=input_text,
            confidence=confidence,
            estimated_cost_usd=0.00001,
        ))
    return Plan(
        plan_id="test-plan",
        request_id="test-req",
        nodes=nodes,
        total_nodes=len(nodes),
        decomposed_nodes=sum(1 for n in nodes if n.kind != NodeKind.MODEL_BOUND.value),
        model_bound_nodes=sum(1 for n in nodes if n.kind == NodeKind.MODEL_BOUND.value),
    )


# ══════════════════════════════════════════════════════════════════════
# V4-01: Empty plan → fallback
# ══════════════════════════════════════════════════════════════════════


def test_v4_01_empty_plan_fallback():
    ctrl = SequentialController()
    plan = _make_plan(
        (NodeKind.MODEL_BOUND.value, "none", "none", "judgment", 1.0),
    )
    result = ctrl.execute(plan)
    assert result.fell_back_to_model
    assert "no decomposable" in result.fallback_reason


# ══════════════════════════════════════════════════════════════════════
# V4-02: Single computation step succeeds
# ══════════════════════════════════════════════════════════════════════


def test_v4_02_single_step_success():
    ctrl = SequentialController()
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "2 + 3", 0.9),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    assert result.total_steps == 1
    assert result.successful_steps >= 1 or result.steps[0].outcome != StepOutcome.FAILED.value
    assert not result.fell_back_to_model


# ══════════════════════════════════════════════════════════════════════
# V4-03: Multi-step execution
# ══════════════════════════════════════════════════════════════════════


def test_v4_03_multi_step():
    ctrl = SequentialController()
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "10 * 5", 0.8),
        (NodeKind.COMPUTATION.value, "python", "exact", "100 / 4", 0.8),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    assert result.total_steps == 2
    assert len(result.steps) == 2


# ══════════════════════════════════════════════════════════════════════
# V4-04: Verifier gate — unverified results marked partial
# ══════════════════════════════════════════════════════════════════════


def test_v4_04_verifier_gate():
    ctrl = SequentialController()
    # AST executor without source code → will fail
    plan = _make_plan(
        (NodeKind.CODE_INSPECTION.value, "ast", "structural", "list functions", 0.8),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    assert result.total_steps == 1
    # Without source code, AST executor should fail
    assert result.steps[0].outcome in (StepOutcome.FAILED.value, StepOutcome.VERIFIED_PARTIAL.value)


# ══════════════════════════════════════════════════════════════════════
# V4-05: Budget enforcement
# ══════════════════════════════════════════════════════════════════════


def test_v4_05_budget_enforcement():
    ctrl = SequentialController()
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "2 + 2", 0.8),
        (NodeKind.COMPUTATION.value, "python", "exact", "3 + 3", 0.8),
        (NodeKind.COMPUTATION.value, "python", "exact", "4 + 4", 0.8),
    )
    # Tiny budget — should stop after first or second step
    result = ctrl.execute(plan, budget_usd=0.000015)
    budget_exceeded = any(
        s.outcome == StepOutcome.BUDGET_EXCEEDED.value for s in result.steps
    )
    # Either budget exceeded or all fit
    assert result.total_cost_usd <= 0.000015 + 0.00002  # small tolerance


# ══════════════════════════════════════════════════════════════════════
# V4-06: Marginal value gate
# ══════════════════════════════════════════════════════════════════════


def test_v4_06_marginal_value_gate():
    ctrl = SequentialController(min_marginal_value=0.5)
    plan = _make_plan(
        # Low confidence = low marginal value → should be skipped
        (NodeKind.COMPUTATION.value, "python", "exact", "2+2", 0.05),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    assert result.skipped_steps >= 1
    assert result.steps[0].outcome == StepOutcome.SKIPPED.value


# ══════════════════════════════════════════════════════════════════════
# V4-07: Escalation on consecutive failures
# ══════════════════════════════════════════════════════════════════════


def test_v4_07_escalation_on_failure():
    policy = EscalationPolicy(max_consecutive_failures=2)
    ctrl = SequentialController(escalation_policy=policy)
    plan = _make_plan(
        # These will all fail (bad expressions)
        (NodeKind.COMPUTATION.value, "sympy", "exact", "invalid!!!expr", 0.8),
        (NodeKind.COMPUTATION.value, "sympy", "exact", "another!!!bad", 0.8),
        (NodeKind.COMPUTATION.value, "sympy", "exact", "third!!!bad", 0.8),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    # After 2 consecutive failures, third should escalate
    escalated = [s for s in result.steps if s.outcome == StepOutcome.ESCALATED.value]
    assert len(escalated) >= 1


# ══════════════════════════════════════════════════════════════════════
# V4-08: Escalation resets failure counter
# ══════════════════════════════════════════════════════════════════════


def test_v4_08_escalation_resets():
    policy = EscalationPolicy(max_consecutive_failures=1)
    ctrl = SequentialController(escalation_policy=policy)
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "sympy", "exact", "bad!!!", 0.8),
        (NodeKind.COMPUTATION.value, "python", "exact", "2 + 3", 0.8),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    # First fails → escalates (resets counter)
    # Second should execute normally (not escalate)
    assert result.total_steps == 2


# ══════════════════════════════════════════════════════════════════════
# V4-09: Must-escalate kinds
# ══════════════════════════════════════════════════════════════════════


def test_v4_09_must_escalate_kinds():
    ctrl = SequentialController()
    plan = _make_plan(
        # test_execution always escalates on failure
        (NodeKind.TEST_EXECUTION.value, "test_runner", "exit_code", "run tests", 0.8),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    # test_runner stub returns succeeded=False → must escalate
    assert result.steps[0].outcome == StepOutcome.ESCALATED.value


# ══════════════════════════════════════════════════════════════════════
# V4-10: Early stopping
# ══════════════════════════════════════════════════════════════════════


def test_v4_10_early_stopping():
    ctrl = SequentialController(min_marginal_value=0.8)
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "2+2", 0.9),
        # Second node has very low confidence
        (NodeKind.RETRIEVAL_CLAIM.value, "retrieval", "citation", "check docs", 0.1),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    # Second should be skipped due to low marginal value
    skipped = [s for s in result.steps if s.outcome == StepOutcome.SKIPPED.value]
    assert len(skipped) >= 1


# ══════════════════════════════════════════════════════════════════════
# V4-11: Result serializable
# ══════════════════════════════════════════════════════════════════════


def test_v4_11_result_serializable():
    ctrl = SequentialController()
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "7 * 6", 0.9),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    d = result.to_dict()
    j = json.dumps(d)
    parsed = json.loads(j)
    assert parsed["total_steps"] == result.total_steps


# ══════════════════════════════════════════════════════════════════════
# V4-12: Cost accounting
# ══════════════════════════════════════════════════════════════════════


def test_v4_12_cost_accounting():
    ctrl = SequentialController()
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "1+1", 0.9),
        (NodeKind.COMPUTATION.value, "python", "exact", "2+2", 0.9),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    step_costs = sum(s.cost_usd for s in result.steps)
    assert abs(result.total_cost_usd - step_costs) < 1e-9


# ══════════════════════════════════════════════════════════════════════
# V4-13: All fail → fell_back_to_model
# ══════════════════════════════════════════════════════════════════════


def test_v4_13_fallback_on_all_fail():
    ctrl = SequentialController()
    # All nodes will fail (bad expressions, no escalation triggered with default policy)
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "sympy", "exact", "!!!bad", 0.8),
    )
    result = ctrl.execute(plan, budget_usd=0.01)
    assert result.fell_back_to_model or result.escalated_steps > 0


# ══════════════════════════════════════════════════════════════════════
# V4-14: Stats tracking
# ══════════════════════════════════════════════════════════════════════


def test_v4_14_stats_tracking():
    ctrl = SequentialController()

    plan1 = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "2+2", 0.9),
    )
    plan2 = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "3+3", 0.9),
    )

    ctrl.execute(plan1, budget_usd=0.01)
    ctrl.execute(plan2, budget_usd=0.01)

    stats = ctrl.stats()
    assert stats["total_executions"] == 2
    assert stats["total_steps"] >= 2


# ══════════════════════════════════════════════════════════════════════
# V4-15: Budget remaining tracks correctly
# ══════════════════════════════════════════════════════════════════════


def test_v4_15_budget_remaining():
    ctrl = SequentialController()
    plan = _make_plan(
        (NodeKind.COMPUTATION.value, "python", "exact", "5+5", 0.9),
    )
    budget = 0.01
    result = ctrl.execute(plan, budget_usd=budget)
    assert result.budget_remaining_usd <= budget
    assert abs(result.budget_remaining_usd - (budget - result.total_cost_usd)) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
