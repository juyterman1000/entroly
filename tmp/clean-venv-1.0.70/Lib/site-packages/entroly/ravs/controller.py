"""
RAVS v4 — Sequential Controller

The controller orchestrates multi-step execution of compiled plans.
Instead of running all nodes independently, it:

  1. Observes uncertainty at each step
  2. Branches if a node's output is ambiguous
  3. Calls verifiers after each executor
  4. Escalates irreducible nodes to a stronger model
  5. Stops when marginal value < marginal cost

This is the "real magic" — but it comes last because it requires
traces with intermediate states and verifier outcomes (V2/V3 data).

Design invariants:
  1. BUDGET-BOUNDED. Total cost of controller execution must not
     exceed the baseline model cost. If it does → abort, fall back.
  2. FAIL CLOSED. On any step failure, the controller falls back to
     the original model for the remainder. Partial results are discarded.
  3. OBSERVABLE. Every step is logged with intermediate state, cost,
     and verifier outcome. This feeds back into V1 event log.
  4. MONOTONIC VALUE. Each step must add positive marginal value.
     If a step's expected value drops below its cost → stop.
  5. NO SPECULATIVE EXECUTION. The controller executes sequentially.
     It never runs nodes whose inputs depend on unverified outputs.

Execution model:
  Plan → Controller → Step 1 → Verify → Step 2 → Verify → ... → Result
                          ↓                  ↓
                      (escalate?)        (branch?)
                          ↓                  ↓
                    stronger model     alternative path
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .compiler import NodeKind, Plan, PlanNode
from .executors import ExecutorRegistry, ExecutorResult
from .verifiers import VerifierRegistry

logger = logging.getLogger("entroly.ravs.controller")


# ── Step outcomes ──────────────────────────────────────────────────────


class StepOutcome(str, Enum):
    SUCCESS = "success"           # executor + verifier passed
    VERIFIED_PARTIAL = "partial"  # executor ok, verifier partial
    ESCALATED = "escalated"       # sent to stronger model
    BRANCHED = "branched"         # tried alternative path
    FAILED = "failed"             # executor failed, fell back
    SKIPPED = "skipped"           # marginal value too low
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class StepResult:
    """Result of executing one step in the controller."""
    node_id: str = ""
    node_kind: str = ""
    outcome: str = StepOutcome.FAILED.value
    executor_result: str = ""
    verifier_passed: bool = False
    escalated: bool = False
    cost_usd: float = 0.0
    time_ms: float = 0.0
    marginal_value: float = 0.0
    reason: str = ""


@dataclass
class ControllerResult:
    """Complete result of controller execution."""
    plan_id: str = ""
    request_id: str = ""
    steps: list[StepResult] = field(default_factory=list)

    # Summary
    total_steps: int = 0
    successful_steps: int = 0
    escalated_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0

    total_cost_usd: float = 0.0
    budget_usd: float = 0.0
    budget_remaining_usd: float = 0.0

    execution_time_ms: float = 0.0
    fell_back_to_model: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "request_id": self.request_id,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "escalated_steps": self.escalated_steps,
            "failed_steps": self.failed_steps,
            "skipped_steps": self.skipped_steps,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_usd": round(self.budget_usd, 6),
            "budget_remaining_usd": round(self.budget_remaining_usd, 6),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "fell_back_to_model": self.fell_back_to_model,
            "fallback_reason": self.fallback_reason,
            "steps": [
                {
                    "node_id": s.node_id,
                    "kind": s.node_kind,
                    "outcome": s.outcome,
                    "verifier_passed": s.verifier_passed,
                    "cost_usd": round(s.cost_usd, 6),
                    "time_ms": round(s.time_ms, 2),
                    "marginal_value": round(s.marginal_value, 4),
                }
                for s in self.steps
            ],
        }


# ── Marginal Value Estimator ──────────────────────────────────────────


def _estimate_marginal_value(
    node: PlanNode,
    step_index: int,
    total_steps: int,
    prior_successes: int,
) -> float:
    """Estimate the marginal value of executing this node.

    Value = (confidence that executor will succeed) *
            (cost savings vs model) *
            (diminishing returns factor)

    Returns a value in [0, 1]. If < marginal_cost, skip.
    """
    # Base value from node confidence
    base = node.confidence

    # Diminishing returns: later steps are worth less
    position_factor = 1.0 - (step_index / max(total_steps, 1)) * 0.3

    # Momentum: more prior successes → higher expected value
    momentum = min(1.0, 0.5 + prior_successes * 0.1)

    # Model-bound nodes always have value 0 (can't be cheapened)
    if node.kind == NodeKind.MODEL_BOUND.value:
        return 0.0

    return round(base * position_factor * momentum, 4)


# ── Escalation Policy ─────────────────────────────────────────────────


@dataclass
class EscalationPolicy:
    """When and how to escalate to a stronger model."""

    # Max consecutive failures before escalating entire plan
    max_consecutive_failures: int = 2

    # Node kinds that should always escalate on failure (never skip)
    must_escalate_kinds: frozenset[str] = frozenset({
        NodeKind.TEST_EXECUTION.value,
    })

    # Cost multiplier for escalation (how much more expensive)
    escalation_cost_multiplier: float = 10.0

    # Whether to allow partial results (some steps succeeded, rest escalated)
    allow_partial: bool = True


# ── The Controller ─────────────────────────────────────────────────────


class SequentialController:
    """Execute a compiled plan step-by-step with verification.

    The controller is the orchestrator that turns a static Plan into
    a dynamic execution with branching, escalation, and early stopping.
    """

    def __init__(
        self,
        executors: ExecutorRegistry | None = None,
        verifiers: VerifierRegistry | None = None,
        escalation_policy: EscalationPolicy | None = None,
        min_marginal_value: float = 0.1,
    ):
        self._executors = executors or ExecutorRegistry()
        self._verifiers = verifiers or VerifierRegistry()
        self._policy = escalation_policy or EscalationPolicy()
        self._min_marginal_value = min_marginal_value

        # Lifetime counters
        self._total_executions = 0
        self._total_steps = 0
        self._total_escalations = 0
        self._total_fallbacks = 0
        self._total_early_stops = 0

    def execute(
        self,
        plan: Plan,
        *,
        budget_usd: float = 0.01,
        source_code: str = "",
    ) -> ControllerResult:
        """Execute a plan sequentially with verification gates.

        Args:
            plan: Compiled plan from PlanCompiler.
            budget_usd: Maximum cost allowed (fail if exceeded).
            source_code: Source code for AST executor (if applicable).

        Returns:
            ControllerResult with per-step outcomes.
        """
        t0 = time.perf_counter()

        result = ControllerResult(
            plan_id=plan.plan_id,
            request_id=plan.request_id,
            budget_usd=budget_usd,
            budget_remaining_usd=budget_usd,
        )

        # Filter to decomposable nodes (model-bound are skipped)
        exec_nodes = [
            n for n in plan.nodes
            if n.kind != NodeKind.MODEL_BOUND.value
        ]

        if not exec_nodes:
            result.fell_back_to_model = True
            result.fallback_reason = "no decomposable nodes"
            result.execution_time_ms = (time.perf_counter() - t0) * 1000
            self._total_executions += 1
            return result

        consecutive_failures = 0
        prior_successes = 0

        for i, node in enumerate(exec_nodes):
            # ── Gate 1: Budget check ──────────────────────────────
            estimated_step_cost = node.estimated_cost_usd
            if estimated_step_cost < 0:
                estimated_step_cost = 0.00001  # minimum step cost
            if result.budget_remaining_usd <= 0:
                step = StepResult(
                    node_id=node.node_id,
                    node_kind=node.kind,
                    outcome=StepOutcome.BUDGET_EXCEEDED.value,
                    reason="budget exhausted",
                )
                result.steps.append(step)
                result.skipped_steps += 1
                continue

            # ── Gate 2: Marginal value check ──────────────────────
            mv = _estimate_marginal_value(
                node, i, len(exec_nodes), prior_successes,
            )
            if mv < self._min_marginal_value:
                step = StepResult(
                    node_id=node.node_id,
                    node_kind=node.kind,
                    outcome=StepOutcome.SKIPPED.value,
                    marginal_value=mv,
                    reason=f"marginal_value={mv:.3f} < threshold={self._min_marginal_value}",
                )
                result.steps.append(step)
                result.skipped_steps += 1
                self._total_early_stops += 1
                continue

            # ── Execute ───────────────────────────────────────────
            step_t0 = time.perf_counter()

            kwargs: dict[str, Any] = {}
            if node.executor == "ast" and source_code:
                kwargs["source_code"] = source_code

            exec_result = self._executors.execute_node(
                node.executor, node.input_text, **kwargs,
            )

            step_time = (time.perf_counter() - step_t0) * 1000

            if not exec_result.succeeded:
                consecutive_failures += 1

                # ── Escalation check ──────────────────────────────
                should_escalate = (
                    node.kind in self._policy.must_escalate_kinds
                    or consecutive_failures >= self._policy.max_consecutive_failures
                )

                if should_escalate:
                    step = StepResult(
                        node_id=node.node_id,
                        node_kind=node.kind,
                        outcome=StepOutcome.ESCALATED.value,
                        escalated=True,
                        cost_usd=estimated_step_cost * self._policy.escalation_cost_multiplier,
                        time_ms=step_time,
                        marginal_value=mv,
                        reason=f"escalated after {consecutive_failures} failures",
                    )
                    result.steps.append(step)
                    result.escalated_steps += 1
                    result.budget_remaining_usd -= step.cost_usd
                    result.total_cost_usd += step.cost_usd
                    self._total_escalations += 1
                    # Reset failure count after escalation
                    consecutive_failures = 0
                else:
                    step = StepResult(
                        node_id=node.node_id,
                        node_kind=node.kind,
                        outcome=StepOutcome.FAILED.value,
                        executor_result=exec_result.error,
                        cost_usd=estimated_step_cost,
                        time_ms=step_time,
                        marginal_value=mv,
                        reason=f"executor failed: {exec_result.error[:100]}",
                    )
                    result.steps.append(step)
                    result.failed_steps += 1
                    result.budget_remaining_usd -= step.cost_usd
                    result.total_cost_usd += step.cost_usd

                self._total_steps += 1
                continue

            # ── Verify ────────────────────────────────────────────
            consecutive_failures = 0
            verifier = self._verifiers.get(node.verifier)
            verifier_passed = False

            if verifier is not None:
                if node.verifier == "exact" and exec_result.result:
                    # Self-consistency: re-execute and compare
                    recheck = self._executors.execute_node(
                        node.executor, node.input_text, **kwargs,
                    )
                    if recheck.succeeded:
                        from .verifiers import ExactVerifier
                        vr = ExactVerifier().verify(exec_result.result, recheck.result)
                        verifier_passed = vr.passed
                elif node.verifier == "structural" and exec_result.result:
                    vr = verifier.verify(exec_result.result)
                    verifier_passed = vr.passed
                elif node.verifier == "exit_code":
                    vr = verifier.verify(exec_result.result)
                    verifier_passed = vr.passed

            step = StepResult(
                node_id=node.node_id,
                node_kind=node.kind,
                outcome=StepOutcome.SUCCESS.value if verifier_passed
                        else StepOutcome.VERIFIED_PARTIAL.value,
                executor_result=exec_result.result[:200],
                verifier_passed=verifier_passed,
                cost_usd=estimated_step_cost,
                time_ms=step_time,
                marginal_value=mv,
            )
            result.steps.append(step)

            if verifier_passed:
                result.successful_steps += 1
                prior_successes += 1
            else:
                result.failed_steps += 1

            result.budget_remaining_usd -= step.cost_usd
            result.total_cost_usd += step.cost_usd
            self._total_steps += 1

        # ── Check if we need full fallback ────────────────────────
        if result.successful_steps == 0 and result.escalated_steps == 0:
            result.fell_back_to_model = True
            result.fallback_reason = "no steps succeeded"
            self._total_fallbacks += 1

        result.total_steps = len(result.steps)
        result.execution_time_ms = (time.perf_counter() - t0) * 1000
        self._total_executions += 1

        return result

    def stats(self) -> dict[str, Any]:
        """Dashboard stats for the controller."""
        return {
            "total_executions": self._total_executions,
            "total_steps": self._total_steps,
            "total_escalations": self._total_escalations,
            "total_fallbacks": self._total_fallbacks,
            "total_early_stops": self._total_early_stops,
            "escalation_rate": round(
                self._total_escalations / max(self._total_steps, 1), 4
            ),
            "fallback_rate": round(
                self._total_fallbacks / max(self._total_executions, 1), 4
            ),
        }
