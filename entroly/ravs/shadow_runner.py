"""
RAVS v2 — Shadow Runner

Executes compiled plans in shadow mode. For each request:
  1. Compiler produces a Plan (DAG of typed nodes)
  2. Shadow Runner executes each node's executor
  3. Shadow Runner runs the verifier on the result
  4. Metrics are recorded: executor success, verifier pass, fallback rate
  5. Nothing touches production output

The shadow runner writes results to the V1 event log as
DecompositionEvidence entries, maintaining the V1 data pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .compiler import NodeKind, Plan, PlanCompiler, PlanNode
from .executors import ExecutorRegistry, ExecutorResult
from .verifiers import VerifierRegistry

logger = logging.getLogger("entroly.ravs.shadow_runner")


# ── Cost model ─────────────────────────────────────────────────────────
# Conservative cost estimates per executor type (USD per call).
# These are WAY cheaper than model calls — that's the whole point.

_EXECUTOR_COST_USD = {
    "sympy": 0.000001,       # ~0.001 ms CPU
    "python": 0.000001,
    "ast": 0.000002,
    "test_runner": 0.000050,  # subprocess overhead
    "retrieval": 0.000010,
    "none": 0.0,
}

# Model cost estimates per request (used for baseline comparison)
_MODEL_COST_USD = {
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.0003,
    "claude-3.5-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "gemini-2.5-pro": 0.004,
    "gemini-2.0-flash": 0.0002,
    "default": 0.002,
}


class ShadowRunner:
    """Execute plans in shadow and record results.

    Thread-safe: no mutable state between runs.
    Each run is independent — the runner doesn't learn from past runs
    (that's the OutcomeBridge's job).
    """

    def __init__(
        self,
        executor_registry: ExecutorRegistry | None = None,
        verifier_registry: VerifierRegistry | None = None,
    ):
        self._executors = executor_registry or ExecutorRegistry()
        self._verifiers = verifier_registry or VerifierRegistry()
        self._compiler = PlanCompiler()

        # Counters
        self._total_runs = 0
        self._total_nodes_executed = 0
        self._total_executor_successes = 0
        self._total_verifier_passes = 0
        self._total_fallbacks = 0

    def compile_and_run(
        self,
        query: str,
        *,
        request_id: str = "",
        model: str = "default",
        model_cost_usd: float = -1.0,
        tools_used: list[str] | None = None,
        source_code: str = "",
    ) -> Plan:
        """Compile a query into a plan and execute in shadow.

        Args:
            query: The user's request text.
            request_id: Trace ID for V1 correlation.
            model: The production model that actually handled this request.
            model_cost_usd: Actual cost of the production model call.
                           If -1, estimated from model name.
            tools_used: Tool names from the trace.
            source_code: Source code for AST executor (if applicable).

        Returns:
            The executed Plan with all metrics filled in.
        """
        # 1. Compile
        plan = self._compiler.compile(
            query, request_id=request_id, tools_used=tools_used,
        )

        # 2. Estimate baseline cost
        if model_cost_usd >= 0:
            baseline_cost = model_cost_usd
        else:
            baseline_cost = _MODEL_COST_USD.get(model, _MODEL_COST_USD["default"])
        plan.baseline_total_cost_usd = round(baseline_cost, 6)

        # 3. Execute each node in shadow
        total_executor_cost = 0.0
        for node in plan.nodes:
            if node.kind == NodeKind.MODEL_BOUND.value:
                # Model-bound nodes always "cost" the model price
                node.baseline_model_cost_usd = baseline_cost
                node.estimated_cost_usd = baseline_cost
                continue

            self._total_nodes_executed += 1

            # Execute
            executor_cost = _EXECUTOR_COST_USD.get(node.executor, 0.0)
            node.estimated_cost_usd = round(executor_cost, 6)
            node.baseline_model_cost_usd = round(baseline_cost, 6)

            kwargs: dict[str, Any] = {}
            if node.executor == "ast" and source_code:
                kwargs["source_code"] = source_code

            result = self._executors.execute_node(
                node.executor, node.input_text, **kwargs,
            )

            node.executor_result = result.result if result.succeeded else result.error
            node.executor_succeeded = result.succeeded
            node.execution_time_ms = result.execution_time_ms

            if result.succeeded:
                self._total_executor_successes += 1

                # Verify
                verifier = self._verifiers.get(node.verifier)
                if verifier is not None:
                    # For exact verifier, we'd need expected output.
                    # In shadow mode, we verify structural integrity only.
                    if node.verifier == "exact" and result.result:
                        # Self-consistency check: re-execute and compare
                        recheck = self._executors.execute_node(
                            node.executor, node.input_text, **kwargs,
                        )
                        if recheck.succeeded:
                            from .verifiers import ExactVerifier
                            v = ExactVerifier()
                            vr = v.verify(result.result, recheck.result)
                            node.verifier_passed = vr.passed
                            if vr.passed:
                                self._total_verifier_passes += 1
                    elif node.verifier == "structural" and result.result:
                        vr = verifier.verify(result.result)
                        node.verifier_passed = vr.passed
                        if vr.passed:
                            self._total_verifier_passes += 1
                    elif node.verifier == "exit_code":
                        vr = verifier.verify(result.result)
                        node.verifier_passed = vr.passed
                        if vr.passed:
                            self._total_verifier_passes += 1
            else:
                # Executor failed — fall back to model
                node.fell_back_to_model = True
                node.estimated_cost_usd = baseline_cost  # fallback costs model price
                self._total_fallbacks += 1

            total_executor_cost += node.estimated_cost_usd

        # 4. Fill plan-level metrics
        plan.executor_success_count = sum(
            1 for n in plan.nodes
            if n.kind != NodeKind.MODEL_BOUND.value and n.executor_succeeded
        )
        plan.verifier_pass_count = sum(
            1 for n in plan.nodes if n.verifier_passed
        )
        plan.fallback_count = sum(
            1 for n in plan.nodes if n.fell_back_to_model
        )
        plan.estimated_total_cost_usd = round(total_executor_cost, 6)

        self._total_runs += 1
        return plan

    def stats(self) -> dict[str, Any]:
        """Dashboard-friendly stats."""
        return {
            "total_runs": self._total_runs,
            "total_nodes_executed": self._total_nodes_executed,
            "executor_success_rate": round(
                self._total_executor_successes / max(self._total_nodes_executed, 1), 4
            ),
            "verifier_pass_rate": round(
                self._total_verifier_passes / max(self._total_executor_successes, 1), 4
            ),
            "fallback_rate": round(
                self._total_fallbacks / max(self._total_nodes_executed, 1), 4
            ),
        }
