"""
RAVS E2E Integration Test — Full Pipeline Smoke Test

Simulates the complete lifecycle:
  1. Shadow compiler decomposes a query
  2. Executors run on decomposed nodes
  3. Verifiers check results
  4. Decomposition evidence written to event log
  5. Honest outcome arrives → PRISM posterior corrected
  6. Offline eval report generated from the log
  7. Metrics match expectations
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from entroly.online_learner import OnlinePrism
from entroly.ravs.compiler import PlanCompiler, NodeKind
from entroly.ravs.events import (
    AppendOnlyEventLog,
    DecompositionEvidence,
    OutcomeEvent,
    TraceEvent,
)
from entroly.ravs.executors import ExecutorRegistry
from entroly.ravs.outcome_bridge import OutcomeBridge
from entroly.ravs.report import generate_report, format_report_text
from entroly.ravs.shadow_runner import ShadowRunner
from entroly.ravs.verifiers import VerifierRegistry


def main():
    print("\n  ══════════════════════════════════════════════════════════")
    print("  RAVS E2E Integration Test")
    print("  ══════════════════════════════════════════════════════════\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "events.jsonl"
        log = AppendOnlyEventLog(log_path)

        # ── Setup ─────────────────────────────────────────────────
        prism = OnlinePrism(
            prior_weights={
                "w_recency": 0.30, "w_frequency": 0.20,
                "w_semantic": 0.35, "w_entropy": 0.15,
            },
            prior_strength=10.0,
        )
        bridge = OutcomeBridge(prism)
        runner = ShadowRunner()
        compiler = PlanCompiler()

        print("  ✓ Components initialized (PRISM, Bridge, Shadow Runner)\n")

        # ── Simulate 5 requests ───────────────────────────────────
        requests = [
            {
                "id": "r1", "query": "calculate 2 + 3 * 4",
                "model": "gpt-4o-mini", "cost": 0.0003,
                "outcome": ("test_result", "passed", "strong"),
            },
            {
                "id": "r2", "query": "list the functions in server.py",
                "model": "gpt-4o-mini", "cost": 0.0004,
                "outcome": ("user_acceptance", "accepted", "strong"),
            },
            {
                "id": "r3", "query": "what is the average of 10, 20, 30, 40, 50",
                "model": "gpt-4o", "cost": 0.005,
                "outcome": ("ci_result", "passed", "strong"),
            },
            {
                "id": "r4", "query": "explain the strategy pattern",
                "model": "gpt-4o", "cost": 0.006,
                "outcome": ("agent_self_report", "success", "weak"),
            },
            {
                "id": "r5", "query": "run the tests and check if they pass",
                "model": "gpt-4o-mini", "cost": 0.0003,
                "outcome": ("test_result", "failed", "strong"),
            },
        ]

        initial_weights = prism.weights()
        print(f"  Initial PRISM weights: {_fmt(initial_weights)}\n")

        for req in requests:
            rid = req["id"]
            query = req["query"]
            model = req["model"]
            cost = req["cost"]

            # 1. Shadow compile + run
            plan = runner.compile_and_run(
                query, request_id=rid, model=model,
                model_cost_usd=cost,
            )

            # 2. Write trace with decomposition evidence
            decomp_evidence = [
                {"kind": n.kind, "source": "shadow_compiler",
                 "executor": n.executor, "confidence": round(n.confidence, 2)}
                for n in plan.nodes if n.kind != NodeKind.MODEL_BOUND.value
            ]

            log.write_trace(TraceEvent(
                request_id=rid,
                model=model,
                cost_usd=cost,
                latency_ms=150.0,
                context_size_tokens=5000,
                timestamp=time.time(),
                retrieved_fragments=[],
                decomposition_evidence=decomp_evidence,
                shadow_recommendations={},
            ))

            # 3. Simulate PRISM observation + bridge caching
            implicit_reward = 0.5  # proxy
            contributions = {"w_recency": 0.25, "w_frequency": 0.25,
                           "w_semantic": 0.25, "w_entropy": 0.25}
            prism.observe(implicit_reward, contributions)

            bridge.cache_observation(
                request_id=rid,
                implicit_reward=implicit_reward,
                implicit_advantage=implicit_reward - prism._reward_ema,
                contributions=contributions,
                weights=prism.weights(),
            )

            # 4. Honest outcome arrives → bridge corrects PRISM
            etype, value, strength = req["outcome"]
            log.write_outcome(OutcomeEvent(
                request_id=rid,
                event_type=etype,
                value=value,
                strength=strength,
                source="e2e_test",
                include_in_default_training=(strength != "weak"),
                timestamp=time.time(),
            ))

            correction = bridge.on_honest_outcome(rid, etype, value, strength)

            # Print per-request summary
            decomp_str = f"{plan.decomposed_nodes} decomposed" if plan.decomposed_nodes > 0 else "model-bound only"
            exec_str = f"{plan.executor_success_count}/{plan.decomposed_nodes} executors succeeded" if plan.decomposed_nodes > 0 else ""
            corr_str = ""
            if correction:
                corr_str = f"  Δadv={correction['delta_advantage']:+.3f}"
            elif strength == "weak":
                corr_str = "  (weak, no correction)"

            print(f"  [{rid}] {query[:45]:45s}")
            print(f"       plan: {decomp_str}  {exec_str}")
            print(f"       outcome: {etype}={value} ({strength}){corr_str}")

        # ── Results ───────────────────────────────────────────────
        final_weights = prism.weights()
        print(f"\n  Final PRISM weights:   {_fmt(final_weights)}")
        print(f"  Weight delta:         {_fmt_delta(initial_weights, final_weights)}")

        # Bridge stats
        bstats = bridge.stats()
        print(f"\n  Bridge: {bstats['corrections_applied']} corrections applied, "
              f"{bstats['corrections_skipped']} skipped")

        # Shadow runner stats
        sstats = runner.stats()
        print(f"  Shadow: {sstats['total_runs']} runs, "
              f"{sstats['total_nodes_executed']} nodes executed, "
              f"executor_success={sstats['executor_success_rate']:.0%}, "
              f"fallback={sstats['fallback_rate']:.0%}")

        # ── Generate offline report ───────────────────────────────
        report = generate_report(str(log_path))
        print(f"\n  ── Offline Report ────────────────────────────────────")
        print(f"  Total requests:          {report['total_requests']}")
        print(f"  Honest label coverage:   {report['honest_label_coverage']*100:.0f}%")
        print(f"  Weak label coverage:     {report['weak_label_coverage']*100:.0f}%")
        print(f"  Unknown label rate:      {report['unknown_label_rate']*100:.0f}%")
        print(f"  Success rate:            {report['success_rate']*100:.0f}%")
        bd = report['label_breakdown']
        print(f"    successes={bd['successes']}  failures={bd['failures']}  unknown={bd['unknown']}")
        print(f"  Decomp evidence rate:    {report['decomposition_evidence_rate']*100:.0f}%")
        print(f"  Decomp evidence count:   {report['decomposition_evidence_count']}")

        # Cost by model
        for model, entry in report.get("cost_by_model", {}).items():
            print(f"  {model:25s}  reqs={entry['requests']}  "
                  f"cost=${entry['total_cost_usd']:.4f}  "
                  f"avg=${entry['avg_cost_usd']:.6f}")

        # ── Assertions ────────────────────────────────────────────
        print(f"\n  ── Assertions ────────────────────────────────────────")
        ok = True

        def check(name, cond):
            nonlocal ok
            status = "✓" if cond else "✗"
            if not cond:
                ok = False
            print(f"  {status} {name}")

        check("total_requests == 5", report["total_requests"] == 5)
        check("honest_label_coverage == 80% (4/5, weak excluded)",
              report["honest_label_coverage"] == 0.8)
        check("weak_label_coverage == 100% (5/5 with weak)",
              report["weak_label_coverage"] == 1.0)
        check("success_rate >= 50%", report["success_rate"] >= 0.5)
        check("decomp_evidence_rate > 0", report["decomposition_evidence_rate"] > 0)
        check("bridge corrections >= 3", bstats["corrections_applied"] >= 3)
        check("shadow executor_success_rate > 0", sstats["executor_success_rate"] > 0)
        check("PRISM weights changed from initial",
              final_weights != initial_weights)

        # JSON byte-stability
        r1 = generate_report(str(log_path))
        r2 = generate_report(str(log_path))
        r1.pop("generated_at_utc", None)
        r2.pop("generated_at_utc", None)
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        check("byte-stable JSON output", j1 == j2)

        print(f"\n  ══════════════════════════════════════════════════════════")
        if ok:
            print(f"  ALL ASSERTIONS PASSED ✓")
        else:
            print(f"  SOME ASSERTIONS FAILED ✗")
        print(f"  ══════════════════════════════════════════════════════════\n")

        return 0 if ok else 1


def _fmt(w: dict) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in w.items())


def _fmt_delta(a: dict, b: dict) -> str:
    return "  ".join(f"{k}={b[k]-a[k]:+.4f}" for k in a)


if __name__ == "__main__":
    sys.exit(main())
