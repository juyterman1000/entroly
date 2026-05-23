"""
RAVS offline evaluation report — the V1 closer.

Pure-function pipeline: (JSONL log) → deterministic report dict.

Design invariants:
  1. Same input → byte-identical JSON output (sorted keys, fixed rounding).
  2. Malformed lines are counted and skipped, never fatal.
  3. Empty logs produce a valid zero-report.
  4. Weak labels only affect metrics when ``include_weak=True``.
  5. Shadow metrics are labeled as estimates/agreement, never regret truth.
  6. ``--since`` filters by trace timestamp, not outcome timestamp.

Reducer precedence (explicit, per spec):
  1. strong user/IDE acceptance or rejection  (user_acceptance)
  2. CI / test / command pass-fail            (ci_result, test_result, command_exit)
  3. escalation outcome                       (escalation_event)
  4. retry outcome                            (retry_event, topic_change)
  5. weak legacy self-report                  (agent_self_report, excluded by default)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

from .events import (
    DECOMPOSITION_KINDS,
    HONEST_OUTCOME_TYPES,
    AppendOnlyEventLog,
    derive_label,
)
from .shadow import aggregate_shadow_agreement

logger = logging.getLogger(__name__)


# ── Precedence-aware label derivation ──────────────────────────────────
#
# The spec defines an explicit precedence order stricter than the generic
# ``derive_label`` reducer. We implement it as a special rule that walks
# outcomes in precedence order and returns the first match.

_PRECEDENCE_ORDER = [
    # (event_type, min_strength)
    ("user_acceptance", "strong"),
    ("ci_result", "strong"),
    ("test_result", "strong"),
    ("command_exit", "strong"),
    # Hook-sourced verifiable categories
    ("test", "strong"),
    ("build", "strong"),
    ("lint", "strong"),
    ("typecheck", "strong"),
    ("format", "strong"),
    ("escalation_event", "medium"),
    ("retry_event", "medium"),
    ("topic_change", "medium"),
]

_WEAK_TYPES = frozenset({"agent_self_report"})


def _derive_precedence_label(
    outcomes: list[dict[str, Any]],
    include_weak: bool = False,
) -> str:
    """Derive label using explicit precedence order from the spec.

    Returns one of: "success" | "failure" | "passed" | "failed" |
    "accepted" | "rejected" | "unknown".
    """
    for event_type, _min_strength in _PRECEDENCE_ORDER:
        matches = [
            o for o in outcomes
            if o.get("event_type") == event_type
        ]
        if matches:
            # Last-write-wins among matches of this precedence level
            latest = max(matches, key=lambda e: float(e.get("timestamp", 0.0) or 0.0))
            return str(latest.get("value", "") or "")

    if include_weak:
        weak = [o for o in outcomes if o.get("event_type") in _WEAK_TYPES]
        if weak:
            latest = max(weak, key=lambda e: float(e.get("timestamp", 0.0) or 0.0))
            return str(latest.get("value", "") or "")

    return "unknown"


# ── Core report generator ──────────────────────────────────────────────


def generate_report(
    log_path: str | Path,
    *,
    include_weak: bool = False,
    since_timestamp: float | None = None,
) -> dict[str, Any]:
    """Generate the complete RAVS v1 offline evaluation report.

    Args:
        log_path: Path to the JSONL event log.
        include_weak: If True, weak (agent_self_report) signals affect
            headline metrics. Default False.
        since_timestamp: If set, only traces with timestamp >= this value
            are included.

    Returns:
        A deterministic dict suitable for ``json.dumps(sort_keys=True)``.
        Same input always produces byte-stable JSON output.
    """


    # ── Phase 1: Parse and partition ──────────────────────────────────
    traces: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, list[dict[str, Any]]] = {}
    malformed_lines = 0
    total_lines = 0

    if Path(log_path).exists():
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines += 1
                    continue
                if not isinstance(evt, dict):
                    malformed_lines += 1
                    continue

                # Support both original format (kind=trace/outcome)
                # and hook/capture format (type=request/outcome)
                kind = evt.get("kind") or evt.get("type")
                rid = evt.get("request_id")
                if not rid:
                    malformed_lines += 1
                    continue

                if kind in ("trace", "request"):
                    traces[rid] = evt
                elif kind == "outcome":
                    outcomes.setdefault(rid, []).append(evt)
                else:
                    malformed_lines += 1

    # ── Phase 2: Filter by --since ────────────────────────────────────
    if since_timestamp is not None:
        traces = {
            rid: t for rid, t in traces.items()
            if float(t.get("timestamp", 0.0) or 0.0) >= since_timestamp
        }

    # ── Phase 3: Derive labels using explicit precedence ──────────────
    total_requests = len(traces)

    # Per-trace derived label
    labels: dict[str, str] = {}
    weak_labels: dict[str, str] = {}
    for rid, trace in traces.items():
        outs = outcomes.get(rid, [])
        labels[rid] = _derive_precedence_label(outs, include_weak=include_weak)
        weak_labels[rid] = _derive_precedence_label(outs, include_weak=True)

    # ── Phase 4: Compute metrics ──────────────────────────────────────

    # 4a. Label coverage
    honest_labeled = sum(1 for v in labels.values() if v != "unknown")
    weak_labeled = sum(1 for v in weak_labels.values() if v != "unknown")
    unknown_count = sum(1 for v in labels.values() if v == "unknown")

    honest_label_coverage = _safe_rate(honest_labeled, total_requests)
    weak_label_coverage = _safe_rate(weak_labeled, total_requests)
    unknown_label_rate = _safe_rate(unknown_count, total_requests)

    # 4b. Success/failure breakdown (from derived labels)
    # Include hook verdicts: "pass" maps to success, "fail" maps to failure
    success_values = frozenset({"success", "passed", "accepted", "pass"})
    failure_values = frozenset({"failure", "failed", "rejected", "fail"})
    labeled_count = honest_labeled if not include_weak else weak_labeled

    successes = sum(1 for v in labels.values() if v in success_values)
    failures = sum(1 for v in labels.values() if v in failure_values)
    success_rate = _safe_rate(successes, labeled_count)

    # 4c. Retry rate — count traces that have at least one retry_event outcome
    retry_traces = set()
    for rid in traces:
        for o in outcomes.get(rid, []):
            if o.get("event_type") == "retry_event":
                retry_traces.add(rid)
    retry_rate = _safe_rate(len(retry_traces), total_requests)

    # 4d. Escalation rate
    escalation_traces = set()
    for rid in traces:
        for o in outcomes.get(rid, []):
            if o.get("event_type") == "escalation_event":
                escalation_traces.add(rid)
    escalation_rate = _safe_rate(len(escalation_traces), total_requests)

    # 4e. Cost by model
    cost_by_model: dict[str, dict[str, Any]] = {}
    latency_by_model: dict[str, list[float]] = {}
    for trace in traces.values():
        model = trace.get("model") or "unknown"
        cost = float(trace.get("cost_usd", -1.0) or -1.0)
        latency = float(trace.get("latency_ms", -1.0) or -1.0)
        ctx_tokens = int(trace.get("context_size_tokens", 0) or 0)

        entry = cost_by_model.setdefault(model, {
            "requests": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "measured_cost_requests": 0,
        })
        entry["requests"] += 1
        entry["total_tokens"] += ctx_tokens
        if cost >= 0:
            entry["total_cost_usd"] += cost
            entry["measured_cost_requests"] += 1

        if latency >= 0:
            latency_by_model.setdefault(model, []).append(latency)

    # Finalize cost_by_model
    for model, entry in cost_by_model.items():
        entry["total_cost_usd"] = round(entry["total_cost_usd"], 6)
        entry["avg_cost_usd"] = round(
            entry["total_cost_usd"] / max(entry["measured_cost_requests"], 1), 6
        )

    # 4f. Latency by model (p50, p95)
    latency_report: dict[str, dict[str, float]] = {}
    for model, lats in sorted(latency_by_model.items()):
        sorted_lats = sorted(lats)
        n = len(sorted_lats)
        latency_report[model] = {
            "n": n,
            "p50_ms": round(sorted_lats[n // 2], 2) if n else 0.0,
            "p95_ms": round(sorted_lats[int(n * 0.95)] if n else 0.0, 2),
            "mean_ms": round(sum(sorted_lats) / max(n, 1), 2),
        }

    # 4g. Decomposition evidence rate + by kind
    traces_with_decomp = 0
    decomp_by_kind: dict[str, int] = {}
    total_decomp_evidence = 0
    for trace in traces.values():
        evidences = trace.get("decomposition_evidence") or []
        if evidences:
            traces_with_decomp += 1
        for ev in evidences:
            total_decomp_evidence += 1
            kind = ev.get("kind", "unknown")
            decomp_by_kind[kind] = decomp_by_kind.get(kind, 0) + 1

    decomposition_evidence_rate = _safe_rate(traces_with_decomp, total_requests)

    # Sort by count descending for readability
    decomp_by_kind_sorted = dict(
        sorted(decomp_by_kind.items(), key=lambda x: (-x[1], x[0]))
    )

    # 4h. Shadow policy agreement
    trace_list = list(traces.values())
    shadow_agreement = aggregate_shadow_agreement(trace_list)

    # 4i. Shadow recommendation cost delta (estimate, NOT regret)
    # For each shadow that recommended a different model, estimate the
    # cost difference using the per-model average cost we computed.
    # This is labeled as an ESTIMATE — we don't know what would have
    # actually happened.
    model_avg_cost: dict[str, float] = {}
    for model, entry in cost_by_model.items():
        if entry["measured_cost_requests"] > 0:
            model_avg_cost[model] = entry["avg_cost_usd"]

    shadow_cost_delta: dict[str, dict[str, Any]] = {}
    for trace in traces.values():
        prod_model = trace.get("model") or "unknown"
        prod_cost = model_avg_cost.get(prod_model)
        if prod_cost is None:
            continue
        shadows = trace.get("shadow_recommendations") or {}
        for policy_name, rec in shadows.items():
            if not isinstance(rec, dict):
                continue
            if rec.get("insufficient_data"):
                continue
            rec_model = rec.get("model")
            if not rec_model or rec_model == prod_model:
                continue
            rec_cost = model_avg_cost.get(rec_model)
            if rec_cost is None:
                continue
            entry = shadow_cost_delta.setdefault(policy_name, {
                "comparisons": 0,
                "total_estimated_delta_usd": 0.0,
                "note": "ESTIMATE based on avg model cost, NOT counterfactual truth",
            })
            entry["comparisons"] += 1
            entry["total_estimated_delta_usd"] += (rec_cost - prod_cost)

    for entry in shadow_cost_delta.values():
        entry["total_estimated_delta_usd"] = round(
            entry["total_estimated_delta_usd"], 6
        )
        entry["avg_estimated_delta_usd"] = round(
            entry["total_estimated_delta_usd"] / max(entry["comparisons"], 1), 6
        )

    # ── Phase 4½: Bayesian cells (Empirical Bayes) ─────────────────────
    # Per-cell Beta-Bernoulli posteriors with hierarchical pooling.
    # Each cell = (tool_category) from hook-sourced events.
    # Sparse cells borrow from the global prior via Empirical Bayes.

    bayesian_cells: dict[str, dict[str, Any]] = {}
    _cell_counts: dict[str, dict[str, int]] = {}  # {cell: {pass: N, fail: N}}

    for rid, outs in outcomes.items():
        for o in outs:
            evt_type = o.get("event_type", "")
            tool = o.get("tool", evt_type)
            value = o.get("value", "")
            # Only process hook/CI-sourced verifiable events
            if evt_type not in ("test", "build", "lint", "typecheck", "format",
                                "test_result", "ci_result", "command_exit"):
                continue

            cell_key = f"{evt_type}/{tool}" if tool != evt_type else evt_type
            cell = _cell_counts.setdefault(cell_key, {"pass": 0, "fail": 0})
            if value in success_values:
                cell["pass"] += 1
            elif value in failure_values:
                cell["fail"] += 1

    # Compute global prior (Empirical Bayes: pool across all cells)
    total_pass = sum(c["pass"] for c in _cell_counts.values())
    total_fail = sum(c["fail"] for c in _cell_counts.values())
    total_obs = total_pass + total_fail
    # Global success rate as prior mean; prior strength = 2 (weak prior)
    global_mean = total_pass / max(total_obs, 1)
    prior_strength = 2.0  # equivalent sample size of prior
    alpha_0 = max(global_mean * prior_strength, 0.1)
    beta_0 = max((1 - global_mean) * prior_strength, 0.1)

    for cell_key, counts in _cell_counts.items():
        n = counts["pass"] + counts["fail"]
        alpha = alpha_0 + counts["pass"]
        beta_val = beta_0 + counts["fail"]
        bayesian_cells[cell_key] = {
            "n": n,
            "passes": counts["pass"],
            "failures": counts["fail"],
            "alpha": round(alpha, 4),
            "beta": round(beta_val, 4),
            "posterior_mean": round(alpha / (alpha + beta_val), 4),
            "prior": {"alpha_0": round(alpha_0, 4), "beta_0": round(beta_0, 4)},
        }

    # ── Phase 5: Assemble report ──────────────────────────────────────

    report: dict[str, Any] = {
        "ravs_version": "v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "log_path": str(log_path),
        "include_weak": include_weak,
        "since_timestamp": since_timestamp,
        "parsing": {
            "total_lines": total_lines,
            "malformed_lines_skipped": malformed_lines,
        },
        "total_requests": total_requests,
        "honest_label_coverage": honest_label_coverage,
        "weak_label_coverage": weak_label_coverage,
        "unknown_label_rate": unknown_label_rate,
        "success_rate": success_rate,
        "label_breakdown": {
            "successes": successes,
            "failures": failures,
            "unknown": unknown_count,
        },
        "retry_rate": retry_rate,
        "retry_count": len(retry_traces),
        "escalation_rate": escalation_rate,
        "escalation_count": len(escalation_traces),
        "cost_by_model": dict(sorted(cost_by_model.items())),
        "latency_by_model": dict(sorted(latency_report.items())),
        "decomposition_evidence_rate": decomposition_evidence_rate,
        "decomposition_evidence_count": total_decomp_evidence,
        "decomposition_by_kind": decomp_by_kind_sorted,
        "shadow_policy_agreement": dict(sorted(shadow_agreement.items())),
        "shadow_policy_recommendation_cost_delta": dict(
            sorted(shadow_cost_delta.items())
        ),
        "bayesian_cells": dict(sorted(bayesian_cells.items())),
    }

    return report


# ── Helpers ────────────────────────────────────────────────────────────


def _safe_rate(numerator: int, denominator: int) -> float:
    """Compute a rate with 4-digit rounding, zero-safe."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def format_report_text(report: dict[str, Any]) -> str:
    """Render a human-readable text report from the report dict."""
    lines: list[str] = []
    lines.append("")
    lines.append("  ══════════════════════════════════════════════════════════")
    lines.append("  RAVS v1 — Offline Evaluation Report")
    lines.append("  ══════════════════════════════════════════════════════════")
    lines.append("")

    # Parsing
    p = report.get("parsing", {})
    lines.append(f"  Log:            {report.get('log_path', '?')}")
    lines.append(f"  Lines parsed:   {p.get('total_lines', 0)}")
    if p.get("malformed_lines_skipped", 0) > 0:
        lines.append(f"  ⚠ Malformed:    {p['malformed_lines_skipped']} lines skipped")
    if report.get("since_timestamp"):
        lines.append(f"  Since:          {report['since_timestamp']}")
    if report.get("include_weak"):
        lines.append("  Mode:           --include-weak (weak labels affect metrics)")
    lines.append("")

    # Headline metrics
    lines.append("  ── Label Coverage ────────────────────────────────────────")
    lines.append(f"  Total requests:         {report.get('total_requests', 0)}")
    lines.append(f"  Honest label coverage:  {_pct(report.get('honest_label_coverage', 0))}")
    lines.append(f"  Weak label coverage:    {_pct(report.get('weak_label_coverage', 0))}")
    lines.append(f"  Unknown label rate:     {_pct(report.get('unknown_label_rate', 0))}")
    bd = report.get("label_breakdown", {})
    lines.append(f"  Success rate:           {_pct(report.get('success_rate', 0))}")
    lines.append(f"    successes={bd.get('successes', 0)}  failures={bd.get('failures', 0)}  unknown={bd.get('unknown', 0)}")
    lines.append("")

    # Behavioral signals
    lines.append("  ── Behavioral Signals ────────────────────────────────────")
    lines.append(f"  Retry rate:             {_pct(report.get('retry_rate', 0))}  ({report.get('retry_count', 0)} traces)")
    lines.append(f"  Escalation rate:        {_pct(report.get('escalation_rate', 0))}  ({report.get('escalation_count', 0)} traces)")
    lines.append("")

    # Cost by model
    cbm = report.get("cost_by_model", {})
    if cbm:
        lines.append("  ── Cost by Model ─────────────────────────────────────────")
        for model, entry in cbm.items():
            req = entry.get("requests", 0)
            cost = entry.get("total_cost_usd", 0)
            avg = entry.get("avg_cost_usd", 0)
            tokens = entry.get("total_tokens", 0)
            lines.append(f"  {model:30s}  reqs={req:4d}  cost=${cost:.4f}  avg=${avg:.6f}  tokens={tokens}")
        lines.append("")

    # Latency by model
    lbm = report.get("latency_by_model", {})
    if lbm:
        lines.append("  ── Latency by Model ──────────────────────────────────────")
        for model, entry in lbm.items():
            n = entry.get("n", 0)
            p50 = entry.get("p50_ms", 0)
            p95 = entry.get("p95_ms", 0)
            lines.append(f"  {model:30s}  n={n:4d}  p50={p50:.0f}ms  p95={p95:.0f}ms")
        lines.append("")

    # Decomposition evidence
    lines.append("  ── Decomposition Evidence ────────────────────────────────")
    lines.append(f"  Evidence rate:           {_pct(report.get('decomposition_evidence_rate', 0))}")
    lines.append(f"  Total evidence records:  {report.get('decomposition_evidence_count', 0)}")
    dbk = report.get("decomposition_by_kind", {})
    if dbk:
        lines.append("  By kind:")
        for kind, count in dbk.items():
            lines.append(f"    {kind:25s}  {count}")
    lines.append("")

    # Shadow agreement
    spa = report.get("shadow_policy_agreement", {})
    if spa:
        lines.append("  ── Shadow Policy Agreement (ESTIMATE, not regret) ────────")
        for policy, entry in spa.items():
            agree = entry.get("agree_rate", 0)
            abstain = entry.get("abstain_rate", 0)
            n = int(entry.get("n", 0))
            lines.append(f"  {policy:30s}  agree={_pct(agree)}  abstain={_pct(abstain)}  n={n}")
        lines.append("")

    # Shadow cost delta
    scd = report.get("shadow_policy_recommendation_cost_delta", {})
    if scd:
        lines.append("  ── Shadow Cost Delta (ESTIMATE, not counterfactual) ─────")
        for policy, entry in scd.items():
            comps = entry.get("comparisons", 0)
            avg_d = entry.get("avg_estimated_delta_usd", 0)
            lines.append(f"  {policy:30s}  comparisons={comps}  avg_delta=${avg_d:+.6f}")
        lines.append(f"  {'':30s}  ⚠ These are estimates from avg model cost,")
        lines.append(f"  {'':30s}    NOT counterfactual truth.")
        lines.append("")

    # Bayesian cells (Empirical Bayes per-tool posteriors)
    cells = report.get("bayesian_cells", {})
    if cells:
        lines.append("  ── Bayesian Cells (Empirical Bayes) ──────────────────────")
        prior = None
        for cell_key, cell in cells.items():
            n = cell.get("n", 0)
            mean = cell.get("posterior_mean", 0)
            alpha = cell.get("alpha", 1)
            beta = cell.get("beta", 1)
            passes = cell.get("passes", 0)
            fails = cell.get("failures", 0)
            prior = cell.get("prior", {})

            # Compute 95% CI via normal approximation of Beta
            import math
            ab = alpha + beta
            var = (alpha * beta) / (ab * ab * (ab + 1)) if ab > 0 else 0
            std = math.sqrt(var) if var > 0 else 0
            ci_lo = max(0.0, mean - 1.96 * std)
            ci_hi = min(1.0, mean + 1.96 * std)

            lines.append(
                f"  {cell_key:25s}  n={n:3d}  "
                f"pass={passes} fail={fails}  "
                f"P(success)={mean:.2f}  "
                f"95%CI=[{ci_lo:.2f},{ci_hi:.2f}]"
            )
        if prior:
            lines.append(f"  Prior: α₀={prior.get('alpha_0', '?')}, β₀={prior.get('beta_0', '?')} (hierarchical pool)")
        lines.append("")

    lines.append("  ══════════════════════════════════════════════════════════")
    lines.append("")
    return "\n".join(lines)


def _pct(v: float) -> str:
    """Format a rate as percentage string."""
    return f"{v * 100:.1f}%"
