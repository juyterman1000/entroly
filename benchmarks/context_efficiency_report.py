"""Render an honest public report from a Context Efficiency Frontier JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.context_efficiency_frontier import REPORT_VERSION


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _interval(value: list[float] | None) -> str:
    return "n/a" if value is None else f"[{value[0]:.1%}, {value[1]:.1%}]"


def _number(value: float | None, *, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def render_markdown(report: dict[str, Any]) -> str:
    if report.get("schema_version") != REPORT_VERSION:
        raise ValueError(f"schema_version must be {REPORT_VERSION!r}")
    methodology = report.get("methodology")
    aggregates = report.get("aggregates")
    comparisons = report.get("comparisons_to_raw")
    caveats = report.get("caveats")
    if not isinstance(methodology, dict):
        raise ValueError("report is missing methodology")
    if not isinstance(aggregates, dict) or not aggregates:
        raise ValueError("report is missing aggregates")
    if not isinstance(comparisons, dict):
        raise ValueError("report is missing comparisons_to_raw")
    if not isinstance(caveats, list):
        raise ValueError("report is missing caveats")

    lines = [
        "# Context Efficiency Frontier",
        "",
        "Paired evaluation of task quality, evidence retention, provider-observed context usage, cost, and latency.",
        "",
        "## Operating Points",
        "",
        "| Condition | Trials | Errors | Successful tasks | Task score | Evidence recall | Unsupported-output proxy | Context tokens | Usage observed | Cost (USD) | Cost / successful task | Latency (ms) | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    frontier = set(report.get("pareto_frontier", []))
    for condition, aggregate in aggregates.items():
        lines.append(
            "| {condition} | {trials} | {errors} | {successes} | {score:.3f} | {evidence:.3f} | "
            "{unsupported:.3f} | {context} | {usage}/{trials} | {cost} | {cost_per_success} | {latency:,.1f} | {pareto} |".format(
                condition=condition,
                trials=aggregate["trials"],
                errors=aggregate["errors"],
                successes=aggregate["successful_tasks"],
                score=aggregate["mean_task_score"],
                evidence=aggregate["mean_evidence_recall"],
                unsupported=aggregate["mean_unsupported_claim_rate"],
                context=_number(aggregate["mean_context_tokens"]),
                usage=aggregate["usage_observations"],
                cost=_number(aggregate["mean_billed_cost_usd"], digits=6),
                cost_per_success=_number(
                    aggregate["cost_per_successful_task_usd"], digits=6
                ),
                latency=aggregate["mean_latency_ms"],
                pareto="yes" if condition in frontier else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Paired Results Versus Raw Context",
            "",
            "`PASS` requires complete provider usage, the minimum pair count, descriptive paired-bootstrap bounds, and simultaneous exact one-sided risk bounds for task regressions, evidence regressions, unsupported-claim regressions, and per-task context wins. Smaller runs are `SMOKE ONLY`; imprecise larger runs remain `INSUFFICIENT PRECISION`.",
            "",
            "| Condition | Pairs | Quality delta (95% CI) | Evidence delta (95% CI) | Context reduction (95% CI) | Cost reduction | Result |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for condition, comparison in comparisons.items():
        lines.append(
            "| {condition} | {pairs} | {quality} {quality_ci} | {evidence} "
            "{evidence_ci} | {context} {context_ci} | {cost} | **{result}** |".format(
                condition=condition,
                pairs=comparison["paired_trials"],
                quality=_percent(comparison["mean_quality_delta"]),
                quality_ci=_interval(comparison["quality_delta_95ci"]),
                evidence=_percent(comparison["mean_evidence_recall_delta"]),
                evidence_ci=_interval(comparison["evidence_recall_delta_95ci"]),
                context=_percent(comparison["mean_context_reduction"]),
                context_ci=_interval(comparison["context_reduction_95ci"]),
                cost=_percent(comparison["mean_billed_cost_reduction"]),
                result={
                    "pass": "PASS",
                    "no_claim": "NO CLAIM",
                    "insufficient_data": "SMOKE ONLY",
                    "measurement_incomplete": "MEASUREMENT INCOMPLETE",
                    "non_publishable_runtime": "NON-PUBLISHABLE RUNTIME",
                    "insufficient_precision": "INSUFFICIENT PRECISION",
                }[comparison["claim_status"]],
            )
        )

    lines.extend(
        [
            "",
            "## Exact Per-Task Risk Gates",
            "",
            "Bounds are one-sided Clopper-Pearson intervals with Bonferroni correction across the four primary gates.",
            "",
            "| Condition | Task regressions (upper) | Evidence regressions (upper) | Unsupported-output regressions (upper) | Context wins (lower) | Blockers |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for condition, comparison in comparisons.items():
        blockers = ", ".join(comparison["claim_blockers"]) or "none"
        lines.append(
            "| {condition} | {task}/{pairs} ({task_upper}) | {evidence}/{pairs} "
            "({evidence_upper}) | {unsupported}/{pairs} ({unsupported_upper}) | "
            "{wins}/{usage_pairs} ({win_lower}) | {blockers} |".format(
                condition=condition,
                task=comparison["task_regressions"],
                pairs=comparison["paired_trials"],
                task_upper=_percent(comparison["task_regression_rate_upper_bound"]),
                evidence=comparison["evidence_regressions"],
                evidence_upper=_percent(
                    comparison["evidence_regression_rate_upper_bound"]
                ),
                unsupported=comparison["unsupported_claim_regressions"],
                unsupported_upper=_percent(
                    comparison["unsupported_claim_regression_rate_upper_bound"]
                ),
                wins=comparison["context_token_wins"],
                usage_pairs=comparison["usage_observed_pairs"],
                win_lower=_percent(comparison["context_token_win_rate_lower_bound"]),
                blockers=blockers,
            )
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"- Baseline: `{methodology['baseline']}`",
            f"- Pair count: {report['pair_count']}",
            f"- Descriptive confidence interval: {methodology['descriptive_confidence_interval']}",
            f"- Bootstrap samples: {methodology['bootstrap_samples']}",
            f"- Quality tolerance: {_percent(methodology['quality_tolerance'])}",
            f"- Minimum pairs for a public claim: {methodology['minimum_claim_pairs']}",
            f"- Familywise alpha: {methodology['familywise_alpha']}",
            f"- Exact-risk correction: {methodology['risk_gate_correction']}",
            f"- Maximum allowed per-task regression risk: {_percent(methodology['max_regression_rate'])}",
            f"- Minimum context-token win rate: {_percent(methodology['minimum_context_win_rate'])}",
            f"- Usage sources: {', '.join(report['provenance']['usage_sources'])}",
            f"- Cost sources: {', '.join(report['provenance']['cost_sources'])}",
            "- Cost source references: "
            + ", ".join(report["provenance"]["cost_source_references"]),
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in caveats)
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m benchmarks.context_efficiency_frontier trials.jsonl --output report.json",
            "python -m benchmarks.context_efficiency_report report.json --output report.md",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Frontier report JSON")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report must be a JSON object")
    rendered = render_markdown(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
