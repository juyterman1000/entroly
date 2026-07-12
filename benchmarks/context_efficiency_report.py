"""Render an honest public report from a Context Efficiency Frontier JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.context_efficiency_frontier import REPORT_VERSION


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _interval(value: list[float]) -> str:
    return f"[{value[0]:.1%}, {value[1]:.1%}]"


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
        "| Condition | Trials | Task score | Evidence recall | Unsupported claims | Context tokens | Cost (USD) | Latency (ms) | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    frontier = set(report.get("pareto_frontier", []))
    for condition, aggregate in aggregates.items():
        lines.append(
            "| {condition} | {trials} | {score:.3f} | {evidence:.3f} | "
            "{unsupported:.3f} | {context:,.0f} | {cost:.6f} | {latency:,.1f} | {pareto} |".format(
                condition=condition,
                trials=aggregate["trials"],
                score=aggregate["mean_task_score"],
                evidence=aggregate["mean_evidence_recall"],
                unsupported=aggregate["mean_unsupported_claim_rate"],
                context=aggregate["mean_context_tokens"],
                cost=aggregate["mean_billed_cost_usd"],
                latency=aggregate["mean_latency_ms"],
                pareto="yes" if condition in frontier else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Paired Results Versus Raw Context",
            "",
            "`PASS` requires the 95% paired-bootstrap bounds to preserve task score and evidence recall, avoid excess unsupported claims, and reduce context beyond zero.",
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
                result=(
                    "PASS"
                    if comparison["quality_preserving_context_win"]
                    else "NO CLAIM"
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Methodology",
            "",
            f"- Baseline: `{methodology['baseline']}`",
            f"- Pair count: {report['pair_count']}",
            f"- Confidence interval: {methodology['confidence_interval']}",
            f"- Bootstrap samples: {methodology['bootstrap_samples']}",
            f"- Quality tolerance: {_percent(methodology['quality_tolerance'])}",
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
