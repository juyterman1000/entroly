"""Analyze paired quality-cost trials for context-control systems.

The runner is deliberately model-neutral. Provider adapters write one JSONL
record per task and condition; this module validates pairing, computes paired
bootstrap intervals, and reports the quality/context Pareto frontier.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "entroly.context-efficiency-trial.v1"
REPORT_VERSION = "entroly.context-efficiency-frontier.v1"
BASELINE = "raw"
CONDITIONS = ("raw", "native_compaction", "entroly", "combined")
USAGE_SOURCES = (
    "provider_response",
    "provider_log",
    "provider_ledger",
    "deterministic_fixture",
)
COST_SOURCES = (
    "provider_invoice",
    "provider_ledger",
    "pricing_snapshot",
    "zero_cost_fixture",
)


def _number(payload: dict[str, Any], name: str, *, minimum: float = 0.0) -> float:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value


def _text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(payload: dict[str, Any], name: str, *, minimum: int = 0) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Trial:
    workload: str
    workload_version: str
    task_id: str
    provider: str
    model: str
    provider_request_id: str
    usage_source: str
    cost_source: str
    replicate: int
    condition: str
    scorer: str
    task_score: float
    evidence_recall: float
    unsupported_claim_rate: float
    context_tokens: int
    reasoning_tokens: int
    output_tokens: int
    billed_cost_usd: float
    latency_ms: float
    context_commit_id: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trial:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        condition = _text(payload, "condition")
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")

        replicate = payload.get("replicate")
        if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
            raise ValueError("replicate must be a non-negative integer")

        commit_id = payload.get("context_commit_id")
        if commit_id is not None and (
            not isinstance(commit_id, str) or not commit_id.startswith("ctx_")
        ):
            raise ValueError("context_commit_id must start with 'ctx_'")
        if condition in {"entroly", "combined"} and commit_id is None:
            raise ValueError(f"{condition} trials require context_commit_id")

        usage_source = _text(payload, "usage_source")
        if usage_source not in USAGE_SOURCES:
            raise ValueError(f"usage_source must be one of {USAGE_SOURCES}")
        cost_source = _text(payload, "cost_source")
        if cost_source not in COST_SOURCES:
            raise ValueError(f"cost_source must be one of {COST_SOURCES}")

        scores = {
            name: _number(payload, name)
            for name in ("task_score", "evidence_recall", "unsupported_claim_rate")
        }
        for name, value in scores.items():
            if value > 1.0:
                raise ValueError(f"{name} must be <= 1.0")

        return cls(
            workload=_text(payload, "workload"),
            workload_version=_text(payload, "workload_version"),
            task_id=_text(payload, "task_id"),
            provider=_text(payload, "provider"),
            model=_text(payload, "model"),
            provider_request_id=_text(payload, "provider_request_id"),
            usage_source=usage_source,
            cost_source=cost_source,
            replicate=replicate,
            condition=condition,
            scorer=_text(payload, "scorer"),
            task_score=scores["task_score"],
            evidence_recall=scores["evidence_recall"],
            unsupported_claim_rate=scores["unsupported_claim_rate"],
            context_tokens=_integer(payload, "context_tokens", minimum=1),
            reasoning_tokens=_integer(payload, "reasoning_tokens"),
            output_tokens=_integer(payload, "output_tokens"),
            billed_cost_usd=_number(payload, "billed_cost_usd"),
            latency_ms=_number(payload, "latency_ms"),
            context_commit_id=commit_id,
        )

    @property
    def pair_key(self) -> tuple[str, str, str, str, str, int, str]:
        return (
            self.workload,
            self.workload_version,
            self.task_id,
            self.provider,
            self.model,
            self.replicate,
            self.scorer,
        )

    def as_record(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def load_trials(path: Path) -> list[Trial]:
    trials: list[Trial] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("record must be a JSON object")
            trials.append(Trial.from_dict(payload))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not trials:
        raise ValueError(f"{path}: no trial records")
    return trials


def _paired(trials: Iterable[Trial]) -> dict[tuple[Any, ...], dict[str, Trial]]:
    pairs: dict[tuple[Any, ...], dict[str, Trial]] = {}
    for trial in trials:
        conditions = pairs.setdefault(trial.pair_key, {})
        if trial.condition in conditions:
            raise ValueError(
                f"duplicate {trial.condition!r} trial for pair {trial.pair_key!r}"
            )
        conditions[trial.condition] = trial
    for key, conditions in pairs.items():
        if BASELINE not in conditions:
            raise ValueError(f"pair {key!r} is missing the raw baseline")
    expected_conditions = set().union(*(set(conditions) for conditions in pairs.values()))
    if expected_conditions == {BASELINE}:
        raise ValueError("at least one paired non-baseline trial is required")
    for key, conditions in pairs.items():
        missing = expected_conditions.difference(conditions)
        if missing:
            raise ValueError(
                f"pair {key!r} has an incomplete condition matrix; "
                f"missing {sorted(missing)!r}"
            )
    return pairs


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _percentile(sorted_values: list[float], probability: float) -> float:
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    weight = index - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_mean_ci(
    values: list[float], *, samples: int, rng: random.Random
) -> list[float]:
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    estimates = [
        _mean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    estimates.sort()
    return [
        round(_percentile(estimates, 0.025), 6),
        round(_percentile(estimates, 0.975), 6),
    ]


def _reduction(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return 1.0 - candidate / baseline


def _aggregate(trials: list[Trial]) -> dict[str, Any]:
    return {
        "trials": len(trials),
        "mean_task_score": round(_mean(t.task_score for t in trials), 6),
        "mean_evidence_recall": round(_mean(t.evidence_recall for t in trials), 6),
        "mean_unsupported_claim_rate": round(
            _mean(t.unsupported_claim_rate for t in trials), 6
        ),
        "mean_context_tokens": round(_mean(t.context_tokens for t in trials), 3),
        "mean_reasoning_tokens": round(_mean(t.reasoning_tokens for t in trials), 3),
        "mean_output_tokens": round(_mean(t.output_tokens for t in trials), 3),
        "mean_billed_cost_usd": round(_mean(t.billed_cost_usd for t in trials), 8),
        "mean_latency_ms": round(_mean(t.latency_ms for t in trials), 3),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        left["mean_task_score"] >= right["mean_task_score"]
        and left["mean_evidence_recall"] >= right["mean_evidence_recall"]
        and left["mean_unsupported_claim_rate"]
        <= right["mean_unsupported_claim_rate"]
        and left["mean_context_tokens"] <= right["mean_context_tokens"]
        and left["mean_billed_cost_usd"] <= right["mean_billed_cost_usd"]
    )
    strictly_better = any(
        (
            left[name] > right[name]
            if name in {"mean_task_score", "mean_evidence_recall"}
            else left[name] < right[name]
        )
        for name in (
            "mean_task_score",
            "mean_evidence_recall",
            "mean_unsupported_claim_rate",
            "mean_context_tokens",
            "mean_billed_cost_usd",
        )
    )
    return no_worse and strictly_better


def analyze_frontier(
    trials: Iterable[Trial],
    *,
    bootstrap_samples: int = 2_000,
    seed: int = 42,
    quality_tolerance: float = 0.01,
) -> dict[str, Any]:
    if not 0.0 <= quality_tolerance <= 1.0:
        raise ValueError("quality_tolerance must be between 0 and 1")
    materialized = list(trials)
    pairs = _paired(materialized)
    by_condition = {
        condition: [trial for trial in materialized if trial.condition == condition]
        for condition in CONDITIONS
        if any(trial.condition == condition for trial in materialized)
    }
    aggregates = {
        condition: _aggregate(condition_trials)
        for condition, condition_trials in by_condition.items()
    }

    comparisons: dict[str, Any] = {}
    rng = random.Random(seed)
    for condition in CONDITIONS:
        if condition == BASELINE:
            continue
        paired_rows = [
            (conditions[BASELINE], conditions[condition])
            for conditions in pairs.values()
            if condition in conditions
        ]
        if not paired_rows:
            continue
        quality_delta = [candidate.task_score - raw.task_score for raw, candidate in paired_rows]
        evidence_delta = [
            candidate.evidence_recall - raw.evidence_recall
            for raw, candidate in paired_rows
        ]
        unsupported_delta = [
            candidate.unsupported_claim_rate - raw.unsupported_claim_rate
            for raw, candidate in paired_rows
        ]
        context_reduction = [
            _reduction(candidate.context_tokens, raw.context_tokens)
            for raw, candidate in paired_rows
        ]
        latency_reduction = [
            _reduction(candidate.latency_ms, raw.latency_ms)
            for raw, candidate in paired_rows
        ]
        cost_reduction = [
            _reduction(candidate.billed_cost_usd, raw.billed_cost_usd)
            for raw, candidate in paired_rows
            if raw.billed_cost_usd > 0
        ]
        quality_ci = _bootstrap_mean_ci(quality_delta, samples=bootstrap_samples, rng=rng)
        evidence_ci = _bootstrap_mean_ci(evidence_delta, samples=bootstrap_samples, rng=rng)
        context_ci = _bootstrap_mean_ci(
            context_reduction, samples=bootstrap_samples, rng=rng
        )
        unsupported_ci = _bootstrap_mean_ci(
            unsupported_delta, samples=bootstrap_samples, rng=rng
        )
        comparisons[condition] = {
            "paired_trials": len(paired_rows),
            "mean_quality_delta": round(_mean(quality_delta), 6),
            "quality_delta_95ci": quality_ci,
            "mean_evidence_recall_delta": round(_mean(evidence_delta), 6),
            "evidence_recall_delta_95ci": evidence_ci,
            "mean_unsupported_claim_rate_delta": round(_mean(unsupported_delta), 6),
            "unsupported_claim_rate_delta_95ci": unsupported_ci,
            "mean_context_reduction": round(_mean(context_reduction), 6),
            "context_reduction_95ci": context_ci,
            "mean_latency_reduction": round(_mean(latency_reduction), 6),
            "mean_billed_cost_reduction": (
                round(_mean(cost_reduction), 6) if cost_reduction else None
            ),
            "quality_preserving_context_win": (
                quality_ci[0] >= -quality_tolerance
                and evidence_ci[0] >= -quality_tolerance
                and unsupported_ci[1] <= quality_tolerance
                and context_ci[0] > 0.0
            ),
        }

    frontier = [
        condition
        for condition, aggregate in aggregates.items()
        if not any(
            other != condition and _dominates(other_aggregate, aggregate)
            for other, other_aggregate in aggregates.items()
        )
    ]
    return {
        "schema_version": REPORT_VERSION,
        "methodology": {
            "baseline": BASELINE,
            "pairing_unit": [
                "workload",
                "workload_version",
                "task_id",
                "provider",
                "model",
                "replicate",
                "scorer",
            ],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "confidence_interval": "paired percentile bootstrap 95%",
            "quality_tolerance": quality_tolerance,
        },
        "pair_count": len(pairs),
        "provenance": {
            "workloads": sorted({t.workload for t in materialized}),
            "providers": sorted({t.provider for t in materialized}),
            "models": sorted({t.model for t in materialized}),
            "usage_sources": sorted({t.usage_source for t in materialized}),
            "cost_sources": sorted({t.cost_source for t in materialized}),
        },
        "aggregates": aggregates,
        "comparisons_to_raw": comparisons,
        "pareto_frontier": frontier,
        "caveats": [
            "A frontier win applies only to the recorded models, workloads, and scorer.",
            "Provider token and cost fields must come from actual response usage or invoices.",
            "Context Commit IDs prove artifact integrity, not task-score validity or signer identity.",
            "A non-dominated point is not necessarily statistically better than every alternative.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL file containing paired trials")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    args = parser.parse_args()
    report = analyze_frontier(
        load_trials(args.input),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        quality_tolerance=args.quality_tolerance,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
