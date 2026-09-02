"""Analyze paired quality-cost trials for context-control systems.

The runner is deliberately model-neutral. Provider adapters write one JSONL
record per task and condition; this module validates pairing, computes paired
bootstrap intervals, and reports the quality/context Pareto frontier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "entroly.context-efficiency-trial.v2"
REPORT_VERSION = "entroly.context-efficiency-frontier.v2"
BASELINE = "raw"
MINIMUM_CLAIM_PAIRS = 20
FAMILYWISE_ALPHA = 0.05
MAX_REGRESSION_RATE = 0.05
MINIMUM_CONTEXT_WIN_RATE = 0.5
PRIMARY_RISK_GATES = 4
CONDITIONS = ("raw", "native_compaction", "entroly", "combined")
USAGE_SOURCES = (
    "provider_response",
    "provider_log",
    "provider_ledger",
    "provider_error",
    "runner_error",
    "deterministic_fixture",
)
OUTCOMES = ("success", "error")
COST_SOURCES = (
    "provider_invoice",
    "provider_ledger",
    "pricing_snapshot",
    "self_hosted_no_api_fee",
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


def _boolean(payload: dict[str, Any], name: str) -> bool:
    value = payload.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256(payload: dict[str, Any], name: str) -> str:
    value = _text(payload, name).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json_object(payload: dict[str, Any], name: str) -> str:
    value = _text(payload, name)
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be canonical JSON") from error
    if not isinstance(decoded, dict) or not decoded:
        raise ValueError(f"{name} must encode a non-empty JSON object")
    canonical = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    if value != canonical:
        raise ValueError(f"{name} must use canonical JSON object encoding")
    return canonical


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
    cost_source_reference: str
    outcome: str
    error_type: str | None
    context_sha256: str
    response_text: str | None
    response_sha256: str | None
    replicate: int
    condition: str
    scorer: str
    task_score: float
    task_success: bool
    evidence_recall: float
    unsupported_claim_rate: float
    experiment_config: str
    usage_observed: bool
    cost_observed: bool
    error_fingerprint: str | None
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
        outcome = _text(payload, "outcome")
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        error_type = payload.get("error_type")
        error_fingerprint = payload.get("error_fingerprint")
        if outcome == "success":
            if error_type is not None:
                raise ValueError("successful trials must not set error_type")
            minimum_context_tokens = 1
            response_text = payload.get("response_text")
            if not isinstance(response_text, str):
                raise ValueError("successful trials require response_text")
            response_sha256 = _sha256(payload, "response_sha256")
            if response_sha256 != _sha256_text(response_text):
                raise ValueError("response_sha256 does not match response_text")
            if error_fingerprint is not None:
                raise ValueError("successful trials must not set error_fingerprint")
        else:
            if not isinstance(error_type, str) or not error_type.strip():
                raise ValueError("error trials require error_type")
            error_type = error_type.strip()
            minimum_context_tokens = 0
            response_text = payload.get("response_text")
            response_sha256 = payload.get("response_sha256")
            if response_text is not None or response_sha256 is not None:
                raise ValueError("error trials must not contain a provider response")
            error_fingerprint = _sha256(payload, "error_fingerprint")

        scores = {
            name: _number(payload, name)
            for name in ("task_score", "evidence_recall", "unsupported_claim_rate")
        }
        for name, value in scores.items():
            if value > 1.0:
                raise ValueError(f"{name} must be <= 1.0")
        task_success = _boolean(payload, "task_success")
        usage_observed = _boolean(payload, "usage_observed")
        cost_observed = _boolean(payload, "cost_observed")
        if outcome == "error" and task_success:
            raise ValueError("error trials cannot set task_success")

        context_tokens = _integer(
            payload, "context_tokens", minimum=minimum_context_tokens
        )
        reasoning_tokens = _integer(payload, "reasoning_tokens")
        output_tokens = _integer(payload, "output_tokens")
        billed_cost_usd = _number(payload, "billed_cost_usd")
        if not usage_observed and any(
            (context_tokens, reasoning_tokens, output_tokens)
        ):
            raise ValueError("unobserved usage must use zero token placeholders")
        if not cost_observed and billed_cost_usd != 0.0:
            raise ValueError("unobserved cost must use a zero placeholder")

        return cls(
            workload=_text(payload, "workload"),
            workload_version=_text(payload, "workload_version"),
            task_id=_text(payload, "task_id"),
            provider=_text(payload, "provider"),
            model=_text(payload, "model"),
            provider_request_id=_text(payload, "provider_request_id"),
            usage_source=usage_source,
            cost_source=cost_source,
            cost_source_reference=_text(payload, "cost_source_reference"),
            outcome=outcome,
            error_type=error_type,
            context_sha256=_sha256(payload, "context_sha256"),
            response_text=response_text,
            response_sha256=response_sha256,
            replicate=replicate,
            condition=condition,
            scorer=_text(payload, "scorer"),
            task_score=scores["task_score"],
            task_success=task_success,
            evidence_recall=scores["evidence_recall"],
            unsupported_claim_rate=scores["unsupported_claim_rate"],
            experiment_config=_canonical_json_object(payload, "experiment_config"),
            usage_observed=usage_observed,
            cost_observed=cost_observed,
            error_fingerprint=error_fingerprint,
            context_tokens=context_tokens,
            reasoning_tokens=reasoning_tokens,
            output_tokens=output_tokens,
            billed_cost_usd=billed_cost_usd,
            latency_ms=_number(payload, "latency_ms"),
            context_commit_id=commit_id,
        )

    @property
    def pair_key(self) -> tuple[str, str, str, str, str, int, str, str]:
        return (
            self.workload,
            self.workload_version,
            self.task_id,
            self.provider,
            self.model,
            self.replicate,
            self.scorer,
            self.experiment_config,
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


def _binomial_cdf(events: int, trials: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 1.0 if events >= trials else 0.0
    return sum(
        math.comb(trials, index)
        * probability**index
        * (1.0 - probability) ** (trials - index)
        for index in range(events + 1)
    )


def clopper_pearson_upper(
    events: int, trials: int, *, alpha: float
) -> float:
    """One-sided exact upper bound for a binomial event probability."""
    if trials < 1 or not 0 <= events <= trials:
        raise ValueError("events must be between zero and positive trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if events == trials:
        return 1.0
    if events == 0:
        return 1.0 - alpha ** (1.0 / trials)
    lower = events / trials
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(events, trials, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def clopper_pearson_lower(
    events: int, trials: int, *, alpha: float
) -> float:
    """One-sided exact lower bound for a binomial event probability."""
    return 1.0 - clopper_pearson_upper(trials - events, trials, alpha=alpha)


def zero_event_sample_size(*, upper_rate: float, alpha: float) -> int:
    """Pairs required for an exact upper bound when no adverse event occurs."""
    if not 0.0 < upper_rate < 1.0:
        raise ValueError("upper_rate must be between zero and one")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    return math.ceil(math.log(alpha) / math.log(1.0 - upper_rate))


def _reduction(candidate: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return 1.0 - candidate / baseline


def _aggregate(trials: list[Trial]) -> dict[str, Any]:
    usage_trials = [trial for trial in trials if trial.usage_observed]
    cost_trials = [trial for trial in trials if trial.cost_observed]
    successful_tasks = sum(trial.task_success for trial in trials)
    cost_complete = len(cost_trials) == len(trials)
    total_observed_cost = sum(trial.billed_cost_usd for trial in cost_trials)
    return {
        "trials": len(trials),
        "errors": sum(t.outcome == "error" for t in trials),
        "successful_tasks": successful_tasks,
        "mean_task_score": round(_mean(t.task_score for t in trials), 6),
        "mean_evidence_recall": round(_mean(t.evidence_recall for t in trials), 6),
        "mean_unsupported_claim_rate": round(
            _mean(t.unsupported_claim_rate for t in trials), 6
        ),
        "usage_observations": len(usage_trials),
        "usage_observation_complete": len(usage_trials) == len(trials),
        "mean_context_tokens": (
            round(_mean(t.context_tokens for t in usage_trials), 3)
            if usage_trials
            else None
        ),
        "mean_reasoning_tokens": (
            round(_mean(t.reasoning_tokens for t in usage_trials), 3)
            if usage_trials
            else None
        ),
        "mean_output_tokens": (
            round(_mean(t.output_tokens for t in usage_trials), 3)
            if usage_trials
            else None
        ),
        "cost_observations": len(cost_trials),
        "cost_observation_complete": cost_complete,
        "mean_billed_cost_usd": (
            round(_mean(t.billed_cost_usd for t in cost_trials), 8)
            if cost_trials
            else None
        ),
        "cost_per_successful_task_usd": (
            round(total_observed_cost / successful_tasks, 8)
            if cost_complete and successful_tasks
            else None
        ),
        "mean_latency_ms": round(_mean(t.latency_ms for t in trials), 3),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    required = ("mean_context_tokens", "mean_billed_cost_usd")
    if any(left[name] is None or right[name] is None for name in required):
        return False
    if not (
        left["usage_observation_complete"]
        and right["usage_observation_complete"]
        and left["cost_observation_complete"]
        and right["cost_observation_complete"]
    ):
        return False
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
    minimum_claim_pairs: int = MINIMUM_CLAIM_PAIRS,
    familywise_alpha: float = FAMILYWISE_ALPHA,
    max_regression_rate: float = MAX_REGRESSION_RATE,
    minimum_context_win_rate: float = MINIMUM_CONTEXT_WIN_RATE,
) -> dict[str, Any]:
    if not 0.0 <= quality_tolerance <= 1.0:
        raise ValueError("quality_tolerance must be between 0 and 1")
    if minimum_claim_pairs < 1:
        raise ValueError("minimum_claim_pairs must be positive")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be between zero and one")
    if not 0.0 < max_regression_rate < 1.0:
        raise ValueError("max_regression_rate must be between zero and one")
    if not 0.0 <= minimum_context_win_rate < 1.0:
        raise ValueError("minimum_context_win_rate must be in [0, 1)")
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
        usage_pairs = [
            (raw, candidate)
            for raw, candidate in paired_rows
            if raw.usage_observed and candidate.usage_observed
        ]
        context_reduction = [
            _reduction(candidate.context_tokens, raw.context_tokens)
            for raw, candidate in usage_pairs
        ]
        latency_reduction = [
            _reduction(candidate.latency_ms, raw.latency_ms)
            for raw, candidate in paired_rows
        ]
        cost_reduction = [
            _reduction(candidate.billed_cost_usd, raw.billed_cost_usd)
            for raw, candidate in paired_rows
            if raw.cost_observed
            and candidate.cost_observed
            and raw.billed_cost_usd > 0
        ]
        quality_ci = _bootstrap_mean_ci(quality_delta, samples=bootstrap_samples, rng=rng)
        evidence_ci = _bootstrap_mean_ci(evidence_delta, samples=bootstrap_samples, rng=rng)
        context_ci = (
            _bootstrap_mean_ci(context_reduction, samples=bootstrap_samples, rng=rng)
            if context_reduction
            else None
        )
        unsupported_ci = _bootstrap_mean_ci(
            unsupported_delta, samples=bootstrap_samples, rng=rng
        )
        descriptive_bounds_pass = (
            quality_ci[0] >= -quality_tolerance
            and evidence_ci[0] >= -quality_tolerance
            and unsupported_ci[1] <= quality_tolerance
            and context_ci is not None
            and context_ci[0] > 0.0
        )
        adjusted_alpha = familywise_alpha / PRIMARY_RISK_GATES
        task_regressions = sum(
            raw.task_success and not candidate.task_success
            for raw, candidate in paired_rows
        )
        evidence_regressions = sum(
            candidate.evidence_recall < raw.evidence_recall
            for raw, candidate in paired_rows
        )
        unsupported_regressions = sum(
            candidate.unsupported_claim_rate > raw.unsupported_claim_rate
            for raw, candidate in paired_rows
        )
        context_wins = sum(
            candidate.context_tokens < raw.context_tokens
            for raw, candidate in usage_pairs
        )
        task_regression_upper = clopper_pearson_upper(
            task_regressions, len(paired_rows), alpha=adjusted_alpha
        )
        evidence_regression_upper = clopper_pearson_upper(
            evidence_regressions, len(paired_rows), alpha=adjusted_alpha
        )
        unsupported_regression_upper = clopper_pearson_upper(
            unsupported_regressions, len(paired_rows), alpha=adjusted_alpha
        )
        context_win_lower = (
            clopper_pearson_lower(context_wins, len(usage_pairs), alpha=adjusted_alpha)
            if usage_pairs
            else None
        )
        exact_risk_bounds_pass = (
            task_regression_upper <= max_regression_rate
            and evidence_regression_upper <= max_regression_rate
            and unsupported_regression_upper <= max_regression_rate
            and context_win_lower is not None
            and context_win_lower > minimum_context_win_rate
        )
        has_enough_pairs = len(paired_rows) >= minimum_claim_pairs
        usage_complete = len(usage_pairs) == len(paired_rows)
        blockers: list[str] = []
        if not has_enough_pairs:
            blockers.append("below_minimum_pair_count")
        if not usage_complete:
            blockers.append("incomplete_usage_observation")
        if has_enough_pairs and not exact_risk_bounds_pass:
            blockers.append("exact_risk_bounds_not_met")
        if has_enough_pairs and usage_complete and not descriptive_bounds_pass:
            blockers.append("paired_effect_bounds_not_met")
        if not has_enough_pairs:
            claim_status = "insufficient_data"
        elif not usage_complete:
            claim_status = "measurement_incomplete"
        elif not exact_risk_bounds_pass:
            claim_status = "insufficient_precision"
        elif not descriptive_bounds_pass:
            claim_status = "no_claim"
        else:
            claim_status = "pass"
        comparisons[condition] = {
            "paired_trials": len(paired_rows),
            "mean_quality_delta": round(_mean(quality_delta), 6),
            "quality_delta_95ci": quality_ci,
            "mean_evidence_recall_delta": round(_mean(evidence_delta), 6),
            "evidence_recall_delta_95ci": evidence_ci,
            "mean_unsupported_claim_rate_delta": round(_mean(unsupported_delta), 6),
            "unsupported_claim_rate_delta_95ci": unsupported_ci,
            "usage_observed_pairs": len(usage_pairs),
            "usage_observation_complete": usage_complete,
            "mean_context_reduction": (
                round(_mean(context_reduction), 6) if context_reduction else None
            ),
            "context_reduction_95ci": context_ci,
            "mean_latency_reduction": round(_mean(latency_reduction), 6),
            "mean_billed_cost_reduction": (
                round(_mean(cost_reduction), 6) if cost_reduction else None
            ),
            "minimum_claim_pairs": minimum_claim_pairs,
            "familywise_alpha": familywise_alpha,
            "per_gate_alpha": adjusted_alpha,
            "max_regression_rate": max_regression_rate,
            "minimum_context_win_rate": minimum_context_win_rate,
            "task_regressions": task_regressions,
            "task_regression_rate_upper_bound": round(task_regression_upper, 6),
            "evidence_regressions": evidence_regressions,
            "evidence_regression_rate_upper_bound": round(
                evidence_regression_upper, 6
            ),
            "unsupported_claim_regressions": unsupported_regressions,
            "unsupported_claim_regression_rate_upper_bound": round(
                unsupported_regression_upper, 6
            ),
            "context_token_wins": context_wins,
            "context_token_win_rate_lower_bound": (
                round(context_win_lower, 6) if context_win_lower is not None else None
            ),
            "zero_regression_pairs_required": zero_event_sample_size(
                upper_rate=max_regression_rate, alpha=adjusted_alpha
            ),
            "claim_blockers": blockers,
            "claim_status": claim_status,
            "quality_preserving_context_win": claim_status == "pass",
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
                "experiment_config",
            ],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "descriptive_confidence_interval": "paired percentile bootstrap 95%",
            "quality_tolerance": quality_tolerance,
            "minimum_claim_pairs": minimum_claim_pairs,
            "familywise_alpha": familywise_alpha,
            "risk_gate_correction": "Bonferroni across four one-sided exact bounds",
            "per_gate_alpha": familywise_alpha / PRIMARY_RISK_GATES,
            "max_regression_rate": max_regression_rate,
            "minimum_context_win_rate": minimum_context_win_rate,
        },
        "pair_count": len(pairs),
        "provenance": {
            "workloads": sorted({t.workload for t in materialized}),
            "providers": sorted({t.provider for t in materialized}),
            "models": sorted({t.model for t in materialized}),
            "usage_sources": sorted({t.usage_source for t in materialized}),
            "cost_sources": sorted({t.cost_source for t in materialized}),
            "cost_source_references": sorted(
                {t.cost_source_reference for t in materialized}
            ),
            "experiment_configs": sorted({t.experiment_config for t in materialized}),
        },
        "aggregates": aggregates,
        "comparisons_to_raw": comparisons,
        "pareto_frontier": frontier,
        "caveats": [
            "A frontier win applies only to the recorded models, workloads, and scorer.",
            "Provider token and cost fields must come from actual response usage or invoices.",
            "Self-hosted zero API cost excludes hardware, energy, operations, and depreciation.",
            "Context Commit IDs prove artifact integrity, not task-score validity or signer identity.",
            "A non-dominated point is not necessarily statistically better than every alternative.",
            "Runs below the minimum pair count are smoke tests and cannot produce a public PASS claim.",
            "Percentile-bootstrap effect intervals are descriptive; PASS additionally requires finite-sample exact binomial risk bounds.",
            "Missing provider usage blocks an efficiency claim instead of being interpreted as zero tokens.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL file containing paired trials")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--minimum-claim-pairs", type=int, default=MINIMUM_CLAIM_PAIRS
    )
    parser.add_argument("--familywise-alpha", type=float, default=FAMILYWISE_ALPHA)
    parser.add_argument(
        "--max-regression-rate", type=float, default=MAX_REGRESSION_RATE
    )
    parser.add_argument(
        "--minimum-context-win-rate", type=float, default=MINIMUM_CONTEXT_WIN_RATE
    )
    args = parser.parse_args()
    report = analyze_frontier(
        load_trials(args.input),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        quality_tolerance=args.quality_tolerance,
        minimum_claim_pairs=args.minimum_claim_pairs,
        familywise_alpha=args.familywise_alpha,
        max_regression_rate=args.max_regression_rate,
        minimum_context_win_rate=args.minimum_context_win_rate,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
