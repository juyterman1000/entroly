"""Forecast-only cache retention economics.

This module never contacts a provider. It turns observed conversation pauses
into a conservative expected-cost comparison for explicitly supported cache
retention plans.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CacheRetentionPlan:
    name: str
    ttl_seconds: float
    write_multiplier: float
    read_multiplier: float

    def __post_init__(self) -> None:
        if not self.name or self.ttl_seconds < 0:
            raise ValueError("retention plan requires a name and non-negative TTL")
        if self.write_multiplier < 0 or self.read_multiplier < 0:
            raise ValueError("retention price multipliers must be non-negative")


@dataclass(frozen=True, slots=True)
class CacheRetentionEstimate:
    plan: CacheRetentionPlan
    projected_micro_usd: int
    warm_probability: float
    expected_warm_resumes: float
    expected_cold_resumes: float


@dataclass(frozen=True, slots=True)
class CacheRetentionForecast:
    recommended_plan: str
    baseline_plan: str
    projected_savings_micro_usd: int
    confidence: float
    sample_count: int
    tier: str
    estimates: tuple[CacheRetentionEstimate, ...]


class CacheRetentionForecaster:
    """Bounded pause observer and Bayesian retention-cost forecaster."""

    def __init__(self, *, max_conversations: int = 10_000, max_samples: int = 64) -> None:
        if max_conversations < 1 or max_samples < 1:
            raise ValueError("forecast bounds must be positive")
        self.max_conversations = max_conversations
        self.max_samples = max_samples
        self._last_activity: OrderedDict[str, float] = OrderedDict()
        self._pauses: dict[str, deque[float]] = {}
        self._lock = threading.RLock()

    def observe_activity(
        self,
        conversation_id: str,
        *,
        observed_at: float | None = None,
    ) -> float | None:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        timestamp = time.time() if observed_at is None else float(observed_at)
        with self._lock:
            previous = self._last_activity.get(conversation_id)
            pause = max(0.0, timestamp - previous) if previous is not None else None
            if pause is not None:
                self._pauses.setdefault(
                    conversation_id, deque(maxlen=self.max_samples)
                ).append(pause)
            self._last_activity[conversation_id] = timestamp
            self._last_activity.move_to_end(conversation_id)
            while len(self._last_activity) > self.max_conversations:
                expired, _ = self._last_activity.popitem(last=False)
                self._pauses.pop(expired, None)
            return pause

    def pauses(self, conversation_id: str) -> tuple[float, ...]:
        with self._lock:
            return tuple(self._pauses.get(conversation_id, ()))

    def forecast(
        self,
        conversation_id: str,
        *,
        prefix_tokens: int,
        input_micro_usd_per_million: int,
        plans: Iterable[CacheRetentionPlan],
        expected_resumes: int = 3,
        baseline_plan: str | None = None,
        minimum_savings_micro_usd: int = 0,
    ) -> CacheRetentionForecast:
        """Compare plans using posterior warm probability for observed pauses.

        A Beta(1, 1) prior prevents one short pause from producing a confident
        long-retention recommendation. The reported tier is always estimated.
        """
        if prefix_tokens < 0 or input_micro_usd_per_million < 0:
            raise ValueError("token count and price must be non-negative")
        if expected_resumes < 1 or minimum_savings_micro_usd < 0:
            raise ValueError("forecast horizon must be positive")
        options = tuple(plans)
        if not options:
            raise ValueError("at least one retention plan is required")
        if len({plan.name for plan in options}) != len(options):
            raise ValueError("retention plan names must be unique")
        samples = self.pauses(conversation_id)
        estimates: list[CacheRetentionEstimate] = []
        token_base_cost = prefix_tokens * input_micro_usd_per_million / 1_000_000.0
        for plan in options:
            warm_count = sum(pause <= plan.ttl_seconds for pause in samples)
            warm_probability = (warm_count + 1.0) / (len(samples) + 2.0)
            warm_resumes = expected_resumes * warm_probability
            cold_resumes = expected_resumes - warm_resumes
            multiplier = (
                plan.write_multiplier
                + warm_resumes * plan.read_multiplier
                + cold_resumes * plan.write_multiplier
            )
            estimates.append(
                CacheRetentionEstimate(
                    plan=plan,
                    projected_micro_usd=max(0, int(round(token_base_cost * multiplier))),
                    warm_probability=round(warm_probability, 6),
                    expected_warm_resumes=round(warm_resumes, 6),
                    expected_cold_resumes=round(cold_resumes, 6),
                )
            )
        by_name = {estimate.plan.name: estimate for estimate in estimates}
        baseline_name = baseline_plan or options[0].name
        if baseline_name not in by_name:
            raise ValueError("baseline_plan must name one of the supplied plans")
        baseline = by_name[baseline_name]
        best = min(estimates, key=lambda estimate: (estimate.projected_micro_usd, estimate.plan.name))
        savings = baseline.projected_micro_usd - best.projected_micro_usd
        if savings < minimum_savings_micro_usd:
            best = baseline
            savings = 0
        confidence = 1.0 - math.exp(-len(samples) / 8.0)
        return CacheRetentionForecast(
            recommended_plan=best.plan.name,
            baseline_plan=baseline.plan.name,
            projected_savings_micro_usd=max(0, savings),
            confidence=round(confidence, 6),
            sample_count=len(samples),
            tier="estimated",
            estimates=tuple(sorted(estimates, key=lambda estimate: estimate.plan.name)),
        )


def anthropic_retention_plans(*, include_one_hour: bool = True) -> tuple[CacheRetentionPlan, ...]:
    """Documented Anthropic prompt-cache price multipliers."""
    plans = [CacheRetentionPlan("anthropic_5m", 300.0, 1.25, 0.10)]
    if include_one_hour:
        plans.append(CacheRetentionPlan("anthropic_1h", 3600.0, 2.00, 0.10))
    return tuple(plans)


__all__ = [
    "CacheRetentionEstimate",
    "CacheRetentionForecast",
    "CacheRetentionForecaster",
    "CacheRetentionPlan",
    "anthropic_retention_plans",
]
