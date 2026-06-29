"""Conversation-level cache economics for model routing.

A cheap model is not necessarily the cheap decision when switching invalidates a
large warm prefix. This module projects billed input/output cost over a bounded
multi-turn horizon and applies quality, risk, availability, TTL, and hysteresis
constraints before permitting a switch.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .cache_retention import CacheRetentionForecaster


@dataclass(frozen=True, slots=True)
class CachePrice:
    """USD rates per one million tokens."""

    input_per_million: float
    cache_read_per_million: float
    output_per_million: float
    cache_write_per_million: float | None = None

    def __post_init__(self) -> None:
        rates = (
            self.input_per_million,
            self.cache_read_per_million,
            self.output_per_million,
        )
        if any(rate < 0 for rate in rates):
            raise ValueError("token prices must be non-negative")
        if self.cache_write_per_million is not None and self.cache_write_per_million < 0:
            raise ValueError("cache-write price must be non-negative")

    @property
    def write_rate(self) -> float:
        if self.cache_write_per_million is None:
            return self.input_per_million
        return self.cache_write_per_million


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    model: str
    provider: str
    price: CachePrice
    quality: float = 1.0
    available: bool = True
    capabilities_satisfied: bool = True

    def __post_init__(self) -> None:
        if not self.model or not self.provider:
            raise ValueError("candidate model and provider are required")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")


@dataclass(slots=True)
class ConversationCacheLease:
    conversation_id: str
    model: str
    provider: str
    prefix_hash: str
    cached_prefix_tokens: int
    last_used_at: float
    expires_at: float
    hits: int = 0
    misses: int = 0

    def is_warm(
        self,
        *,
        model: str,
        provider: str,
        prefix_hash: str,
        now: float,
    ) -> bool:
        return (
            now < self.expires_at
            and self.model == model
            and self.provider == provider
            and self.prefix_hash == prefix_hash
            and self.cached_prefix_tokens > 0
        )


@dataclass(frozen=True, slots=True)
class CacheRoutingPolicy:
    default_ttl_seconds: float = 300.0
    provider_ttl_seconds: Mapping[str, float] = field(default_factory=dict)
    expected_turns: int = 3
    projected_turn_interval_seconds: float = 60.0
    switch_hysteresis_usd: float = 0.001
    max_quality_drop_low: float = 0.05
    max_quality_drop_standard: float = 0.02
    max_quality_drop_high: float = 0.0
    max_leases: int = 10_000

    def __post_init__(self) -> None:
        if self.default_ttl_seconds < 0:
            raise ValueError("default TTL must be non-negative")
        if self.expected_turns < 1:
            raise ValueError("expected_turns must be at least one")
        if self.projected_turn_interval_seconds < 0:
            raise ValueError("projected turn interval must be non-negative")
        if self.switch_hysteresis_usd < 0:
            raise ValueError("switch hysteresis must be non-negative")
        if self.max_leases < 1:
            raise ValueError("max_leases must be at least one")
        drops = (
            self.max_quality_drop_low,
            self.max_quality_drop_standard,
            self.max_quality_drop_high,
        )
        if any(drop < 0 or drop > 1 for drop in drops):
            raise ValueError("quality drops must be between zero and one")

    def ttl_for(self, provider: str) -> float:
        ttl = float(self.provider_ttl_seconds.get(provider, self.default_ttl_seconds))
        return max(0.0, ttl)

    def quality_drop_for(self, risk: str) -> float:
        if risk == "high":
            return self.max_quality_drop_high
        if risk == "low":
            return self.max_quality_drop_low
        return self.max_quality_drop_standard


@dataclass(frozen=True, slots=True)
class CacheRoutingDecision:
    selected_model: str
    selected_provider: str
    reason: str
    cache_warm: bool
    stayed_for_cache: bool
    switch_savings_usd: float
    projected_costs_usd: Mapping[str, float]


class CacheAwareRouter:
    """Thread-safe controller for observed cache leases and routing decisions."""

    def __init__(
        self,
        policy: CacheRoutingPolicy | None = None,
        *,
        retention_forecaster: CacheRetentionForecaster | None = None,
    ) -> None:
        self.policy = policy or CacheRoutingPolicy()
        self.retention_forecaster = retention_forecaster or CacheRetentionForecaster(
            max_conversations=self.policy.max_leases
        )
        self._leases: dict[str, ConversationCacheLease] = {}
        self._lock = threading.RLock()

    def observe(
        self,
        conversation_id: str,
        *,
        model: str,
        provider: str,
        prefix_hash: str,
        cached_prefix_tokens: int,
        cache_hit: bool,
        observed_at: float | None = None,
        ttl_seconds: float | None = None,
    ) -> ConversationCacheLease:
        """Record provider-reported cache state; never infer a hit."""
        if not conversation_id:
            raise ValueError("conversation_id is required")
        now = time.time() if observed_at is None else float(observed_at)
        ttl = self.policy.ttl_for(provider) if ttl_seconds is None else max(0.0, ttl_seconds)
        self.retention_forecaster.observe_activity(
            conversation_id,
            observed_at=now,
        )
        with self._lock:
            previous = self._leases.get(conversation_id)
            if previous is not None and now <= previous.last_used_at:
                previous.hits += int(cache_hit)
                previous.misses += int(not cache_hit)
                return previous
            hits = (previous.hits if previous else 0) + int(cache_hit)
            misses = (previous.misses if previous else 0) + int(not cache_hit)
            lease = ConversationCacheLease(
                conversation_id=conversation_id,
                model=model,
                provider=provider,
                prefix_hash=prefix_hash,
                cached_prefix_tokens=max(0, int(cached_prefix_tokens)),
                last_used_at=now,
                expires_at=now + ttl,
                hits=hits,
                misses=misses,
            )
            self._leases[conversation_id] = lease
            self._evict(now)
            return lease

    def get_lease(self, conversation_id: str) -> ConversationCacheLease | None:
        with self._lock:
            return self._leases.get(conversation_id)

    def invalidate(self, conversation_id: str) -> None:
        with self._lock:
            self._leases.pop(conversation_id, None)

    @staticmethod
    def _turn_cost(
        candidate: ModelCandidate,
        *,
        prefix_tokens: int,
        cached_prefix_tokens: int,
        new_input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Price cached and uncached prefix portions separately."""
        price = candidate.price
        prefix = max(0, int(prefix_tokens))
        cached = min(prefix, max(0, int(cached_prefix_tokens)))
        uncached_prefix = prefix - cached
        billed = (
            cached * price.cache_read_per_million
            + uncached_prefix * price.write_rate
            + max(0, int(new_input_tokens)) * price.input_per_million
            + max(0, int(output_tokens)) * price.output_per_million
        )
        return billed / 1_000_000.0

    def _projected_cost(
        self,
        candidate: ModelCandidate,
        *,
        prefix_tokens: int,
        cached_prefix_tokens_now: int,
        new_input_tokens: int,
        output_tokens: int,
        turns: int,
        cache_reusable_between_turns: bool,
    ) -> float:
        """Project cost without inventing cache hits outside provider TTL."""
        total = self._turn_cost(
            candidate,
            prefix_tokens=prefix_tokens,
            cached_prefix_tokens=cached_prefix_tokens_now,
            new_input_tokens=new_input_tokens,
            output_tokens=output_tokens,
        )
        if turns <= 1:
            return total

        future_cached = prefix_tokens if cache_reusable_between_turns else 0
        future = self._turn_cost(
            candidate,
            prefix_tokens=prefix_tokens,
            cached_prefix_tokens=future_cached,
            new_input_tokens=new_input_tokens,
            output_tokens=output_tokens,
        )
        return total + future * (turns - 1)

    def decide(
        self,
        conversation_id: str,
        *,
        current_model: str,
        candidates: Iterable[ModelCandidate],
        prefix_hash: str,
        prefix_tokens: int,
        new_input_tokens: int,
        expected_output_tokens: int,
        risk: str = "standard",
        expected_turns: int | None = None,
        force_model: str | None = None,
        provider_failed: bool = False,
        now: float | None = None,
    ) -> CacheRoutingDecision:
        """Choose the lowest projected-cost eligible model without cache thrash."""
        timestamp = time.time() if now is None else float(now)
        choices = list(candidates)
        by_model = {candidate.model: candidate for candidate in choices}
        if len(by_model) != len(choices):
            raise ValueError(
                "candidate model identifiers must be unique across providers"
            )
        if current_model not in by_model:
            raise ValueError("current_model must be included in candidates")
        current = by_model[current_model]
        turns = max(1, expected_turns or self.policy.expected_turns)

        eligible = [
            candidate
            for candidate in choices
            if candidate.available and candidate.capabilities_satisfied
        ]
        if force_model is not None:
            forced = by_model.get(force_model)
            if forced is None or forced not in eligible:
                raise ValueError("forced model is not an eligible candidate")
            return CacheRoutingDecision(
                selected_model=forced.model,
                selected_provider=forced.provider,
                reason="forced_model",
                cache_warm=False,
                stayed_for_cache=False,
                switch_savings_usd=0.0,
                projected_costs_usd={},
            )

        if not eligible:
            raise RuntimeError("no eligible routing candidates")
        if provider_failed:
            eligible = [
                candidate
                for candidate in eligible
                if candidate.provider != current.provider
            ]
            if not eligible:
                raise RuntimeError("provider failed and no failover candidate is eligible")

        allowed_drop = self.policy.quality_drop_for(risk)
        quality_floor = current.quality - allowed_drop
        quality_eligible = [
            candidate for candidate in eligible if candidate.quality >= quality_floor
        ]
        if quality_eligible:
            eligible = quality_eligible
        elif provider_failed:
            raise RuntimeError(
                "provider failed and no failover candidate satisfies the quality floor"
            )
        elif current in eligible:
            eligible = [current]
        else:
            raise RuntimeError("no available routing candidate satisfies the quality floor")

        with self._lock:
            lease = self._leases.get(conversation_id)

        projected: dict[str, float] = {}
        warm_by_model: dict[str, bool] = {}
        cached_by_model: dict[str, int] = {}
        for candidate in eligible:
            warm = bool(
                lease
                and lease.is_warm(
                    model=candidate.model,
                    provider=candidate.provider,
                    prefix_hash=prefix_hash,
                    now=timestamp,
                )
            )
            cached_now = (
                min(max(0, prefix_tokens), lease.cached_prefix_tokens)
                if warm and lease is not None
                else 0
            )
            provider_ttl = self.policy.ttl_for(candidate.provider)
            cache_reusable = (
                provider_ttl > 0
                and provider_ttl >= self.policy.projected_turn_interval_seconds
            )
            warm_by_model[candidate.model] = warm
            cached_by_model[candidate.model] = cached_now
            projected[candidate.model] = self._projected_cost(
                candidate,
                prefix_tokens=max(0, prefix_tokens),
                cached_prefix_tokens_now=cached_now,
                new_input_tokens=max(0, new_input_tokens),
                output_tokens=max(0, expected_output_tokens),
                turns=turns,
                cache_reusable_between_turns=cache_reusable,
            )

        best = min(
            eligible,
            key=lambda candidate: (
                projected[candidate.model],
                -candidate.quality,
                candidate.model,
            ),
        )
        current_cost = projected.get(current_model)
        if current_cost is None:
            current_cost = self._projected_cost(
                current,
                prefix_tokens=max(0, prefix_tokens),
                new_input_tokens=max(0, new_input_tokens),
                output_tokens=max(0, expected_output_tokens),
                cached_prefix_tokens_now=0,
                turns=turns,
                cache_reusable_between_turns=(
                    self.policy.ttl_for(current.provider) > 0
                    and self.policy.ttl_for(current.provider)
                    >= self.policy.projected_turn_interval_seconds
                ),
            )

        savings = current_cost - projected[best.model]
        current_warm = warm_by_model.get(current_model, False)
        if (
            not provider_failed
            and best.model != current_model
            and savings <= self.policy.switch_hysteresis_usd
        ):
            best = current
            reason = "stay:hysteresis"
            savings = 0.0
        elif provider_failed:
            reason = "switch:provider_failure"
        elif best.model == current_model and current_warm:
            reason = "stay:warm_cache_economics"
        elif best.model == current_model:
            reason = "stay:lowest_projected_cost"
        else:
            reason = "switch:projected_savings"

        return CacheRoutingDecision(
            selected_model=best.model,
            selected_provider=best.provider,
            reason=reason,
            cache_warm=warm_by_model.get(best.model, False),
            stayed_for_cache=best.model == current_model and current_warm,
            switch_savings_usd=round(max(0.0, savings), 9),
            projected_costs_usd={
                model: round(cost, 9) for model, cost in sorted(projected.items())
            },
        )

    def stats(self, *, now: float | None = None) -> dict[str, float | int]:
        timestamp = time.time() if now is None else float(now)
        with self._lock:
            leases = list(self._leases.values())
        hits = sum(lease.hits for lease in leases)
        misses = sum(lease.misses for lease in leases)
        retention = self.retention_forecaster.stats()
        return {
            "active_leases": sum(lease.expires_at > timestamp for lease in leases),
            "tracked_leases": len(leases),
            "observed_hits": hits,
            "observed_misses": misses,
            "observed_hit_rate": hits / max(1, hits + misses),
            "retention_tracked_conversations": retention["tracked_conversations"],
            "retention_pause_samples": retention["pause_samples"],
        }

    def _evict(self, now: float) -> None:
        expired = [
            key for key, lease in self._leases.items() if lease.expires_at <= now
        ]
        for key in expired:
            self._leases.pop(key, None)
        overflow = len(self._leases) - self.policy.max_leases
        if overflow > 0:
            oldest = sorted(
                self._leases,
                key=lambda key: self._leases[key].last_used_at,
            )
            for key in oldest[:overflow]:
                self._leases.pop(key, None)


__all__ = [
    "CacheAwareRouter",
    "CachePrice",
    "CacheRoutingDecision",
    "CacheRoutingPolicy",
    "ConversationCacheLease",
    "ModelCandidate",
]
