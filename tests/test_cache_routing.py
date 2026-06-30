from __future__ import annotations

from entroly.cache_routing import (
    CacheAwareRouter,
    CachePrice,
    CacheRoutingPolicy,
    ModelCandidate,
)


CURRENT = ModelCandidate(
    "flagship",
    "provider-a",
    CachePrice(3.0, 0.3, 15.0),
    quality=1.0,
)
CHEAP = ModelCandidate(
    "fast",
    "provider-b",
    CachePrice(0.5, 0.05, 2.0),
    quality=0.99,
)


def test_warm_cache_prevents_a_short_horizon_switch() -> None:
    router = CacheAwareRouter(
        CacheRoutingPolicy(expected_turns=1, switch_hysteresis_usd=0.0)
    )
    router.observe(
        "conversation",
        model="flagship",
        provider="provider-a",
        prefix_hash="prefix",
        cached_prefix_tokens=100_000,
        cache_hit=True,
        observed_at=100.0,
        ttl_seconds=300.0,
    )

    decision = router.decide(
        "conversation",
        current_model="flagship",
        candidates=[CURRENT, CHEAP],
        prefix_hash="prefix",
        prefix_tokens=100_000,
        new_input_tokens=1_000,
        expected_output_tokens=1_000,
        expected_turns=1,
        now=101.0,
    )

    assert decision.selected_model == "flagship"
    assert decision.reason == "stay:warm_cache_economics"
    assert decision.stayed_for_cache


def test_expired_cache_frees_router_to_choose_cheaper_model() -> None:
    router = CacheAwareRouter(CacheRoutingPolicy(switch_hysteresis_usd=0.0))
    router.observe(
        "conversation",
        model="flagship",
        provider="provider-a",
        prefix_hash="prefix",
        cached_prefix_tokens=100_000,
        cache_hit=True,
        observed_at=100.0,
        ttl_seconds=10.0,
    )

    decision = router.decide(
        "conversation",
        current_model="flagship",
        candidates=[CURRENT, CHEAP],
        prefix_hash="prefix",
        prefix_tokens=100_000,
        new_input_tokens=1_000,
        expected_output_tokens=1_000,
        now=111.0,
    )

    assert decision.selected_model == "fast"
    assert decision.reason == "switch:projected_savings"


def test_provider_failure_overrides_stickiness() -> None:
    router = CacheAwareRouter()
    router.observe(
        "conversation",
        model="flagship",
        provider="provider-a",
        prefix_hash="prefix",
        cached_prefix_tokens=100_000,
        cache_hit=True,
        observed_at=100.0,
    )

    decision = router.decide(
        "conversation",
        current_model="flagship",
        candidates=[CURRENT, CHEAP],
        prefix_hash="prefix",
        prefix_tokens=100_000,
        new_input_tokens=100,
        expected_output_tokens=100,
        provider_failed=True,
        now=101.0,
    )

    assert decision.selected_model == "fast"
    assert decision.reason == "switch:provider_failure"


def test_high_risk_request_disallows_quality_drop() -> None:
    router = CacheAwareRouter(CacheRoutingPolicy(switch_hysteresis_usd=0.0))

    decision = router.decide(
        "new-conversation",
        current_model="flagship",
        candidates=[CURRENT, CHEAP],
        prefix_hash="prefix",
        prefix_tokens=10_000,
        new_input_tokens=100,
        expected_output_tokens=100,
        risk="high",
    )

    assert decision.selected_model == "flagship"


def test_partial_cache_hit_prices_uncached_prefix_at_write_rate() -> None:
    router = CacheAwareRouter(
        CacheRoutingPolicy(expected_turns=1, switch_hysteresis_usd=0.0)
    )
    partial = ModelCandidate(
        "partial",
        "provider-a",
        CachePrice(10.0, 1.0, 0.0),
        quality=1.0,
    )
    alternative = ModelCandidate(
        "alternative",
        "provider-b",
        CachePrice(6.0, 6.0, 0.0),
        quality=1.0,
    )
    router.observe(
        "conversation",
        model="partial",
        provider="provider-a",
        prefix_hash="prefix",
        cached_prefix_tokens=4_000,
        cache_hit=True,
        observed_at=100.0,
        ttl_seconds=300.0,
    )

    decision = router.decide(
        "conversation",
        current_model="partial",
        candidates=[partial, alternative],
        prefix_hash="prefix",
        prefix_tokens=40_000,
        new_input_tokens=0,
        expected_output_tokens=0,
        expected_turns=1,
        now=101.0,
    )

    assert decision.projected_costs_usd["partial"] == 0.364
    assert decision.projected_costs_usd["alternative"] == 0.24
    assert decision.selected_model == "alternative"


def test_zero_ttl_does_not_assume_future_cache_hits() -> None:
    router = CacheAwareRouter(
        CacheRoutingPolicy(
            expected_turns=3,
            provider_ttl_seconds={"provider-a": 0.0, "provider-b": 0.0},
            switch_hysteresis_usd=0.0,
        )
    )

    decision = router.decide(
        "conversation",
        current_model="flagship",
        candidates=[CURRENT, CHEAP],
        prefix_hash="prefix",
        prefix_tokens=100_000,
        new_input_tokens=0,
        expected_output_tokens=0,
        now=100.0,
    )

    assert decision.projected_costs_usd["flagship"] == 0.9
    assert decision.projected_costs_usd["fast"] == 0.15


def test_provider_failure_fails_closed_below_quality_floor() -> None:
    low_quality = ModelCandidate(
        "unsafe-backup",
        "provider-b",
        CachePrice(0.1, 0.01, 0.1),
        quality=0.5,
    )
    router = CacheAwareRouter()

    import pytest

    with pytest.raises(RuntimeError, match="quality floor"):
        router.decide(
            "conversation",
            current_model="flagship",
            candidates=[CURRENT, low_quality],
            prefix_hash="prefix",
            prefix_tokens=10_000,
            new_input_tokens=100,
            expected_output_tokens=100,
            risk="high",
            provider_failed=True,
        )
