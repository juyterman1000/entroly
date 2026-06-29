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
