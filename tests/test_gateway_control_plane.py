from __future__ import annotations

from entroly.cache_routing import CachePrice, ModelCandidate
from entroly.gateway_control_plane import GatewayControlPlane
from entroly.provider_policy import (
    CanonicalGatewayRequest,
    Capability,
    GatewayRedactionPolicy,
    ProviderTarget,
)
from entroly.stable_prefix import CanonicalPrefixBuilder
from entroly.usage_ledger import UsageLedger, UsagePricing


def test_control_plane_composes_policy_routing_cache_and_ledger() -> None:
    ledger = UsageLedger()
    control = GatewayControlPlane(
        redaction_policy=GatewayRedactionPolicy(enabled=True),
        usage_ledger=ledger,
    )
    request = CanonicalGatewayRequest(
        model="primary",
        messages=(
            {"role": "system", "content": "coding policy"},
            {"role": "user", "content": "contact dev@example.com"},
        ),
    )
    prompt = CanonicalPrefixBuilder().add("policy", "coding policy").build(
        dynamic_tail="new turn"
    )
    candidates = [
        ModelCandidate(
            "primary",
            "openai",
            CachePrice(10, 1, 20),
            quality=1.0,
        ),
        ModelCandidate(
            "backup",
            "anthropic",
            CachePrice(8, 0.8, 16),
            quality=1.0,
        ),
    ]
    targets = [
        ProviderTarget("openai", "primary", frozenset({Capability.CHAT})),
        ProviderTarget("anthropic", "backup", frozenset({Capability.CHAT})),
    ]

    plan = control.plan(
        request,
        stable_prompt=prompt,
        current_model="primary",
        candidates=candidates,
        targets=targets,
        prefix_tokens=1_000,
        new_input_tokens=100,
        expected_output_tokens=100,
    )
    assert "[REDACTED_EMAIL]" in plan.request.messages[1]["content"]
    assert plan.redaction.changed

    event = control.observe_response(
        plan,
        request_id="request-1",
        provider="openai",
        model="primary",
        usage_payload={
            "usage": {
                "input_tokens": 1_100,
                "output_tokens": 100,
                "input_tokens_details": {"cached_tokens": 1_000},
            }
        },
        pricing=UsagePricing.from_values(
            input_per_million=10,
            cache_read_per_million=1,
            output_per_million=20,
        ),
        observed_at=100.0,
        cache_ttl_seconds=300.0,
        team="platform",
    )

    assert event.usage.cache_read_tokens == 1_000
    assert control.cache_router.get_lease(plan.conversation_id) is not None
    assert ledger.summary(team="platform")["requests"] == 1
    ledger.close()


def test_routed_target_is_first_executable_attempt() -> None:
    control = GatewayControlPlane()
    request = CanonicalGatewayRequest(
        model="primary",
        messages=({"role": "user", "content": "routine task"},),
    )
    prompt = CanonicalPrefixBuilder().add("policy", "stable").build()
    candidates = [
        ModelCandidate(
            "primary",
            "openai",
            CachePrice(20, 2, 40),
            quality=1.0,
        ),
        ModelCandidate(
            "backup",
            "anthropic",
            CachePrice(1, 0.1, 2),
            quality=1.0,
        ),
    ]
    targets = [
        ProviderTarget("openai", "primary", frozenset({Capability.CHAT})),
        ProviderTarget("anthropic", "backup", frozenset({Capability.CHAT})),
    ]

    plan = control.plan(
        request,
        stable_prompt=prompt,
        current_model="primary",
        candidates=candidates,
        targets=targets,
        prefix_tokens=10_000,
        new_input_tokens=100,
        expected_output_tokens=100,
    )

    assert plan.routing.selected_model == "backup"
    assert plan.failover.primary.key == "anthropic:backup"


def test_redaction_covers_stable_and_dynamic_prompt_content() -> None:
    control = GatewayControlPlane(
        redaction_policy=GatewayRedactionPolicy(enabled=True)
    )
    request = CanonicalGatewayRequest(
        model="primary",
        messages=({"role": "user", "content": "hello"},),
    )
    original = (
        CanonicalPrefixBuilder()
        .add("policy", "token sk-abcdefghijklmnopqrstuvwxyz")
        .build(dynamic_tail="contact dev@example.com")
    )
    candidate = ModelCandidate(
        "primary",
        "openai",
        CachePrice(10, 1, 20),
        quality=1.0,
    )
    target = ProviderTarget(
        "openai",
        "primary",
        frozenset({Capability.CHAT}),
    )

    plan = control.plan(
        request,
        stable_prompt=original,
        current_model="primary",
        candidates=[candidate],
        targets=[target],
    )

    assert "sk-abcdefghijklmnopqrstuvwxyz" not in plan.stable_prompt.rendered
    assert "dev@example.com" not in plan.stable_prompt.rendered
    assert "[REDACTED]" in plan.stable_prompt.stable_prefix
    assert "[REDACTED_EMAIL]" in plan.stable_prompt.dynamic_tail
    assert plan.stable_prompt.prefix_hash != original.prefix_hash
    assert plan.redaction.changed


def test_cache_creation_observation_warms_next_turn() -> None:
    control = GatewayControlPlane()
    request = CanonicalGatewayRequest(
        model="claude",
        messages=({"role": "user", "content": "hello"},),
    )
    prompt = CanonicalPrefixBuilder().add("policy", "stable").build()
    candidate = ModelCandidate(
        "claude",
        "anthropic",
        CachePrice(10, 1, 20, cache_write_per_million=12.5),
        quality=1.0,
    )
    target = ProviderTarget(
        "anthropic",
        "claude",
        frozenset({Capability.CHAT}),
    )
    plan = control.plan(
        request,
        stable_prompt=prompt,
        current_model="claude",
        candidates=[candidate],
        targets=[target],
    )

    control.observe_response(
        plan,
        request_id="cache-create",
        provider="anthropic",
        model="claude",
        usage_payload={
            "usage": {
                "input_tokens": 50,
                "cache_creation_input_tokens": 900,
                "output_tokens": 25,
            }
        },
        pricing=UsagePricing.from_values(
            input_per_million=10,
            cache_read_per_million=1,
            cache_write_per_million=12.5,
            output_per_million=20,
        ),
        observed_at=100.0,
        cache_ttl_seconds=300.0,
    )

    lease = control.cache_router.get_lease(plan.conversation_id)
    assert lease is not None
    assert lease.cached_prefix_tokens == 900
    assert lease.misses == 1
