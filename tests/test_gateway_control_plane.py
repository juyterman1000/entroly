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
