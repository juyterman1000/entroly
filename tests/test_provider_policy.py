from __future__ import annotations

from entroly.provider_policy import (
    CanonicalGatewayRequest,
    Capability,
    GatewayRedactionPolicy,
    ProviderFailoverPlanner,
    ProviderTarget,
)


def test_failover_excludes_incompatible_and_open_targets() -> None:
    request = CanonicalGatewayRequest(
        model="logical-model",
        messages=({"role": "user", "content": "call the tool"},),
        tools=({"name": "search"},),
        stream=True,
    )
    targets = [
        ProviderTarget(
            "a",
            "model-a",
            frozenset({Capability.CHAT, Capability.STREAMING, Capability.TOOLS}),
            priority=20,
        ),
        ProviderTarget(
            "b",
            "model-b",
            frozenset({Capability.CHAT, Capability.STREAMING}),
            priority=1,
        ),
        ProviderTarget(
            "c",
            "model-c",
            frozenset({Capability.CHAT, Capability.STREAMING, Capability.TOOLS}),
            priority=0,
            circuit_open=True,
        ),
    ]

    plan = ProviderFailoverPlanner().plan(request, targets)

    assert plan.primary.key == "a:model-a"
    assert plan.excluded["b:model-b"].startswith("missing_capabilities")
    assert plan.excluded["c:model-c"] == "circuit_open"


def test_redaction_is_explicit_and_receipt_keeps_only_digests() -> None:
    request = CanonicalGatewayRequest(
        model="model",
        messages=(
            {
                "role": "user",
                "content": "email me at a@example.com with sk-abcdefghijklmnopqrstuvwxyz",
            },
        ),
    )

    unchanged, disabled = GatewayRedactionPolicy().apply(request)
    assert unchanged == request
    assert not disabled.enabled

    redacted, receipt = GatewayRedactionPolicy(enabled=True).apply(request)
    content = redacted.messages[0]["content"]
    assert "[REDACTED_EMAIL]" in content
    assert "[REDACTED]" in content
    assert receipt.changed
    assert receipt.counts == {"email": 1, "openai_api_key": 1}
    assert all(len(item.digest) == 16 for item in receipt.findings)
    assert "a@example.com" not in repr(receipt)
