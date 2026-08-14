from __future__ import annotations

import pytest

from entroly import proxy_traffic_session as attribution
from entroly.usage_ledger import TokenUsage, UsageEvent, UsageLedger


def test_public_contribution_cannot_claim_measured_or_headline_value() -> None:
    state = attribution.AttributionState(request_id="outer")
    token = attribution.CURRENT_ATTRIBUTION.set(state)
    try:
        assert attribution.record_value_contribution(
            "custom optimizer",
            tokens=123,
            micro_usd=456,
            tier="measured",
            details={"count": 2, "content": "must-not-survive"},
        )
    finally:
        attribution.CURRENT_ATTRIBUTION.reset(token)

    item = state.contributions[0]
    assert item.source == "custom_optimizer"
    assert item.tier is attribution.ValueTier.ESTIMATED
    assert item.role is attribution.AccountingRole.EXPLANATORY
    assert item.headline_included is False
    assert item.details == {"count": 2}


def test_headline_role_and_signed_adjustment_invariants() -> None:
    with pytest.raises(ValueError):
        attribution.ValueContribution(
            "bad",
            attribution.ValueTier.MEASURED,
            attribution.AccountingRole.EXPLANATORY,
            tokens=1,
            headline_included=True,
        )
    with pytest.raises(ValueError):
        attribution.ValueContribution(
            "bad",
            attribution.ValueTier.MEASURED,
            attribution.AccountingRole.ADJUSTMENT,
            tokens=1,
        )

    rows = attribution.aggregate_contributions(
        [
            attribution.ValueContribution(
                "context_optimization",
                attribution.ValueTier.MEASURED,
                attribution.AccountingRole.ADDITIVE,
                tokens=600,
                headline_included=True,
            ),
            attribution.ValueContribution(
                "recovery_adjustment",
                attribution.ValueTier.MEASURED,
                attribution.AccountingRole.ADJUSTMENT,
                tokens=-40,
            ),
        ]
    )
    assert sum(row["tokens"] for row in rows if row["headline_included"]) == 600
    assert next(row for row in rows if row["source"] == "recovery_adjustment")[
        "tokens"
    ] == -40


def test_extra_provider_usage_is_request_local_cost_debit() -> None:
    attribution.clear_attribution_state()
    state = attribution.AttributionState(request_id="outer-request")
    ledger = UsageLedger()
    token = attribution.CURRENT_ATTRIBUTION.set(state)
    try:
        assert ledger.record(
            UsageEvent(
                request_id="internal-call",
                provider="openai",
                model="test-model",
                usage=TokenUsage(uncached_input_tokens=80, output_tokens=20),
                cost_micro_usd=2500,
                cache_savings_micro_usd=0,
                pricing_source="test-catalog:openai:test-model",
            )
        )
        assert ledger.record(
            UsageEvent(
                request_id="outer-request",
                provider="openai",
                model="test-model",
                usage=TokenUsage(uncached_input_tokens=10, output_tokens=5),
                cost_micro_usd=500,
                cache_savings_micro_usd=0,
                pricing_source="test-catalog:openai:test-model",
            )
        )
    finally:
        attribution.CURRENT_ATTRIBUTION.reset(token)
        ledger.close()

    assert state.extra_provider_calls == 1
    assert state.extra_provider_tokens == 100
    assert state.extra_provider_cost_micro_usd == 2500
    rows = attribution.aggregate_contributions(state.contributions)
    extra = next(row for row in rows if row["source"] == "extra_provider_call")
    assert extra["tier"] == "measured"
    assert extra["role"] == "adjustment"
    assert extra["micro_usd"] == -2500
    assert extra["tokens"] == 0
    assert extra["evidence_source"] == "provider_usage"
