from __future__ import annotations

from types import SimpleNamespace

from entroly import proxy_traffic_session as attribution
from entroly import proxy_traffic_value as value
from entroly import proxy_value_projection as projection


def _receipt(receipt_id: str = "tr_projection") -> SimpleNamespace:
    return SimpleNamespace(
        receipt_id=receipt_id,
        observed_at=1000.0,
        original_context_tokens=1000,
        entroly_context_tokens=400,
        tokens_avoided=600,
        executed_model="unknown-test-model",
        verification="PASS",
        recovery_receipts=0,
        cache_hit=True,
        input_cost_micro_usd=2000,
        cache_benefit_micro_usd=500,
        warm_prefix_protected_tokens=0,
    )


def test_projection_preserves_headline_and_subtracts_observed_overhead() -> None:
    attribution.clear_attribution_state()
    receipt = _receipt()
    attribution.remember_receipt_meta(
        receipt.receipt_id,
        {
            "attribution_reconciled": True,
            "extra_provider_calls": 1,
            "extra_provider_tokens": 100,
            "extra_provider_cost_micro_usd": 2500,
            "value_contributions": [
                {
                    "source": "context_optimization",
                    "tier": "measured",
                    "role": "additive",
                    "evidence_source": "local_observation",
                    "headline_included": True,
                    "events": 1,
                    "tokens": 600,
                    "micro_usd": 0,
                    "priced_events": 0,
                },
                {
                    "source": "extra_provider_call",
                    "tier": "measured",
                    "role": "adjustment",
                    "evidence_source": "provider_usage",
                    "headline_included": False,
                    "events": 1,
                    "tokens": 0,
                    "micro_usd": -2500,
                    "priced_events": 1,
                },
            ],
        },
    )

    delta = value._receipt_delta(receipt)
    assert delta["tokens_avoided"] == 600
    assert delta["extra_provider_calls"] == 1
    assert delta["extra_provider_cost_usd"] == 0.0025

    metrics = value._empty_metrics()
    value._accumulate(metrics, delta)
    result = value._finalize_metrics(metrics)
    assert result["tokens_avoided"] == 600
    assert result["measured_cache_benefit_usd"] == 0.0005
    assert result["net_value_after_observed_extra_provider_cost_usd"] == -0.002
    assert {row["source"] for row in result["value_by_source"]} == {
        "context_optimization",
        "extra_provider_call",
    }


def test_prometheus_uses_bounded_source_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        projection._value,
        "build_traffic_value_snapshot",
        lambda: {
            "windows": {
                "lifetime": {
                    "extra_provider_cost_usd": 0.0025,
                    "value_by_source": [
                        {
                            "source": "unbounded-custom-source",
                            "tier": "estimated",
                            "role": "explanatory",
                            "evidence_source": "modelled",
                            "events": 1,
                            "tokens": 123,
                            "micro_usd": 0,
                            "priced_events": 0,
                        }
                    ],
                }
            }
        },
    )
    text = projection._prometheus_rows()
    assert 'source="other"' in text
    assert "unbounded-custom-source" not in text
    assert "entroly_value_extra_provider_cost_usd 0.002500" in text


def test_dashboard_is_a_projection_not_a_second_calculator() -> None:
    assert "Value attribution by source" in value._TRAFFIC_VALUE_HTML
    assert "value_by_source" in value._TRAFFIC_VALUE_HTML
