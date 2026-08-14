from __future__ import annotations

import hashlib

from entroly import proxy_traffic_receipt as traffic
from entroly import proxy_traffic_session as attribution


def _unsigned() -> dict:
    return {
        "schema_version": "entroly.traffic-receipt.v1",
        "receipt_id": "tr_integrity",
        "request_correlation": "corr",
        "client": "AI client",
        "provider": "openai",
        "requested_model": "test-model",
        "executed_model": "test-model",
        "original_context_tokens": 1000,
        "entroly_context_tokens": 400,
        "tokens_avoided": 600,
        "evidence_retained_pct": None,
        "evidence_retained_source": "withheld_shared_state",
        "recoverable": False,
        "recovery_receipts": 0,
        "warm_prefix_protected_tokens": 0,
        "cache_hit": None,
        "cache_read_tokens": 0,
        "routing_decision": "STAY",
        "routing_reason": "requested model preserved",
        "input_cost_micro_usd": None,
        "cache_benefit_micro_usd": None,
        "net_measured_saving_micro_usd": None,
        "money_source": "provider_usage_unavailable",
        "context_risk": "UNKNOWN",
        "verification": "NOT_REPORTED",
        "response_status": 200,
        "streaming": False,
        "latency_ms": 1.0,
        "observed_at": 1.0,
    }


def test_attribution_metadata_is_inside_receipt_digest() -> None:
    attribution.clear_attribution_state()
    attribution.remember_receipt_meta(
        "tr_integrity",
        {
            "attribution_schema_version": attribution.ATTRIBUTION_SCHEMA,
            "attribution_reconciled": True,
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
                }
            ],
        },
    )
    unsigned = _unsigned()
    draft = traffic.TrafficReceipt(receipt_digest="", **unsigned)
    digest = hashlib.sha256(traffic._canonical_json(draft.payload())).hexdigest()
    receipt = traffic.TrafficReceipt(receipt_digest=digest, **unsigned)

    assert receipt.verify()
    assert receipt.payload()["attribution_reconciled"] is True

    attribution.remember_receipt_meta(
        "tr_integrity",
        {
            "attribution_schema_version": attribution.ATTRIBUTION_SCHEMA,
            "attribution_reconciled": False,
            "value_contributions": [],
        },
    )
    assert not receipt.verify()
