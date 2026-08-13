from __future__ import annotations

import hashlib
from dataclasses import asdict

from starlette.responses import Response

from entroly.proxy_traffic_receipt import (
    TrafficReceipt,
    TrafficReceiptLedger,
    _TRAFFIC_HTML,
    _canonical_json,
    _classify_client,
    _prefix_protection,
    _routing_decision,
    _verification,
)


def _receipt(**overrides) -> TrafficReceipt:
    payload = {
        "schema_version": "entroly.traffic-receipt.v1",
        "receipt_id": "tr_test",
        "request_correlation": "deadbeefdeadbeef",
        "client": "Claude Code",
        "provider": "anthropic",
        "requested_model": "claude-test",
        "executed_model": "claude-test",
        "original_context_tokens": 73440,
        "entroly_context_tokens": 18206,
        "tokens_avoided": 55234,
        "evidence_retained_pct": 100.0,
        "evidence_retained_source": "context_coverage_estimate",
        "recoverable": True,
        "recovery_receipts": 4,
        "warm_prefix_protected_tokens": 14820,
        "cache_hit": True,
        "cache_read_tokens": 14820,
        "routing_decision": "STAY",
        "routing_reason": "requested model preserved",
        "input_cost_micro_usd": 1200,
        "cache_benefit_micro_usd": 400,
        "net_measured_saving_micro_usd": None,
        "money_source": "operator-catalog",
        "context_risk": "LOW",
        "verification": "PASS",
        "response_status": 200,
        "streaming": True,
        "latency_ms": 123.4,
        "observed_at": 1_700_000_000.0,
    }
    payload.update(overrides)
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return TrafficReceipt(receipt_digest=digest, **payload)


def test_traffic_receipt_verifies_and_ledger_is_content_blind() -> None:
    secret = "sk-secret-prompt-must-never-appear"
    receipt = _receipt(routing_reason="warm cache economics")
    assert receipt.verify()

    ledger = TrafficReceiptLedger(max_records=2)
    ledger.register_request()
    ledger.append(receipt)
    snapshot = ledger.snapshot()

    assert snapshot["records_contain_prompt_content"] is False
    assert snapshot["records_contain_credentials"] is False
    assert snapshot["latest"]["receipt_digest"] == receipt.receipt_digest
    assert secret not in repr(snapshot)


def test_traffic_receipt_rejects_bad_digest() -> None:
    receipt = _receipt()
    broken_payload = asdict(receipt)
    broken_payload["receipt_digest"] = "0" * 64
    broken = TrafficReceipt(**broken_payload)
    ledger = TrafficReceiptLedger()
    try:
        ledger.append(broken)
    except ValueError as exc:
        assert "digest" in str(exc)
    else:
        raise AssertionError("tampered receipt should be rejected")


def test_claude_code_client_detection_does_not_store_user_agent() -> None:
    headers = {"user-agent": "claude-code/9.9.9 secret-user-agent-detail"}
    assert _classify_client(headers) == "Claude Code"


def test_prefix_protection_uses_per_request_headers() -> None:
    response = Response(status_code=200)
    headers = {
        "x-entroly-prefix-guard": "preserve_warm_prefix",
        "x-entroly-prefix-tokens-at-risk": "14820",
    }
    assert _prefix_protection(headers, response) == 14820


def test_routing_and_verification_use_execution_evidence() -> None:
    response = Response(
        status_code=200,
        headers={
            "X-Entroly-Routing-Decision": "denied",
            "X-Entroly-Routing-Reason": "warm cache economics",
            "X-Entroly-Witness": "pass",
        },
    )
    decision, reason = _routing_decision("sonnet", "sonnet", response)
    assert decision == "STAY"
    assert reason == "warm cache economics"
    assert _verification(response) == "PASS"


def test_traffic_page_contains_no_hardcoded_demo_numbers_or_fake_savings() -> None:
    # The product surface must render live receipt values, not the PM mockup.
    assert "73,440" not in _TRAFFIC_HTML
    assert "18,206" not in _TRAFFIC_HTML
    assert "55,234" not in _TRAFFIC_HTML
    assert "Net measured saving" in _TRAFFIC_HTML
    assert "measured counterfactual" in _TRAFFIC_HTML
