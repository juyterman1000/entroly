from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from types import SimpleNamespace

from entroly.proxy_traffic_value import (
    _TRAFFIC_VALUE_HTML,
    _default_window,
    build_traffic_value_snapshot,
    record_traffic_value_receipt,
)


def _metrics(**overrides):
    row = {
        "requests_observed": 0,
        "requests_optimized": 0,
        "tokens_received": 0,
        "tokens_sent": 0,
        "tokens_avoided": 0,
        "estimated_value_avoided_usd": 0.0,
        "measured_cache_benefit_usd": 0.0,
        "provider_input_spend_usd": 0.0,
        "verified_requests": 0,
        "verification_passed": 0,
        "recovery_invoked": 0,
        "recovery_succeeded": 0,
        "warm_cache_protected_tokens": 0,
        "cache_observed": 0,
        "cache_hits": 0,
        "provider_priced_requests": 0,
    }
    row.update(overrides)
    return row


class _FakeTracker:
    def __init__(self, state: dict | None = None):
        self._data = {
            "version": 4,
            "traffic_assurance": state
            or {
                "started_at": 0.0,
                "lifetime": _metrics(),
                "daily": {},
                "seen_receipts": [],
            },
        }
        self.saved = 0

    def reload_if_changed(self) -> bool:
        return False

    @contextmanager
    def _mutation(self):
        yield

    def _save(self) -> None:
        self.saved += 1


def _utc_ts(day: dt.date) -> float:
    return dt.datetime.combine(
        day,
        dt.time(12, 0),
        tzinfo=dt.timezone.utc,
    ).timestamp()


def test_default_window_adapts_to_collection_age() -> None:
    assert _default_window(3) == "7d"
    assert _default_window(29) == "7d"
    assert _default_window(45) == "30d"
    assert _default_window(70) == "60d"
    assert _default_window(100) == "90d"
    assert _default_window(365) == "90d"


def test_traffic_value_rolls_up_full_30_day_executive_surface(monkeypatch) -> None:
    today = dt.date(2026, 8, 12)
    now = _utc_ts(today)
    state = {
        "started_at": now - 45 * 86400,
        "lifetime": _metrics(
            requests_observed=500,
            requests_optimized=420,
            tokens_received=2_000_000,
            tokens_sent=700_000,
            tokens_avoided=1_300_000,
            estimated_value_avoided_usd=420.0,
            measured_cache_benefit_usd=80.0,
            provider_input_spend_usd=210.0,
            verified_requests=490,
            verification_passed=485,
            recovery_invoked=20,
            recovery_succeeded=19,
            warm_cache_protected_tokens=300_000,
            cache_observed=400,
            cache_hits=320,
            provider_priced_requests=400,
        ),
        "daily": {
            "2026-08-12": _metrics(
                requests_observed=100,
                requests_optimized=90,
                tokens_received=1_000,
                tokens_sent=400,
                tokens_avoided=600,
                estimated_value_avoided_usd=6.0,
                measured_cache_benefit_usd=2.0,
                provider_input_spend_usd=3.0,
                verified_requests=99,
                verification_passed=98,
                recovery_invoked=10,
                recovery_succeeded=9,
                warm_cache_protected_tokens=200,
                cache_observed=80,
                cache_hits=60,
                provider_priced_requests=80,
            ),
            "2026-07-20": _metrics(
                requests_observed=50,
                requests_optimized=40,
                tokens_received=500,
                tokens_sent=200,
                tokens_avoided=300,
                estimated_value_avoided_usd=3.0,
                measured_cache_benefit_usd=1.0,
                provider_input_spend_usd=1.5,
                verified_requests=49,
                verification_passed=48,
                recovery_invoked=5,
                recovery_succeeded=5,
                warm_cache_protected_tokens=100,
                cache_observed=40,
                cache_hits=30,
                provider_priced_requests=40,
            ),
            # Outside the rolling 30-day window.
            "2026-07-01": _metrics(
                requests_observed=999,
                requests_optimized=999,
                tokens_received=9_999,
                tokens_sent=1,
                tokens_avoided=9_998,
                estimated_value_avoided_usd=999.0,
            ),
        },
        "seen_receipts": [],
    }
    tracker = _FakeTracker(state)
    monkeypatch.setattr(
        "entroly.proxy_traffic_value.pricing_provenance",
        lambda: {"as_of": "2026-08", "source": "test-catalog"},
    )

    snapshot = build_traffic_value_snapshot(
        tracker,
        today=today,
        now=now,
    )
    thirty = snapshot["windows"]["30d"]

    assert snapshot["default_window"] == "30d"
    assert snapshot["always_show_lifetime"] is True
    assert snapshot["window_order"] == [
        "today",
        "7d",
        "30d",
        "60d",
        "90d",
        "lifetime",
    ]
    assert thirty["requests_optimized"] == 130
    assert thirty["tokens_received"] == 1500
    assert thirty["tokens_sent"] == 600
    assert thirty["tokens_avoided"] == 900
    assert thirty["context_reduction_pct"] == 60.0
    assert thirty["estimated_value_avoided_usd"] == 9.0
    assert thirty["measured_cache_benefit_usd"] == 3.0
    assert thirty["provider_input_spend_usd"] == 4.5
    assert thirty["requests_verified_pct"] == 98.67
    assert thirty["recovery_invoked_pct"] == 10.0
    assert thirty["recovery_succeeded_pct"] == 93.33
    assert thirty["warm_cache_protected_tokens"] == 300
    assert thirty["cache_hit_request_pct"] == 75.0
    assert thirty["total_ai_value_protected_usd"] == 12.0

    lifetime = snapshot["windows"]["lifetime"]
    assert lifetime["estimated_value_avoided_usd"] == 420.0
    assert lifetime["measured_cache_benefit_usd"] == 80.0
    assert lifetime["total_ai_value_protected_usd"] == 500.0
    assert snapshot["pricing"]["source"] == "test-catalog"


def test_receipt_persistence_is_idempotent_and_evidence_classified(monkeypatch) -> None:
    tracker = _FakeTracker()
    monkeypatch.setattr(
        "entroly.proxy_traffic_value._has_priced_model",
        lambda _model: True,
    )
    monkeypatch.setattr(
        "entroly.proxy_traffic_value.estimate_cost",
        lambda tokens, model, kind="input": 2.5 if tokens == 600 else 0.0,
    )

    receipt = SimpleNamespace(
        receipt_id="tr_same",
        observed_at=_utc_ts(dt.date(2026, 8, 12)),
        executed_model="claude-test",
        original_context_tokens=1000,
        entroly_context_tokens=400,
        tokens_avoided=600,
        verification="PASS",
        recovery_receipts=2,
        warm_prefix_protected_tokens=250,
        cache_hit=True,
        input_cost_micro_usd=1_250_000,
        cache_benefit_micro_usd=300_000,
        verify=lambda: True,
    )

    assert record_traffic_value_receipt(receipt, tracker=tracker) is True
    assert record_traffic_value_receipt(receipt, tracker=tracker) is False

    state = tracker._data["traffic_assurance"]
    lifetime = state["lifetime"]
    assert lifetime["requests_observed"] == 1
    assert lifetime["requests_optimized"] == 1
    assert lifetime["tokens_received"] == 1000
    assert lifetime["tokens_sent"] == 400
    assert lifetime["tokens_avoided"] == 600
    assert lifetime["estimated_value_avoided_usd"] == 2.5
    assert lifetime["measured_cache_benefit_usd"] == 0.3
    assert lifetime["provider_input_spend_usd"] == 1.25
    assert lifetime["verified_requests"] == 1
    assert lifetime["verification_passed"] == 1
    assert lifetime["recovery_invoked"] == 1
    assert lifetime["recovery_succeeded"] == 1
    assert lifetime["warm_cache_protected_tokens"] == 250
    assert lifetime["cache_observed"] == 1
    assert lifetime["cache_hits"] == 1
    assert state["seen_receipts"] == ["tr_same"]
    assert tracker.saved == 1


def test_traffic_value_ui_has_requested_exec_metrics_without_fake_values() -> None:
    # Product surface renders live API values, never the illustrative PM mockup.
    for fake in ("84,291", "2.41B", "$4,812", "$11,487", "$14,705"):
        assert fake not in _TRAFFIC_VALUE_HTML

    for label in (
        "Requests optimized",
        "Tokens received",
        "Tokens sent",
        "Tokens avoided",
        "Context reduction",
        "Estimated value avoided",
        "Measured cache benefit",
        "Provider input spend",
        "Requests verified",
        "Recovery invoked",
        "Recovery succeeded",
        "Warm cache protected",
        "Total AI value protected",
        "ALL TIME",
    ):
        assert label in _TRAFFIC_VALUE_HTML
