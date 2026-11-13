from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from entroly.proxy_traffic_value import (
    _TRAFFIC_VALUE_HTML,
    _default_window,
    _reset_session_state_for_tests,
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
        "estimated_priced_requests": 0,
        "cache_benefit_priced_requests": 0,
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


@pytest.fixture(autouse=True)
def _clean_process_session():
    _reset_session_state_for_tests()
    yield
    _reset_session_state_for_tests()


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
            estimated_priced_requests=420,
            cache_benefit_priced_requests=400,
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
                estimated_priced_requests=90,
                cache_benefit_priced_requests=80,
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
                estimated_priced_requests=40,
                cache_benefit_priced_requests=40,
                provider_priced_requests=40,
            ),
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

    snapshot = build_traffic_value_snapshot(tracker, today=today, now=now)
    thirty = snapshot["windows"]["30d"]

    assert snapshot["default_window"] == "30d"
    assert snapshot["always_show_lifetime"] is True
    assert snapshot["window_order"] == [
        "session",
        "today",
        "7d",
        "30d",
        "60d",
        "90d",
        "lifetime",
    ]
    assert snapshot["windows"]["session"]["requests_observed"] == 0
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


def test_receipt_persistence_and_session_are_idempotent_and_evidence_classified(
    monkeypatch,
) -> None:
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
    assert lifetime["estimated_priced_requests"] == 1
    assert lifetime["cache_benefit_priced_requests"] == 1
    assert lifetime["provider_priced_requests"] == 1
    assert lifetime["verified_requests"] == 1
    assert lifetime["verification_passed"] == 1
    assert lifetime["recovery_invoked"] == 1
    assert lifetime["recovery_succeeded"] == 1
    assert lifetime["warm_cache_protected_tokens"] == 250
    assert lifetime["cache_observed"] == 1
    assert lifetime["cache_hits"] == 1
    assert state["seen_receipts"] == ["tr_same"]
    assert tracker.saved == 1

    session = build_traffic_value_snapshot(tracker)["windows"]["session"]
    assert session["requests_observed"] == 1
    assert session["tokens_avoided"] == 600


def test_traffic_value_ui_has_immediate_proof_without_fake_or_fake_zero_values() -> None:
    for fake in ("84,291", "2.41B", "$4,812", "$11,487", "$14,705"):
        assert fake not in _TRAFFIC_VALUE_HTML

    for label in (
        "Requests optimized",
        "Tokens received by Entroly",
        "Tokens sent to provider",
        "Tokens avoided",
        "Context reduction",
        "Net saving (conservative)",
        "List-price model (ignores caching)",
        "Provider cache discount (measured)",
        "Provider input spend observed",
        "Requests verified",
        "Recovery evidence observed",
        "Warm cache protected",
        "Modeled total (list price + cache discount)",
        "ALL TIME",
        "Session duration",
        "resets on proxy restart",
        "not measured",
    ):
        assert label in _TRAFFIC_VALUE_HTML


# ── the dollar headline is the conservative, cache-aware figure ─────────


def _priced_receipt(receipt_id: str, *, net_micro_usd: int | None, avoided: int = 600):
    from entroly import proxy_traffic_session as session_state

    session_state.remember_receipt_meta(
        receipt_id,
        {"net_saving_conservative_micro_usd": net_micro_usd, "value_contributions": []},
    )
    return SimpleNamespace(
        receipt_id=receipt_id,
        observed_at=_utc_ts(dt.date(2026, 8, 12)),
        executed_model="claude-test",
        original_context_tokens=1000,
        entroly_context_tokens=1000 - avoided,
        tokens_avoided=avoided,
        verification="NOT_REPORTED",
        recovery_receipts=0,
        warm_prefix_protected_tokens=0,
        cache_hit=None,
        input_cost_micro_usd=None,
        cache_benefit_micro_usd=None,
        verify=lambda: True,
    )


@pytest.fixture
def _list_price(monkeypatch):
    """A list-price model that would claim $2.50 per receipt."""
    import entroly.proxy_value_projection  # noqa: F401  (installs the projection)
    from entroly import proxy_traffic_session as session_state

    monkeypatch.setattr("entroly.proxy_traffic_value._has_priced_model", lambda _m: True)
    monkeypatch.setattr(
        "entroly.proxy_traffic_value.estimate_cost", lambda tokens, model, kind="input": 2.5
    )
    yield
    session_state.clear_attribution_state()


def test_headline_is_the_signed_conservative_sum_over_priced_requests_only(_list_price) -> None:
    tracker = _FakeTracker()
    for receipt in (
        _priced_receipt("tr_gain", net_micro_usd=400_000),
        _priced_receipt("tr_loss", net_micro_usd=-150_000),  # Entroly made this one dearer
        _priced_receipt("tr_unpriced", net_micro_usd=None),  # no usage or no verified price
    ):
        assert record_traffic_value_receipt(receipt, tracker=tracker) is True

    snapshot = build_traffic_value_snapshot(tracker)
    assert snapshot["headline_metric"] == "net_saving_conservative_usd"
    for key in ("lifetime", "session"):
        window = snapshot["windows"][key]
        # Losses are subtracted, not clamped, and the unpriced request adds nothing.
        assert window["net_saving_conservative_micro_usd"] == 250_000
        assert window["net_saving_conservative_usd"] == 0.25
        assert window["headline"] == {
            "metric": "net_saving_conservative_usd",
            "usd": 0.25,
            "basis": "provider-calibrated estimate",
            "priced_requests": 2,
            "requests_observed": 3,
            "unpriced_requests": 1,
        }
        assert window["net_saving_coverage_pct"] == 66.67
        # The list-price model is still reported, separately, and is far larger.
        assert window["estimated_value_avoided_usd"] == 7.5
    assert "list-price" in snapshot["truth"]["estimated_value"].lower()
    assert "not the headline" in snapshot["truth"]["estimated_value"]


def test_headline_is_absent_not_zero_and_never_falls_back_to_list_price(_list_price) -> None:
    tracker = _FakeTracker()
    assert record_traffic_value_receipt(
        _priced_receipt("tr_unpriced_only", net_micro_usd=None), tracker=tracker
    )
    lifetime = build_traffic_value_snapshot(tracker)["windows"]["lifetime"]
    assert lifetime["estimated_value_avoided_usd"] == 2.5
    assert lifetime["net_saving_conservative_usd"] is None
    assert lifetime["headline"]["usd"] is None
    assert lifetime["headline"]["priced_requests"] == 0
    assert lifetime["headline"]["unpriced_requests"] == 1


def test_headline_can_be_negative(_list_price) -> None:
    tracker = _FakeTracker()
    assert record_traffic_value_receipt(
        _priced_receipt("tr_only_loss", net_micro_usd=-90_000, avoided=0), tracker=tracker
    )
    lifetime = build_traffic_value_snapshot(tracker)["windows"]["lifetime"]
    assert lifetime["net_saving_conservative_usd"] == -0.09
    assert lifetime["headline"]["usd"] == -0.09


def test_a_priced_zero_is_a_claim_of_zero_not_missing(_list_price) -> None:
    tracker = _FakeTracker()
    assert record_traffic_value_receipt(
        _priced_receipt("tr_zero", net_micro_usd=0, avoided=0), tracker=tracker
    )
    lifetime = build_traffic_value_snapshot(tracker)["windows"]["lifetime"]
    assert lifetime["net_saving_conservative_usd"] == 0.0
    assert lifetime["headline"]["priced_requests"] == 1


def test_stored_rollups_from_before_the_headline_load_as_unpriced() -> None:
    old = _metrics(requests_observed=40, tokens_avoided=9_000, estimated_value_avoided_usd=12.0,
                   estimated_priced_requests=40)
    tracker = _FakeTracker({"started_at": 1.0, "lifetime": old, "daily": {}, "seen_receipts": []})
    lifetime = build_traffic_value_snapshot(tracker)["windows"]["lifetime"]
    assert lifetime["net_saving_conservative_usd"] is None
    assert lifetime["headline"]["unpriced_requests"] == 40


def test_dashboard_leads_with_the_headline_and_explains_missing_prices() -> None:
    html = _TRAFFIC_VALUE_HTML
    assert html.index("${headlineBox(w)}") < html.index("List-price model (ignores caching)")
    assert "No dollar claim" in html
    assert "not a substitute" in html
    assert "priced ${fmt(priced)} of ${fmt(seen)} requests" in html
