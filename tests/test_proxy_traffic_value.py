from __future__ import annotations

import datetime as dt

from entroly.proxy_traffic_value import (
    _TRAFFIC_VALUE_HTML,
    build_traffic_value_snapshot,
)


class _FakeTracker:
    def reload_if_changed(self) -> bool:
        return False

    def get_daily(self, last_n: int = 90):
        assert last_n == 90
        return [
            {
                "date": "2026-08-12",
                "provider_tokens_saved": 100,
                "provider_cost_avoided_usd": 1.25,
                "provider_requests": 2,
                "provider_requests_optimized": 2,
                "provider_unpriced_tokens": 0,
                "provider_unpriced_requests": 0,
                "local_tokens_reduced": 50,
                "local_operations": 1,
            },
            {
                "date": "2026-08-06",
                "provider_tokens_saved": 300,
                "provider_cost_avoided_usd": 3.75,
                "provider_requests": 3,
                "provider_requests_optimized": 3,
                "provider_unpriced_tokens": 25,
                "provider_unpriced_requests": 1,
                "local_tokens_reduced": 70,
                "local_operations": 2,
            },
            {
                "date": "2026-07-14",
                "provider_tokens_saved": 700,
                "provider_cost_avoided_usd": 7.00,
                "provider_requests": 4,
                "provider_requests_optimized": 4,
                "provider_unpriced_tokens": 0,
                "provider_unpriced_requests": 0,
                "local_tokens_reduced": 90,
                "local_operations": 3,
            },
            {
                "date": "2026-06-14",
                "provider_tokens_saved": 900,
                "provider_cost_avoided_usd": 9.00,
                "provider_requests": 5,
                "provider_requests_optimized": 5,
                "provider_unpriced_tokens": 0,
                "provider_unpriced_requests": 0,
                "local_tokens_reduced": 110,
                "local_operations": 4,
            },
            {
                "date": "2026-05-14",
                "provider_tokens_saved": 1100,
                "provider_cost_avoided_usd": 11.00,
                "provider_requests": 6,
                "provider_requests_optimized": 6,
                "provider_unpriced_tokens": 0,
                "provider_unpriced_requests": 0,
                "local_tokens_reduced": 130,
                "local_operations": 5,
            },
        ]

    def get_lifetime(self):
        return {
            "provider_tokens_saved": 9999,
            "provider_cost_avoided_usd": 123.456,
            "provider_requests": 42,
            "provider_requests_optimized": 40,
            "provider_unpriced_tokens": 25,
            "provider_unpriced_requests": 1,
            "local_tokens_reduced": 777,
            "local_operations": 12,
        }


def test_traffic_value_has_rolling_windows_and_lifetime(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.proxy_traffic_value.pricing_provenance",
        lambda: {"as_of": "2026-08", "source": "test-catalog"},
    )
    snapshot = build_traffic_value_snapshot(
        _FakeTracker(),
        today=dt.date(2026, 8, 12),
    )

    assert snapshot["window_order"] == ["today", "7d", "30d", "60d", "90d", "lifetime"]
    assert snapshot["default_window"] == "lifetime"
    assert snapshot["windows"]["today"]["provider_tokens_avoided"] == 100
    assert snapshot["windows"]["7d"]["provider_tokens_avoided"] == 400
    assert snapshot["windows"]["30d"]["provider_tokens_avoided"] == 1100
    assert snapshot["windows"]["60d"]["provider_tokens_avoided"] == 2000
    assert snapshot["windows"]["90d"]["provider_tokens_avoided"] == 3100
    assert snapshot["windows"]["lifetime"]["provider_tokens_avoided"] == 9999
    assert snapshot["windows"]["lifetime"]["estimated_input_value_avoided_usd"] == 123.456
    assert snapshot["pricing"]["source"] == "test-catalog"


def test_traffic_value_does_not_mix_local_tokens_into_dollar_value(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.proxy_traffic_value.pricing_provenance",
        lambda: {"as_of": "2026-08", "source": "test-catalog"},
    )
    snapshot = build_traffic_value_snapshot(
        _FakeTracker(),
        today=dt.date(2026, 8, 12),
    )
    today = snapshot["windows"]["today"]
    assert today["estimated_input_value_avoided_usd"] == 1.25
    assert today["local_tokens_reduced"] == 50
    assert "not a provider invoice" in snapshot["truth"]["estimated_usd"]


def test_traffic_value_ui_has_no_hardcoded_demo_savings() -> None:
    assert "73,440" not in _TRAFFIC_VALUE_HTML
    assert "$4,812" not in _TRAFFIC_VALUE_HTML
    assert "Today" not in _TRAFFIC_VALUE_HTML  # labels come from the live API
    assert "estimated provider input value avoided" in _TRAFFIC_VALUE_HTML
    assert "not presented as invoice-verified savings" in _TRAFFIC_VALUE_HTML
