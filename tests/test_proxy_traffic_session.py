from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import entroly.proxy_traffic_session as session_compat
import entroly.proxy_traffic_value as value


class _FakeTracker:
    def __init__(self, *, started_at: float = 0.0):
        self._data = {
            "version": 4,
            "traffic_assurance": {
                "started_at": started_at,
                "lifetime": value._empty_metrics(),
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
def _clean_session():
    value._reset_session_state_for_tests(started_at=1000.0)
    yield
    value._reset_session_state_for_tests()


def _receipt(receipt_id: str = "tr_session", *, observed_at: float = 1500.0):
    return SimpleNamespace(
        receipt_id=receipt_id,
        observed_at=observed_at,
        verify=lambda: True,
        original_context_tokens=1000,
        entroly_context_tokens=400,
        tokens_avoided=600,
        executed_model="unknown-test-model",
        verification="PASS",
        recovery_receipts=0,
        cache_hit=True,
        input_cost_micro_usd=2000,
        cache_benefit_micro_usd=500,
        warm_prefix_protected_tokens=120,
    )


def test_session_value_is_native_immediate_and_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(value, "_has_priced_model", lambda _model: False)
    tracker = _FakeTracker()

    assert value.record_traffic_value_receipt(_receipt(), tracker=tracker) is True
    assert value.record_traffic_value_receipt(_receipt(), tracker=tracker) is False

    rollup = value._session_rollup(now=1600.0)
    assert rollup["requests_observed"] == 1
    assert rollup["requests_optimized"] == 1
    assert rollup["tokens_received"] == 1000
    assert rollup["tokens_sent"] == 400
    assert rollup["tokens_avoided"] == 600
    assert rollup["context_reduction_pct"] == 60.0
    assert rollup["warm_cache_protected_tokens"] == 120
    assert rollup["cache_hit_request_pct"] == 100.0
    assert rollup["session_elapsed_seconds"] == 600
    assert rollup["session_status"] == "measurable"
    assert "measurable value" in rollup["session_status_message"]


def test_first_day_defaults_to_session_but_all_time_stays_visible(monkeypatch) -> None:
    monkeypatch.setattr(value, "pricing_provenance", lambda: {"source": "test"})
    tracker = _FakeTracker(started_at=1500.0)
    assert value.record_traffic_value_receipt(
        _receipt("tr_fresh", observed_at=1500.0),
        tracker=tracker,
    )

    snapshot = value.build_traffic_value_snapshot(tracker, now=1600.0)
    assert snapshot["default_window"] == "session"
    assert snapshot["window_order"][0] == "session"
    assert snapshot["windows"]["session"]["tokens_avoided"] == 600
    assert "lifetime" in snapshot["windows"]
    assert "resets on proxy restart" in snapshot["truth"]["session"]


def test_mature_install_keeps_age_adaptive_default_with_session_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(value, "pricing_provenance", lambda: {"source": "test"})
    now = 10_000_000.0
    tracker = _FakeTracker(started_at=now - 45 * 86400)
    assert value.record_traffic_value_receipt(
        _receipt("tr_old", observed_at=now),
        tracker=tracker,
    )

    snapshot = value.build_traffic_value_snapshot(tracker, now=now)
    assert snapshot["default_window"] == "30d"
    assert snapshot["window_order"][0] == "session"
    assert snapshot["windows"]["session"]["requests_optimized"] == 1


def test_session_reports_waiting_without_claiming_value() -> None:
    rollup = value._session_rollup(now=1600.0)
    assert rollup["requests_observed"] == 0
    assert rollup["session_status"] == "waiting"
    assert "Send traffic through Entroly" in rollup["session_status_message"]


def test_compatibility_module_does_not_monkeypatch_value_ownership() -> None:
    record_before = value.record_traffic_value_receipt
    snapshot_before = value.build_traffic_value_snapshot

    assert session_compat.install_session_value() is None

    assert value.record_traffic_value_receipt is record_before
    assert value.build_traffic_value_snapshot is snapshot_before
    assert session_compat._session_rollup(now=1600.0)["key"] == "session"
