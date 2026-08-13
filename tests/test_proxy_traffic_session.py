from __future__ import annotations

from types import SimpleNamespace

import entroly.proxy_traffic_session as session


def _receipt(receipt_id: str = "tr_session") -> SimpleNamespace:
    return SimpleNamespace(
        receipt_id=receipt_id,
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


def _base_snapshot(installed_days: int) -> dict:
    return {
        "schema_version": "entroly.traffic-value.v2",
        "installed_days": installed_days,
        "default_window": "7d",
        "windows": {
            "today": {"key": "today", "label": "Today"},
            "7d": {"key": "7d", "label": "7 days"},
            "lifetime": {"key": "lifetime", "label": "All time"},
        },
        "window_order": ["today", "7d", "lifetime"],
        "truth": {},
    }


def test_session_value_is_immediate_and_idempotent(monkeypatch) -> None:
    session._reset_session_state_for_tests(started_at=1000.0)
    monkeypatch.setattr(session, "_ORIGINAL_RECORD", lambda receipt, tracker=None: True)

    assert session._record_with_session(_receipt()) is True
    assert session._record_with_session(_receipt()) is True  # durable layer may accept; session must not double count

    rollup = session._session_rollup(now=1600.0)
    assert rollup["requests_observed"] == 1
    assert rollup["requests_optimized"] == 1
    assert rollup["tokens_received"] == 1000
    assert rollup["tokens_sent"] == 400
    assert rollup["tokens_avoided"] == 600
    assert rollup["context_reduction_pct"] == 60.0
    assert rollup["warm_cache_protected_tokens"] == 120
    assert rollup["cache_hit_request_pct"] == 100.0
    assert rollup["session_elapsed_seconds"] == 600


def test_first_day_defaults_to_session_but_all_time_stays_visible(monkeypatch) -> None:
    session._reset_session_state_for_tests(started_at=1000.0)
    session._record_session_receipt(_receipt("tr_fresh"))
    monkeypatch.setattr(
        session,
        "_ORIGINAL_SNAPSHOT",
        lambda tracker=None, today=None, now=None: _base_snapshot(installed_days=0),
    )

    snapshot = session._snapshot_with_session(now=1600.0)
    assert snapshot["default_window"] == "session"
    assert snapshot["window_order"][0] == "session"
    assert snapshot["windows"]["session"]["tokens_avoided"] == 600
    assert "lifetime" in snapshot["windows"]
    assert "resets on proxy restart" in snapshot["truth"]["session"]


def test_older_install_keeps_age_adaptive_default_with_session_available(monkeypatch) -> None:
    session._reset_session_state_for_tests(started_at=1000.0)
    session._record_session_receipt(_receipt("tr_old"))
    monkeypatch.setattr(
        session,
        "_ORIGINAL_SNAPSHOT",
        lambda tracker=None, today=None, now=None: _base_snapshot(installed_days=3),
    )

    snapshot = session._snapshot_with_session(now=1600.0)
    assert snapshot["default_window"] == "7d"
    assert snapshot["window_order"][0] == "session"
    assert snapshot["windows"]["session"]["requests_optimized"] == 1
