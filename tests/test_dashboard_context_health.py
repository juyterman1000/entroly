from __future__ import annotations

import entroly.context_health as context_health
import entroly.dashboard as dashboard


def test_dashboard_exposes_context_health_panel_and_refresh_loop():
    assert 'id="contextHealth"' in dashboard.DASHBOARD_HTML
    assert "function renderContextHealth" in dashboard.DASHBOARD_HTML
    assert "fetch('/api/context/health')" in dashboard.DASHBOARD_HTML
    assert "Copy privacy-safe proof" in dashboard.DASHBOARD_HTML
    assert "Zero means no event observed" in dashboard.DASHBOARD_HTML
    assert "measured_net_tokens||0)<0?'hv-rose':'hv-green'" in dashboard.DASHBOARD_HTML


def test_context_health_endpoint_returns_aggregate_report(monkeypatch):
    expected = {
        "schema_version": "entroly.context-health.v1",
        "privacy": {"content_blind": True},
    }
    monkeypatch.setattr(context_health, "build_context_health", lambda: expected)
    handler = object.__new__(dashboard.DashboardHandler)
    responses: list[tuple[int, dict]] = []
    handler._send_json = lambda status, payload: responses.append((status, payload))

    handler._handle_context_health()

    assert responses == [(200, expected)]


def test_context_health_endpoint_fails_without_leaking_error_text(monkeypatch):
    def fail():
        raise RuntimeError("private path C:/customer/secret.py")

    monkeypatch.setattr(context_health, "build_context_health", fail)
    handler = object.__new__(dashboard.DashboardHandler)
    responses: list[tuple[int, dict]] = []
    handler._send_json = lambda status, payload: responses.append((status, payload))

    handler._handle_context_health()

    assert responses == [
        (
            503,
            {
                "error": "context health is temporarily unavailable",
                "type": "RuntimeError",
            },
        )
    ]
    assert "secret.py" not in str(responses)
