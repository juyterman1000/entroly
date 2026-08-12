from __future__ import annotations

import json

import entroly.context_health as context_health
import entroly.dashboard as dashboard


def test_dashboard_exposes_context_health_panel_and_refresh_loop():
    assert 'id="contextHealth"' in dashboard.DASHBOARD_HTML
    assert "function renderContextHealth" in dashboard.DASHBOARD_HTML
    assert "fetch('/api/context/health')" in dashboard.DASHBOARD_HTML
    assert "Copy privacy-safe proof" in dashboard.DASHBOARD_HTML
    assert "Zero means no event observed" in dashboard.DASHBOARD_HTML
    assert "retrieval_adjusted_net_tokens||0)<0?'hv-rose':'hv-green'" in dashboard.DASHBOARD_HTML
    assert "Provider request cache hits" in dashboard.DASHBOARD_HTML
    assert "Cached input-token ratio" in dashboard.DASHBOARD_HTML
    assert "Optimizer interference guard" in dashboard.DASHBOARD_HTML
    assert "Requires a paired baseline" in dashboard.DASHBOARD_HTML


def test_context_health_endpoint_returns_aggregate_report(monkeypatch):
    expected = {
        "schema_version": "entroly.context-health.v2",
        "privacy": {"content_blind": True},
    }
    captured = {}

    def build_context_health(*, provider_economics):
        captured.update(provider_economics)
        return expected

    monkeypatch.setattr(context_health, "build_context_health", build_context_health)
    monkeypatch.setattr(
        dashboard,
        "_fetch_context_economics_snapshot",
        lambda: {"provider_cache": {"observed_hits": 2}},
    )
    handler = object.__new__(dashboard.DashboardHandler)
    responses: list[tuple[int, dict]] = []
    handler._send_json = lambda status, payload: responses.append((status, payload))

    handler._handle_context_health()

    assert responses == [(200, expected)]
    assert captured == {"provider_cache": {"observed_hits": 2}}


def test_context_health_endpoint_fails_without_leaking_error_text(monkeypatch):
    def fail(*, provider_economics):
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


def test_proxy_economics_fetcher_filters_paths_content_and_identifiers(monkeypatch):
    private = "C:/customer/private-prompt.py"
    payload = {
        "provider_cache": {
            "observed_hits": 3,
            "observed_misses": 1,
            "last_decision": {"model": "private-model", "query": private},
        },
        "usage_accounting": {
            "enabled": True,
            "pricing_catalog": private,
            "live": {
                "scope": "process_local_content_blind",
                "requests": 4,
                "cache_read_tokens": 300,
                "private": private,
            },
            "ledger": {
                "requests": 4,
                "cache_read_tokens": 300,
                "private": private,
            },
        },
        "optimizer_interference": {
            "guard_interventions": 2,
            "private": private,
        },
        "last_query": private,
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())

    filtered = dashboard._fetch_context_economics_snapshot()

    assert filtered["provider_cache"] == {
        "observed_hits": 3,
        "observed_misses": 1,
    }
    assert filtered["usage_accounting"]["ledger"]["requests"] == 4
    assert filtered["usage_accounting"]["live"]["requests"] == 4
    assert filtered["optimizer_interference"]["guard_interventions"] == 2
    assert private not in json.dumps(filtered, sort_keys=True)
    assert "private-model" not in json.dumps(filtered, sort_keys=True)
