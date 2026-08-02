"""Offline-safe local runtime status reporting.

This module only probes loopback Entroly endpoints. It never contacts provider
APIs and never includes raw exception messages, credentials, or local paths in
its machine-readable output.
"""

from __future__ import annotations

import json
from typing import Any, Callable
import urllib.error
import urllib.request

SCHEMA_VERSION = "entroly.runtime-status.v1"


def _probe_json(
    url: str,
    *,
    timeout: float,
    expected_service: str | None = None,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    try:
        with opener(url, timeout=timeout) as response:
            status_code = int(getattr(response, "status", 0) or 0)
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"ok": False, "state": "http_error", "status_code": int(exc.code)}
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return {"ok": False, "state": "unreachable", "status_code": None}
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"ok": False, "state": "invalid_response", "status_code": None}

    if not isinstance(payload, dict):
        return {"ok": False, "state": "invalid_response", "status_code": status_code}
    if status_code != 200:
        return {"ok": False, "state": "http_error", "status_code": status_code}
    if expected_service is not None and (
        payload.get("status") != "ok" or payload.get("service") != expected_service
    ):
        return {"ok": False, "state": "identity_mismatch", "status_code": status_code}
    return {
        "ok": True,
        "state": "ready",
        "status_code": status_code,
        "payload": payload,
    }


def runtime_status(
    port: int = 9377,
    *,
    timeout: float = 2.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    """Return a stable, sanitized report for local Entroly services."""
    proxy_url = f"http://127.0.0.1:{port}"
    dashboard_url = "http://127.0.0.1:9378"

    proxy = _probe_json(
        f"{proxy_url}/health",
        timeout=timeout,
        expected_service="entroly-proxy",
        opener=opener,
    )
    dashboard = _probe_json(
        f"{dashboard_url}/health",
        timeout=timeout,
        expected_service="entroly-dashboard",
        opener=opener,
    )

    stats: dict[str, Any] | None = None
    engine_stats: dict[str, Any] | None = None
    if proxy["ok"]:
        stats_result = _probe_json(
            f"{proxy_url}/stats", timeout=timeout, opener=opener
        )
        engine_result = _probe_json(
            f"{proxy_url}/engine-stats", timeout=timeout, opener=opener
        )
        if stats_result["ok"]:
            stats = stats_result["payload"]
        if engine_result["ok"]:
            engine_stats = engine_result["payload"]

    return {
        "schema_version": SCHEMA_VERSION,
        "healthy": bool(proxy["ok"]),
        "proxy_port": int(port),
        "services": {
            "proxy": {key: value for key, value in proxy.items() if key != "payload"},
            "dashboard": {
                key: value for key, value in dashboard.items() if key != "payload"
            },
        },
        "stats": stats,
        "engine_stats": engine_stats,
        "claims": {
            "provider_connectivity_verified": False,
            "production_readiness_implied": False,
        },
    }


__all__ = ["SCHEMA_VERSION", "runtime_status"]
