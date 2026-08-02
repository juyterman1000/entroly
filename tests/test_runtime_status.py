from __future__ import annotations

import io
import json
import urllib.error

from entroly.runtime_status import SCHEMA_VERSION, runtime_status


class _Response:
    def __init__(self, payload: object, status: int = 200):
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body


def test_runtime_status_is_sanitized_when_services_are_down() -> None:
    def opener(_url: str, *, timeout: float):
        assert timeout == 0.25
        raise urllib.error.URLError("secret-bearing failure")

    report = runtime_status(8123, timeout=0.25, opener=opener)

    assert report["schema_version"] == SCHEMA_VERSION
    assert report["healthy"] is False
    assert report["proxy_port"] == 8123
    assert report["services"]["proxy"] == {
        "ok": False,
        "state": "unreachable",
        "status_code": None,
    }
    assert "secret-bearing" not in json.dumps(report)
    assert report["claims"]["provider_connectivity_verified"] is False
    assert report["claims"]["production_readiness_implied"] is False


def test_runtime_status_reads_local_stats_only_after_proxy_identity_matches() -> None:
    calls: list[str] = []

    def opener(url: str, *, timeout: float):
        calls.append(url)
        assert timeout == 1.0
        if url.endswith(":9377/health"):
            return _Response({"status": "ok", "service": "entroly-proxy"})
        if url.endswith(":9378/health"):
            return _Response({"status": "ok", "service": "entroly-dashboard"})
        if url.endswith("/stats"):
            return _Response({"requests_total": 4, "requests_optimized": 3})
        if url.endswith("/engine-stats"):
            return _Response({"resonance": {"tracked_pairs": 2}})
        raise AssertionError(url)

    report = runtime_status(timeout=1.0, opener=opener)

    assert report["healthy"] is True
    assert report["services"]["proxy"]["state"] == "ready"
    assert report["stats"] == {"requests_total": 4, "requests_optimized": 3}
    assert report["engine_stats"] == {"resonance": {"tracked_pairs": 2}}
    assert calls == [
        "http://127.0.0.1:9377/health",
        "http://127.0.0.1:9378/health",
        "http://127.0.0.1:9377/stats",
        "http://127.0.0.1:9377/engine-stats",
    ]


def test_identity_mismatch_is_not_reported_ready() -> None:
    def opener(url: str, *, timeout: float):
        return _Response({"status": "ok", "service": "not-entroly"})

    report = runtime_status(opener=opener)

    assert report["healthy"] is False
    assert report["services"]["proxy"]["state"] == "identity_mismatch"
    assert report["stats"] is None
    assert report["engine_stats"] is None


def test_status_json_emits_machine_readable_report(monkeypatch, capsys) -> None:
    from types import SimpleNamespace
    import entroly.cli as cli
    import entroly.runtime_status as status_module

    expected = {
        "schema_version": SCHEMA_VERSION,
        "healthy": True,
        "proxy_port": 9444,
        "services": {},
        "stats": None,
        "engine_stats": None,
        "claims": {
            "provider_connectivity_verified": False,
            "production_readiness_implied": False,
        },
    }
    monkeypatch.setattr(status_module, "runtime_status", lambda *, port: expected)

    result = cli.cmd_status(
        SimpleNamespace(port=9444, json_output=True, require_running=True)
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_status_require_running_returns_nonzero_when_proxy_is_down(
    monkeypatch, capsys
) -> None:
    from types import SimpleNamespace
    import entroly.cli as cli
    import entroly.runtime_status as status_module

    report = {
        "schema_version": SCHEMA_VERSION,
        "healthy": False,
        "proxy_port": 9377,
        "services": {},
        "stats": None,
        "engine_stats": None,
        "claims": {
            "provider_connectivity_verified": False,
            "production_readiness_implied": False,
        },
    }
    monkeypatch.setattr(status_module, "runtime_status", lambda *, port: report)

    result = cli.cmd_status(
        SimpleNamespace(port=None, json_output=True, require_running=True)
    )

    assert result == 1
    assert json.loads(capsys.readouterr().out)["healthy"] is False
