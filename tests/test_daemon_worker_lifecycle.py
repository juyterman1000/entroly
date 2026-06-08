"""Regression tests for daemon worker lifecycle bookkeeping."""

import os
import sys
from types import SimpleNamespace

from entroly.daemon import EntrolyDaemon


def test_proxy_worker_clears_running_state_after_normal_exit(monkeypatch):
    class ProxyConfig:
        @classmethod
        def from_env(cls):
            return cls()

        def _apply_quality_dial(self, quality):
            self.quality = quality

    class Server:
        def __init__(self, config):
            self.config = config

        def run(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "entroly.proxy",
        SimpleNamespace(create_proxy_app=lambda *_args, **_kwargs: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "entroly.proxy_config",
        SimpleNamespace(ProxyConfig=ProxyConfig, resolve_quality=lambda _mode: 0.5),
    )
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(Config=lambda *_args, **_kwargs: object(), Server=Server),
    )

    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon._engine = object()
    daemon._start_proxy_worker()
    worker = daemon._workers["proxy"]
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert daemon.state.proxy.running is False


def test_mcp_worker_clears_running_state_after_normal_exit(monkeypatch):
    class Mcp:
        settings = SimpleNamespace(port=None, host=None)

        def run(self, transport=None):
            return None

    monkeypatch.setitem(
        sys.modules,
        "entroly.server",
        SimpleNamespace(create_mcp_server=lambda **_kwargs: (Mcp(), object())),
    )

    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon._engine = object()
    daemon._start_mcp_worker()
    worker = daemon._workers["mcp"]
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert daemon.state.mcp.running is False


def test_bypass_controls_update_state_env_and_live_proxy(monkeypatch):
    monkeypatch.delenv("ENTROLY_BYPASS", raising=False)
    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    live_proxy = SimpleNamespace(_bypass=False)
    daemon._proxy_runtime = live_proxy

    try:
        daemon.set_bypass(True)

        assert daemon.state.bypass_mode is True
        assert daemon.state.optimization_enabled is False
        assert os.environ["ENTROLY_BYPASS"] == "1"
        assert live_proxy._bypass is True

        daemon.set_optimization(True)

        assert daemon.state.bypass_mode is False
        assert daemon.state.optimization_enabled is True
        assert "ENTROLY_BYPASS" not in os.environ
        assert live_proxy._bypass is False
    finally:
        os.environ.pop("ENTROLY_BYPASS", None)


def test_standalone_dashboard_controls_attach_live_proxy(monkeypatch):
    from entroly.daemon import _register_control_api, get_daemon
    from entroly.dashboard import start_dashboard

    monkeypatch.delenv("ENTROLY_BYPASS", raising=False)
    _register_control_api(None)
    live_proxy = SimpleNamespace(_bypass=False, config=SimpleNamespace())
    server = start_dashboard(engine=object(), port=0, daemon=True, proxy_runtime=live_proxy)
    try:
        daemon = get_daemon()
        assert daemon is not None

        daemon.set_bypass(True)

        assert live_proxy._bypass is True
        assert daemon.state.bypass_mode is True
        assert daemon.state.optimization_enabled is False
        assert os.environ["ENTROLY_BYPASS"] == "1"
    finally:
        os.environ.pop("ENTROLY_BYPASS", None)
        server.shutdown()
        _register_control_api(None)


def test_proxy_app_passes_runtime_to_dashboard(monkeypatch):
    import entroly.dashboard as dashboard
    from entroly.proxy import create_proxy_app
    from entroly.proxy_config import ProxyConfig

    captured = {}

    def fake_start_dashboard(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(dashboard, "start_dashboard", fake_start_dashboard)

    app = create_proxy_app(object(), ProxyConfig(), start_dashboard=True)

    assert captured["proxy_runtime"] is app.state.proxy
