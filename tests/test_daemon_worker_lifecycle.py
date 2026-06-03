"""Regression tests for daemon worker lifecycle bookkeeping."""

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
