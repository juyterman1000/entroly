"""MCP startup readiness regression tests."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import entroly.auto_index as auto_index_module
import entroly.server as server


def test_background_services_do_not_block_mcp_startup(monkeypatch):
    index_started = threading.Event()
    allow_index_to_finish = threading.Event()
    watcher_started = threading.Event()
    autotune_started = threading.Event()

    def _blocking_auto_index(_engine):
        index_started.set()
        assert allow_index_to_finish.wait(timeout=2)
        return {
            "status": "indexed",
            "files_indexed": 1,
            "total_tokens": 10,
            "duration_s": 0.1,
        }

    monkeypatch.setattr(auto_index_module, "auto_index", _blocking_auto_index)
    monkeypatch.setattr(
        auto_index_module,
        "start_incremental_watcher",
        lambda _engine: watcher_started.set(),
    )
    monkeypatch.setattr(
        server,
        "_start_autotune_daemon",
        lambda _engine: autotune_started.set(),
    )

    startup_thread = server._start_background_services(object())

    assert index_started.wait(timeout=1)
    assert startup_thread.is_alive()
    assert not watcher_started.is_set()
    assert not autotune_started.is_set()

    allow_index_to_finish.set()
    startup_thread.join(timeout=2)

    assert not startup_thread.is_alive()
    assert watcher_started.is_set()
    assert autotune_started.is_set()


def test_background_services_start_autotune_after_index_failure(monkeypatch):
    autotune_started = threading.Event()

    def _failing_auto_index(_engine):
        raise RuntimeError("index failed")

    monkeypatch.setattr(auto_index_module, "auto_index", _failing_auto_index)
    monkeypatch.setattr(
        server,
        "_start_autotune_daemon",
        lambda _engine: autotune_started.set(),
    )

    startup_thread = server._start_background_services(object())
    startup_thread.join(timeout=2)

    assert not startup_thread.is_alive()
    assert autotune_started.is_set()


def test_background_services_start_attached_belief_listener(monkeypatch):
    calls = []

    class Listener:
        def start(self, **kwargs):
            calls.append(kwargs)
            return {"status": "started"}

    monkeypatch.setattr(
        auto_index_module,
        "auto_index",
        lambda _engine: {
            "status": "skipped",
            "files_indexed": 1,
            "total_tokens": 10,
            "duration_s": 0.0,
        },
    )
    monkeypatch.setattr(auto_index_module, "start_incremental_watcher", lambda _engine: None)
    monkeypatch.setattr(server, "_start_autotune_daemon", lambda _engine: None)
    engine = SimpleNamespace(_workspace_listener=Listener())

    startup_thread = server._start_background_services(engine)
    startup_thread.join(timeout=2)

    assert calls == [{"interval_s": 120, "max_files": 100, "force_initial": False}]
