"""MCP startup readiness regression tests."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import entroly.auto_index as auto_index_module
import entroly.server as server


def test_passive_background_services_skip_autonomous_work(monkeypatch):
    calls = []

    monkeypatch.setenv("ENTROLY_MCP_PASSIVE", "1")
    monkeypatch.setattr(
        auto_index_module,
        "auto_index",
        lambda _engine: calls.append("index"),
    )
    monkeypatch.setattr(
        auto_index_module,
        "start_incremental_watcher",
        lambda _engine: calls.append("watcher"),
    )
    monkeypatch.setattr(
        server,
        "_start_autotune_daemon",
        lambda _engine: calls.append("autotune"),
    )
    engine = SimpleNamespace(
        _workspace_listener=SimpleNamespace(
            start=lambda **_kwargs: calls.append("belief-listener")
        )
    )

    startup_thread = server._start_background_services(engine)
    startup_thread.join(timeout=2)

    assert not startup_thread.is_alive()
    assert calls == []


def test_passive_mcp_server_does_not_construct_evolution_daemon(
    monkeypatch,
    tmp_path,
):
    class UnexpectedEvolutionDaemon:
        def __init__(self, **_kwargs):
            raise AssertionError("passive MCP startup constructed EvolutionDaemon")

    project_dir = tmp_path / "project"
    state_home = tmp_path / "home"
    project_dir.mkdir()
    state_home.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(state_home))
    monkeypatch.setenv("USERPROFILE", str(state_home))
    monkeypatch.setenv("ENTROLY_MCP_PASSIVE", "true")
    monkeypatch.setattr(server, "EvolutionDaemon", UnexpectedEvolutionDaemon)

    mcp, _engine = server.create_mcp_server(allowed_tools={"recall_relevant"})

    assert mcp.instructions is not None
    assert "call recall_relevant once" in mcp.instructions
    assert not (project_dir / ".entroly").exists()


def test_primary_mcp_surface_includes_work_graph_continuity(
    monkeypatch,
    tmp_path,
):
    project_dir = tmp_path / "project"
    state_home = tmp_path / "home"
    project_dir.mkdir()
    state_home.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.setenv("HOME", str(state_home))
    monkeypatch.setenv("USERPROFILE", str(state_home))
    monkeypatch.setenv("ENTROLY_MCP_PASSIVE", "true")

    calls = []

    def resume(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "kind": "work_resume"}

    monkeypatch.setattr(
        "entroly.work_graph_mcp.work_resume",
        resume,
    )

    mcp, _engine = server.create_mcp_server(allowed_tools={"work_resume"})
    tools = mcp._tool_manager._tools

    assert set(tools) == {"work_resume"}
    assert "work_resume" in mcp.instructions
    payload = tools["work_resume"].fn(to_agent="codex")
    assert json.loads(payload)["kind"] == "work_resume"
    assert calls == [{
        "project": "",
        "workstream_id": "",
        "max_evidence": 128,
        "to_agent": "codex",
    }]


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
