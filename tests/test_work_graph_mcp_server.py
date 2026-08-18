from __future__ import annotations

import builtins

import pytest

from entroly import work_graph_mcp_server as server


def test_transport_json_is_deterministic():
    assert server._json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_mcp_dependency_is_lazy(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "mcp.server.fastmcp":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="MCP SDK not installed"):
        server.create_mcp_server()


def test_resume_tool_refreshes_repository_before_delegate(monkeypatch, tmp_path):
    import sys
    import types

    registered = {}

    class FakeInner:
        version = ""

    class FakeMCP:
        def __init__(self, *_args, **_kwargs):
            self._mcp_server = FakeInner()

        def tool(self):
            def decorate(fn):
                registered[fn.__name__] = fn
                return fn
            return decorate

    fastmcp = types.ModuleType("mcp.server.fastmcp")
    fastmcp.FastMCP = FakeMCP
    server_pkg = types.ModuleType("mcp.server")
    mcp_pkg = types.ModuleType("mcp")
    monkeypatch.setitem(sys.modules, "mcp", mcp_pkg)
    monkeypatch.setitem(sys.modules, "mcp.server", server_pkg)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp)

    events = []

    class FakeStore:
        def __init__(self, repo_id):
            events.append(("store", repo_id))

        def submit_observation(self, observation):
            events.append(("observe", observation))

    monkeypatch.setenv("ENTROLY_SOURCE", str(tmp_path))
    monkeypatch.setattr(server, "discover_repository_identity", lambda _p: {"repo_id": "repo:test"})
    monkeypatch.setattr(server, "discover_repository_observation", lambda _p: {"repo_id": "repo:test"})
    monkeypatch.setattr(server, "WorkGraphStore", FakeStore)
    monkeypatch.setattr(server._work, "work_resume", lambda **kwargs: {"status": "ok", "kind": "work_resume", "args": kwargs})

    server.create_mcp_server()
    payload = registered["work_resume"](workstream_id="w", max_evidence=9)

    assert events == [("store", "repo:test"), ("observe", {"repo_id": "repo:test"})]
    assert '"workstream_id":"w"' in payload
    assert '"max_evidence":9' in payload
