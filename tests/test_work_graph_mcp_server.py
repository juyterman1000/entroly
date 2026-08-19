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


def _fake_fastmcp(monkeypatch):
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
    monkeypatch.setitem(sys.modules, "mcp", types.ModuleType("mcp"))
    monkeypatch.setitem(sys.modules, "mcp.server", types.ModuleType("mcp.server"))
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp)
    return registered


def test_resume_transport_delegates_exactly_once(monkeypatch):
    registered = _fake_fastmcp(monkeypatch)
    calls = []

    def resume(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "kind": "work_resume", "args": kwargs}

    monkeypatch.setattr(server._work, "work_resume", resume)
    server.create_mcp_server()
    payload = registered["work_resume"](
        project="subdir",
        workstream_id="w",
        max_evidence=9,
    )

    assert calls == [
        {"project": "subdir", "workstream_id": "w", "max_evidence": 9}
    ]
    assert '"workstream_id":"w"' in payload
    assert '"max_evidence":9' in payload


def test_resume_transport_preserves_helper_validation_error(monkeypatch):
    registered = _fake_fastmcp(monkeypatch)
    calls = []

    def resume(**kwargs):
        calls.append(kwargs)
        return {
            "status": "error",
            "kind": "work_resume",
            "error": "invalid_work_graph_request",
            "detail": "max_evidence out of bounds",
        }

    monkeypatch.setattr(server._work, "work_resume", resume)
    server.create_mcp_server()
    payload = registered["work_resume"](max_evidence=-1)

    assert calls == [{"project": "", "workstream_id": "", "max_evidence": -1}]
    assert '"error":"invalid_work_graph_request"' in payload
