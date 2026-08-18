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
