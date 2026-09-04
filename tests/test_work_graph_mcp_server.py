from __future__ import annotations

import builtins

import pytest

from entroly import work_graph_mcp_server as server


def test_transport_json_is_deterministic():
    assert server._json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_mcp_dependency_is_lazy(monkeypatch):
    """Importing this module must not require the SDK; building a server may.

    The failure is raised from ``create_mcp_server``, not at import time.

    This asserted the literal "MCP SDK not installed", which is only true when
    the SDK is absent. Here it is installed and only the submodule is blocked,
    and reporting "not installed" would send the operator to reinstall a
    package they already have. The message is now chosen from the actual state,
    so the assertion checks the state this test creates.
    """
    from entroly.mcp_sdk import installed_version

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "mcp.server.fastmcp":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(RuntimeError) as excinfo:
        server.create_mcp_server()

    message = str(excinfo.value)
    assert "MCP SDK" in message, message
    if installed_version() is not None:
        assert "not installed" not in message, (
            f"the SDK is installed; the error claims otherwise: {message}"
        )


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
        to_agent="codex",
    )

    assert calls == [
        {
            "project": "subdir",
            "workstream_id": "w",
            "max_evidence": 9,
            "to_agent": "codex",
        }
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

    assert calls == [
        {"project": "", "workstream_id": "", "max_evidence": -1, "to_agent": ""}
    ]
    assert '"error":"invalid_work_graph_request"' in payload


def test_execution_transport_exposes_complete_product_loop(monkeypatch):
    registered = _fake_fastmcp(monkeypatch)
    calls = []

    def record(**kwargs):
        calls.append(kwargs)
        return {"status": "ok", "kind": "work_record_execution"}

    monkeypatch.setattr(server._work, "work_record_execution", record)
    server.create_mcp_server()
    expected_tools = {
        "work_state",
        "work_claim",
        "work_resume",
        "work_handoff",
        "work_compile_context",
        "work_context_fault",
        "work_record_context",
        "work_record_memory",
        "work_record_execution",
    }
    assert expected_tools <= registered.keys()

    payload = registered["work_record_execution"](
        {"routing_id": "route_1"},
        {"outcome_id": "outcome_1"},
        {"verification_id": "verify_1"},
        project="repo",
        invalidated_commitments=["sha256:old"],
    )
    assert calls == [{
        "route": {"routing_id": "route_1"},
        "outcome": {"outcome_id": "outcome_1"},
        "verification": {"verification_id": "verify_1"},
        "project": "repo",
        "invalidated_commitments": ["sha256:old"],
    }]
    assert '"kind":"work_record_execution"' in payload
