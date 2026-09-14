from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_lobehub_score.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_lobehub_score", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_has_complete_local_lobehub_evidence() -> None:
    module = _load_module()
    report = module.collect(ROOT, protocol_validated=True)
    evidence = report["evidence"]

    for criterion in (
        "deployment",
        "deployMoreThanManual",
        "license",
        "readme",
        "tools",
        "prompts",
        "resources",
    ):
        assert evidence[criterion]["check"], criterion

    assert evidence["validated"]["check"] is True
    assert evidence["validated"]["external"] is True
    assert report["local_readiness"]["score"] == 96
    assert report["local_readiness"]["grade_if_lobehub_confirmed_same_flags"] == "A"
    assert report["local_readiness"]["warning"].startswith(
        "This is repository/local-protocol"
    )


def test_claimed_point_remains_external_not_fabricated() -> None:
    module = _load_module()
    report = module.collect(ROOT, protocol_validated=False)

    assert report["evidence"]["claimed"] == {
        "check": False,
        "source": "Only LobeHub can confirm ownership; its detail score currently does not receive claimed state.",
        "classification": "external implementation/index state",
        "external": True,
    }
    assert report["evidence"]["validated"]["check"] is False
    assert report["local_readiness"]["grade_if_lobehub_confirmed_same_flags"] == "F"


def test_public_mcp_profile_keeps_marketplace_tool_surface_small(tmp_path, monkeypatch) -> None:
    """Marketplace startup should not expose the whole research control plane."""
    monkeypatch.setenv("ENTROLY_MCP_PROFILE", "public")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly"))
    from entroly.server import _PUBLIC_MCP_TOOLS, create_mcp_server

    server, _engine = create_mcp_server()
    tools = set(server._tool_manager._tools)

    assert tools == set(_PUBLIC_MCP_TOOLS)
    assert len(tools) == 19
    assert "entroly_retrieve" in tools
    assert "remember_fragment" in tools
    assert "eicv_suppress_hallucinations" not in tools
    assert "export_training_data" not in tools
    assert "start_workspace_listener" not in tools
    assert server._prompt_manager._prompts
    assert server._resource_manager._resources


def test_unconfigured_mcp_profile_preserves_existing_clients(tmp_path, monkeypatch) -> None:
    """A direct or legacy entrypoint must retain the established full contract."""
    monkeypatch.delenv("ENTROLY_MCP_PROFILE", raising=False)
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly"))
    from entroly.server import _PUBLIC_MCP_TOOLS, create_mcp_server

    server, _engine = create_mcp_server()
    tools = set(server._tool_manager._tools)

    assert len(tools) > len(_PUBLIC_MCP_TOOLS)
    assert _PUBLIC_MCP_TOOLS < tools
    assert {"entroly_retrieve", "remember_fragment", "work_handoff", "work_state"} <= tools


def test_full_mcp_profile_preserves_advanced_tool_surface(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENTROLY_MCP_PROFILE", "full")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly"))
    from entroly.server import _PUBLIC_MCP_TOOLS, create_mcp_server

    server, _engine = create_mcp_server()
    tools = set(server._tool_manager._tools)

    assert len(tools) > len(_PUBLIC_MCP_TOOLS)
    assert _PUBLIC_MCP_TOOLS < tools
    assert "eicv_suppress_hallucinations" in tools
    assert "export_training_data" in tools
    assert "start_workspace_listener" in tools


def test_high_risk_tools_publish_actionable_parameter_contracts(tmp_path, monkeypatch) -> None:
    """Keep marketplace-facing schemas useful for first-attempt tool calls."""
    monkeypatch.setenv("ENTROLY_MCP_PROFILE", "full")
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "entroly"))
    from entroly.server import create_mcp_server

    server, _engine = create_mcp_server()
    expected = {
        "record_ci_result": {
            "request_id": "Exact request_id returned by optimize_context",
            "passed": "True only when every required CI check passed",
            "pipeline": "Short CI provider or pipeline name",
            "url": "Optional CI run URL stored as provenance",
        },
        "render_context_receipt": {
            "receipt_json": "Complete JSON text returned by create_context_receipt",
        },
        "resume_state": {
            "query": "Optional task terms used to select the most relevant checkpoint",
            "project": "Optional project path or identifier",
        },
        "sync_workspace_changes": {
            "directory": "Workspace directory to scan",
            "force": "Rescan all discovered source files",
            "max_files": "Maximum changed files processed in this pass",
        },
        "start_workspace_listener": {
            "directory": "Workspace directory to watch",
            "interval_s": "Polling interval in seconds",
            "force_initial": "Run an initial full synchronization",
            "max_files": "Maximum changed files processed per poll",
        },
    }

    for tool_name, fields in expected.items():
        tool = server._tool_manager._tools[tool_name]
        schema = json.loads(json.dumps(tool.parameters))
        properties = schema["properties"]
        for field_name, description_start in fields.items():
            assert properties[field_name]["description"].startswith(description_start)

    listener_schema = server._tool_manager._tools["start_workspace_listener"].parameters
    assert listener_schema["properties"]["interval_s"]["minimum"] == 1
    assert listener_schema["properties"]["interval_s"]["maximum"] == 86400
    assert listener_schema["properties"]["max_files"]["minimum"] == 1
    assert listener_schema["properties"]["max_files"]["maximum"] == 10000
