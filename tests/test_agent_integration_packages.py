"""Packaging and contract guards for supported agent runtimes."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_opencode_package_loads_local_mcp_and_compaction_hook() -> None:
    integration = ROOT / "integrations" / "opencode"
    config = json.loads((integration / "opencode.jsonc").read_text(encoding="utf-8"))
    plugin = (
        integration
        / ".opencode"
        / "plugins"
        / "entroly-context-assurance.ts"
    ).read_text(encoding="utf-8")

    server = config["mcp"]["entroly"]
    assert server["type"] == "local"
    assert server["command"] == ["entroly", "serve"]
    assert server["environment"]["ENTROLY_NO_DOCKER"] == "1"
    assert config["permission"]["entroly_*"] == "ask"
    assert '"experimental.session.compacting"' in plugin
    assert "ccr:<24-hex>" in plugin
    assert "entroly_entroly_retrieve" in plugin
    assert "do not add a query or source path" in plugin.lower()


def test_openclaw_remains_a_first_class_context_engine() -> None:
    integration = ROOT / "integrations" / "openclaw"
    entry = (integration / "index.js").read_text(encoding="utf-8")
    manifest = json.loads(
        (integration / "openclaw.plugin.json").read_text(encoding="utf-8")
    )
    bridge = (ROOT / "entroly" / "openclaw_bridge.py").read_text(encoding="utf-8")

    assert 'api.registerContextEngine("entroly"' in entry
    assert 'api.on("llm_output"' in entry
    assert 'api.on("before_agent_finalize"' in entry
    assert manifest["id"] == "entroly"
    assert manifest["activation"]["onStartup"] is True
    assert manifest["configSchema"]["additionalProperties"] is False
    assert 'operation == "assemble"' in bridge
    assert 'operation == "verify_proof_guided_output"' in bridge


def test_hermes_exports_current_contract_adapter() -> None:
    package = (
        ROOT
        / "entroly"
        / "integrations"
        / "hermes_context_engine"
        / "__init__.py"
    ).read_text(encoding="utf-8")
    modern = (
        ROOT
        / "entroly"
        / "integrations"
        / "hermes_context_engine"
        / "modern.py"
    ).read_text(encoding="utf-8")

    assert "ModernHermesContextMixin" in package
    for method in (
        "select_context",
        "on_turn_complete",
        "update_model",
        "get_tool_schemas",
        "handle_tool_call",
        "get_status",
    ):
        assert f"def {method}(" in modern
