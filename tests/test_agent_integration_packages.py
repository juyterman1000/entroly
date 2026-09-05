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


def test_openclaw_docs_use_a_publicly_verifiable_install_path() -> None:
    readme = (ROOT / "integrations" / "openclaw" / "README.md").read_text(
        encoding="utf-8"
    )
    landing = (ROOT / "docs" / "openclaw-context-engine.html").read_text(
        encoding="utf-8"
    )

    for document in (readme, landing):
        assert "openclaw plugins install npm:entroly-openclaw" in document
        assert "openclaw plugins enable entroly" in document
        assert "openclaw gateway restart" in document
        assert "openclaw plugins install clawhub:entroly-openclaw" not in document


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


def test_codex_bundle_has_manifest_mcp_and_narrow_valid_skill() -> None:
    integration = ROOT / "integrations" / "codex" / "entroly"
    manifest = json.loads(
        (integration / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    mcp = json.loads((integration / ".mcp.json").read_text(encoding="utf-8"))
    skill = (
        integration / "skills" / "entroly-evidence-operations" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert manifest["name"] == "entroly"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert len(manifest["interface"]["defaultPrompt"]) <= 3
    assert mcp["mcpServers"]["entroly"]["args"] == ["serve"]
    assert mcp["mcpServers"]["entroly"]["env"]["ENTROLY_NO_DOCKER"] == "1"
    assert "process exit code" in skill.lower()
    assert "provider billing" in skill.lower()


def test_claude_and_gemini_bundles_share_evidence_contract() -> None:
    claude_manifest = json.loads(
        (ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
    )
    gemini_root = ROOT / "integrations" / "gemini" / "entroly"
    gemini_manifest = json.loads(
        (gemini_root / "gemini-extension.json").read_text(encoding="utf-8")
    )
    gemini_skill = (
        gemini_root / "skills" / "entroly-evidence-operations" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert claude_manifest["skills"] == "./skills/"
    assert gemini_manifest["name"] == "entroly"
    assert gemini_manifest["contextFileName"] == "GEMINI.md"
    assert "matched operational experiment" in gemini_skill


def test_bundle_installers_are_reversible_and_marker_gated() -> None:
    powershell = (ROOT / "scripts" / "install-agent-bundles.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "scripts" / "install-agent-bundles.sh").read_text(encoding="utf-8")
    for script in (powershell, shell):
        assert "entroly-bundle.json" in script
        assert "backup" in script.lower()
        assert "disabled" in script.lower()
        assert "uninstall" in script.lower()
