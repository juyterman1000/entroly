"""Packaging and contract guards for supported agent runtimes."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRODUCT_VERSION = re.search(
    r'^version\s*=\s*"([^"]+)"',
    (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    re.MULTILINE,
).group(1)


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

    assert "openclaw plugins install npm:entroly-openclaw" in readme
    assert "openclaw plugins enable entroly" in readme
    assert "openclaw gateway restart" in readme
    assert "openclaw plugins install clawhub:entroly-openclaw" not in readme


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
    hooks = json.loads(
        (integration / "hooks" / "hooks.json").read_text(encoding="utf-8")
    )
    skill = (
        integration / "skills" / "entroly-evidence-operations" / "SKILL.md"
    ).read_text(encoding="utf-8")

    assert manifest["name"] == "entroly"
    assert manifest["skills"] == "./skills/"
    assert manifest["mcpServers"] == "./.mcp.json"
    assert len(manifest["interface"]["defaultPrompt"]) <= 3
    assert mcp["mcpServers"]["entroly"]["command"] == "node"
    assert mcp["mcpServers"]["entroly"]["args"] == [
        "${PLUGIN_ROOT}/scripts/entroly-plugin-launch.mjs"
    ]
    assert mcp["mcpServers"]["entroly"]["env"]["ENTROLY_NO_DOCKER"] == "1"
    assert mcp["mcpServers"]["entroly"]["env"]["ENTROLY_MCP_PASSIVE"] == "1"
    assert mcp["mcpServers"]["entroly"]["env"]["ENTROLY_MCP_PROFILE"] == "public"
    assert mcp["mcpServers"]["entroly"]["env"]["ENTROLY_MAX_FILES"] == "200"
    command = hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    assert command["type"] == "command"
    assert "entroly-plugin-launch.mjs" in command["command"]
    assert "activation hook --host codex" in command["command"]
    assert (integration / "scripts" / "entroly-plugin-launch.mjs").read_bytes() == (
        ROOT / "scripts" / "entroly-plugin-launch.mjs"
    ).read_bytes()
    assert "process exit code" in skill.lower()
    assert "provider billing" in skill.lower()


def test_codex_portable_and_marketplace_surfaces_are_synchronized() -> None:
    integration = ROOT / "integrations" / "codex" / "entroly"
    npm_plugin = ROOT / "entroly" / "npm-alias"
    marketplace = json.loads(
        (ROOT / ".agents" / "plugins" / "marketplace.json").read_text(
            encoding="utf-8"
        )
    )

    portable = json.loads((integration / "plugin.json").read_text(encoding="utf-8"))
    portable_mcp = json.loads((integration / "mcp.json").read_text(encoding="utf-8"))
    entry = marketplace["plugins"][0]
    package = json.loads((npm_plugin / "package.json").read_text(encoding="utf-8"))

    assert portable["$schema"].endswith("/plugin.schema.json")
    assert portable_mcp["$schema"].endswith("/mcp.schema.json")
    assert portable["extensions"]["com.openai"]["hooks"] == "./hooks/hooks.json"
    assert entry["source"] == {
        "source": "local",
        "path": "./integrations/codex/entroly",
    }
    assert package["version"] == portable["version"]
    for relative in (
        "plugin.json",
        "mcp.json",
        ".codex-plugin/plugin.json",
        ".mcp.json",
        "hooks/hooks.json",
        "scripts/entroly-plugin-launch.mjs",
        "skills/entroly-evidence-operations/SKILL.md",
    ):
        assert (npm_plugin / relative).read_bytes() == (integration / relative).read_bytes()

    launcher = (npm_plugin / "scripts" / "entroly-plugin-launch.mjs").read_text(
        encoding="utf-8"
    )
    assert 'require.resolve("entroly-wasm/bin/entroly-wasm.js")' in launcher
    assert 'args: [packagedCli, "serve"]' in launcher
    assert "plugin.json" in package["files"]
    assert "hooks/hooks.json" in package["files"]


def test_root_agent_plugin_surface_is_cursor_installable() -> None:
    plugin = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    mcp = json.loads((ROOT / "mcp.json").read_text(encoding="utf-8"))

    assert plugin["$schema"].endswith("/plugin.schema.json")
    assert plugin["name"] == "entroly"
    assert plugin["version"] == PRODUCT_VERSION
    assert mcp["$schema"].endswith("/mcp.schema.json")
    server = mcp["mcpServers"]["entroly"]
    assert server["type"] == "stdio"
    assert server["command"] == "node"
    assert server["args"] == ["${PLUGIN_ROOT}/scripts/entroly-plugin-launch.mjs"]
    assert server["env"]["ENTROLY_MCP_PASSIVE"] == "1"
    assert server["env"]["ENTROLY_MCP_PROFILE"] == "public"


def test_root_gemini_extension_is_github_installable() -> None:
    manifest = json.loads(
        (ROOT / "gemini-extension.json").read_text(encoding="utf-8")
    )
    context = ROOT / "GEMINI.md"

    assert manifest["name"] == "entroly"
    assert manifest["version"] == PRODUCT_VERSION
    assert manifest["contextFileName"] == "GEMINI.md"
    assert context.is_file()
    server = manifest["mcpServers"]["entroly"]
    assert server["command"] == "entroly"
    assert server["args"] == ["serve"]
    assert server["env"]["ENTROLY_MCP_PASSIVE"] == "1"
    assert server["env"]["ENTROLY_MCP_PROFILE"] == "public"


def test_root_dot_mcp_is_directory_installable() -> None:
    config = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
    server = config["mcpServers"]["entroly"]

    assert server["command"] == "npx"
    assert server["args"] == ["-y", f"entroly-mcp@{PRODUCT_VERSION}", "serve"]
    assert server["env"]["ENTROLY_MCP_PASSIVE"] == "1"
    assert server["env"]["ENTROLY_MCP_PROFILE"] == "public"


def test_glama_metadata_declares_repository_maintainer() -> None:
    metadata = json.loads((ROOT / "glama.json").read_text(encoding="utf-8"))

    assert metadata["$schema"] == "https://glama.ai/mcp/schemas/server.json"
    assert metadata["maintainers"] == ["juyterman1000"]


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
    claude_hooks = json.loads(
        (ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8")
    )
    gemini_hooks = json.loads(
        (gemini_root / "hooks" / "hooks.json").read_text(encoding="utf-8")
    )

    assert claude_manifest["skills"] == "./skills/"
    claude_env = claude_manifest["mcpServers"]["entroly"]["env"]
    assert claude_env["ENTROLY_MCP_PASSIVE"] == "1"
    assert claude_env["ENTROLY_MCP_PROFILE"] == "public"
    assert claude_env["ENTROLY_MAX_FILES"] == "200"
    claude_command = claude_hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    assert claude_command["type"] == "command"
    assert "entroly-plugin-launch.mjs" in claude_command["command"]
    assert "activation hook --host auto" in claude_command["command"]
    assert gemini_manifest["name"] == "entroly"
    assert gemini_manifest["contextFileName"] == "GEMINI.md"
    gemini_env = gemini_manifest["mcpServers"]["entroly"]["env"]
    assert gemini_env["ENTROLY_MCP_PASSIVE"] == "1"
    assert gemini_env["ENTROLY_MCP_PROFILE"] == "public"
    assert gemini_env["ENTROLY_MAX_FILES"] == "200"
    gemini_command = gemini_hooks["hooks"]["BeforeAgent"][0]["hooks"][0]
    assert "entroly activation hook" in gemini_command["command"]
    assert "matched operational experiment" in gemini_skill


def test_kiro_prompt_hook_is_project_installable_and_context_injecting() -> None:
    integration = ROOT / "integrations" / "kiro" / "entroly"
    hooks = json.loads(
        (integration / ".kiro" / "hooks" / "entroly-activation.json").read_text(
            encoding="utf-8"
        )
    )

    assert hooks["version"] == "v1"
    hook = hooks["hooks"][0]
    assert hook["trigger"] == "PromptSubmit"
    assert hook["action"]["type"] == "command"
    assert "activation hook --host kiro" in hook["action"]["command"]
    assert "--output-format context" in hook["action"]["command"]
    assert hook["timeout"] == 30


def test_cursor_bundle_uses_claude_compatible_context_injection() -> None:
    integration = ROOT / "integrations" / "cursor" / "entroly"
    settings = json.loads(
        (integration / ".claude" / "settings.local.json").read_text(
            encoding="utf-8"
        )
    )
    command = settings["hooks"]["UserPromptSubmit"][0]["hooks"][0]
    assert command["type"] == "command"
    assert "activation hook --host cursor" in command["command"]
    assert command["timeout"] == 30


def test_bundle_installers_are_reversible_and_marker_gated() -> None:
    powershell = (ROOT / "scripts" / "install-agent-bundles.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "scripts" / "install-agent-bundles.sh").read_text(encoding="utf-8")
    for script in (powershell, shell):
        assert "entroly-bundle.json" in script
        assert "backup" in script.lower()
        assert "disabled" in script.lower()
        assert "uninstall" in script.lower()
