from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPOSITORY = "https://github.com/juyterman1000/entroly"


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _master_version() -> str:
    text = (ROOT / "entroly" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    assert match is not None, "no __version__ in entroly/__init__.py"
    return match.group(1)


def test_marketplace_manifest_has_required_fields() -> None:
    manifest = _json(".claude-plugin/marketplace.json")

    assert manifest["name"] == "entroly"
    assert manifest["owner"]["name"] == "juyterman1000"
    assert isinstance(manifest["plugins"], list) and manifest["plugins"]


def test_marketplace_lists_the_plugin_at_the_master_version() -> None:
    manifest = _json(".claude-plugin/marketplace.json")
    entry = next(p for p in manifest["plugins"] if p["name"] == "entroly")

    assert entry["source"] == "./"
    assert entry["version"] == _master_version()


def test_plugin_declares_the_launcher_shim_not_a_bare_runner() -> None:
    plugin = _json(".claude-plugin/plugin.json")
    server = plugin["mcpServers"]["entroly"]

    # A bare `uvx` here is the silent-dead-plugin failure mode: MCP config has
    # no fallback chain, so a machine without uv gets a plugin that starts
    # nothing and reports nothing.
    assert server["command"] == "node"
    assert server["args"] == [
        "${CLAUDE_PLUGIN_ROOT}/scripts/entroly-plugin-launch.mjs"
    ]
    assert server["env"]["ENTROLY_NO_DOCKER"] == "1"


def test_plugin_identity_matches_the_canonical_repository() -> None:
    plugin = _json(".claude-plugin/plugin.json")

    assert plugin["name"] == "entroly"
    assert plugin["repository"] == CANONICAL_REPOSITORY
    assert plugin["version"] == _master_version()
