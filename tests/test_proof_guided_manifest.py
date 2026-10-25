"""The static discovery manifest must match the live proof tool catalog."""

from __future__ import annotations

import asyncio
import inspect
import json
import socket
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROOF_TOOLS = {
    "prepare_proof_guided_context",
    "advance_proof_guided_context",
    "inspect_proof_guided_context",
}


def test_proof_manifest_matches_live_mcp_catalog(tmp_path, monkeypatch):
    from entroly.server import EntrolyConfig, EntrolyEngine, create_mcp_server

    def deny_network(*args, **kwargs):
        raise AssertionError("catalog verification must not contact a provider")

    monkeypatch.setenv("ENTROLY_SOURCE", str(tmp_path))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_MCP_PASSIVE", "1")
    async def list_catalog():
        # Windows creates the event loop's local self-pipe with socketpair.
        # Install the provider guard after that plumbing, before server creation.
        monkeypatch.setattr(socket.socket, "connect", deny_network)
        monkeypatch.setattr(socket, "create_connection", deny_network)
        server, _ = create_mcp_server(
            engine=EntrolyEngine(config=EntrolyConfig()),
            allowed_tools=PROOF_TOOLS,
        )
        return await server.list_tools()

    live_tools = {tool.name: tool for tool in asyncio.run(list_catalog())}
    manifest = json.loads((ROOT / "lhm.plugin.json").read_text(encoding="utf-8"))
    advertised = {tool["name"]: tool for tool in manifest["tools"]}
    for name in PROOF_TOOLS:
        assert advertised[name]["inputSchema"] == live_tools[name].inputSchema
        # Python 3.13+ strips common docstring indentation at compilation.
        # Compare identical text/paragraphs after that same normalization.
        assert inspect.cleandoc(advertised[name]["description"]) == inspect.cleandoc(live_tools[name].description)
    advance = advertised["advance_proof_guided_context"]
    assert "expected_revision" in advance["inputSchema"]["properties"]
    assert "request_sha256" in advance["inputSchema"]["properties"]
    assert "Missing or invalid bindings are rejected" in advance["description"]


@pytest.mark.parametrize("field", ["expected_revision", "request_sha256"])
def test_advance_manifest_preserves_strict_binding_types(field):
    manifest = json.loads((ROOT / "lhm.plugin.json").read_text(encoding="utf-8"))
    advance = next(tool for tool in manifest["tools"] if tool["name"] == "advance_proof_guided_context")
    expected = "integer" if field == "expected_revision" else "string"
    assert advance["inputSchema"]["properties"][field]["anyOf"] == [
        {"type": expected}, {"type": "null"},
    ]


def test_description_normalization_preserves_wording_and_paragraphs():
    indented = "Verify one model round.\n\n        Missing bindings are rejected.\n        Keep the issued revision.\n        "
    compiled = "Verify one model round.\n\nMissing bindings are rejected.\nKeep the issued revision."
    assert inspect.cleandoc(indented) == inspect.cleandoc(compiled)
    assert inspect.cleandoc(indented) != inspect.cleandoc(compiled.replace("rejected", "accepted"))
    assert inspect.cleandoc(indented) != inspect.cleandoc(compiled.replace("\n\n", "\n"))
