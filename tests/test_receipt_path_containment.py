"""`create_context_receipt_from_path` must not read outside the project root.

Every other path-taking tool on the MCP server rejects escapes -- smart_read,
export_training_data, coverage_gaps, compile_docs, prefetch_related and
prepare_proof_guided_context all return "must remain within the project root".
This one passed `path` straight to the ingester, so `../secret.md`, `../../` and
a bare `..` each read files outside the root, and the directory form turned that
into a bulk read ranked by an attacker-chosen query.

It is reachable from the default MCP server and from the `receipts` attach
scope, whose name reads as audit-oriented.

The ingester's extension allowlist bounds what is readable, but an allowlist is
not a path guard: it was never meant to be the boundary.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from entroly.server import create_mcp_server

CANARY = "AWS_CANARY_hunter2"


def _tool(name: str):
    built = create_mcp_server()
    server = next(x for x in built if hasattr(x, "_tool_manager"))
    return server._tool_manager._tools.get(name)


def _run(tool, path: str, query: str) -> str:
    out = asyncio.run(tool.run({"path": path, "query": query}))
    return out if isinstance(out, str) else json.dumps(out, default=str)


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    outside = tmp_path / "OUTSIDE_SECRET.md"
    outside.write_text(f"{CANARY}\n", encoding="utf-8")
    root = tmp_path / "proj"
    root.mkdir()
    (root / "README.md").write_text("# inside\nordinary project documentation\n", encoding="utf-8")
    monkeypatch.setenv("ENTROLY_SOURCE", str(root))
    monkeypatch.chdir(root)
    return root


@pytest.mark.parametrize("escape", ["../OUTSIDE_SECRET.md", "..", "../../"])
def test_paths_outside_the_root_are_refused(rooted, escape) -> None:
    tool = _tool("create_context_receipt_from_path")
    assert tool is not None, "tool must be registered"

    # The query deliberately avoids the canary string so a match can only come
    # from file content, never from the query being echoed back in the receipt.
    result = _run(tool, escape, "summarise the documentation")

    assert CANARY not in result, f"{escape} leaked content from outside the project root"
    assert "must remain within the project root" in result


def test_a_path_inside_the_root_still_works(rooted) -> None:
    """Containment must not break the legitimate case."""
    tool = _tool("create_context_receipt_from_path")
    result = _run(tool, "README.md", "summarise the documentation")

    assert "must remain within the project root" not in result
    assert CANARY not in result
