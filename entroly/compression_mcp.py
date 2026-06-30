"""MCP tools for Entroly compression retrieval.

This module provides a focused MCP server for retrieving omitted spans produced
by Evidence-Locked Compression. It can run independently or be mounted by the
main Entroly MCP server in a future normal-code patch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .compression_retrieval_store import CompressionRetrievalStore


def _default_store_path() -> Path:
    configured = os.environ.get("ENTROLY_COMPRESSION_STORE")
    if configured:
        return Path(configured)
    return Path(os.environ.get("ENTROLY_DIR", ".entroly")) / "compression-store.json"


def create_compression_mcp_server(store_path: str | None = None):
    """Create a focused MCP server for compressed-span retrieval."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise RuntimeError("MCP SDK not installed. Install with: pip install mcp") from None

    mcp = FastMCP(
        "entroly-compression",
        instructions=(
            "Retrieve omitted spans from Entroly Evidence-Locked Compression receipts. "
            "Use these tools when a compressed prompt says more context is recoverable."
        ),
    )

    def _store(path_override: str = "") -> CompressionRetrievalStore:
        path = Path(path_override) if path_override else Path(store_path) if store_path else _default_store_path()
        return CompressionRetrievalStore(path)

    @mcp.tool()
    def retrieve_compressed_span(receipt_id: str, span_id: str, store_path_override: str = "") -> str:
        """Retrieve one omitted span by receipt id and span id."""
        store = _store(store_path_override)
        span = store.get_span(receipt_id, span_id)
        if span is None:
            return json.dumps(
                {"status": "not_found", "receipt_id": receipt_id, "span_id": span_id},
                indent=2,
            )
        return json.dumps({"status": "ok", "span": span.as_dict()}, indent=2, ensure_ascii=False)

    @mcp.tool()
    def search_compressed_spans(query: str, limit: int = 5, store_path_override: str = "") -> str:
        """Search recoverable omitted spans by keyword."""
        store = _store(store_path_override)
        spans = store.search(query, limit=max(1, min(int(limit), 20)))
        return json.dumps(
            {"status": "ok", "query": query, "spans": [span.as_dict() for span in spans]},
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def list_compression_receipts(store_path_override: str = "") -> str:
        """List locally stored compression receipts and span counts."""
        store = _store(store_path_override)
        return json.dumps(
            {"status": "ok", "receipts": store.list_receipts()},
            indent=2,
            ensure_ascii=False,
        )

    return mcp


def main() -> None:
    create_compression_mcp_server().run()


if __name__ == "__main__":
    main()


__all__ = ["create_compression_mcp_server"]
