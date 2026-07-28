"""MCP tools for scoped Entroly compression retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .compression_retrieval_store_secure import CompressionRetrievalStore
from .optimization_ledger import OptimizationLedger

_MAX_PATH_CHARS = 4096


def _default_store_path() -> Path:
    configured = os.environ.get("ENTROLY_COMPRESSION_STORE")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("ENTROLY_DIR", ".entroly")).expanduser() / "compression-store.json"


def _default_ledger_path() -> Path:
    configured = os.environ.get("ENTROLY_OPTIMIZATION_LEDGER")
    if configured:
        return Path(configured).expanduser()
    return Path(os.environ.get("ENTROLY_DIR", ".entroly")).expanduser() / "optimization-ledger.sqlite3"


def _resolved_nonexistent_path(path: Path) -> Path:
    """Resolve existing symlinks while allowing bounded descendants to be created."""
    raw = str(path)
    if not raw or len(raw) > _MAX_PATH_CHARS or "\x00" in raw:
        raise ValueError("store path is not a safe bounded path")
    candidate = Path(os.path.abspath(os.fspath(path.expanduser())))
    ancestor = candidate
    suffix: list[str] = []
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise ValueError("store path has no accessible existing ancestor")
        if ancestor.name in {"", ".", ".."}:
            raise ValueError("store path contains an unsafe component")
        suffix.append(ancestor.name)
        ancestor = ancestor.parent
    resolved = ancestor.resolve(strict=True)
    for component in reversed(suffix):
        resolved /= component
    return resolved


def _safe_store_path(path_override: str, configured_path: str | None) -> Path:
    selected = Path(configured_path).expanduser() if configured_path else _default_store_path()
    try:
        selected = _resolved_nonexistent_path(selected)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"configured compression store is unavailable: {exc}") from exc

    if not path_override:
        return selected
    if os.environ.get("ENTROLY_ALLOW_STORE_PATH_OVERRIDE") != "1":
        raise ValueError(
            "store_path_override is disabled; set ENTROLY_ALLOW_STORE_PATH_OVERRIDE=1 "
            "only for a trusted operator-controlled MCP client"
        )
    override = Path(path_override).expanduser()
    try:
        resolved = _resolved_nonexistent_path(override)
        allowed_root = _resolved_nonexistent_path(
            Path(os.environ.get("ENTROLY_STORE_OVERRIDE_ROOT", str(selected.parent)))
        )
        resolved.relative_to(allowed_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("store_path_override escapes the configured recovery root") from exc
    return resolved


def _error_payload(error: Exception, *, operation: str) -> str:
    return json.dumps(
        {
            "status": "error",
            "operation": operation,
            "error": str(error)[:500],
        },
        indent=2,
        ensure_ascii=False,
    )


def create_compression_mcp_server(store_path: str | None = None):
    """Create a focused MCP server for scope-bound compressed-span retrieval."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise RuntimeError("MCP SDK not installed. Install with: pip install mcp") from None

    mcp = FastMCP(
        "entroly-compression",
        instructions=(
            "Retrieve omitted spans from Entroly Evidence-Locked Compression receipts. "
            "Recovery is restricted to this MCP process's store/workspace/session scope."
        ),
    )

    def _store(path_override: str = "") -> CompressionRetrievalStore:
        path = _safe_store_path(path_override, store_path)
        return CompressionRetrievalStore(
            path,
            optimization_ledger=OptimizationLedger(_default_ledger_path()),
            require_scope=True,
        )

    @mcp.tool()
    def retrieve_compressed_span(
        receipt_id: str,
        span_id: str,
        store_path_override: str = "",
        retrieval_id: str = "",
    ) -> str:
        """Retrieve one in-scope span and debit returned tokens from savings."""
        try:
            store = _store(store_path_override)
            span = store.retrieve_span(
                receipt_id,
                span_id,
                retrieval_id=retrieval_id or None,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error_payload(exc, operation="retrieve_compressed_span")
        if span is None:
            return json.dumps(
                {"status": "not_found", "receipt_id": receipt_id, "span_id": span_id},
                indent=2,
            )
        return json.dumps(
            {"status": "ok", "span": span.as_dict()},
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def search_compressed_spans(
        query: str,
        limit: int = 5,
        store_path_override: str = "",
        retrieval_id: str = "",
        max_tokens_per_span: int = 600,
    ) -> str:
        """Search in-scope spans and return bounded exact excerpts."""
        try:
            store = _store(store_path_override)
            spans = store.search_exact_excerpts(
                query,
                limit=max(1, min(int(limit), 20)),
                max_tokens_per_span=max(32, min(int(max_tokens_per_span), 8_000)),
                record_retrieval=True,
                retrieval_id=retrieval_id or None,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error_payload(exc, operation="search_compressed_spans")
        return json.dumps(
            {"status": "ok", "query": query, "spans": [span.as_dict() for span in spans]},
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def list_compression_receipts(store_path_override: str = "") -> str:
        """List only receipts visible to this workspace/session scope."""
        try:
            store = _store(store_path_override)
            receipts = store.list_receipts()
            savings = store.savings_summary()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return _error_payload(exc, operation="list_compression_receipts")
        return json.dumps(
            {
                "status": "ok",
                "receipts": receipts,
                "savings": savings,
            },
            indent=2,
            ensure_ascii=False,
        )

    return mcp


def main() -> None:
    create_compression_mcp_server().run()


if __name__ == "__main__":
    main()


__all__ = ["create_compression_mcp_server"]
