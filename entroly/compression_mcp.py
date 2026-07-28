"""MCP tools for scoped Entroly compression retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Package initialization imports the core proxy before this focused MCP module.
# Activating the boundary layer here patches only transport contracts after the
# proxy class is fully defined and before normal users construct a proxy instance.
from . import proxy_transport_safe as _proxy_transport_safe  # noqa: F401
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

    try:
        allowed_root = Path(
            os.environ.get("ENTROLY_STORE_OVERRIDE_ROOT", str(selected.parent))
        ).expanduser().resolve(strict=True)
        if not allowed_root.is_dir():
            raise ValueError("override root is not a directory")

        override = Path(path_override).expanduser()
        raw_override = str(override)
        if (
            not raw_override
            or len(raw_override) > _MAX_PATH_CHARS
            or "\x00" in raw_override
            or override.name in {"", ".", ".."}
        ):
            raise ValueError("override path is not safe")
        # Requiring the parent to exist removes the create-time symlink race that
        # exists when several not-yet-created path components are accepted.
        resolved_parent = override.parent.resolve(strict=True)
        if not resolved_parent.is_dir():
            raise ValueError("override parent is not a directory")
        resolved_parent.relative_to(allowed_root)
        resolved = resolved_parent / override.name
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
