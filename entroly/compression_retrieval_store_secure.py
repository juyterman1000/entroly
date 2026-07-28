"""Final scoped public facade for the hardened compression retrieval store.

This thin layer closes compatibility edges that require changing receipt inputs
before the underlying engine derives its content address. It intentionally keeps
all storage, locking, snapshot, and accounting logic in
``compression_retrieval_store_safe``.
"""

from __future__ import annotations

import copy
import sys
from typing import Any

from . import compression_retrieval_store as _legacy_module
from . import compression_retrieval_store_safe as _safe_module
from .compression_retrieval_store import StoredCompression, StoredSpan
from .compression_retrieval_store_safe import (
    CompressionRetrievalStore as _SafeCompressionRetrievalStore,
    _MAX_QUERY_CHARS,
    derive_recovery_scope,
    sanitize_recovery_metadata,
)


class CompressionRetrievalStore(_SafeCompressionRetrievalStore):
    """Public secure recovery store with scope-derived content addresses."""

    def _scoped_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(receipt, dict):
            raise ValueError("receipt must be an object")
        scoped = copy.deepcopy(receipt)
        existing = scoped.get("_entroly_scope_sha256")
        if existing is not None and str(existing) != self.scope_hash:
            raise ValueError("receipt scope does not match this recovery store")
        scoped["_entroly_scope_sha256"] = self.scope_hash
        return scoped

    def put(
        self,
        *,
        original_text: str,
        compressed_text: str,
        receipt: dict[str, Any],
        metadata: dict[str, object] | None = None,
    ) -> StoredCompression:
        return super().put(
            original_text=original_text,
            compressed_text=compressed_text,
            receipt=self._scoped_receipt(receipt),
            metadata=metadata,
        )

    def put_exact_spans(
        self,
        *,
        original_text: str,
        compressed_text: str,
        receipt: dict[str, Any],
        spans: list[dict[str, Any]],
        metadata: dict[str, object] | None = None,
    ) -> StoredCompression:
        return super().put_exact_spans(
            original_text=original_text,
            compressed_text=compressed_text,
            receipt=self._scoped_receipt(receipt),
            spans=spans,
            metadata=metadata,
        )

    def search_exact_excerpts(
        self,
        query: str,
        *,
        limit: int = 5,
        max_tokens_per_span: int = 600,
        record_retrieval: bool = False,
        retrieval_id: str | None = None,
    ) -> list[StoredSpan]:
        if not isinstance(query, str) or len(query) > _MAX_QUERY_CHARS:
            raise ValueError("query must be bounded text")
        if isinstance(max_tokens_per_span, bool):
            raise ValueError("max_tokens_per_span must be an integer")
        try:
            token_limit = int(max_tokens_per_span)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_tokens_per_span must be an integer") from exc
        if token_limit < 32 or token_limit > 100_000:
            raise ValueError("max_tokens_per_span must be between 32 and 100000")
        return super().search_exact_excerpts(
            query,
            limit=limit,
            max_tokens_per_span=token_limit,
            record_retrieval=record_retrieval,
            retrieval_id=retrieval_id,
        )

    def list_receipts(self) -> list[dict[str, object]]:
        receipts = super().list_receipts()
        for receipt in receipts:
            metadata = receipt.get("metadata")
            if isinstance(metadata, dict):
                receipt["metadata"] = {
                    key: copy.deepcopy(value)
                    for key, value in metadata.items()
                    if not str(key).startswith("_")
                }
        return receipts

    def realized_savings_summary(self) -> dict[str, object]:
        with self._lock:
            self._refresh_if_changed()
            records = [
                item.savings_record()
                for item in self._items.values()
                if self._scope_allows(item)
            ]
        total: dict[str, object] = {
            "receipts": len(records),
            "gross_saved_tokens": 0,
            "retrieved_tokens": 0,
            "repeated_expansion_tokens": 0,
            "net_realized_saved_tokens": 0,
            "by_confidence": {},
        }
        by_confidence: dict[str, dict[str, int]] = {}
        for record in records:
            confidence = str(record["confidence"])
            bucket = by_confidence.setdefault(
                confidence,
                {
                    "receipts": 0,
                    "gross_saved_tokens": 0,
                    "retrieved_tokens": 0,
                    "repeated_expansion_tokens": 0,
                    "net_realized_saved_tokens": 0,
                },
            )
            bucket["receipts"] += 1
            for key in (
                "gross_saved_tokens",
                "retrieved_tokens",
                "repeated_expansion_tokens",
                "net_realized_saved_tokens",
            ):
                value = int(record[key])
                total[key] = int(total[key]) + value
                bucket[key] += value
        total["by_confidence"] = by_confidence
        return total


# Keep every historical import path and already-loaded product module aligned
# with this final facade. This matters because package initialization imports the
# proxy before the focused MCP surface that activates the secure store.
_safe_module.CompressionRetrievalStore = CompressionRetrievalStore
_legacy_module.CompressionRetrievalStore = CompressionRetrievalStore
for _module_name in (
    "entroly.compression_proxy",
    "entroly.compression_proxy_direct",
    "entroly.compression_dashboard",
    "entroly.compression_verification_loop",
    "entroly.session_rescue",
    "entroly.neural_evidence_selector",
    "entroly.proxy",
):
    _module = sys.modules.get(_module_name)
    if _module is not None and hasattr(_module, "CompressionRetrievalStore"):
        setattr(_module, "CompressionRetrievalStore", CompressionRetrievalStore)

__all__ = [
    "CompressionRetrievalStore",
    "StoredCompression",
    "StoredSpan",
    "derive_recovery_scope",
    "sanitize_recovery_metadata",
]
