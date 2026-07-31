"""Dashboard surface for Entroly compression receipts.

This module turns ELC/proxy receipts and retrieval-store metadata into a compact
JSON/Markdown dashboard. It is intentionally framework-free so it can power a
CLI, HTTP `/dashboard` route, docs page, or MCP tool without introducing a web UI
stack.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .compression_retrieval_store import CompressionRetrievalStore


@dataclass(slots=True)
class CompressionDashboard:
    receipts_seen: int
    compressed_blocks: int
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    savings_ratio: float
    recoverable_spans: int
    retrieval_receipts: int
    recent_receipts: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [
            "# Entroly Compression Dashboard",
            "",
            f"- Receipts seen: **{self.receipts_seen}**",
            f"- Compressed blocks: **{self.compressed_blocks}**",
            f"- Original tokens: **{self.original_tokens:,}**",
            f"- Compressed tokens: **{self.compressed_tokens:,}**",
            f"- Tokens saved: **{self.tokens_saved:,}**",
            f"- Savings ratio: **{self.savings_ratio:.1%}**",
            f"- Recoverable spans: **{self.recoverable_spans}**",
            f"- Retrieval receipts: **{self.retrieval_receipts}**",
            "",
        ]
        if self.recent_receipts:
            lines.append("## Recent Retrieval Receipts")
            lines.append("")
            lines.append("| receipt_id | spans | original_tokens | compressed_tokens |")
            lines.append("|---|---:|---:|---:|")
            for item in self.recent_receipts[:10]:
                lines.append(
                    "| {receipt_id} | {span_count} | {original_tokens} | {compressed_tokens} |".format(
                        receipt_id=str(item.get("receipt_id", ""))[:16],
                        span_count=int(item.get("span_count", 0)),
                        original_tokens=int(item.get("original_tokens", 0)),
                        compressed_tokens=int(item.get("compressed_tokens", 0)),
                    )
                )
        return "\n".join(lines)


def dashboard_from_proxy_receipts(
    receipts: list[dict[str, Any]],
    *,
    store: CompressionRetrievalStore | None = None,
) -> CompressionDashboard:
    original = 0
    compressed = 0
    recoverable_spans = 0
    retrieval_receipts = 0
    compressed_blocks = 0

    for receipt in receipts:
        original += int(receipt.get("original_tokens", 0) or 0)
        compressed += int(receipt.get("compressed_tokens", 0) or 0)
        if receipt.get("compression_level", 0):
            compressed_blocks += 1
        omitted = receipt.get("omitted_spans", []) or []
        if isinstance(omitted, list):
            recoverable_spans += len(omitted)
        retrieval = receipt.get("retrieval")
        if isinstance(retrieval, dict):
            retrieval_receipts += 1
            recoverable_spans += int(retrieval.get("span_count", 0) or 0)

    recent: list[dict[str, object]] = store.list_receipts() if store is not None else []
    saved = max(0, original - compressed)
    return CompressionDashboard(
        receipts_seen=len(receipts),
        compressed_blocks=compressed_blocks,
        original_tokens=original,
        compressed_tokens=compressed,
        tokens_saved=saved,
        savings_ratio=0.0 if original == 0 else saved / original,
        recoverable_spans=recoverable_spans,
        retrieval_receipts=retrieval_receipts,
        recent_receipts=recent,
    )


def dashboard_from_store(store: CompressionRetrievalStore) -> CompressionDashboard:
    receipts = store.list_receipts()
    original = sum(int(item.get("original_tokens", 0) or 0) for item in receipts)
    compressed = sum(int(item.get("compressed_tokens", 0) or 0) for item in receipts)
    spans = sum(int(item.get("span_count", 0) or 0) for item in receipts)
    saved = max(0, original - compressed)
    return CompressionDashboard(
        receipts_seen=len(receipts),
        compressed_blocks=len(receipts),
        original_tokens=original,
        compressed_tokens=compressed,
        tokens_saved=saved,
        savings_ratio=0.0 if original == 0 else saved / original,
        recoverable_spans=spans,
        retrieval_receipts=len(receipts),
        recent_receipts=receipts,
    )


__all__ = [
    "CompressionDashboard",
    "dashboard_from_proxy_receipts",
    "dashboard_from_store",
]
