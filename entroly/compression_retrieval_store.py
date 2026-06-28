"""Local retrieval store for Evidence-Locked Compression.

Headroom-style reversible compression requires two pieces:

1. a compressed prompt that keeps enough evidence for the first pass,
2. a local store that can retrieve omitted spans when the model needs more.

This module provides the second piece for Entroly. It is deterministic,
dependency-free, local-first, and intentionally small enough to audit.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StoredSpan:
    span_id: str
    receipt_id: str
    start_line: int
    end_line: int
    content: str
    reason: str = "budget"
    created_ns: int = field(default_factory=time.time_ns)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class StoredCompression:
    receipt_id: str
    original_hash: str
    original_tokens: int
    compressed_tokens: int
    spans: list[StoredSpan]
    metadata: dict[str, object] = field(default_factory=dict)
    created_ns: int = field(default_factory=time.time_ns)

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["spans"] = [span.as_dict() for span in self.spans]
        return data


class CompressionRetrievalStore:
    """Local store for compressed omitted spans."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._items: dict[str, StoredCompression] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def put(
        self,
        *,
        original_text: str,
        compressed_text: str,
        receipt: dict[str, Any],
        metadata: dict[str, object] | None = None,
    ) -> StoredCompression:
        """Store omitted spans from an ELC receipt.

        The receipt must contain ``omitted_spans`` with 1-based line ranges.
        The original text never leaves the local store.
        """
        original_hash = _sha256_text(original_text)
        receipt_id = _short_hash(
            json.dumps(receipt, sort_keys=True, default=str) + original_hash
        )
        lines = original_text.splitlines()
        spans: list[StoredSpan] = []
        for raw in receipt.get("omitted_spans", []) or []:
            if not isinstance(raw, dict):
                continue
            start = int(raw.get("start_line", 1))
            end = int(raw.get("end_line", start))
            start = max(1, min(start, len(lines) or 1))
            end = max(start, min(end, len(lines) or start))
            content = "\n".join(lines[start - 1 : end])
            span_id = _short_hash(f"{receipt_id}:{start}:{end}:{_sha256_text(content)}")
            spans.append(
                StoredSpan(
                    span_id=span_id,
                    receipt_id=receipt_id,
                    start_line=start,
                    end_line=end,
                    content=content,
                    reason=str(raw.get("reason", "budget")),
                )
            )

        stored = StoredCompression(
            receipt_id=receipt_id,
            original_hash=original_hash,
            original_tokens=int(receipt.get("original_tokens", 0)),
            compressed_tokens=int(receipt.get("compressed_tokens", 0)),
            spans=spans,
            metadata=dict(metadata or {}),
        )
        self._items[receipt_id] = stored
        self._persist()
        return stored

    def get_receipt(self, receipt_id: str) -> StoredCompression | None:
        return self._items.get(receipt_id)

    def get_span(self, receipt_id: str, span_id: str) -> StoredSpan | None:
        item = self._items.get(receipt_id)
        if item is None:
            return None
        for span in item.spans:
            if span.span_id == span_id:
                return span
        return None

    def search(self, query: str, *, limit: int = 5) -> list[StoredSpan]:
        terms = {part.lower() for part in query.split() if len(part) >= 3}
        scored: list[tuple[int, StoredSpan]] = []
        for item in self._items.values():
            for span in item.spans:
                text = span.content.lower()
                score = sum(1 for term in terms if term in text)
                if score:
                    scored.append((score, span))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_ns), reverse=True)
        return [span for _score, span in scored[:limit]]

    def list_receipts(self) -> list[dict[str, object]]:
        return [
            {
                "receipt_id": item.receipt_id,
                "original_tokens": item.original_tokens,
                "compressed_tokens": item.compressed_tokens,
                "span_count": len(item.spans),
                "created_ns": item.created_ns,
                "metadata": item.metadata,
            }
            for item in sorted(self._items.values(), key=lambda i: i.created_ns, reverse=True)
        ]

    def _load(self) -> None:
        assert self.path is not None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw.get("items", []):
            spans = [StoredSpan(**span) for span in item.get("spans", [])]
            stored = StoredCompression(
                receipt_id=item["receipt_id"],
                original_hash=item["original_hash"],
                original_tokens=int(item.get("original_tokens", 0)),
                compressed_tokens=int(item.get("compressed_tokens", 0)),
                spans=spans,
                metadata=dict(item.get("metadata", {})),
                created_ns=int(item.get("created_ns", time.time_ns())),
            )
            self._items[stored.receipt_id] = stored

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": [item.as_dict() for item in self._items.values()]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _short_hash(text: str) -> str:
    return _sha256_text(text)[:16]


__all__ = [
    "CompressionRetrievalStore",
    "StoredCompression",
    "StoredSpan",
]
