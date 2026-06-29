"""Local retrieval store for Evidence-Locked Compression.

Headroom-style reversible compression requires two pieces:

1. a compressed prompt that keeps enough evidence for the first pass,
2. a local store that can retrieve omitted spans when the model needs more.

This module provides the second piece for Entroly. It is deterministic,
dependency-free, local-first, and intentionally small enough to audit.

The store also records omitted-span retrievals so Entroly reports net realized
savings rather than inflated gross compression savings.
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
    retrieval_count: int = 0
    retrieved_tokens: int = 0
    last_retrieved_ns: int = 0

    @property
    def token_estimate(self) -> int:
        return _estimate_tokens(self.content)

    def record_retrieval(self, *, tokens: int | None = None, retrieved_ns: int | None = None) -> None:
        self.retrieval_count += 1
        self.retrieved_tokens += max(0, self.token_estimate if tokens is None else int(tokens))
        self.last_retrieved_ns = time.time_ns() if retrieved_ns is None else int(retrieved_ns)

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

    @property
    def gross_saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def retrieved_tokens(self) -> int:
        return sum(max(0, span.retrieved_tokens) for span in self.spans)

    @property
    def repeated_expansion_tokens(self) -> int:
        return sum(
            max(0, span.retrieval_count - 1) * span.token_estimate
            for span in self.spans
        )

    @property
    def net_realized_saved_tokens(self) -> int:
        return max(
            0,
            self.gross_saved_tokens
            - self.retrieved_tokens
            - self.repeated_expansion_tokens,
        )

    def savings_record(self) -> dict[str, object]:
        confidence = str(self.metadata.get("savings_confidence", "measured"))
        return {
            "receipt_id": self.receipt_id,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "gross_saved_tokens": self.gross_saved_tokens,
            "retrieved_tokens": self.retrieved_tokens,
            "repeated_expansion_tokens": self.repeated_expansion_tokens,
            "net_realized_saved_tokens": self.net_realized_saved_tokens,
            "confidence": confidence,
        }

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
            metadata={"savings_confidence": "measured", **dict(metadata or {})},
        )
        self._items[receipt_id] = stored
        self._persist()
        return stored

    def get_receipt(self, receipt_id: str) -> StoredCompression | None:
        return self._items.get(receipt_id)

    def get_span(
        self,
        receipt_id: str,
        span_id: str,
        *,
        record_retrieval: bool = True,
    ) -> StoredSpan | None:
        item = self._items.get(receipt_id)
        if item is None:
            return None
        for span in item.spans:
            if span.span_id == span_id:
                if record_retrieval:
                    span.record_retrieval()
                    self._persist()
                return span
        return None

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        record_retrieval: bool = True,
    ) -> list[StoredSpan]:
        terms = {part.lower() for part in query.split() if len(part) >= 3}
        scored: list[tuple[int, StoredSpan]] = []
        for item in self._items.values():
            for span in item.spans:
                text = span.content.lower()
                score = sum(1 for term in terms if term in text)
                if score:
                    scored.append((score, span))
        scored.sort(key=lambda pair: (pair[0], pair[1].created_ns), reverse=True)
        spans = [span for _score, span in scored[:limit]]
        if record_retrieval:
            for span in spans:
                span.record_retrieval()
            if spans:
                self._persist()
        return spans

    def realized_savings(self, receipt_id: str) -> dict[str, object] | None:
        item = self._items.get(receipt_id)
        if item is None:
            return None
        return item.savings_record()

    def realized_savings_summary(self) -> dict[str, object]:
        records = [item.savings_record() for item in self._items.values()]
        total = {
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

    def list_receipts(self) -> list[dict[str, object]]:
        return [
            {
                "receipt_id": item.receipt_id,
                "original_tokens": item.original_tokens,
                "compressed_tokens": item.compressed_tokens,
                "gross_saved_tokens": item.gross_saved_tokens,
                "retrieved_tokens": item.retrieved_tokens,
                "repeated_expansion_tokens": item.repeated_expansion_tokens,
                "net_realized_saved_tokens": item.net_realized_saved_tokens,
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


def _estimate_tokens(text: str) -> int:
    return max(1, len((text or "").encode("utf-8")) // 4)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _short_hash(text: str) -> str:
    return _sha256_text(text)[:16]


__all__ = [
    "CompressionRetrievalStore",
    "StoredCompression",
    "StoredSpan",
]
