"""Local retrieval store for Evidence-Locked Compression.

The store records gross compression once and debits every span actually returned
to an agent. Plain inspection methods remain side-effect free; callers use the
explicit ``retrieve_*`` methods when content crosses the model boundary.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .optimization_ledger import OptimizationLedger


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
    last_retrieved_ns: int | None = None

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
    retrieved_tokens: int = 0
    retrieval_count: int = 0
    last_retrieved_ns: int | None = None
    savings_tier: str = "measured"

    @property
    def gross_tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def net_tokens_saved(self) -> int:
        return self.gross_tokens_saved - self.retrieved_tokens

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["spans"] = [span.as_dict() for span in self.spans]
        data["gross_tokens_saved"] = self.gross_tokens_saved
        data["net_tokens_saved"] = self.net_tokens_saved
        return data


class CompressionRetrievalStore:
    """Thread-safe local store with retrieval-adjusted savings accounting."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        optimization_ledger: OptimizationLedger | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.optimization_ledger = optimization_ledger
        self._items: dict[str, StoredCompression] = {}
        self._lock = threading.RLock()
        if self.path is not None and self.path.exists():
            self._load()
        if self.optimization_ledger is not None:
            for item in self._items.values():
                self._record_compression(item)

    def put(
        self,
        *,
        original_text: str,
        compressed_text: str,
        receipt: dict[str, Any],
        metadata: dict[str, object] | None = None,
    ) -> StoredCompression:
        """Store omitted spans and record measured gross savings once."""
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
        with self._lock:
            previous = self._items.get(receipt_id)
            if previous is not None:
                return previous
            self._items[receipt_id] = stored
            self._persist()
        self._record_compression(stored)
        return stored

    def get_receipt(self, receipt_id: str) -> StoredCompression | None:
        with self._lock:
            return self._items.get(receipt_id)

    def get_span(self, receipt_id: str, span_id: str) -> StoredSpan | None:
        """Inspect a span without charging retrieval accounting."""
        with self._lock:
            item = self._items.get(receipt_id)
            if item is None:
                return None
            return next((span for span in item.spans if span.span_id == span_id), None)

    def retrieve_span(
        self,
        receipt_id: str,
        span_id: str,
        *,
        retrieval_id: str | None = None,
    ) -> StoredSpan | None:
        """Return a span and debit its tokens from measured gross savings."""
        with self._lock:
            span = self.get_span(receipt_id, span_id)
            item = self._items.get(receipt_id)
            if span is None or item is None:
                return None
            token_count = _estimate_tokens(span.content)
            now_ns = time.time_ns()
            span.retrieval_count += 1
            span.retrieved_tokens += token_count
            span.last_retrieved_ns = now_ns
            item.retrieval_count += 1
            item.retrieved_tokens += token_count
            item.last_retrieved_ns = now_ns
            self._persist()
        self._record_retrieval(
            receipt_id,
            span_id,
            token_count,
            retrieval_id=retrieval_id or f"{receipt_id}:{span_id}:{now_ns}",
        )
        return span

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        record_retrieval: bool = False,
        retrieval_id: str | None = None,
    ) -> list[StoredSpan]:
        terms = {part.lower() for part in query.split() if len(part) >= 3}
        with self._lock:
            scored: list[tuple[int, StoredSpan]] = []
            for item in self._items.values():
                for span in item.spans:
                    text = span.content.lower()
                    score = sum(1 for term in terms if term in text)
                    if score:
                        scored.append((score, span))
            scored.sort(key=lambda pair: (pair[0], pair[1].created_ns), reverse=True)
            selected = [span for _score, span in scored[: max(0, int(limit))]]
        if not record_retrieval:
            return selected
        base_id = retrieval_id or f"search:{_short_hash(query)}:{time.time_ns()}"
        recorded: list[StoredSpan] = []
        for index, span in enumerate(selected):
            returned = self.retrieve_span(
                span.receipt_id,
                span.span_id,
                retrieval_id=f"{base_id}:{index}",
            )
            if returned is not None:
                recorded.append(returned)
        return recorded

    def list_receipts(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "receipt_id": item.receipt_id,
                    "original_tokens": item.original_tokens,
                    "compressed_tokens": item.compressed_tokens,
                    "gross_tokens_saved": item.gross_tokens_saved,
                    "retrieved_tokens": item.retrieved_tokens,
                    "net_tokens_saved": item.net_tokens_saved,
                    "retrieval_count": item.retrieval_count,
                    "savings_tier": item.savings_tier,
                    "span_count": len(item.spans),
                    "created_ns": item.created_ns,
                    "metadata": item.metadata,
                }
                for item in sorted(
                    self._items.values(), key=lambda entry: entry.created_ns, reverse=True
                )
            ]

    def savings_summary(self) -> dict[str, int | str]:
        with self._lock:
            gross = sum(item.gross_tokens_saved for item in self._items.values())
            retrieved = sum(item.retrieved_tokens for item in self._items.values())
        return {
            "tier": "measured",
            "gross_tokens_saved": gross,
            "retrieved_tokens": retrieved,
            "net_tokens_saved": gross - retrieved,
        }

    def _record_compression(self, item: StoredCompression) -> None:
        if self.optimization_ledger is None:
            return
        from .optimization_ledger import OptimizationEvent, SavingsTier

        self.optimization_ledger.record(
            OptimizationEvent(
                event_id=f"compression:{item.receipt_id}",
                feature="evidence_locked_compression",
                tier=SavingsTier.MEASURED,
                gross_tokens_saved=max(0, item.gross_tokens_saved),
                session_id=str(item.metadata.get("session_id", "")),
                conversation_id=str(item.metadata.get("conversation_id", "")),
                provider=str(item.metadata.get("provider", "")),
                model=str(item.metadata.get("model", "")),
                metadata={"receipt_id": item.receipt_id},
            )
        )

    def _record_retrieval(
        self,
        receipt_id: str,
        span_id: str,
        token_count: int,
        *,
        retrieval_id: str,
    ) -> None:
        if self.optimization_ledger is None:
            return
        from .optimization_ledger import OptimizationAdjustment

        self.optimization_ledger.adjust(
            OptimizationAdjustment(
                adjustment_id=retrieval_id,
                event_id=f"compression:{receipt_id}",
                tokens_reexpanded=token_count,
                metadata={"receipt_id": receipt_id, "span_id": span_id},
            )
        )

    def _load(self) -> None:
        assert self.path is not None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw.get("items", []):
            spans = [
                StoredSpan(
                    span_id=span["span_id"],
                    receipt_id=span["receipt_id"],
                    start_line=int(span["start_line"]),
                    end_line=int(span["end_line"]),
                    content=str(span.get("content", "")),
                    reason=str(span.get("reason", "budget")),
                    created_ns=int(span.get("created_ns", time.time_ns())),
                    retrieval_count=int(span.get("retrieval_count", 0)),
                    retrieved_tokens=int(span.get("retrieved_tokens", 0)),
                    last_retrieved_ns=span.get("last_retrieved_ns"),
                )
                for span in item.get("spans", [])
            ]
            stored = StoredCompression(
                receipt_id=item["receipt_id"],
                original_hash=item["original_hash"],
                original_tokens=int(item.get("original_tokens", 0)),
                compressed_tokens=int(item.get("compressed_tokens", 0)),
                spans=spans,
                metadata=dict(item.get("metadata", {})),
                created_ns=int(item.get("created_ns", time.time_ns())),
                retrieved_tokens=int(item.get("retrieved_tokens", 0)),
                retrieval_count=int(item.get("retrieval_count", 0)),
                last_retrieved_ns=item.get("last_retrieved_ns"),
                savings_tier=str(item.get("savings_tier", "measured")),
            )
            self._items[stored.receipt_id] = stored

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 2, "items": [item.as_dict() for item in self._items.values()]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _short_hash(text: str) -> str:
    return _sha256_text(text)[:16]


__all__ = ["CompressionRetrievalStore", "StoredCompression", "StoredSpan"]
