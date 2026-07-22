"""Local retrieval store for Evidence-Locked Compression.

Reversible compression requires two pieces:

1. a compressed prompt that keeps enough evidence for the first pass,
2. a local store that can retrieve omitted spans when the model needs more.

This module provides the second piece for Entroly. It is deterministic,
dependency-free, local-first, and intentionally small enough to audit.

The store records gross compression once and debits every span returned to an
agent. Explicit retrieval IDs make those debits idempotent across retries.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

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
    source_id: str = ""
    start_char: int | None = None
    end_char: int | None = None
    content_sha256: str = ""
    created_ns: int = field(default_factory=time.time_ns)
    retrieval_count: int = 0
    retrieved_tokens: int = 0
    last_retrieved_ns: int = 0
    retrieval_ids: list[str] = field(default_factory=list)

    @property
    def token_estimate(self) -> int:
        return _estimate_tokens(self.content)

    def record_retrieval(
        self,
        *,
        tokens: int | None = None,
        retrieved_ns: int | None = None,
        retrieval_id: str | None = None,
    ) -> bool:
        if retrieval_id is not None and retrieval_id in self.retrieval_ids:
            return False
        self.retrieval_count += 1
        self.retrieved_tokens += max(0, self.token_estimate if tokens is None else int(tokens))
        self.last_retrieved_ns = time.time_ns() if retrieved_ns is None else int(retrieved_ns)
        if retrieval_id is not None:
            self.retrieval_ids.append(retrieval_id)
        return True

    def as_dict(self, *, include_internal: bool = False) -> dict[str, object]:
        data = asdict(self)
        if not include_internal:
            data.pop("retrieval_ids", None)
        return data


@dataclass(slots=True)
class StoredCompression:
    receipt_id: str
    original_hash: str
    original_tokens: int
    compressed_tokens: int
    spans: list[StoredSpan]
    metadata: dict[str, object] = field(default_factory=dict)
    created_ns: int = field(default_factory=time.time_ns)
    retrieval_count: int = 0
    last_retrieved_ns: int | None = None
    savings_tier: str = "measured"

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

    @property
    def gross_tokens_saved(self) -> int:
        return self.gross_saved_tokens

    @property
    def net_tokens_saved(self) -> int:
        return self.net_realized_saved_tokens

    def as_dict(self, *, include_internal: bool = False) -> dict[str, object]:
        data = asdict(self)
        data["spans"] = [
            span.as_dict(include_internal=include_internal) for span in self.spans
        ]
        data["gross_tokens_saved"] = self.gross_tokens_saved
        data["net_tokens_saved"] = self.net_tokens_saved
        return data


class CompressionRetrievalStore:
    """Process-safe local store with retrieval-adjusted savings accounting."""

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
        self._disk_signature: tuple[int, int, int] | None = None
        if self.path is not None and self.path.exists():
            self._load()
        if self.optimization_ledger is not None:
            for item in self._items.values():
                self._record_compression(item)
                for span in item.spans:
                    token_count = _estimate_tokens(span.content)
                    for retrieval_id in span.retrieval_ids:
                        self._record_retrieval(
                            item.receipt_id,
                            span.span_id,
                            token_count,
                            retrieval_id=retrieval_id,
                        )

    @property
    def _lock_path(self) -> Path | None:
        if self.path is None:
            return None
        return self.path.with_name(self.path.name + ".lock")

    @staticmethod
    def _signature(stat: os.stat_result) -> tuple[int, int, int]:
        return (int(stat.st_mtime_ns), int(stat.st_size), int(stat.st_ino))

    @contextmanager
    def _interprocess_lock(self) -> Iterator[None]:
        """Serialize disk mutations across independent Entroly processes."""
        lock_path = self._lock_path
        if lock_path is None:
            yield
            return
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                deadline = time.monotonic() + 30.0
                delay = 0.001
                while True:
                    try:
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as error:
                        if error.errno not in {errno.EACCES, errno.EAGAIN}:
                            raise
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"timed out acquiring recovery-store lock {lock_path}"
                            ) from error
                        time.sleep(delay)
                        delay = min(0.05, delay * 1.5)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _refresh_if_changed(self, *, force: bool = False) -> None:
        if self.path is None or not self.path.exists():
            return
        current = self._signature(self.path.stat())
        if force or current != self._disk_signature:
            self._load()

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
            metadata={"savings_confidence": "measured", **dict(metadata or {})},
        )
        with self._lock:
            with self._interprocess_lock():
                self._refresh_if_changed(force=True)
                previous = self._items.get(receipt_id)
                if previous is not None:
                    return previous
                self._items[receipt_id] = stored
                try:
                    self._persist()
                except Exception:
                    self._items.pop(receipt_id, None)
                    raise
        self._record_compression(stored)
        return stored

    def put_exact_spans(
        self,
        *,
        original_text: str,
        compressed_text: str,
        receipt: dict[str, Any],
        spans: list[dict[str, Any]],
        metadata: dict[str, object] | None = None,
    ) -> StoredCompression:
        """Atomically persist exact character-addressed omitted source spans.

        Every span is validated against ``original_text`` before any state is
        mutated. This prevents a stale offset or forged content hash from
        creating a partially recoverable receipt.
        """
        original_hash = _sha256_text(original_text)
        receipt_id = _short_hash(
            json.dumps(receipt, sort_keys=True, default=str) + original_hash
        )
        receipt_spans = receipt.get("spans")
        if not isinstance(receipt_spans, list):
            raise ValueError("exact span persistence requires receipt span records")
        receipt_by_id: dict[str, dict[str, Any]] = {}
        expected_omitted_ids: set[str] = set()
        selected_records: list[dict[str, Any]] = []
        for record in receipt_spans:
            if not isinstance(record, dict):
                raise ValueError("receipt span records must be objects")
            span_id = str(record.get("span_id", ""))
            if not span_id or span_id in receipt_by_id:
                raise ValueError(
                    "receipt spans require unique non-empty span_id values"
                )
            start = int(record.get("start_char", -1))
            end_value = record.get("end_char")
            end = int(end_value) if end_value is not None else -1
            if start < 0 or end <= start or end > len(original_text):
                raise ValueError(f"invalid receipt character range for span {span_id}")
            content_hash = _sha256_text(original_text[start:end])
            if str(record.get("content_sha256", "")) != content_hash:
                raise ValueError(f"receipt content hash mismatch for span {span_id}")
            receipt_by_id[span_id] = record
            if bool(record.get("selected")):
                selected_records.append(record)
            else:
                expected_omitted_ids.add(span_id)

        expected_input_tokens = sum(
            int(record["token_count"]) for record in receipt_spans
        )
        expected_selected_tokens = sum(
            int(record["token_count"]) for record in selected_records
        )
        if int(receipt.get("input_tokens", -1)) != expected_input_tokens:
            raise ValueError("receipt input token count does not match its spans")
        if int(receipt.get("selected_tokens", -1)) != expected_selected_tokens:
            raise ValueError("receipt selected token count does not match its spans")
        ordered_selected = sorted(
            selected_records,
            key=lambda record: (
                int(record.get("ordinal", 0)),
                str(record.get("source", "")),
                int(record.get("start_char", 0)),
                str(record.get("span_id", "")),
            ),
        )
        expected_compressed_text = "\n\n".join(
            original_text[int(record["start_char"]) : int(record["end_char"])]
            for record in ordered_selected
        )
        if compressed_text != expected_compressed_text:
            raise ValueError(
                "compressed text does not match receipt-selected source spans"
            )

        supplied_ids = {str(raw.get("span_id", "")) for raw in spans}
        if supplied_ids != expected_omitted_ids or len(supplied_ids) != len(spans):
            raise ValueError("supplied exact spans do not match every receipt omission")
        validated: list[StoredSpan] = []
        seen_ids: set[str] = set()
        for raw in spans:
            span_id = str(raw.get("span_id", ""))
            if not span_id or span_id in seen_ids:
                raise ValueError("exact spans require unique non-empty span_id values")
            seen_ids.add(span_id)
            start = int(raw.get("start_char", -1))
            end = int(raw.get("end_char", -1))
            if start < 0 or end <= start or end > len(original_text):
                raise ValueError(f"invalid character range for span {span_id}")
            content = original_text[start:end]
            expected_hash = str(raw.get("content_sha256", ""))
            actual_hash = _sha256_text(content)
            if not expected_hash or expected_hash != actual_hash:
                raise ValueError(f"content hash mismatch for span {span_id}")
            receipt_record = receipt_by_id[span_id]
            receipt_contract = {
                "source": str(receipt_record.get("source", "")),
                "start_char": int(receipt_record["start_char"]),
                "end_char": int(receipt_record["end_char"]),
                "content_sha256": str(receipt_record["content_sha256"]),
            }
            supplied_contract = {
                "source": str(raw.get("source", "")),
                "start_char": start,
                "end_char": end,
                "content_sha256": expected_hash,
            }
            if supplied_contract != receipt_contract:
                raise ValueError(f"span {span_id} does not match its receipt record")
            start_line = original_text.count("\n", 0, start) + 1
            end_line = original_text.count("\n", 0, end - 1) + 1
            validated.append(
                StoredSpan(
                    span_id=span_id,
                    receipt_id=receipt_id,
                    start_line=start_line,
                    end_line=max(start_line, end_line),
                    content=content,
                    reason=str(raw.get("reason", "neural_budget")),
                    source_id=str(raw.get("source", "")),
                    start_char=start,
                    end_char=end,
                    content_sha256=actual_hash,
                )
            )

        stored = StoredCompression(
            receipt_id=receipt_id,
            original_hash=original_hash,
            original_tokens=int(
                receipt.get("input_tokens", receipt.get("original_tokens", 0))
            ),
            compressed_tokens=int(
                receipt.get("selected_tokens", receipt.get("compressed_tokens", 0))
            ),
            spans=validated,
            metadata={
                "savings_confidence": "measured",
                "compressed_sha256": _sha256_text(compressed_text),
                **dict(metadata or {}),
            },
        )
        with self._lock:
            with self._interprocess_lock():
                self._refresh_if_changed(force=True)
                previous = self._items.get(receipt_id)
                if previous is not None:
                    return previous
                self._items[receipt_id] = stored
                try:
                    self._persist()
                except Exception:
                    self._items.pop(receipt_id, None)
                    raise
        self._record_compression(stored)
        return stored

    def get_receipt(self, receipt_id: str) -> StoredCompression | None:
        with self._lock:
            self._refresh_if_changed()
            return self._items.get(receipt_id)

    def get_span(
        self,
        receipt_id: str,
        span_id: str,
        *,
        record_retrieval: bool = False,
    ) -> StoredSpan | None:
        with self._lock:
            self._refresh_if_changed()
            item = self._items.get(receipt_id)
            if item is None:
                return None
            span = next((entry for entry in item.spans if entry.span_id == span_id), None)
        if span is None or not record_retrieval:
            return span
        return self.retrieve_span(receipt_id, span_id)

    def retrieve_span(
        self,
        receipt_id: str,
        span_id: str,
        *,
        retrieval_id: str | None = None,
    ) -> StoredSpan | None:
        """Return a span and debit its tokens from measured gross savings."""
        with self._lock:
            with self._interprocess_lock():
                self._refresh_if_changed(force=True)
                item = self._items.get(receipt_id)
                if item is None:
                    return None
                span = next(
                    (entry for entry in item.spans if entry.span_id == span_id),
                    None,
                )
                if span is None:
                    return None
                token_count = _estimate_tokens(span.content)
                now_ns = time.time_ns()
                effective_id = retrieval_id or (
                    f"{receipt_id}:{span_id}:{span.retrieval_count + 1}"
                )
                if effective_id in span.retrieval_ids:
                    return span
                previous_span_state = (
                    span.retrieval_count,
                    span.retrieved_tokens,
                    span.last_retrieved_ns,
                    list(span.retrieval_ids),
                )
                previous_item_state = (item.retrieval_count, item.last_retrieved_ns)
                if not span.record_retrieval(
                    tokens=token_count,
                    retrieved_ns=now_ns,
                    retrieval_id=effective_id,
                ):
                    return span
                item.retrieval_count += 1
                item.last_retrieved_ns = now_ns
                try:
                    self._persist()
                except Exception:
                    (
                        span.retrieval_count,
                        span.retrieved_tokens,
                        span.last_retrieved_ns,
                        span.retrieval_ids,
                    ) = previous_span_state
                    item.retrieval_count, item.last_retrieved_ns = previous_item_state
                    raise
        self._record_retrieval(
            receipt_id,
            span_id,
            token_count,
            retrieval_id=effective_id,
        )
        return span

    def retrieve_span_excerpt(
        self,
        receipt_id: str,
        span_id: str,
        *,
        query: str,
        max_tokens: int = 600,
        retrieval_id: str | None = None,
    ) -> StoredSpan | None:
        """Return an exact query-local excerpt and debit only emitted tokens.

        The full stored span remains untouched and is still available through
        :meth:`retrieve_span`. Oversized spans are sliced with the same exact
        lead-plus-query-window policy used by live CCR recovery; no generated
        summary is introduced and every omitted gap is marked explicitly.
        """
        if int(max_tokens) < 32:
            raise ValueError("max_tokens must be at least 32")
        with self._lock:
            with self._interprocess_lock():
                self._refresh_if_changed(force=True)
                item = self._items.get(receipt_id)
                if item is None:
                    return None
                span = next(
                    (entry for entry in item.spans if entry.span_id == span_id),
                    None,
                )
                if span is None:
                    return None
                excerpt, was_sliced = _bounded_exact_excerpt(
                    span.content,
                    query,
                    int(max_tokens),
                )
                if not excerpt:
                    return None
                token_count = _count_o200k_tokens(excerpt)
                now_ns = time.time_ns()
                effective_id = retrieval_id or (
                    f"{receipt_id}:{span_id}:excerpt:{span.retrieval_count + 1}"
                )
                already_recorded = effective_id in span.retrieval_ids
                if not already_recorded:
                    previous_span_state = (
                        span.retrieval_count,
                        span.retrieved_tokens,
                        span.last_retrieved_ns,
                        list(span.retrieval_ids),
                    )
                    previous_item_state = (item.retrieval_count, item.last_retrieved_ns)
                    if not span.record_retrieval(
                        tokens=token_count,
                        retrieved_ns=now_ns,
                        retrieval_id=effective_id,
                    ):
                        already_recorded = True
                    else:
                        item.retrieval_count += 1
                        item.last_retrieved_ns = now_ns
                        try:
                            self._persist()
                        except Exception:
                            (
                                span.retrieval_count,
                                span.retrieved_tokens,
                                span.last_retrieved_ns,
                                span.retrieval_ids,
                            ) = previous_span_state
                            item.retrieval_count, item.last_retrieved_ns = (
                                previous_item_state
                            )
                            raise
                view = replace(
                    span,
                    content=excerpt,
                    reason=(
                        f"{span.reason}:exact_excerpt" if was_sliced else span.reason
                    ),
                    start_char=None if was_sliced else span.start_char,
                    end_char=None if was_sliced else span.end_char,
                    content_sha256=_sha256_text(excerpt),
                    retrieval_ids=list(span.retrieval_ids),
                )
        if not already_recorded:
            self._record_retrieval(
                receipt_id,
                span_id,
                token_count,
                retrieval_id=effective_id,
            )
        return view

    def search_exact_excerpts(
        self,
        query: str,
        *,
        limit: int = 5,
        max_tokens_per_span: int = 600,
        record_retrieval: bool = False,
        retrieval_id: str | None = None,
    ) -> list[StoredSpan]:
        """Search omitted spans and return bounded exact query-local excerpts."""
        selected = self.search(query, limit=limit, record_retrieval=False)
        if not record_retrieval:
            excerpts: list[StoredSpan] = []
            for span in selected:
                content, was_sliced = _bounded_exact_excerpt(
                    span.content,
                    query,
                    int(max_tokens_per_span),
                )
                if content:
                    excerpts.append(
                        replace(
                            span,
                            content=content,
                            reason=(
                                f"{span.reason}:exact_excerpt"
                                if was_sliced
                                else span.reason
                            ),
                            start_char=None if was_sliced else span.start_char,
                            end_char=None if was_sliced else span.end_char,
                            content_sha256=_sha256_text(content),
                            retrieval_ids=list(span.retrieval_ids),
                        )
                    )
            return excerpts
        base_id = (
            retrieval_id or f"excerpt-search:{_short_hash(query)}:{time.time_ns()}"
        )
        recorded: list[StoredSpan] = []
        for index, span in enumerate(selected):
            returned = self.retrieve_span_excerpt(
                span.receipt_id,
                span.span_id,
                query=query,
                max_tokens=max_tokens_per_span,
                retrieval_id=f"{base_id}:{index}",
            )
            if returned is not None:
                recorded.append(returned)
        return recorded

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        record_retrieval: bool = False,
        retrieval_id: str | None = None,
    ) -> list[StoredSpan]:
        terms = {term.casefold() for term in _EXCERPT_TERM_RE.findall(query)}
        with self._lock:
            self._refresh_if_changed()
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

    def realized_savings(self, receipt_id: str) -> dict[str, object] | None:
        with self._lock:
            self._refresh_if_changed()
            item = self._items.get(receipt_id)
            return item.savings_record() if item is not None else None

    def realized_savings_summary(self) -> dict[str, object]:
        with self._lock:
            self._refresh_if_changed()
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
        with self._lock:
            self._refresh_if_changed()
            return [
                {
                    "receipt_id": item.receipt_id,
                    "original_tokens": item.original_tokens,
                    "compressed_tokens": item.compressed_tokens,
                    "gross_tokens_saved": item.gross_tokens_saved,
                    "retrieved_tokens": item.retrieved_tokens,
                    "net_tokens_saved": item.net_tokens_saved,
                    "gross_saved_tokens": item.gross_saved_tokens,
                    "repeated_expansion_tokens": item.repeated_expansion_tokens,
                    "net_realized_saved_tokens": item.net_realized_saved_tokens,
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
            self._refresh_if_changed()
            gross = sum(item.gross_tokens_saved for item in self._items.values())
            retrieved = sum(item.retrieved_tokens for item in self._items.values())
        return {
            "tier": "measured",
            "gross_tokens_saved": gross,
            "retrieved_tokens": retrieved,
            "net_tokens_saved": max(0, gross - retrieved),
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
        with self.path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
            signature = self._signature(os.fstat(handle.fileno()))
        schema_version = int(raw.get("schema_version", 1))
        if schema_version not in {1, 2, 3}:
            raise ValueError(
                f"unsupported recovery store schema_version {schema_version}"
            )
        loaded: dict[str, StoredCompression] = {}
        for item in raw.get("items", []):
            spans = [
                StoredSpan(
                    span_id=span["span_id"],
                    receipt_id=span["receipt_id"],
                    start_line=int(span["start_line"]),
                    end_line=int(span["end_line"]),
                    content=str(span.get("content", "")),
                    reason=str(span.get("reason", "budget")),
                    source_id=str(span.get("source_id", "")),
                    start_char=(
                        int(span["start_char"])
                        if span.get("start_char") is not None
                        else None
                    ),
                    end_char=(
                        int(span["end_char"])
                        if span.get("end_char") is not None
                        else None
                    ),
                    content_sha256=str(span.get("content_sha256", "")),
                    created_ns=int(span.get("created_ns", time.time_ns())),
                    retrieval_count=int(span.get("retrieval_count", 0)),
                    retrieved_tokens=int(span.get("retrieved_tokens", 0)),
                    last_retrieved_ns=span.get("last_retrieved_ns"),
                    retrieval_ids=[str(value) for value in span.get("retrieval_ids", [])],
                )
                for span in item.get("spans", [])
            ]
            for span in spans:
                if span.receipt_id != item["receipt_id"]:
                    raise ValueError(
                        f"stored span {span.span_id} belongs to a different receipt"
                    )
                if span.content_sha256 and span.content_sha256 != _sha256_text(
                    span.content
                ):
                    raise ValueError(
                        f"stored span content hash mismatch for {span.span_id}"
                    )
            if len({span.span_id for span in spans}) != len(spans):
                raise ValueError(f"duplicate span ids in receipt {item['receipt_id']}")
            stored = StoredCompression(
                receipt_id=item["receipt_id"],
                original_hash=item["original_hash"],
                original_tokens=int(item.get("original_tokens", 0)),
                compressed_tokens=int(item.get("compressed_tokens", 0)),
                spans=spans,
                metadata=dict(item.get("metadata", {})),
                created_ns=int(item.get("created_ns", time.time_ns())),
                retrieval_count=int(item.get("retrieval_count", 0)),
                last_retrieved_ns=item.get("last_retrieved_ns"),
                savings_tier=str(item.get("savings_tier", "measured")),
            )
            if stored.receipt_id in loaded:
                raise ValueError(f"duplicate recovery receipt {stored.receipt_id}")
            loaded[stored.receipt_id] = stored
        self._items = loaded
        self._disk_signature = signature

    def _persist(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 3,
            "items": [
                item.as_dict(include_internal=True) for item in self._items.values()
            ],
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        tmp = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}."
            f"{time.time_ns()}.tmp"
        )
        try:
            with tmp.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
            self._disk_signature = self._signature(self.path.stat())
            self._sync_parent_directory()
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _sync_parent_directory(self) -> None:
        """Durably record the atomic rename where the platform permits it."""
        if self.path is None or os.name == "nt":
            return
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _estimate_tokens(text: str) -> int:
    return max(1, len((text or "").encode("utf-8")) // 4)


def _count_o200k_tokens(text: str) -> int:
    try:
        import tiktoken
    except ImportError:
        # A byte is a conservative upper bound for a tokenizer piece. Base
        # installs intentionally keep benchmark tokenizers optional, so this
        # path may return less context but can never violate the public cap.
        return len(text.encode("utf-8"))

    return len(tiktoken.get_encoding("o200k_base").encode(text))


def _bounded_exact_excerpt(
    content: str,
    query: str,
    max_tokens: int,
) -> tuple[str, bool]:
    """Slice exact source windows under an exact or conservative token cap."""
    from .ccr import slice_recovery_content

    budget = int(max_tokens)
    if budget < 32:
        raise ValueError("max_tokens must be at least 32")
    structured = _query_local_json_object(content, query, budget)
    if structured is not None:
        return structured, True
    complete_line = _query_local_complete_line(content, query, budget)
    if complete_line is not None:
        return complete_line, True
    excerpt, was_sliced = slice_recovery_content(content, query, budget)
    for _ in range(4):
        observed = _count_o200k_tokens(excerpt) if excerpt else 0
        if observed <= budget:
            return excerpt, was_sliced
        budget = max(32, int(budget * max_tokens / observed * 0.95))
        excerpt, was_sliced = slice_recovery_content(content, query, budget)
    if excerpt and _count_o200k_tokens(excerpt) > max_tokens:
        raise RuntimeError("exact recovery excerpt could not satisfy its token cap")
    return excerpt, was_sliced


_EXCERPT_TERM_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{2,}")
_EXACT_EXCERPT_GAP = (
    "\n\n[... exact excerpt gap; retrieve full source by handle ...]\n\n"
)


def _query_local_complete_line(
    content: str,
    query: str,
    max_tokens: int,
) -> str | None:
    """Keep the strongest complete source line, then spend spare budget on lead.

    A character-ratio slice can end inside an identifier even while satisfying
    its exact token cap. Recovery evidence is safer when a fitting query-local
    line is indivisible: preserve it byte-for-byte and only trim optional lead
    context around the explicit provenance gap.
    """
    if not content or _count_o200k_tokens(content) <= max_tokens:
        return None
    terms = {term.casefold() for term in _EXCERPT_TERM_RE.findall(query)}
    if not terms:
        return None
    lowered = content.casefold()
    frequencies = {term: max(1, lowered.count(term)) for term in terms}
    offset = 0
    candidates: list[tuple[float, int, int, int, str]] = []
    for line_with_ending in content.splitlines(keepends=True):
        line = line_with_ending.rstrip("\r\n")
        folded = line.casefold()
        score = sum(
            len(term) / frequencies[term]
            for term in sorted(terms)
            if term in folded
        )
        if score > 0:
            tokens = _count_o200k_tokens(line)
            candidates.append((score, -tokens, -offset, offset, line))
        offset += len(line_with_ending)
    if not candidates:
        return None

    _, negated_tokens, _, local_start, local = max(candidates)
    if -negated_tokens > max_tokens:
        return None
    gap = _EXACT_EXCERPT_GAP
    if local_start == 0:
        candidate = local + gap
        return candidate if _count_o200k_tokens(candidate) <= max_tokens else None
    if _count_o200k_tokens(gap + local) > max_tokens:
        return None

    # Binary-search the largest exact source prefix that leaves the complete
    # local line and marker inside the public token ceiling.
    low = 0
    high = local_start
    while low < high:
        middle = (low + high + 1) // 2
        candidate = content[:middle] + gap + local
        if _count_o200k_tokens(candidate) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return content[:low] + gap + local


def _json_object_ranges(content: str) -> list[tuple[int, int]]:
    """Return exact balanced JSON object byte ranges, including nested ones."""
    ranges: list[tuple[int, int]] = []
    stack: list[int] = []
    in_string = False
    escaped = False
    for index, character in enumerate(content):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            stack.append(index)
        elif character == "}" and stack:
            ranges.append((stack.pop(), index + 1))
    return ranges


def _query_local_json_object(
    content: str,
    query: str,
    max_tokens: int,
) -> str | None:
    """Select one complete, exact source JSON object when it fits the cap."""
    stripped = content.lstrip()
    if not stripped.startswith(("[", "{")):
        return None
    terms = {term.casefold() for term in _EXCERPT_TERM_RE.findall(query)}
    if not terms:
        return None
    lowered = content.casefold()
    frequencies = {term: max(1, lowered.count(term)) for term in terms}
    candidates: list[tuple[float, int, int, int, str]] = []
    for start, end in _json_object_ranges(content):
        exact = content[start:end]
        try:
            value = json.loads(exact)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        candidate = exact.casefold()
        score = sum(
            len(term) / frequencies[term]
            for term in sorted(terms)
            if term in candidate
        )
        if score <= 0:
            continue
        tokens = _count_o200k_tokens(exact)
        candidates.append((score, -tokens, -start, tokens, exact))
    if not candidates:
        return None
    best = max(candidates)
    return best[4] if best[3] <= max_tokens else None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _short_hash(text: str) -> str:
    return _sha256_text(text)[:16]


__all__ = ["CompressionRetrievalStore", "StoredCompression", "StoredSpan"]
