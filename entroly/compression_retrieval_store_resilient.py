"""Failure-isolated facade for Entroly's exact recovery store.

Exact evidence remains fail-closed per record: a hash mismatch, malformed span,
or cross-receipt reference is never normalized into something recoverable. The
availability improvement is narrower—healthy receipts remain usable while the
invalid raw record is retained locally in a bounded forensic quarantine.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import sys
import threading
import time
from typing import Any

from . import compression_retrieval_store as _legacy
from . import compression_retrieval_store_safe as _safe
from . import compression_retrieval_store_secure as _secure
from .compression_retrieval_store import StoredCompression, StoredSpan

_MAX_QUARANTINED_RECORDS = 128
_MAX_QUARANTINE_BYTES = 16 * 1024 * 1024


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8", errors="replace")


def _quarantine_record(raw: Any, reason: str) -> dict[str, Any]:
    payload = copy.deepcopy(raw)
    return {
        "reason": str(reason)[:160],
        "sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        "item": payload,
    }


def _decode_span(raw: Any, receipt_id: str) -> StoredSpan:
    if not isinstance(raw, dict):
        raise ValueError("span_not_object")
    span_id = str(raw.get("span_id", ""))
    stored_receipt = str(raw.get("receipt_id", ""))
    if not span_id:
        raise ValueError("span_id_missing")
    if stored_receipt != receipt_id:
        raise ValueError("span_receipt_mismatch")
    content = str(raw.get("content", ""))
    expected_hash = str(raw.get("content_sha256", ""))
    if expected_hash and expected_hash != _legacy._sha256_text(content):
        raise ValueError("span_content_hash_mismatch")
    start_line = int(raw.get("start_line", 1))
    end_line = int(raw.get("end_line", start_line))
    if start_line < 1 or end_line < start_line:
        raise ValueError("span_line_range_invalid")
    retrieval_count = int(raw.get("retrieval_count", 0))
    retrieved_tokens = int(raw.get("retrieved_tokens", 0))
    if retrieval_count < 0 or retrieved_tokens < 0:
        raise ValueError("span_counters_invalid")
    retrieval_ids = raw.get("retrieval_ids", [])
    if not isinstance(retrieval_ids, list):
        raise ValueError("retrieval_ids_not_array")
    return StoredSpan(
        span_id=span_id,
        receipt_id=receipt_id,
        start_line=start_line,
        end_line=end_line,
        content=content,
        reason=str(raw.get("reason", "budget")),
        source_id=str(raw.get("source_id", "")),
        start_char=(int(raw["start_char"]) if raw.get("start_char") is not None else None),
        end_char=(int(raw["end_char"]) if raw.get("end_char") is not None else None),
        content_sha256=expected_hash,
        created_ns=int(raw.get("created_ns", time.time_ns())),
        retrieval_count=retrieval_count,
        retrieved_tokens=retrieved_tokens,
        last_retrieved_ns=int(raw.get("last_retrieved_ns") or 0),
        retrieval_ids=[str(value) for value in retrieval_ids],
    )


def _decode_item(raw: Any) -> StoredCompression:
    if not isinstance(raw, dict):
        raise ValueError("receipt_not_object")
    receipt_id = str(raw.get("receipt_id", ""))
    if not receipt_id:
        raise ValueError("receipt_id_missing")
    raw_spans = raw.get("spans", [])
    if not isinstance(raw_spans, list):
        raise ValueError("spans_not_array")
    spans = [_decode_span(span, receipt_id) for span in raw_spans]
    if len({span.span_id for span in spans}) != len(spans):
        raise ValueError("duplicate_span_id")
    original_tokens = int(raw.get("original_tokens", 0))
    compressed_tokens = int(raw.get("compressed_tokens", 0))
    retrieval_count = int(raw.get("retrieval_count", 0))
    if min(original_tokens, compressed_tokens, retrieval_count) < 0:
        raise ValueError("receipt_counters_invalid")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata_not_object")
    return StoredCompression(
        receipt_id=receipt_id,
        original_hash=str(raw.get("original_hash", "")),
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        spans=spans,
        metadata=copy.deepcopy(metadata),
        created_ns=int(raw.get("created_ns", time.time_ns())),
        retrieval_count=retrieval_count,
        last_retrieved_ns=raw.get("last_retrieved_ns"),
        savings_tier=str(raw.get("savings_tier", "measured")),
    )


class CompressionRetrievalStore(_secure.CompressionRetrievalStore):
    """Secure recovery store with per-record forensic quarantine."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._quarantined_items: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _validate_quarantine(self) -> None:
        if len(self._quarantined_items) > _MAX_QUARANTINED_RECORDS:
            raise ValueError("recovery quarantine record limit exceeded")
        size = sum(len(_canonical_bytes(record)) for record in self._quarantined_items)
        if size > _MAX_QUARANTINE_BYTES:
            raise ValueError("recovery quarantine byte limit exceeded")

    def _load(self) -> None:
        assert self.path is not None
        with self.path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
            signature = self._signature(os.fstat(handle.fileno()))
        if not isinstance(raw, dict):
            raise ValueError("recovery store root must be an object")
        schema_version = int(raw.get("schema_version", 1))
        if schema_version not in {1, 2, 3}:
            raise ValueError(f"unsupported recovery store schema_version {schema_version}")
        items = raw.get("items", [])
        if not isinstance(items, list):
            raise ValueError("recovery store items must be an array")

        quarantine: list[dict[str, Any]] = []
        existing_quarantine = raw.get("quarantined_items", [])
        if existing_quarantine is not None:
            if not isinstance(existing_quarantine, list):
                raise ValueError("quarantined_items must be an array")
            for record in existing_quarantine:
                if not isinstance(record, dict):
                    raise ValueError("quarantine record must be an object")
                quarantine.append(copy.deepcopy(record))

        loaded: dict[str, StoredCompression] = {}
        for raw_item in items:
            try:
                item = _decode_item(raw_item)
                _safe._validate_identifier(item.receipt_id, name="stored receipt_id")
                for span in item.spans:
                    _safe._validate_identifier(span.span_id, name="stored span_id")
                if item.receipt_id in loaded:
                    raise ValueError("duplicate_receipt_id")
                loaded[item.receipt_id] = item
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                quarantine.append(
                    _quarantine_record(raw_item, f"{type(exc).__name__}:{exc}")
                )

        self._items = loaded
        self._quarantined_items = quarantine
        self._validate_quarantine()
        self._disk_signature = signature

    def _persist(self) -> None:
        if self.path is None:
            return
        self._validate_quarantine()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 3,
            "items": [item.as_dict(include_internal=True) for item in self._items.values()],
            "quarantined_items": copy.deepcopy(self._quarantined_items),
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        serialized_bytes = len(serialized.encode("utf-8"))
        if self.max_bytes is not None and serialized_bytes > self.max_bytes:
            raise OSError(
                errno.ENOSPC,
                f"recovery store write would exceed its configured {self.max_bytes}-byte limit",
                str(self.path),
            )
        tmp = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
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

    def quarantine_summary(self) -> dict[str, Any]:
        """Return content-blind quarantine health for diagnostics."""
        by_reason: dict[str, int] = {}
        for record in self._quarantined_items:
            reason = str(record.get("reason", "unknown"))[:160]
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return {
            "quarantined_records": len(self._quarantined_items),
            "by_reason": dict(sorted(by_reason.items())),
        }


# Keep historical import surfaces aligned without making package import itself
# perform I/O. Existing references are updated only when their module is loaded.
_secure.CompressionRetrievalStore = CompressionRetrievalStore
_safe.CompressionRetrievalStore = CompressionRetrievalStore
_legacy.CompressionRetrievalStore = CompressionRetrievalStore
for _module_name in (
    "entroly",
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


__all__ = ["CompressionRetrievalStore", "StoredCompression", "StoredSpan"]
