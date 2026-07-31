"""Trust-hardened compression recovery store.

This module preserves the public ``CompressionRetrievalStore`` API while adding
security and integrity properties that the original storage engine did not
encode at its boundary:

- returned objects are defensive snapshots, never mutable internal state;
- records are bound to a canonical store/workspace/session scope;
- prompt-, auth-, and session-shaped metadata is hashed instead of disclosed;
- retrieval-id history is bounded and becomes conservatively zero-savings when
  saturated rather than growing without limit or overclaiming savings;
- post-rename directory-sync failures are warnings, not logical rollbacks of a
  write that is already visible on disk;
- reads tolerate a replace/unlink race and fail closed on malformed identifiers.

The underlying on-disk schema remains compatible with the existing engine.
"""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import compression_retrieval_store as _legacy_module
from .compression_retrieval_store import (
    CompressionRetrievalStore as _LegacyCompressionRetrievalStore,
    StoredCompression,
    StoredSpan,
    _EXCERPT_TERM_RE,
    _bounded_exact_excerpt,
    _count_o200k_tokens,
    _estimate_tokens,
    _sha256_text,
    _short_hash,
)

logger = logging.getLogger("entroly.compression_retrieval_store")

_SCOPE_METADATA_KEY = "_entroly_scope_sha256"
_ACCOUNTING_EXHAUSTED_KEY = "retrieval_accounting_exhausted"
_ACCOUNTING_CONFIDENCE_KEY = "retrieval_accounting_confidence"
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_RETRIEVAL_IDS = 4096
_MAX_IDENTIFIER_CHARS = 512
_MAX_QUERY_CHARS = 32_768
_MAX_SEARCH_LIMIT = 100
_MAX_METADATA_ITEMS = 64
_MAX_METADATA_DEPTH = 4
_MAX_METADATA_STRING_CHARS = 2048
_METADATA_SENSITIVE_WORDS = frozenset(
    {
        "AUTH",
        "COOKIE",
        "CREDENTIAL",
        "CREDENTIALS",
        "PASSWORD",
        "PASSWD",
        "PROMPT",
        "QUERY",
        "SECRET",
        "SESSION",
        "TEXT",
        "TOKEN",
    }
)
_METADATA_CONTENT_WORDS = frozenset({"CONTENT", "INPUT", "OUTPUT", "PAYLOAD"})
_SAFE_METADATA_KEY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _env_positive_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= minimum else default


def _canonical_path(path: str | Path | None) -> str:
    if path is None:
        return "memory"
    try:
        return str(Path(path).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return str(path)


def derive_recovery_scope(
    path: str | Path | None,
    *,
    scope_id: str | None = None,
) -> tuple[str, str]:
    """Return the canonical scope description and its SHA-256 capability hash."""
    explicit = scope_id or os.environ.get("ENTROLY_RECOVERY_SCOPE", "")
    try:
        workspace = str(
            Path(os.environ.get("ENTROLY_WORKSPACE", os.getcwd()))
            .expanduser()
            .resolve(strict=False)
        )
    except (OSError, RuntimeError, ValueError):
        workspace = os.getcwd()
    components = {
        "version": 1,
        "store": _canonical_path(path),
        "workspace": workspace,
        "scope": explicit,
        "session": os.environ.get("ENTROLY_SESSION_ID", ""),
        "conversation": os.environ.get("ENTROLY_CONVERSATION_ID", ""),
    }
    canonical = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return canonical, _hash_text(canonical)


def _validate_identifier(value: str | None, *, name: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if (
        len(value) > _MAX_IDENTIFIER_CHARS
        or "\x00" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"{name} is not a safe bounded identifier")
    return value


def _metadata_key_words(key: str) -> set[str]:
    return {word for word in re.split(r"[^A-Z0-9]+", key.upper()) if word}


def _safe_metadata_value(value: object, *, depth: int) -> object:
    if depth > _MAX_METADATA_DEPTH:
        return "[depth-limit]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value[:_MAX_METADATA_STRING_CHARS]
    if isinstance(value, dict):
        output: dict[str, object] = {}
        for index, (raw_key, child) in enumerate(value.items()):
            if index >= _MAX_METADATA_ITEMS:
                output["_truncated"] = True
                break
            key = str(raw_key)
            if not _SAFE_METADATA_KEY_RE.fullmatch(key):
                continue
            output[key] = _safe_metadata_value(child, depth=depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        return [
            _safe_metadata_value(child, depth=depth + 1)
            for child in list(value)[:_MAX_METADATA_ITEMS]
        ]
    return str(value)[:_MAX_METADATA_STRING_CHARS]


def sanitize_recovery_metadata(metadata: dict[str, object] | None) -> dict[str, object]:
    """Return bounded metadata without raw prompts, secrets, or identity tokens."""
    output: dict[str, object] = {}
    for index, (raw_key, value) in enumerate(dict(metadata or {}).items()):
        if index >= _MAX_METADATA_ITEMS:
            output["_truncated"] = True
            break
        key = str(raw_key)
        if not _SAFE_METADATA_KEY_RE.fullmatch(key):
            continue
        if key == _SCOPE_METADATA_KEY:
            continue
        words = _metadata_key_words(key)
        if words & (_METADATA_SENSITIVE_WORDS | _METADATA_CONTENT_WORDS):
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            output[f"{key}_sha256"] = _hash_text(serialized)
            output[f"{key}_length"] = len(serialized)
            continue
        output[key] = _safe_metadata_value(value, depth=0)
    return output


def _snapshot_span(span: StoredSpan | None) -> StoredSpan | None:
    if span is None:
        return None
    return replace(span, retrieval_ids=list(span.retrieval_ids))


def _snapshot_item(item: StoredCompression | None) -> StoredCompression | None:
    if item is None:
        return None
    return replace(
        item,
        spans=[_snapshot_span(span) for span in item.spans if span is not None],
        metadata=copy.deepcopy(item.metadata),
    )


class CompressionRetrievalStore(_LegacyCompressionRetrievalStore):
    """Scoped, immutable-boundary recovery store with bounded accounting state."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        optimization_ledger=None,
        max_bytes: int | None = None,
        scope_id: str | None = None,
        require_scope: bool = True,
        allow_legacy_unscoped: bool | None = None,
        max_retrieval_ids: int | None = None,
    ) -> None:
        configured_max = max_bytes
        if configured_max is None:
            configured_max = _env_positive_int(
                "ENTROLY_RECOVERY_STORE_MAX_BYTES",
                _DEFAULT_MAX_BYTES,
                minimum=1024,
            )
        self.scope_description, self.scope_hash = derive_recovery_scope(
            path,
            scope_id=scope_id,
        )
        self.require_scope = bool(require_scope)
        self.allow_legacy_unscoped = (
            os.environ.get("ENTROLY_ALLOW_LEGACY_UNSCOPED_RECOVERY") == "1"
            if allow_legacy_unscoped is None
            else bool(allow_legacy_unscoped)
        )
        configured_history = (
            _env_positive_int(
                "ENTROLY_RECOVERY_MAX_RETRIEVAL_IDS",
                _DEFAULT_MAX_RETRIEVAL_IDS,
                minimum=32,
            )
            if max_retrieval_ids is None
            else int(max_retrieval_ids)
        )
        if configured_history < 32:
            raise ValueError("max_retrieval_ids must be at least 32")
        self.max_retrieval_ids = configured_history
        super().__init__(
            path,
            optimization_ledger=optimization_ledger,
            max_bytes=configured_max,
        )
        self._validate_loaded_state()

    def _scope_allows(self, item: StoredCompression) -> bool:
        stored_scope = str(item.metadata.get(_SCOPE_METADATA_KEY, ""))
        if stored_scope:
            return stored_scope == self.scope_hash
        return not self.require_scope or self.allow_legacy_unscoped

    def _metadata_for_write(
        self,
        metadata: dict[str, object] | None,
    ) -> dict[str, object]:
        cleaned = sanitize_recovery_metadata(metadata)
        cleaned[_SCOPE_METADATA_KEY] = self.scope_hash
        cleaned.setdefault(_ACCOUNTING_CONFIDENCE_KEY, "exact")
        return cleaned

    def _validate_loaded_state(self) -> None:
        with self._lock:
            for item in self._items.values():
                _validate_identifier(item.receipt_id, name="stored receipt_id")
                if item.original_tokens < 0 or item.compressed_tokens < 0:
                    raise ValueError("stored token counts must be non-negative")
                if item.retrieval_count < 0:
                    raise ValueError("stored retrieval_count must be non-negative")
                if not isinstance(item.metadata, dict):
                    raise ValueError("stored metadata must be an object")
                for span in item.spans:
                    _validate_identifier(span.span_id, name="stored span_id")
                    if span.receipt_id != item.receipt_id:
                        raise ValueError("stored span belongs to a different receipt")
                    if span.start_line < 1 or span.end_line < span.start_line:
                        raise ValueError("stored span line range is invalid")
                    if span.retrieval_count < 0 or span.retrieved_tokens < 0:
                        raise ValueError("stored retrieval counters must be non-negative")
                    if len(span.retrieval_ids) > self.max_retrieval_ids:
                        span.retrieval_ids = span.retrieval_ids[-self.max_retrieval_ids :]
                        span.retrieved_tokens = max(
                            span.retrieved_tokens,
                            item.gross_saved_tokens,
                        )
                        item.metadata[_ACCOUNTING_EXHAUSTED_KEY] = True
                        item.metadata[_ACCOUNTING_CONFIDENCE_KEY] = "conservative_zero"

    def _refresh_if_changed(self, *, force: bool = False) -> None:
        try:
            super()._refresh_if_changed(force=force)
        except FileNotFoundError:
            # A concurrent atomic replace/unlink can race with exists/stat. Retry
            # once; if the file truly disappeared, clear the stale in-memory view.
            time.sleep(0)
            if self.path is not None and self.path.exists():
                super()._refresh_if_changed(force=True)
            else:
                self._items = {}
                self._disk_signature = None
        self._validate_loaded_state()

    def _sync_parent_directory(self) -> None:
        """Best-effort durability barrier after a rename already committed.

        Once ``os.replace`` succeeds, propagating a directory-fsync error causes
        callers to roll memory back even though the new file is visible on disk.
        That creates split-brain. Keep the committed state and surface a warning.
        """
        try:
            super()._sync_parent_directory()
        except OSError as exc:
            logger.warning(
                "recovery-store rename committed but parent fsync was unavailable: %s",
                exc,
            )

    def put(self, *, original_text: str, compressed_text: str, receipt: dict[str, Any], metadata=None):
        stored = super().put(
            original_text=original_text,
            compressed_text=compressed_text,
            receipt=receipt,
            metadata=self._metadata_for_write(metadata),
        )
        return _snapshot_item(stored)

    def put_exact_spans(
        self,
        *,
        original_text: str,
        compressed_text: str,
        receipt: dict[str, Any],
        spans: list[dict[str, Any]],
        metadata=None,
    ):
        stored = super().put_exact_spans(
            original_text=original_text,
            compressed_text=compressed_text,
            receipt=receipt,
            spans=spans,
            metadata=self._metadata_for_write(metadata),
        )
        return _snapshot_item(stored)

    def get_receipt(self, receipt_id: str):
        receipt_id = _validate_identifier(receipt_id, name="receipt_id")
        with self._lock:
            self._refresh_if_changed()
            item = self._items.get(receipt_id)
            if item is None or not self._scope_allows(item):
                return None
            return _snapshot_item(item)

    def get_span(
        self,
        receipt_id: str,
        span_id: str,
        *,
        record_retrieval: bool = False,
    ):
        if record_retrieval:
            return self.retrieve_span(receipt_id, span_id)
        receipt_id = _validate_identifier(receipt_id, name="receipt_id")
        span_id = _validate_identifier(span_id, name="span_id")
        with self._lock:
            self._refresh_if_changed()
            item = self._items.get(receipt_id)
            if item is None or not self._scope_allows(item):
                return None
            span = next((entry for entry in item.spans if entry.span_id == span_id), None)
            return _snapshot_span(span)

    def _effective_retrieval_id(
        self,
        retrieval_id: str | None,
    ) -> str:
        validated = _validate_identifier(
            retrieval_id,
            name="retrieval_id",
            optional=True,
        )
        return validated or f"auto:{uuid.uuid4().hex}"

    def _record_locked(
        self,
        item: StoredCompression,
        span: StoredSpan,
        *,
        token_count: int,
        retrieval_id: str,
        now_ns: int,
    ) -> tuple[bool, str, int]:
        if retrieval_id in span.retrieval_ids:
            return False, retrieval_id, token_count

        if len(span.retrieval_ids) >= self.max_retrieval_ids:
            saturation_id = f"recovery-history-saturated:{item.receipt_id}:{span.span_id}"
            item.metadata[_ACCOUNTING_EXHAUSTED_KEY] = True
            item.metadata[_ACCOUNTING_CONFIDENCE_KEY] = "conservative_zero"
            span.retrieved_tokens = max(span.retrieved_tokens, item.gross_saved_tokens)
            span.retrieval_count += 1
            span.last_retrieved_ns = now_ns
            item.retrieval_count += 1
            item.last_retrieved_ns = now_ns
            return True, saturation_id, item.gross_saved_tokens

        span.record_retrieval(
            tokens=max(0, int(token_count)),
            retrieved_ns=now_ns,
            retrieval_id=retrieval_id,
        )
        item.retrieval_count += 1
        item.last_retrieved_ns = now_ns
        return True, retrieval_id, token_count

    def retrieve_span(
        self,
        receipt_id: str,
        span_id: str,
        *,
        retrieval_id: str | None = None,
    ):
        receipt_id = _validate_identifier(receipt_id, name="receipt_id")
        span_id = _validate_identifier(span_id, name="span_id")
        effective_id = self._effective_retrieval_id(retrieval_id)
        ledger_id = effective_id
        ledger_tokens = 0
        recorded = False
        with self._lock:
            with self._interprocess_lock():
                self._refresh_if_changed(force=True)
                item = self._items.get(receipt_id)
                if item is None or not self._scope_allows(item):
                    return None
                span = next((entry for entry in item.spans if entry.span_id == span_id), None)
                if span is None:
                    return None
                previous_item = copy.deepcopy(item)
                recorded, ledger_id, ledger_tokens = self._record_locked(
                    item,
                    span,
                    token_count=_estimate_tokens(span.content),
                    retrieval_id=effective_id,
                    now_ns=time.time_ns(),
                )
                if recorded:
                    try:
                        self._persist()
                    except Exception:
                        self._items[receipt_id] = previous_item
                        raise
                view = _snapshot_span(span)
        if recorded:
            self._record_retrieval(
                receipt_id,
                span_id,
                ledger_tokens,
                retrieval_id=ledger_id,
            )
        return view

    def retrieve_span_excerpt(
        self,
        receipt_id: str,
        span_id: str,
        *,
        query: str,
        max_tokens: int = 600,
        retrieval_id: str | None = None,
    ):
        receipt_id = _validate_identifier(receipt_id, name="receipt_id")
        span_id = _validate_identifier(span_id, name="span_id")
        if not isinstance(query, str) or len(query) > _MAX_QUERY_CHARS:
            raise ValueError("query must be bounded text")
        if isinstance(max_tokens, bool) or int(max_tokens) < 32 or int(max_tokens) > 100_000:
            raise ValueError("max_tokens must be between 32 and 100000")
        effective_id = self._effective_retrieval_id(retrieval_id)
        ledger_id = effective_id
        ledger_tokens = 0
        recorded = False
        with self._lock:
            with self._interprocess_lock():
                self._refresh_if_changed(force=True)
                item = self._items.get(receipt_id)
                if item is None or not self._scope_allows(item):
                    return None
                span = next((entry for entry in item.spans if entry.span_id == span_id), None)
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
                previous_item = copy.deepcopy(item)
                recorded, ledger_id, ledger_tokens = self._record_locked(
                    item,
                    span,
                    token_count=token_count,
                    retrieval_id=effective_id,
                    now_ns=time.time_ns(),
                )
                if recorded:
                    try:
                        self._persist()
                    except Exception:
                        self._items[receipt_id] = previous_item
                        raise
                view = replace(
                    span,
                    content=excerpt,
                    reason=f"{span.reason}:exact_excerpt" if was_sliced else span.reason,
                    start_char=None if was_sliced else span.start_char,
                    end_char=None if was_sliced else span.end_char,
                    content_sha256=_sha256_text(excerpt),
                    retrieval_ids=list(span.retrieval_ids),
                )
        if recorded:
            self._record_retrieval(
                receipt_id,
                span_id,
                ledger_tokens,
                retrieval_id=ledger_id,
            )
        return view

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        record_retrieval: bool = False,
        retrieval_id: str | None = None,
    ):
        if not isinstance(query, str) or len(query) > _MAX_QUERY_CHARS:
            raise ValueError("query must be bounded text")
        if isinstance(limit, bool):
            raise ValueError("limit must be an integer")
        bounded_limit = max(0, min(int(limit), _MAX_SEARCH_LIMIT))
        terms = {
            term.casefold()
            for term in list(_EXCERPT_TERM_RE.findall(query))[:256]
        }
        with self._lock:
            self._refresh_if_changed()
            scored: list[tuple[int, StoredSpan]] = []
            for item in self._items.values():
                if not self._scope_allows(item):
                    continue
                for span in item.spans:
                    folded = span.content.casefold()
                    score = sum(1 for term in terms if term in folded)
                    if score:
                        scored.append((score, span))
            scored.sort(key=lambda pair: (pair[0], pair[1].created_ns), reverse=True)
            selected = [_snapshot_span(span) for _score, span in scored[:bounded_limit]]
        selected = [span for span in selected if span is not None]
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

    def search_exact_excerpts(
        self,
        query: str,
        *,
        limit: int = 5,
        max_tokens_per_span: int = 600,
        record_retrieval: bool = False,
        retrieval_id: str | None = None,
    ):
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
        base_id = retrieval_id or f"excerpt-search:{_short_hash(query)}:{time.time_ns()}"
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

    def realized_savings(self, receipt_id: str):
        item = self.get_receipt(receipt_id)
        return item.savings_record() if item is not None else None

    def list_receipts(self):
        with self._lock:
            self._refresh_if_changed()
            items = [item for item in self._items.values() if self._scope_allows(item)]
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
                    "metadata": copy.deepcopy(item.metadata),
                }
                for item in sorted(items, key=lambda entry: entry.created_ns, reverse=True)
            ]

    def savings_summary(self):
        with self._lock:
            self._refresh_if_changed()
            items = [item for item in self._items.values() if self._scope_allows(item)]
            gross = sum(item.gross_tokens_saved for item in items)
            retrieved = sum(item.retrieved_tokens for item in items)
        return {
            "tier": "measured",
            "gross_tokens_saved": gross,
            "retrieved_tokens": retrieved,
            "net_tokens_saved": max(0, gross - retrieved),
        }


# Patch the historical module path. Importing any Entroly submodule first loads
# the package, and compression_proxy imports this hardened class before public
# recovery exports are bound.
_legacy_module.CompressionRetrievalStore = CompressionRetrievalStore

__all__ = [
    "CompressionRetrievalStore",
    "StoredCompression",
    "StoredSpan",
    "derive_recovery_scope",
    "sanitize_recovery_metadata",
]
