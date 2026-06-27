"""Context Receipts: auditable context selection for multi-document tasks."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from . import store as _store
from .ingest import ingest_documents as _py_ingest_documents
from .ingest import normalize_document_pairs as _normalize_document_pairs
from .models import ContextIndex, ContextReceipt
from .novelty import NoveltyAssessmentPolicy, novelty_frontier_assessment
from .receipts import attach_novelty_frontier as _attach_novelty_frontier
from .receipts import build_receipt as _py_build_receipt
from .receipts import explain_omitted as _py_explain_omitted
from .receipts import markdown_report as _py_markdown_report
from .recover import build_recovery_bundle as _build_recovery_bundle
from .recover import recover_omitted as _recover_omitted
from .recover import save_recovery_bundle as _save_recovery_bundle


def _int_or_default(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _at_least(value: object, *, default: int, floor: int) -> int:
    return max(floor, _int_or_default(value, default))


def _str_or_default(value: object, default: str = "") -> str:
    return default if value is None else str(value)


def _chunk_tokens_or_default(value: object, default: int = 360) -> int:
    token_count = _int_or_default(value, default)
    return max(40, token_count) if token_count > 0 else 40


def _overlap_tokens_or_default(
    value: object, *, chunk_tokens: int, default: int = 32
) -> int:
    overlap = _at_least(value, default=default, floor=0)
    max_overlap = max(0, chunk_tokens - 1)
    return min(overlap, max_overlap)


def _rust_core(*required: str):
    entroly_core = sys.modules.get("entroly_core")
    if entroly_core is None:
        if importlib.util.find_spec("entroly_core") is None:
            return None
        entroly_core = importlib.import_module("entroly_core")
    if all(hasattr(entroly_core, name) for name in required):
        return entroly_core
    return None


def _has_novelty_summary(receipt: Mapping[str, Any]) -> bool:
    risk_summary = receipt.get("risk_summary", {})
    return isinstance(risk_summary, Mapping) and "novelty_summary" in risk_summary


def _list_contains_only_mappings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, Mapping) for item in value)


def _list_contains_only_strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _mapping_contains_only_string_lists(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and _list_contains_only_strings(items)
        for key, items in value.items()
    )


def _is_rust_report_safe(receipt: Mapping[str, Any]) -> bool:
    """Return whether a receipt can be delegated to Rust report rendering.

    The Rust renderer is fast for canonical receipts, but the Python renderer is
    intentionally more defensive for rehydrated or hand-edited artifacts. Route
    malformed optional containers through Python so public ``markdown_report``
    keeps its tolerant contract instead of surfacing native renderer errors.
    """
    if _has_novelty_summary(receipt):
        return False
    mapping_fields = {"source_fingerprints", "risk_summary", "compression_ratio"}
    list_fields = {
        "selected_context",
        "omitted_context",
        "dependency_links",
        "outcome_links",
    }
    return (
        all(
            field not in receipt or isinstance(receipt[field], Mapping)
            for field in mapping_fields
        )
        and all(
            field not in receipt or _list_contains_only_mappings(receipt[field])
            for field in list_fields
        )
        and (
            "ranking_reasons" not in receipt
            or _mapping_contains_only_string_lists(receipt["ranking_reasons"])
        )
        and (
            "warnings" not in receipt
            or _list_contains_only_strings(receipt["warnings"])
        )
    )


def ingest_documents(
    documents: Iterable[tuple[str, str]],
    *,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Ingest ``(source_path, text)`` pairs into a deterministic receipt index."""

    docs = _normalize_document_pairs(documents)
    chunk_token_count = _chunk_tokens_or_default(chunk_tokens)
    overlap_token_count = _overlap_tokens_or_default(
        overlap_tokens, chunk_tokens=chunk_token_count
    )
    if prefer_rust and (core := _rust_core("context_receipts_ingest")) is not None:
        return json.loads(
            core.context_receipts_ingest(docs, chunk_token_count, overlap_token_count)
        )
    index = _py_ingest_documents(
        docs, chunk_tokens=chunk_token_count, overlap_tokens=overlap_token_count
    )
    return index.to_dict()


def select_from_index(
    index: dict[str, Any] | ContextIndex,
    *,
    query: str,
    token_budget: int,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Select context and produce a machine-readable Context Receipt."""

    budget = _at_least(token_budget, default=0, floor=0)
    safe_query = _str_or_default(query)
    if prefer_rust and (core := _rust_core("context_receipts_select")) is not None:
        index_json = json.dumps(
            index if isinstance(index, dict) else index.to_dict(), sort_keys=True
        )
        receipt = json.loads(
            core.context_receipts_select(index_json, safe_query, budget)
        )
        return _attach_novelty_frontier(receipt, index)
    py_index = ContextIndex.from_dict(index) if isinstance(index, dict) else index
    receipt = _py_build_receipt(py_index, query=safe_query, token_budget=budget)
    return receipt.to_dict()


def run_receipt_pipeline(
    documents: Iterable[tuple[str, str]],
    *,
    query: str,
    token_budget: int,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Ingest documents and immediately produce a Context Receipt."""

    docs = _normalize_document_pairs(documents)
    budget = _at_least(token_budget, default=0, floor=0)
    chunk_token_count = _chunk_tokens_or_default(chunk_tokens)
    overlap_token_count = _overlap_tokens_or_default(
        overlap_tokens, chunk_tokens=chunk_token_count
    )
    if (
        prefer_rust
        and (
            core := _rust_core(
                "context_receipts_run",
                "context_receipts_ingest",
            )
        )
        is not None
    ):
        safe_query = _str_or_default(query)
        raw = core.context_receipts_run(
            docs, safe_query, budget, chunk_token_count, overlap_token_count
        )
        receipt = json.loads(raw)
        rust_index = json.loads(
            core.context_receipts_ingest(docs, chunk_token_count, overlap_token_count)
        )
        return _attach_novelty_frontier(receipt, rust_index)
    index = _py_ingest_documents(
        docs, chunk_tokens=chunk_token_count, overlap_tokens=overlap_token_count
    )
    return _py_build_receipt(
        index, query=_str_or_default(query), token_budget=budget
    ).to_dict()


def markdown_report(
    receipt: dict[str, Any] | ContextReceipt, *, prefer_rust: bool = True
) -> str:
    """Render a human-readable Markdown report from a receipt dict."""

    if (
        prefer_rust
        and isinstance(receipt, dict)
        and _is_rust_report_safe(receipt)
        and (core := _rust_core("context_receipts_report")) is not None
    ):
        return core.context_receipts_report(json.dumps(receipt, sort_keys=True))
    py_receipt = (
        ContextReceipt.from_dict(receipt) if isinstance(receipt, dict) else receipt
    )
    return _py_markdown_report(py_receipt)


def explain_omitted(
    receipt: dict[str, Any] | ContextReceipt,
    chunk_id: str,
    *,
    prefer_rust: bool = True,
) -> str:
    """Explain why a specific chunk was omitted from a receipt."""

    if (
        prefer_rust
        and isinstance(receipt, dict)
        and _is_rust_report_safe(receipt)
        and (core := _rust_core("context_receipts_explain_omitted")) is not None
    ):
        return core.context_receipts_explain_omitted(
            json.dumps(receipt, sort_keys=True), chunk_id
        )
    py_receipt = (
        ContextReceipt.from_dict(receipt) if isinstance(receipt, dict) else receipt
    )
    return _py_explain_omitted(py_receipt, chunk_id)


def assess_novelty_frontier(
    receipt_or_summary: Mapping[str, Any] | object,
    *,
    policy: NoveltyAssessmentPolicy | None = None,
) -> dict[str, object]:
    """Assess novelty-frontier pressure from a receipt dict or novelty summary.

    Passing a full receipt lets callers re-classify stored receipts without
    knowing the internal ``risk_summary.novelty_summary`` location. Passing an
    already-extracted novelty summary keeps the helper useful for lower-level
    integrations.
    """
    payload = receipt_or_summary if isinstance(receipt_or_summary, Mapping) else {}
    risk_summary = payload.get("risk_summary")
    novelty = (
        risk_summary.get("novelty_summary")
        if isinstance(risk_summary, Mapping)
        else None
    )
    if not isinstance(novelty, Mapping):
        novelty = payload
    if policy is None:
        return novelty_frontier_assessment(novelty)
    return novelty_frontier_assessment(novelty, policy=policy)


def run_recoverable_pipeline(
    documents: Iterable[tuple[str, str]],
    *,
    query: str,
    token_budget: int,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    store_dir: str | Path | None = None,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Ingest → select → persist a **recoverable** receipt.

    Writes the receipt plus a project-local recovery bundle so any omitted chunk
    can later be recovered, byte-exact and verified, from the store alone — no
    need to keep the index around. Returns ``{receipt, receipt_path, recovery_path}``.
    """
    docs = _normalize_document_pairs(documents)
    index = ingest_documents(
        docs,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        prefer_rust=prefer_rust,
    )
    receipt = select_from_index(
        index, query=query, token_budget=token_budget, prefer_rust=prefer_rust
    )
    receipt_id = str(receipt.get("receipt_id", ""))
    receipt_path: Path | None = None
    recovery_p: Path | None = None
    if receipt_id:
        base = Path(store_dir) if store_dir is not None else _store.DEFAULT_STORE
        base.mkdir(parents=True, exist_ok=True)
        receipt_path = _store.write_json(base / f"{receipt_id}.json", receipt)
        recovery_p = _save_recovery_bundle(
            receipt_id, _build_recovery_bundle(index), store_dir
        )
    return {
        "receipt": receipt,
        "receipt_path": str(receipt_path) if receipt_path else None,
        "recovery_path": str(recovery_p) if recovery_p else None,
    }


def recover_omitted(
    receipt: dict[str, Any] | ContextReceipt,
    chunk_id: str | None = None,
    *,
    index: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
    store_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Recover the full, fingerprint-verified text of omitted chunk(s) from a receipt.

    Receipts *explain* what was dropped; this *recovers* it. Pass ``chunk_id`` for
    one chunk or omit it for all omitted chunks. Each result's ``verified`` flag is
    True only when the returned text is provably the exact omitted content.
    """
    return _recover_omitted(
        receipt, chunk_id, index=index, bundle=bundle, store_dir=store_dir
    )


__all__ = [
    "ingest_documents",
    "select_from_index",
    "run_receipt_pipeline",
    "run_recoverable_pipeline",
    "markdown_report",
    "explain_omitted",
    "NoveltyAssessmentPolicy",
    "assess_novelty_frontier",
    "recover_omitted",
]
