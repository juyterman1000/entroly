"""Context Receipts: auditable context selection for multi-document tasks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from .ingest import ingest_documents as _py_ingest_documents
from .models import ContextIndex, ContextReceipt
from .receipts import build_receipt as _py_build_receipt
from .receipts import explain_omitted as _py_explain_omitted
from .receipts import markdown_report as _py_markdown_report


def _rust_core():
    try:
        import entroly_core  # type: ignore
    except Exception:
        return None
    required = (
        "context_receipts_ingest",
        "context_receipts_select",
        "context_receipts_report",
        "context_receipts_explain_omitted",
    )
    if all(hasattr(entroly_core, name) for name in required):
        return entroly_core
    return None


def ingest_documents(
    documents: Iterable[tuple[str, str]],
    *,
    chunk_tokens: int = 360,
    overlap_tokens: int = 32,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Ingest ``(source_path, text)`` pairs into a deterministic receipt index."""

    docs = list(documents)
    if prefer_rust and (core := _rust_core()) is not None:
        return json.loads(core.context_receipts_ingest(docs, int(chunk_tokens), int(overlap_tokens)))
    index = _py_ingest_documents(docs, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    return index.to_dict()


def select_from_index(
    index: dict[str, Any] | ContextIndex,
    *,
    query: str,
    token_budget: int,
    prefer_rust: bool = True,
) -> dict[str, Any]:
    """Select context and produce a machine-readable Context Receipt."""

    if prefer_rust and (core := _rust_core()) is not None:
        index_json = json.dumps(index if isinstance(index, dict) else index.to_dict(), sort_keys=True)
        return json.loads(core.context_receipts_select(index_json, query, int(token_budget)))
    py_index = ContextIndex.from_dict(index) if isinstance(index, dict) else index
    receipt = _py_build_receipt(py_index, query=query, token_budget=token_budget)
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

    docs = list(documents)
    if prefer_rust and (core := _rust_core()) is not None:
        raw = core.context_receipts_run(docs, query, int(token_budget), int(chunk_tokens), int(overlap_tokens))
        return json.loads(raw)
    index = _py_ingest_documents(docs, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    return _py_build_receipt(index, query=query, token_budget=token_budget).to_dict()


def markdown_report(receipt: dict[str, Any] | ContextReceipt, *, prefer_rust: bool = True) -> str:
    """Render a human-readable Markdown report from a receipt dict."""

    if prefer_rust and isinstance(receipt, dict) and (core := _rust_core()) is not None:
        return core.context_receipts_report(json.dumps(receipt, sort_keys=True))
    py_receipt = ContextReceipt.from_dict(receipt) if isinstance(receipt, dict) else receipt
    return _py_markdown_report(py_receipt)


def explain_omitted(
    receipt: dict[str, Any] | ContextReceipt,
    chunk_id: str,
    *,
    prefer_rust: bool = True,
) -> str:
    """Explain why a specific chunk was omitted from a receipt."""

    if prefer_rust and isinstance(receipt, dict) and (core := _rust_core()) is not None:
        return core.context_receipts_explain_omitted(json.dumps(receipt, sort_keys=True), chunk_id)
    py_receipt = ContextReceipt.from_dict(receipt) if isinstance(receipt, dict) else receipt
    return _py_explain_omitted(py_receipt, chunk_id)


__all__ = [
    "ingest_documents",
    "select_from_index",
    "run_receipt_pipeline",
    "markdown_report",
    "explain_omitted",
]
