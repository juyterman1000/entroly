"""Optional native fast-path adapter for Evidence-Locked Compression.

This module is intentionally conservative: it uses a Rust/native implementation
only when `entroly_core` exposes a compatible ELC function. Otherwise it falls
back to the audited Python implementation.

Expected native contract, once exported from `entroly-core`:

    elc_compress(text: str, query: str, budget_tokens: int) -> dict

with keys:

    compressed: str
    receipt: dict
    changed: bool

Until that symbol exists, this module keeps the public fast-path API stable and
safe without falsely claiming native acceleration is active.
"""

from __future__ import annotations

from typing import Any

from .evidence_locked_compression import (
    CompressionReceipt,
    CompressionResult,
    OmittedSpan,
    compress_evidence_locked,
)


def native_elc_available() -> bool:
    try:
        import entroly_core  # type: ignore

        return callable(getattr(entroly_core, "elc_compress", None))
    except Exception:
        return False


def compress_evidence_locked_fast(
    text: str,
    *,
    query: str = "",
    budget_tokens: int = 1200,
) -> CompressionResult:
    """Use native ELC when exported; otherwise fall back to Python ELC."""
    try:
        import entroly_core  # type: ignore

        native = getattr(entroly_core, "elc_compress", None)
        if callable(native):
            raw = native(text, query, int(budget_tokens))
            if isinstance(raw, dict):
                return _result_from_native(raw)
    except Exception:
        pass
    return compress_evidence_locked(text, query=query, budget_tokens=budget_tokens)


def _result_from_native(raw: dict[str, Any]) -> CompressionResult:
    receipt_raw = dict(raw.get("receipt") or {})
    spans = []
    for span in receipt_raw.get("omitted_spans", []) or []:
        if not isinstance(span, dict):
            continue
        spans.append(
            OmittedSpan(
                start_line=int(span.get("start_line", 1)),
                end_line=int(span.get("end_line", 1)),
                line_count=int(span.get("line_count", 0)),
                reason=str(span.get("reason", "native")),
            )
        )
    receipt = CompressionReceipt(
        original_tokens=int(receipt_raw.get("original_tokens", 0)),
        compressed_tokens=int(receipt_raw.get("compressed_tokens", 0)),
        savings_ratio=float(receipt_raw.get("savings_ratio", 0.0)),
        compression_level=int(receipt_raw.get("compression_level", 3)),
        content_type=str(receipt_raw.get("content_type", "native")),
        anchors_preserved=dict(receipt_raw.get("anchors_preserved", {}) or {}),
        omitted_spans=spans,
        recoverable=bool(receipt_raw.get("recoverable", True)),
    )
    return CompressionResult(
        compressed=str(raw.get("compressed", "")),
        receipt=receipt,
        changed=bool(raw.get("changed", False)),
    )


__all__ = ["compress_evidence_locked_fast", "native_elc_available"]
