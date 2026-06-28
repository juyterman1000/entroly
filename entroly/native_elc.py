"""Optional native fast-path adapter for Evidence-Locked Compression.

This module is intentionally conservative at the Python/Rust boundary:

1. Prefer a dedicated native ``entroly_core.elc_compress`` export when present.
2. Otherwise use the already-exported PyO3 ``py_compress_block`` compressor as a
   Rust fast-path candidate.
3. Accept native output only if it preserves answer-critical evidence anchors.
4. Fall back to the audited Python ELC implementation on any ambiguity.

This gives current wheels a real PyO3 path without risky edits to the giant
``entroly-core/src/lib.rs`` entrypoint, while keeping the future dedicated
``elc_compress`` ABI stable.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .evidence_locked_compression import (
    CompressionReceipt,
    CompressionResult,
    OmittedSpan,
    compress_evidence_locked,
)

_QUERY_WORD_RE = re.compile(r"\b[\w.-]{4,}\b")
_QUERY_STOPWORDS = frozenset(
    {
        "what",
        "where",
        "when",
        "which",
        "build",
        "fail",
        "failed",
        "failure",
        "error",
        "errors",
        "test",
        "tests",
        "tool",
        "tools",
        "result",
        "results",
        "output",
        "debug",
        "issue",
        "proxy",
        "logs",
        "log",
    }
)
_ANCHOR_RE = re.compile(
    r"(?i)(error|failed|failure|fatal|panic|exception|traceback|warning|timeout|denied|refused|invalid|unexpected|unresolved|exit code|assertion|segfault|not found|no such)"
)
_PATH_RE = re.compile(r"[\w./-]+\.(?:py|rs|ts|tsx|js|jsx|java|kt|go|json|ya?ml|toml)(?::\d+)?")


def native_elc_available() -> bool:
    """Return True when a usable native/PyO3 ELC path is importable."""
    try:
        import entroly_core  # type: ignore

        return callable(getattr(entroly_core, "elc_compress", None)) or callable(
            getattr(entroly_core, "py_compress_block", None)
        )
    except Exception:
        return False


def compress_evidence_locked_fast(
    text: str,
    *,
    query: str = "",
    budget_tokens: int = 1200,
) -> CompressionResult:
    """Use native/PyO3 compression only when it is evidence-safe."""
    native_result = _try_dedicated_native_elc(text, query=query, budget_tokens=budget_tokens)
    if native_result is not None:
        return native_result

    bridge_result = _try_pyo3_compress_block(text, query=query, budget_tokens=budget_tokens)
    if bridge_result is not None:
        return bridge_result

    return compress_evidence_locked(text, query=query, budget_tokens=budget_tokens)


def _try_dedicated_native_elc(
    text: str,
    *,
    query: str,
    budget_tokens: int,
) -> CompressionResult | None:
    try:
        import entroly_core  # type: ignore

        native = getattr(entroly_core, "elc_compress", None)
        if not callable(native):
            return None
        raw = native(text, query, int(budget_tokens))
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            result = _result_from_native(raw)
            if _evidence_safe(text, result.compressed, query):
                return result
    except Exception:
        return None
    return None


def _try_pyo3_compress_block(
    text: str,
    *,
    query: str,
    budget_tokens: int,
) -> CompressionResult | None:
    """Use the existing Rust/PyO3 block compressor as a guarded fast path.

    ``py_compress_block`` is already registered in the native module. It is not
    query-aware, so we treat its output as a candidate and accept it only if all
    detected evidence anchors and query-specific terms remain visible.
    """
    try:
        import entroly_core  # type: ignore

        native = getattr(entroly_core, "py_compress_block", None)
        if not callable(native):
            return None
        original_tokens = _estimate_tokens(text)
        compressed = native("tool", text, int(original_tokens), "skeleton", None)
        if not isinstance(compressed, str):
            return None
        if not compressed.strip() or len(compressed) >= len(text):
            return None
        if _estimate_tokens(compressed) > max(1, int(budget_tokens) + 64):
            # This native path is structural, not budget-solving. Do not pretend
            # it satisfied a tight ELC budget if it did not.
            return None
        if not _evidence_safe(text, compressed, query):
            return None
        original_line_count = len(text.splitlines())
        compressed_line_count = len(compressed.splitlines())
        omitted = []
        if compressed_line_count < original_line_count:
            omitted.append(
                OmittedSpan(
                    start_line=1,
                    end_line=original_line_count,
                    line_count=max(0, original_line_count - compressed_line_count),
                    reason="native_pyo3_skeleton",
                )
            )
        compressed_tokens = _estimate_tokens(compressed)
        original_tokens = _estimate_tokens(text)
        return CompressionResult(
            compressed=compressed,
            receipt=CompressionReceipt(
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                savings_ratio=max(0.0, 1.0 - compressed_tokens / max(original_tokens, 1)),
                compression_level=2,
                content_type="native_pyo3_block",
                anchors_preserved={"evidence": len(_evidence_needles(text, query))},
                omitted_spans=omitted,
                recoverable=True,
            ),
            changed=True,
        )
    except Exception:
        return None


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


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _evidence_safe(original: str, compressed: str, query: str) -> bool:
    lower_compressed = compressed.lower()
    for needle in _evidence_needles(original, query):
        if needle.lower() not in lower_compressed:
            return False
    return True


def _evidence_needles(text: str, query: str) -> list[str]:
    needles: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _ANCHOR_RE.search(stripped):
            needles.append(stripped[:180])
        for match in _PATH_RE.finditer(stripped):
            needles.append(match.group(0))
    query_terms = [
        m.group(0).lower()
        for m in _QUERY_WORD_RE.finditer(query or "")
        if m.group(0).lower() not in _QUERY_STOPWORDS
    ]
    for term in query_terms:
        for line in text.splitlines():
            if term in line.lower():
                needles.append(line.strip()[:180])
                break
    # Preserve order but remove duplicates/very-short anchors.
    seen: set[str] = set()
    out: list[str] = []
    for needle in needles:
        clean = needle.strip()
        if len(clean) < 4 or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
    return out[:32]


__all__ = ["compress_evidence_locked_fast", "native_elc_available"]
