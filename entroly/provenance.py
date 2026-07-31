"""
Entroly Provenance Chain
========================

Wraps optimize_context output with source provenance metadata, enabling
hallucination detection at the LLM output level.

Every selected fragment is source-backed. Exact line/byte coordinates and
content hashes are retained when the ingest path supplies them, while older
callers that only provide file-level provenance remain compatible.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


_FULL_DIAGNOSTICS_ENV = "ENTROLY_MCP_FULL_DIAGNOSTICS"

# Agent-useful wire fields. The engine can keep a much richer in-process
# fragment object, but serializing all scoring features for hundreds of
# fragments makes MCP metadata larger than the selected context itself.
_WIRE_FRAGMENT_KEYS = (
    "id",
    "fragment_id",
    "source",
    "content",
    "text",
    "token_count",
    "tokens",
    "relevance",
    "relevance_score",
    "composite_score",
    "resolution",
    "retrieval_handle",
    "content_sha256",
    "original_tokens",
    "compressed_tokens",
    "is_pinned",
    "start_line",
    "end_line",
    "start_byte",
    "end_byte",
    "byte_start",
    "byte_end",
    "source_version",
    "commit",
    "transform_receipt_id",
    "receipt_id",
    "lines",
    "language",
    "kind",
    "role",
    "quality_issues",
)


def _full_diagnostics_enabled() -> bool:
    return os.environ.get(_FULL_DIAGNOSTICS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _compact_fragment_for_wire(raw: Any) -> Any:
    """Return one canonical, agent-useful fragment representation."""
    if not isinstance(raw, dict):
        return raw

    compact: dict[str, Any] = {
        key: raw[key]
        for key in _WIRE_FRAGMENT_KEYS
        if key in raw and raw[key] is not None
    }

    if "id" not in compact and compact.get("fragment_id"):
        compact["id"] = compact["fragment_id"]

    # Emit one body field, score, token count, and coordinate spelling instead
    # of preserving aliases from the Rust and Python engine paths.
    if "content" not in compact and "text" in compact:
        compact["content"] = compact["text"]
    compact.pop("text", None)

    if "relevance" not in compact:
        for key in ("relevance_score", "composite_score"):
            score = compact.get(key)
            if isinstance(score, (int, float)):
                compact["relevance"] = score
                break
    compact.pop("relevance_score", None)
    compact.pop("composite_score", None)

    if "token_count" not in compact and "tokens" in compact:
        compact["token_count"] = compact["tokens"]
    compact.pop("tokens", None)

    if "start_byte" not in compact and "byte_start" in compact:
        compact["start_byte"] = compact["byte_start"]
    if "end_byte" not in compact and "byte_end" in compact:
        compact["end_byte"] = compact["byte_end"]
    compact.pop("byte_start", None)
    compact.pop("byte_end", None)

    if "source_version" not in compact and compact.get("commit"):
        compact["source_version"] = compact["commit"]
    compact.pop("commit", None)

    if "transform_receipt_id" not in compact and compact.get("receipt_id"):
        compact["transform_receipt_id"] = compact["receipt_id"]
    compact.pop("receipt_id", None)

    return compact


def compact_optimize_result_for_wire(optimize_result: dict[str, Any]) -> None:
    """Compact an optimize result in-place at the MCP serialization boundary.

    ``selected`` and ``selected_fragments`` are compatibility aliases inside
    the engine. Sending both over MCP serializes the same fragment bodies and
    metadata twice. The public wire response keeps ``selected_fragments`` as
    the canonical key and removes the alias. Compact mode strips internal
    scoring vectors while preserving content, exact source locations, token
    counts, hashes, receipt IDs, and exact-recovery handles.

    Set ``ENTROLY_MCP_FULL_DIAGNOSTICS=1`` to retain rich per-fragment fields.
    This function intentionally mutates its argument. Call it only immediately
    before MCP serialization, after all in-process consumers have finished.
    """
    if not isinstance(optimize_result, dict):
        return

    selected = optimize_result.get("selected_fragments")
    if not isinstance(selected, list):
        fallback = optimize_result.get("selected")
        selected = fallback if isinstance(fallback, list) else []

    optimize_result.pop("selected", None)
    if _full_diagnostics_enabled():
        optimize_result["selected_fragments"] = list(selected)
        mode = "diagnostics"
    else:
        optimize_result["selected_fragments"] = [
            _compact_fragment_for_wire(fragment) for fragment in selected
        ]
        mode = "compact"

    response = optimize_result.get("response")
    if not isinstance(response, dict):
        response = {}
        optimize_result["response"] = response
    response.update(
        {
            "mode": mode,
            "canonical_selection_key": "selected_fragments",
            "omitted_duplicate_alias": "selected",
        }
    )
    # `provenance.fragments` mirrors the selection, so leaving it untouched
    # re-serializes every body a third time. Dogfooding an 8,000-token request
    # returned a 378,545-char payload that overflowed the MCP result cap: the
    # agent got an error instead of context, while the selection itself was
    # correct at 7,004 tokens. Removing the `selected` alias alone still left
    # ~410,000 chars because provenance kept 395 full fragments.
    #
    # This is also what the diagnostics hint below already promises the env
    # flag restores -- "per-fragment provenance" -- so compact mode was
    # documented to strip it and did not.
    #
    # The same compactor is reused, which keeps the trust-critical fields:
    # content, source, token counts, hashes, receipt IDs and exact-recovery
    # handles all survive. Only internal scoring vectors are dropped.
    if mode == "compact":
        provenance = optimize_result.get("provenance")
        if isinstance(provenance, dict):
            fragments = provenance.get("fragments")
            if isinstance(fragments, list):
                provenance["fragments"] = [
                    _compact_fragment_for_wire(fragment) for fragment in fragments
                ]

    if mode == "compact":
        response["diagnostics_hint"] = (
            f"Set {_FULL_DIAGNOSTICS_ENV}=1 for full fragment scoring fields "
            "and per-fragment provenance."
        )


@dataclass
class FragmentProvenance:
    """Provenance record for one selected context fragment."""

    fragment_id: str
    source: str
    confidence: float
    token_count: int
    verified: bool
    is_pinned: bool = False
    quality_issues: list[str] = field(default_factory=list)
    start_line: int | None = None
    end_line: int | None = None
    start_byte: int | None = None
    end_byte: int | None = None
    content_sha256: str = ""
    source_version: str = ""
    retrieval_handle: str = ""
    transform_receipt_id: str = ""

    @property
    def risk_contribution(self) -> str:
        if not self.verified:
            return "high"
        if self.confidence < 0.3:
            return "medium"
        return "low"

    @property
    def exact_span(self) -> bool:
        line_exact = (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line >= self.start_line
        )
        byte_exact = (
            self.start_byte is not None
            and self.end_byte is not None
            and self.end_byte > self.start_byte
        )
        return bool(self.content_sha256 and (line_exact or byte_exact))

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.fragment_id,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "tokens": self.token_count,
            "verified": self.verified,
            "pinned": self.is_pinned,
            "risk": self.risk_contribution,
            "exact_span": self.exact_span,
        }
        optional = {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "content_sha256": self.content_sha256 or None,
            "source_version": self.source_version or None,
            "retrieval_handle": self.retrieval_handle or None,
            "transform_receipt_id": self.transform_receipt_id or None,
            "quality_issues": self.quality_issues or None,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


@dataclass
class ContextProvenance:
    """Full provenance record for one optimize_context call."""

    turn: int
    query: str
    refined_query: str | None
    fragments: list[FragmentProvenance]
    token_budget: int
    tokens_used: int

    @property
    def verified_fraction(self) -> float:
        if not self.fragments:
            return 0.0
        return sum(1 for fragment in self.fragments if fragment.verified) / len(self.fragments)

    @property
    def exact_span_fraction(self) -> float:
        if not self.fragments:
            return 0.0
        return sum(1 for fragment in self.fragments if fragment.exact_span) / len(self.fragments)

    @property
    def avg_confidence(self) -> float:
        if not self.fragments:
            return 0.0
        return sum(fragment.confidence for fragment in self.fragments) / len(self.fragments)

    @property
    def source_set(self) -> set[str]:
        return {
            fragment.source
            for fragment in self.fragments
            if fragment.verified and fragment.source
        }

    @property
    def quality_flagged_sources(self) -> list[str]:
        return [fragment.source for fragment in self.fragments if fragment.quality_issues]

    @property
    def hallucination_risk(self) -> str:
        if self.verified_fraction < 0.7:
            return "high"
        if self.avg_confidence < 0.25 or self.verified_fraction < 0.9:
            return "medium"
        return "low"

    def to_dict(self, *, include_fragments: bool = True) -> dict[str, Any]:
        """Serialize provenance without changing the historical SDK default."""
        payload: dict[str, Any] = {
            "turn": self.turn,
            "query": self.query,
            "refined_query": self.refined_query,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "budget_utilization": round(self.tokens_used / max(1, self.token_budget), 3),
            "fragment_count": len(self.fragments),
            "verified_fraction": round(self.verified_fraction, 3),
            "exact_span_fraction": round(self.exact_span_fraction, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "hallucination_risk": self.hallucination_risk,
            "quality_flagged": self.quality_flagged_sources,
        }

        if include_fragments:
            payload["source_set"] = sorted(self.source_set)
            payload["fragments"] = [fragment.as_dict() for fragment in self.fragments]
        else:
            payload["details_omitted"] = {
                "fragment_records": len(self.fragments),
                "source_set_entries": len(self.source_set),
                "exact_span_records": sum(1 for fragment in self.fragments if fragment.exact_span),
                "reason": "duplicated by selected_fragments",
                "diagnostics_env": _FULL_DIAGNOSTICS_ENV,
            }

        return payload

    def to_wire_dict(self) -> dict[str, Any]:
        return self.to_dict(include_fragments=_full_diagnostics_enabled())


def _optional_int(fragment: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = fragment.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _optional_text(fragment: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = fragment.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def build_provenance(
    optimize_result: dict[str, Any],
    query: str,
    refined_query: str | None,
    turn: int,
    token_budget: int,
    quality_scan_fn=None,
) -> ContextProvenance:
    """Build non-mutating provenance from an optimize_context result."""
    selected = optimize_result.get("selected_fragments")
    if not isinstance(selected, list):
        fallback = optimize_result.get("selected")
        selected = fallback if isinstance(fallback, list) else []
    tokens_used = optimize_result.get("tokens_used", 0)

    records: list[FragmentProvenance] = []
    for fragment in selected:
        if not isinstance(fragment, dict):
            continue
        fragment_id = fragment.get("id", fragment.get("fragment_id", ""))
        source = fragment.get("source", "")
        confidence = fragment.get("composite_score", fragment.get("relevance", 0.0))
        token_count = fragment.get("token_count", fragment.get("tokens", 0))
        content = fragment.get("content", fragment.get("text", ""))
        verified = bool(source) and source not in {
            "internal_knowledge",
            "unknown",
            "synthetic",
        }
        issues: list[str] = []
        if quality_scan_fn and content:
            issues = quality_scan_fn(content, source)

        records.append(
            FragmentProvenance(
                fragment_id=str(fragment_id),
                source=str(source),
                confidence=float(confidence),
                token_count=int(token_count),
                verified=verified,
                is_pinned=bool(fragment.get("is_pinned", False)),
                quality_issues=issues,
                start_line=_optional_int(fragment, "start_line", "line_start"),
                end_line=_optional_int(fragment, "end_line", "line_end"),
                start_byte=_optional_int(fragment, "start_byte", "byte_start"),
                end_byte=_optional_int(fragment, "end_byte", "byte_end"),
                content_sha256=_optional_text(fragment, "content_sha256", "sha256"),
                source_version=_optional_text(fragment, "source_version", "commit", "git_commit"),
                retrieval_handle=_optional_text(fragment, "retrieval_handle"),
                transform_receipt_id=_optional_text(
                    fragment,
                    "transform_receipt_id",
                    "context_receipt_id",
                    "receipt_id",
                ),
            )
        )

    return ContextProvenance(
        turn=turn,
        query=query,
        refined_query=refined_query if refined_query != query else None,
        fragments=records,
        token_budget=token_budget,
        tokens_used=int(tokens_used),
    )
