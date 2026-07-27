"""
Provenance Chain for Entroly
===================================

Wraps the optimize_context output with source provenance metadata,
enabling hallucination detection at the LLM output level.

The core idea:
    Every fragment selected by optimize_context is "file-backed" — it came
    from a real source file ingested by the developer. If the LLM cites
    something that isn't in the provenance set, it hallucinated it.

This is a lightweight wrapper — no external dependencies required.
The ProvenanceRecord implements a standard provenance chain design
but is purpose-built for the context selection use case.
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

    # Emit one body field, one score, and one token-count field instead of
    # preserving aliases from the Rust and Python engine paths.
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

    return compact


def compact_optimize_result_for_wire(optimize_result: dict[str, Any]) -> None:
    """Compact an optimize result in-place at the MCP serialization boundary.

    ``selected`` and ``selected_fragments`` are compatibility aliases inside
    the engine. Sending both over MCP serializes the same fragment bodies and
    metadata twice. The public wire response keeps ``selected_fragments`` as
    the canonical key and removes the alias. Compact mode also strips internal
    scoring vectors while preserving content, source locations, token counts,
    and CCR exact-recovery handles.

    Set ``ENTROLY_MCP_FULL_DIAGNOSTICS=1`` to retain rich per-fragment fields.
    The duplicate alias is still removed because it has no diagnostic value.

    This function intentionally mutates its argument. Call it only immediately
    before MCP serialization, after all in-process consumers have finished with
    the rich engine result.
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
    if mode == "compact":
        response["diagnostics_hint"] = (
            f"Set {_FULL_DIAGNOSTICS_ENV}=1 for full fragment scoring fields "
            "and per-fragment provenance."
        )


@dataclass
class FragmentProvenance:
    """Provenance record for a single selected context fragment."""

    fragment_id: str
    source: str               # file path or URL — the external origin
    confidence: float         # composite relevance score [0, 1]
    token_count: int
    verified: bool            # True if source is a real file (not "internal_knowledge")
    is_pinned: bool = False
    quality_issues: list[str] = field(default_factory=list)

    @property
    def risk_contribution(self) -> str:
        """Contribution to hallucination risk."""
        if not self.verified:
            return "high"   # sourced from unknown origin
        if self.confidence < 0.3:
            return "medium"  # low relevance — LLM may extrapolate
        return "low"


@dataclass
class ContextProvenance:
    """
    Full provenance record for one optimize_context call.

    The hallucination_risk is computed from:
    1. Fraction of selected fragments with verified sources
    2. Average confidence of selection
    3. Whether any fragments have quality issues (secrets, TODOs)
    """

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
        return sum(1 for f in self.fragments if f.verified) / len(self.fragments)

    @property
    def avg_confidence(self) -> float:
        if not self.fragments:
            return 0.0
        return sum(f.confidence for f in self.fragments) / len(self.fragments)

    @property
    def source_set(self) -> set:
        """Set of verified source files — use to check LLM citations."""
        return {f.source for f in self.fragments if f.verified and f.source}

    @property
    def quality_flagged_sources(self) -> list[str]:
        """Sources with code quality issues."""
        return [f.source for f in self.fragments if f.quality_issues]

    @property
    def hallucination_risk(self) -> str:
        """
        low    — all fragments file-backed, high confidence
        medium — some low-confidence fragments, or 1-2 unverified
        high   — significant unverified content or very low confidence
        """
        if self.verified_fraction < 0.7:
            return "high"
        if self.avg_confidence < 0.25 or self.verified_fraction < 0.9:
            return "medium"
        return "low"

    def to_dict(self, *, include_fragments: bool = True) -> dict[str, Any]:
        """Serialize provenance without changing the historical SDK default.

        The Python API has always returned ``source_set`` and per-fragment
        provenance from ``to_dict()``. Keep that behavior by default. MCP callers
        that need a bounded envelope must opt into the aggregate-only form via
        ``to_wire_dict()`` or ``include_fragments=False``.
        """
        payload: dict[str, Any] = {
            "turn": self.turn,
            "query": self.query,
            "refined_query": self.refined_query,
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "budget_utilization": round(self.tokens_used / max(1, self.token_budget), 3),
            "fragment_count": len(self.fragments),
            "verified_fraction": round(self.verified_fraction, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "hallucination_risk": self.hallucination_risk,
            "quality_flagged": self.quality_flagged_sources,
        }

        if include_fragments:
            payload["source_set"] = sorted(self.source_set)
            payload["fragments"] = [
                {
                    "id": f.fragment_id,
                    "source": f.source,
                    "confidence": round(f.confidence, 4),
                    "tokens": f.token_count,
                    "verified": f.verified,
                    "pinned": f.is_pinned,
                    "risk": f.risk_contribution,
                    **({"quality_issues": f.quality_issues} if f.quality_issues else {}),
                }
                for f in self.fragments
            ]
        else:
            payload["details_omitted"] = {
                "fragment_records": len(self.fragments),
                "source_set_entries": len(self.source_set),
                "reason": "duplicated by selected_fragments",
                "diagnostics_env": _FULL_DIAGNOSTICS_ENV,
            }

        return payload

    def to_wire_dict(self) -> dict[str, Any]:
        """Serialize bounded MCP provenance, with opt-in full diagnostics."""
        return self.to_dict(include_fragments=_full_diagnostics_enabled())


def build_provenance(
    optimize_result: dict[str, Any],
    query: str,
    refined_query: str | None,
    turn: int,
    token_budget: int,
    quality_scan_fn=None,  # Optional: FragmentGuard.scan
) -> ContextProvenance:
    """
    Build a ContextProvenance from the raw optimize_context result dict.

    This builder is intentionally non-mutating. Wire compaction belongs at the
    MCP serialization boundary, not in this generic Python API.

    Args:
        optimize_result:  The dict returned by EntrolyEngine.optimize_context()
        query:            The original user query
        refined_query:    The expanded query (if any)
        turn:             Current session turn number
        token_budget:     The budget passed to optimize
        quality_scan_fn:  Optional callable(content, source) -> List[str]
    """
    selected = optimize_result.get("selected_fragments")
    if not isinstance(selected, list):
        fallback = optimize_result.get("selected")
        selected = fallback if isinstance(fallback, list) else []
    tokens_used = optimize_result.get("tokens_used", 0)

    frag_provenances = []
    for frag in selected:
        if not isinstance(frag, dict):
            continue
        fid = frag.get("id", frag.get("fragment_id", ""))
        source = frag.get("source", "")
        confidence = frag.get("composite_score", frag.get("relevance", 0.0))
        tokens = frag.get("token_count", frag.get("tokens", 0))
        is_pinned = frag.get("is_pinned", False)
        content = frag.get("content", "")

        # A fragment is "verified" if it has a non-empty source that looks
        # like a real file path (not "internal_knowledge" or blank)
        verified = bool(source) and source not in ("internal_knowledge", "unknown", "synthetic")

        # Quality scan (CodeQualityGuard)
        issues: list[str] = []
        if quality_scan_fn and content:
            issues = quality_scan_fn(content, source)

        frag_provenances.append(FragmentProvenance(
            fragment_id=fid,
            source=source,
            confidence=float(confidence),
            token_count=int(tokens),
            verified=verified,
            is_pinned=is_pinned,
            quality_issues=issues,
        ))

    return ContextProvenance(
        turn=turn,
        query=query,
        refined_query=refined_query if refined_query != query else None,
        fragments=frag_provenances,
        token_budget=token_budget,
        tokens_used=int(tokens_used),
    )
