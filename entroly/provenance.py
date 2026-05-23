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

from dataclasses import dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "source_set": sorted(self.source_set),
            "quality_flagged": self.quality_flagged_sources,
            "fragments": [
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
            ],
        }


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

    Args:
        optimize_result:  The dict returned by EntrolyEngine.optimize_context()
        query:            The original user query
        refined_query:    The expanded query (if any)
        turn:             Current session turn number
        token_budget:     The budget passed to optimize
        quality_scan_fn:  Optional callable(content, source) -> List[str]
    """
    selected = optimize_result.get("selected", [])
    tokens_used = optimize_result.get("tokens_used", 0)

    frag_provenances = []
    for frag in selected:
        fid        = frag.get("id", frag.get("fragment_id", ""))
        source     = frag.get("source", "")
        confidence = frag.get("composite_score", frag.get("relevance", 0.0))
        tokens     = frag.get("token_count", frag.get("tokens", 0))
        is_pinned  = frag.get("is_pinned", False)
        content    = frag.get("content", "")

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
