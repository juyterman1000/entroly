"""Local novelty-frontier accounting for Context Receipts.

The novelty frontier is intentionally deterministic and local-only. It audits
whether selected context contributes concepts beyond the query, then contrasts
that selected concept set with concepts still stranded in omitted chunks.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from collections.abc import Mapping
from typing import Any, Protocol


class _SelectedLike(Protocol):
    chunk_id: str
    source_path: str
    text: str


class _OmittedLike(Protocol):
    chunk_id: str
    source_path: str
    text_preview: str


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
    "does",
    "do",
    "did",
    "after",
    "before",
    "under",
    "over",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


MEDIUM_FRONTIER_SCORE_RATIO = 0.18
HIGH_FRONTIER_SCORE_RATIO = 0.45


@dataclass(frozen=True)
class NoveltyAssessmentPolicy:
    """Thresholds for mapping novelty-frontier metrics to reviewer pressure."""

    medium_score_ratio: float = MEDIUM_FRONTIER_SCORE_RATIO
    high_score_ratio: float = HIGH_FRONTIER_SCORE_RATIO
    minimum_medium_terms: int = 2
    minimum_high_terms: int = 4

    def __post_init__(self) -> None:
        _validate_ratio("medium_score_ratio", self.medium_score_ratio)
        _validate_ratio("high_score_ratio", self.high_score_ratio)
        if self.medium_score_ratio > self.high_score_ratio:
            raise ValueError(
                "medium_score_ratio must be less than or equal to high_score_ratio"
            )
        _validate_non_negative_int("minimum_medium_terms", self.minimum_medium_terms)
        _validate_non_negative_int("minimum_high_terms", self.minimum_high_terms)

    def medium_term_floor(self, selected_terms: int) -> int:
        return max(self.minimum_medium_terms, selected_terms // 2)

    def high_term_floor(self, selected_terms: int) -> int:
        return max(self.minimum_high_terms - 1, selected_terms)


def _validate_ratio(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite numeric ratio between 0.0 and 1.0")
    try:
        ratio = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"{name} must be a finite numeric ratio between 0.0 and 1.0"
        ) from None
    if not math.isfinite(ratio):
        raise ValueError(f"{name} must be a finite numeric ratio between 0.0 and 1.0")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _coerce_non_negative_float(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _summary_float(
    novelty: Mapping[str, Any], primary: str, fallback: str, default: float = 0.0
) -> float:
    for name in (primary, fallback):
        if name in novelty:
            value = _coerce_non_negative_float(novelty[name])
            if value is not None:
                return value
    return default


def _summary_int(novelty: Mapping[str, Any], name: str, default: int = 0) -> int:
    raw = novelty.get(name, default)
    if isinstance(raw, bool):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(default, value)


DEFAULT_NOVELTY_ASSESSMENT_POLICY = NoveltyAssessmentPolicy()


@dataclass(frozen=True)
class _ConceptProfile:
    """Aggregate evidence for a term without letting one chunk dominate."""

    frequency: int
    chunk_count: int
    source_paths: tuple[str, ...]

    @property
    def score(self) -> float:
        # Prefer terms corroborated across chunks, while still giving a small
        # signal to repeated terms.  The square root prevents repeated boilerplate
        # in one chunk from overwhelming a broader counterfactual frontier.
        return self.chunk_count + math.sqrt(self.frequency) / 10.0


def concept_terms(text: str) -> list[str]:
    """Extract stable concept terms for novelty accounting."""
    terms: list[str] = []
    for token in _TOKEN_RE.findall(text):
        term = _normalize_term(token)
        if term and term not in _STOPWORDS:
            terms.append(term)
    return terms


def _normalize_term(token: str) -> str:
    """Normalize lightweight lexical variants without a language model or stemmer."""
    term = token.lower().strip("_-")
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _profiles(
    chunk_term_counts: list[tuple[str, str, Counter[str]]],
) -> dict[str, _ConceptProfile]:
    frequencies: Counter[str] = Counter()
    chunks_by_term: dict[str, set[str]] = defaultdict(set)
    sources_by_term: dict[str, set[str]] = defaultdict(set)
    for chunk_id, source_path, terms in chunk_term_counts:
        frequencies.update(terms)
        for term in terms:
            chunks_by_term[term].add(chunk_id)
            sources_by_term[term].add(source_path)
    return {
        term: _ConceptProfile(
            frequency=frequencies[term],
            chunk_count=len(chunks_by_term[term]),
            source_paths=tuple(sorted(sources_by_term[term])),
        )
        for term in frequencies
    }


def _top_terms(profiles: dict[str, _ConceptProfile], *, limit: int = 8) -> list[str]:
    ranked = sorted(
        profiles.items(),
        key=lambda item: (
            -item[1].score,
            -item[1].chunk_count,
            -item[1].frequency,
            item[0],
        ),
    )
    return [term for term, _profile in ranked[:limit]]


def _serializable_profiles(
    profiles: dict[str, _ConceptProfile], terms: list[str]
) -> dict[str, dict[str, object]]:
    return {
        term: {
            "frequency": profiles[term].frequency,
            "chunk_count": profiles[term].chunk_count,
            "source_paths": list(profiles[term].source_paths),
            "score": round(profiles[term].score, 6),
        }
        for term in terms
        if term in profiles
    }


def novelty_summary(
    query: str,
    selected: list[_SelectedLike],
    omitted: list[_OmittedLike],
    *,
    omitted_text_by_chunk_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Describe selected evidence novelty and omitted evidence frontier gaps.

    ``omitted_text_by_chunk_id`` lets receipt generation use full omitted chunk
    text while report rehydration can still fall back to the recorded preview.
    """
    omitted_text_by_chunk_id = omitted_text_by_chunk_id or {}
    query_terms = set(concept_terms(query))
    selected_chunk_terms: list[tuple[str, str, Counter[str]]] = []
    omitted_chunk_terms: list[tuple[str, str, Counter[str]]] = []
    per_chunk: list[dict[str, object]] = []
    omitted_full_text_chunks = 0

    for item in selected:
        text_terms = Counter(
            term for term in concept_terms(item.text) if term not in query_terms
        )
        selected_chunk_terms.append((item.chunk_id, item.source_path, text_terms))
        profiles = _profiles([(item.chunk_id, item.source_path, text_terms)])
        top_terms = _top_terms(profiles, limit=6)
        per_chunk.append(
            {
                "chunk_id": item.chunk_id,
                "source_path": item.source_path,
                "novel_terms": top_terms,
                "novel_term_count": len(text_terms),
                "novel_term_occurrences": sum(text_terms.values()),
            }
        )

    for item in omitted:
        text = omitted_text_by_chunk_id.get(item.chunk_id)
        if text is not None:
            omitted_full_text_chunks += 1
        else:
            text = item.text_preview
        omitted_chunk_terms.append(
            (
                item.chunk_id,
                item.source_path,
                Counter(
                    term for term in concept_terms(text) if term not in query_terms
                ),
            )
        )

    selected_profiles = _profiles(selected_chunk_terms)
    omitted_profiles = _profiles(omitted_chunk_terms)
    selected_set = set(selected_profiles)
    omitted_set = set(omitted_profiles)
    frontier_gap = omitted_set - selected_set
    overlap = selected_set & omitted_set
    frontier_profiles = {term: omitted_profiles[term] for term in frontier_gap}
    selected_top = _top_terms(selected_profiles)
    frontier_top = _top_terms(frontier_profiles)
    selected_occurrences = sum(
        profile.frequency for profile in selected_profiles.values()
    )
    frontier_score = sum(profile.score for profile in frontier_profiles.values())
    omitted_score = sum(profile.score for profile in omitted_profiles.values())

    return {
        "query_terms": sorted(query_terms),
        "selected_novel_terms": selected_top,
        "omitted_frontier_terms": frontier_top,
        "shared_selected_omitted_terms": sorted(overlap)[:8],
        "selected_novel_term_count": len(selected_set),
        "selected_novel_term_occurrences": selected_occurrences,
        "omitted_frontier_term_count": len(frontier_gap),
        "omitted_full_text_chunks": omitted_full_text_chunks,
        "omitted_truncated_chunks": max(0, len(omitted) - omitted_full_text_chunks),
        "novelty_density": round(len(selected_set) / max(1, selected_occurrences), 6),
        "frontier_gap_ratio": round(len(frontier_gap) / max(1, len(omitted_set)), 6),
        "frontier_gap_score_ratio": round(frontier_score / max(1.0, omitted_score), 6),
        "selected_term_profiles": _serializable_profiles(
            selected_profiles, selected_top
        ),
        "omitted_frontier_profiles": _serializable_profiles(
            frontier_profiles, frontier_top
        ),
        "per_selected_chunk": per_chunk,
        "control": "local_counterfactual_concept_frontier.v3",
    }


def novelty_frontier_assessment(
    novelty: Mapping[str, Any] | object,
    *,
    policy: NoveltyAssessmentPolicy = DEFAULT_NOVELTY_ASSESSMENT_POLICY,
) -> dict[str, object]:
    """Map a novelty summary to an auditable reviewer action.

    The assessment lives with novelty accounting so receipt generation can stay
    orchestration-focused and downstream callers can classify stored novelty
    summaries without rebuilding a receipt.  A policy can be supplied by callers
    that need stricter or more permissive review thresholds.
    """
    summary = novelty if isinstance(novelty, Mapping) else {}
    score_ratio = _summary_float(
        summary, "frontier_gap_score_ratio", "frontier_gap_ratio"
    )
    gap_terms = _summary_int(summary, "omitted_frontier_term_count")
    selected_terms = _summary_int(summary, "selected_novel_term_count")
    term_pressure_floor = policy.high_term_floor(selected_terms)
    medium_term_floor = policy.medium_term_floor(selected_terms)

    pressure = "low"
    rationale = "no omitted frontier concepts were detected"
    action = "No novelty-specific review is required."
    if gap_terms > 0 and score_ratio > 0:
        pressure = "low"
        rationale = "omitted frontier is present but below escalation thresholds"
        action = "Spot-check omitted frontier terms if this receipt supports a high-stakes answer."
        if score_ratio >= policy.high_score_ratio or gap_terms > term_pressure_floor:
            pressure = "high"
            rationale = "omitted frontier is large or score-dominant relative to selected novelty"
            action = "Review omitted frontier terms before relying on this receipt."
        elif score_ratio >= policy.medium_score_ratio or gap_terms >= medium_term_floor:
            pressure = "medium"
            rationale = (
                "omitted frontier is material enough to require reviewer attention"
            )
            action = "Review omitted frontier terms when validating the answer."

    return {
        "pressure": pressure,
        "rationale": rationale,
        "action": action,
        "frontier_gap_score_ratio": round(score_ratio, 6),
        "omitted_frontier_term_count": gap_terms,
        "selected_novel_term_count": selected_terms,
        "thresholds": {
            "medium_score_ratio": policy.medium_score_ratio,
            "high_score_ratio": policy.high_score_ratio,
            "medium_term_floor": medium_term_floor,
            "high_term_floor": term_pressure_floor + 1,
        },
    }
