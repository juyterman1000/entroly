"""Fail-closed optimizer-derived sufficiency certificates.

The current residual-risk thresholds are research signals, not calibrated
probability guarantees. Production callers therefore cannot receive a
``sufficient`` verdict unless they explicitly opt into a separately validated
calibration. Missing discriminative evidence always requests retrieval
expansion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_EPS = 1e-9
RESIDUAL_RISK_LIMIT = 0.0121


@dataclass(frozen=True)
class Candidate:
    unit_id: str
    utility: float
    cost: int
    selected: bool
    anchors: tuple[str, ...] = ()
    neighbourhood: tuple[str, ...] = ()

    @property
    def density(self) -> float:
        return self.utility / self.cost if self.cost > 0 else 0.0


@dataclass(frozen=True)
class SufficiencyCertificate:
    captured_mass: float
    shadow_price: float
    cutoff_ambiguity: float
    boundary_exposure: float
    query_coverage: float
    corpus_gap: float
    verdict: str
    reasons: tuple[str, ...] = field(default=())
    calibrated: bool = False

    @property
    def sufficient(self) -> bool:
        return self.calibrated and self.verdict == "sufficient"

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_mass": round(self.captured_mass, 4),
            "shadow_price": round(self.shadow_price, 6),
            "cutoff_ambiguity": round(self.cutoff_ambiguity, 4),
            "boundary_exposure": round(self.boundary_exposure, 4),
            "query_coverage": round(self.query_coverage, 4),
            "corpus_gap": round(self.corpus_gap, 4),
            "residual_risk": round(
                self.shadow_price / max(self.captured_mass, _EPS), 6
            ),
            "verdict": self.verdict,
            "calibrated": self.calibrated,
            "reasons": list(self.reasons),
        }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    centre = _median(values)
    return _median([abs(v - centre) for v in values])


def captured_mass(candidates: Sequence[Candidate]) -> float:
    total = sum(max(candidate.utility, 0.0) for candidate in candidates)
    if total <= _EPS:
        return 0.0
    kept = sum(
        max(candidate.utility, 0.0)
        for candidate in candidates
        if candidate.selected
    )
    return kept / (total + _EPS)


def shadow_price(candidates: Sequence[Candidate]) -> float:
    excluded = [candidate for candidate in candidates if not candidate.selected]
    if not excluded:
        return 0.0
    return max(
        (max(candidate.utility, 0.0) / candidate.cost)
        if candidate.cost > 0
        else 0.0
        for candidate in excluded
    )


def cutoff_ambiguity(candidates: Sequence[Candidate]) -> float:
    selected = [candidate.density for candidate in candidates if candidate.selected]
    excluded = [candidate.density for candidate in candidates if not candidate.selected]
    if not selected or not excluded:
        return 0.0
    gap = abs(min(selected) - max(excluded))
    scale = _mad([candidate.density for candidate in candidates])
    return math.exp(-gap / (scale + _EPS))


def boundary_exposure(
    candidates: Sequence[Candidate],
    anchor_weights: dict[str, float] | None = None,
) -> float:
    kept = {candidate.unit_id for candidate in candidates if candidate.selected}
    weights = anchor_weights or {}
    numerator = 0.0
    denominator = 0.0
    for candidate in candidates:
        if not candidate.selected:
            continue
        for anchor in candidate.anchors:
            weight = weights.get(anchor, 1.0)
            denominator += weight
            if not set(candidate.neighbourhood).issubset(kept):
                numerator += weight
    return 0.0 if denominator <= _EPS else numerator / denominator


def query_coverage(
    retained_terms: Iterable[str],
    query_term_idf: dict[str, float],
    unattainable_terms: Iterable[str] = (),
) -> float:
    unattainable = set(unattainable_terms)
    attainable_weights = {
        term: max(value, 0.0)
        for term, value in query_term_idf.items()
        if term not in unattainable
    }
    total = sum(attainable_weights.values())
    if total <= _EPS:
        return 0.0
    retained = set(retained_terms)
    covered = sum(
        value for term, value in attainable_weights.items() if term in retained
    )
    return covered / (total + _EPS)


def corpus_gap(
    query_term_idf: dict[str, float],
    unattainable_terms: Iterable[str] = (),
) -> float:
    if not query_term_idf:
        return 0.0
    absent = set(unattainable_terms) & set(query_term_idf)
    return len(absent) / len(query_term_idf)


def certify(
    candidates: Sequence[Candidate],
    *,
    query_term_idf: dict[str, float] | None = None,
    retained_terms: Iterable[str] = (),
    unattainable_terms: Iterable[str] = (),
    anchor_weights: dict[str, float] | None = None,
    budget_exhausted: bool = True,
    shadow_price_limit: float = 0.0,
    calibrated: bool = False,
) -> SufficiencyCertificate:
    """Return an auditable verdict without overstating calibration.

    ``shadow_price_limit`` is retained for API compatibility and workload
    policy, but the provisional repository-wide ratio threshold remains the
    conservative default until held-out calibration exists.
    """
    del shadow_price_limit
    c_b = captured_mass(candidates)
    lambda_b = shadow_price(candidates)
    ambiguity = cutoff_ambiguity(candidates)
    exposure = boundary_exposure(candidates, anchor_weights)
    coverage = query_coverage(
        retained_terms,
        query_term_idf or {},
        unattainable_terms,
    )
    gap = corpus_gap(query_term_idf or {}, unattainable_terms)
    residual_risk = lambda_b / max(c_b, _EPS)

    reasons: list[str] = []
    if budget_exhausted and residual_risk > RESIDUAL_RISK_LIMIT:
        reasons.append(
            f"residual demand {residual_risk:.4f} per unit captured "
            f"(lambda={lambda_b:.5f}, captured={c_b:.3f})"
        )
    if exposure > 0.0:
        reasons.append(
            f"{exposure:.0%} of anchor weight lost its required neighbouring context"
        )
    if coverage < 0.5 and query_term_idf:
        reasons.append(f"query coverage {coverage:.0%}: discriminative terms dropped")

    if gap > 0.0:
        verdict = "expand_required"
        reasons.insert(
            0,
            f"{gap:.0%} of discriminative query terms appears in no candidate: "
            "the evidence was never retrieved",
        )
    elif reasons:
        verdict = "degraded"
    elif calibrated:
        verdict = "sufficient"
    else:
        verdict = "uncalibrated"
        reasons.append(
            "no measured gap was detected, but production sufficiency thresholds "
            "are not calibrated on a held-out workload"
        )

    return SufficiencyCertificate(
        captured_mass=c_b,
        shadow_price=lambda_b,
        cutoff_ambiguity=ambiguity,
        boundary_exposure=exposure,
        query_coverage=coverage,
        corpus_gap=gap,
        verdict=verdict,
        reasons=tuple(reasons),
        calibrated=calibrated,
    )


_QUESTION_WORDS = frozenset(
    """
    how what why when where which who whom whose does did done doing
    and are but for from has have had the this that these those there here
    with without into onto over under after before while during about
    through across between among against along toward towards upon
    can could should would will shall may might must
    its it's their they them then than thus you your our ours
    get gets got use uses used using make makes made
    """.split()
)
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _query_terms(query: str) -> list[str]:
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", query)
    return [
        term
        for term in re.split(r"[^A-Za-z0-9]+", split.lower())
        if len(term) > 2 and term not in _QUESTION_WORDS
    ]


def stem(word: str) -> str | None:
    n = len(word)
    if n >= 6 and word.endswith("ies"):
        return word[: n - 3] + "y"
    if n >= 6 and word.endswith("es"):
        base = word[: n - 2]
        if base.endswith(("s", "x", "z", "ch", "sh")):
            return base
    if n >= 7 and word.endswith("ing"):
        return word[: n - 3]
    if n >= 6 and word.endswith("ed"):
        return word[: n - 2]
    if n >= 5 and word.endswith("s") and not word.endswith(("ss", "us")):
        return word[: n - 1]
    if n >= 6 and word.endswith("e") and not word.endswith("ee"):
        return word[: n - 1]
    return None


def _lexical_terms(text: str) -> set[str]:
    terms: set[str] = set()
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    for token in _TOKEN.findall(split.lower()):
        terms.add(token)
        reduced = stem(token)
        if reduced:
            terms.add(reduced)
    return terms


def attainable(term: str, corpus_lowered: Sequence[str]) -> bool:
    """Use token membership, never unsafe raw-substring matching."""
    reduced = stem(term)
    for text in corpus_lowered:
        terms = _lexical_terms(text)
        if term in terms or (reduced is not None and reduced in terms):
            return True
    return False


def _idf(term: str, corpus_texts: Sequence[str]) -> float:
    n = len(corpus_texts)
    if n == 0:
        return 1.0
    df = sum(1 for text in corpus_texts if term in _lexical_terms(text))
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def candidates_from_selection(
    all_fragments: Sequence[dict[str, Any]],
    selected_fragments: Sequence[dict[str, Any]],
    query: str,
    *,
    neighbourhood_radius: int = 1,
) -> list[Candidate]:
    from .qccr import logical_source

    def key(fragment: dict[str, Any]) -> str:
        return str(fragment.get("id") or fragment.get("source") or id(fragment))

    selected_keys = {key(fragment) for fragment in selected_fragments}
    terms = set(_query_terms(query))
    by_source: dict[str, list[dict[str, Any]]] = {}
    for fragment in all_fragments:
        by_source.setdefault(
            logical_source(str(fragment.get("source") or "")), []
        ).append(fragment)

    positions: dict[str, tuple[str, int]] = {}
    for source, group in by_source.items():
        for index, fragment in enumerate(group):
            positions[key(fragment)] = (source, index)

    result: list[Candidate] = []
    for fragment in all_fragments:
        unit_id = key(fragment)
        content = str(fragment.get("content") or "")
        lexical = _lexical_terms(content)
        utility = fragment.get("relevance")
        if not isinstance(utility, (int, float)):
            utility = fragment.get("relevance_score")
        if not isinstance(utility, (int, float)):
            utility = 0.0
        cost = fragment.get("token_count")
        if not isinstance(cost, (int, float)) or cost <= 0:
            cost = max(1, len(content) // 4)
        anchors = tuple(sorted(term for term in terms if term in lexical))
        source, index = positions.get(unit_id, ("", 0))
        group = by_source.get(source, [])
        neighbourhood = tuple(
            key(group[position])
            for position in range(
                max(0, index - neighbourhood_radius),
                min(len(group), index + neighbourhood_radius + 1),
            )
        )
        result.append(
            Candidate(
                unit_id=unit_id,
                utility=float(utility),
                cost=int(cost),
                selected=unit_id in selected_keys,
                anchors=anchors,
                neighbourhood=neighbourhood,
            )
        )
    return result


def certify_selection(
    all_fragments: Sequence[dict[str, Any]],
    selected_fragments: Sequence[dict[str, Any]],
    query: str,
    *,
    token_budget: int,
    shadow_price_limit: float | None = None,
) -> SufficiencyCertificate:
    candidates = candidates_from_selection(all_fragments, selected_fragments, query)
    corpus = [str(fragment.get("content") or "") for fragment in all_fragments]
    lowered = [text.lower() for text in corpus]
    terms = _query_terms(query)
    term_idf = {term: _idf(term, corpus) for term in terms}
    unattainable = {term for term in terms if not attainable(term, lowered)}
    retained = {
        term
        for candidate in candidates
        if candidate.selected
        for term in candidate.anchors
    }
    delivered = sum(candidate.cost for candidate in candidates if candidate.selected)
    budget_exhausted = delivered >= int(token_budget) * 0.95
    return certify(
        candidates,
        query_term_idf=term_idf,
        retained_terms=retained,
        unattainable_terms=unattainable,
        budget_exhausted=budget_exhausted,
        shadow_price_limit=(
            calibrated_shadow_price_limit(candidates)
            if shadow_price_limit is None
            else shadow_price_limit
        ),
        calibrated=False,
    )


def calibrated_shadow_price_limit(candidates: Sequence[Candidate]) -> float:
    selected = [candidate.density for candidate in candidates if candidate.selected]
    return 0.0 if not selected else _median(selected)
