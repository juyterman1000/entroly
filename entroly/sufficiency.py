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
from functools import lru_cache
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
    calibration_id: str | None = None
    boundary_exposure_measured: bool = True

    @property
    def sufficient(self) -> bool:
        return (
            self.calibrated
            and bool(self.calibration_id)
            and self.verdict == "sufficient"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_mass": round(self.captured_mass, 4),
            "shadow_price": round(self.shadow_price, 6),
            "cutoff_ambiguity": round(self.cutoff_ambiguity, 4),
            "boundary_exposure": round(self.boundary_exposure, 4),
            "boundary_exposure_measured": self.boundary_exposure_measured,
            "query_coverage": round(self.query_coverage, 4),
            "corpus_gap": round(self.corpus_gap, 4),
            "residual_risk": round(
                self.shadow_price / max(self.captured_mass, _EPS), 6
            ),
            "verdict": self.verdict,
            "calibrated": self.calibrated,
            "calibration_id": self.calibration_id,
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
    calibration: CalibrationPolicy | None = None,
    boundary_exposure_measured: bool = True,
) -> SufficiencyCertificate:
    """Compute diagnostics; certify only under an explicit named policy.

    ``calibrated=True`` is accepted for source compatibility but cannot turn
    provisional repository constants into an assurance claim.
    """

    del shadow_price_limit
    term_idf = query_term_idf or {}
    captured = captured_mass(candidates)
    shadow = shadow_price(candidates)
    ambiguity = cutoff_ambiguity(candidates)
    exposure = boundary_exposure(candidates, anchor_weights)
    coverage = query_coverage(
        retained_terms,
        term_idf,
        unattainable_terms,
    )
    gap = corpus_gap(term_idf, unattainable_terms)
    residual_risk = shadow / max(captured, _EPS)

    residual_limit = (
        calibration.residual_risk_limit
        if calibration is not None
        else RESIDUAL_RISK_LIMIT
    )
    coverage_limit = (
        calibration.minimum_query_coverage if calibration is not None else 0.5
    )
    exposure_limit = (
        calibration.maximum_boundary_exposure if calibration is not None else 0.0
    )

    reasons: list[str] = []
    if budget_exhausted and residual_risk > residual_limit:
        reasons.append(
            f"residual demand {residual_risk:.4f} per unit captured "
            f"(lambda={shadow:.5f}, captured={captured:.3f})"
        )
    if boundary_exposure_measured:
        if exposure > exposure_limit:
            reasons.append(
                f"{exposure:.0%} of anchor weight lost its required neighbouring context"
            )
    elif calibration is not None and calibration.require_boundary_measurement:
        reasons.append("boundary exposure was not measured for this selection")
    if coverage < coverage_limit and term_idf:
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
    elif calibration is not None:
        if term_idf or calibration.allow_empty_query:
            verdict = "sufficient"
        else:
            verdict = "uncalibrated"
            reasons.append(
                "the calibration policy does not authorize empty-query sufficiency"
            )
    else:
        verdict = "uncalibrated"
        reasons.append(
            "calibrated=True has no effect without a named CalibrationPolicy"
            if calibrated
            else "no measured gap was detected, but production sufficiency "
            "thresholds are not calibrated on a held-out workload"
        )

    return SufficiencyCertificate(
        captured_mass=captured,
        shadow_price=shadow,
        cutoff_ambiguity=ambiguity,
        boundary_exposure=exposure,
        boundary_exposure_measured=boundary_exposure_measured,
        query_coverage=coverage,
        corpus_gap=gap,
        verdict=verdict,
        reasons=tuple(reasons),
        calibrated=calibration is not None,
        calibration_id=(calibration.calibration_id if calibration else None),
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


# Memoised: `stem` is a pure function of a short token, and a session
# re-derives the same vocabulary on every turn. Profiling a 172-document
# session showed 683,520 calls across four optimize_context calls, almost
# all of them repeats. Bounded so a pathological corpus cannot grow it
# without limit.
@lru_cache(maxsize=100_000)
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
    """Smoothed IDF for one term.

    Retained for callers with a single term. Tokenising the corpus costs the
    same whether one term or twenty are being scored, so anything computing
    IDF for a SET of terms must use `_idf_map` instead -- see the note there.
    """
    n = len(corpus_texts)
    if n == 0:
        return 1.0
    df = sum(1 for text in corpus_texts if term in _lexical_terms(text))
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def _idf_map(
    terms: Iterable[str],
    corpus_texts: Sequence[str],
    document_terms: Sequence[set[str]] | None = None,
) -> dict[str, float]:
    """IDF for many terms, tokenising each document exactly once.

    `_idf` per term re-tokenised the whole corpus per term. On a 172-document
    session that was 20 terms x 172 documents = 3,440 tokenisations of the same
    text, and each one stems every token: `stem` was called 2,004,480 times in
    four optimize_context calls, and certificate construction accounted for 86%
    of total runtime.

    Tokenising once and reusing the sets makes the cost O(documents) instead of
    O(terms x documents). The IDF values are identical.

    `document_terms` lets a caller that has *already* tokenised the corpus hand
    the sets in rather than paying for it twice. `qccr._attach_sufficiency`
    tokenises every candidate file to compute its anchors and then called this,
    which tokenised the identical corpus again -- two full passes per selection.
    Values are unchanged either way; only the second pass disappears.
    """
    terms = list(terms)
    n = len(corpus_texts)
    if n == 0 or not terms:
        return {term: 1.0 for term in terms}
    if document_terms is None:
        document_terms = [_lexical_terms(text) for text in corpus_texts]
    elif len(document_terms) != n:
        raise ValueError(
            "document_terms must align 1:1 with corpus_texts "
            f"({len(document_terms)} vs {n})"
        )
    out: dict[str, float] = {}
    for term in terms:
        df = sum(1 for present in document_terms if term in present)
        out[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
    return out


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
    all_fragments,
    selected_fragments,
    query,
    *,
    token_budget,
    shadow_price_limit=None,
) -> SufficiencyCertificate:
    """Refuse to invent optimizer residue absent from these public inputs."""

    del (
        all_fragments,
        selected_fragments,
        query,
        token_budget,
        shadow_price_limit,
    )
    raise RuntimeError(
        "certify_selection cannot reconstruct optimizer residual state; use the "
        "certificate attached by entroly.qccr.select"
    )


def calibrated_shadow_price_limit(candidates: Sequence[Candidate]) -> float:
    selected = [candidate.density for candidate in candidates if candidate.selected]
    return 0.0 if not selected else _median(selected)


@dataclass(frozen=True)
class CalibrationPolicy:
    """Workload-validated thresholds that may authorize ``sufficient``."""

    calibration_id: str
    residual_risk_limit: float
    minimum_query_coverage: float = 0.5
    maximum_boundary_exposure: float = 0.0
    require_boundary_measurement: bool = True
    allow_empty_query: bool = False

    def __post_init__(self) -> None:
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must be non-empty")
        if not math.isfinite(self.residual_risk_limit) or self.residual_risk_limit < 0:
            raise ValueError("residual_risk_limit must be finite and non-negative")
        if not 0.0 <= self.minimum_query_coverage <= 1.0:
            raise ValueError("minimum_query_coverage must be between 0 and 1")
        if not 0.0 <= self.maximum_boundary_exposure <= 1.0:
            raise ValueError("maximum_boundary_exposure must be between 0 and 1")
