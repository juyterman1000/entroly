"""Optimizer-derived sufficiency certificate for a budgeted selection.

Compression that silently drops the answer is worse than compression that
refuses. Measured on SQuAD 2.0 at a 100-token budget, entroly lost 8 accuracy
points (0.80 -> 0.72); all four regressions were *answerable* questions where
the model answered around the answer -- right topic, missing span -- and one of
them had been compressed by only 8%. Ratio was not the cause and abstention was
not the cure, so neither an adaptive-depth predictor nor a learned sentence
extractor addresses it.

What the optimizer already knows, and throws away, is its own residual state:
which candidates it could not afford, how badly it wanted them, and how close
the cut was. This module turns that residue into five signals at negligible
cost -- no model, no training, no second pass -- and into a fail-closed verdict
the caller can act on and a receipt an auditor can check.

Related work, read rather than assumed
--------------------------------------
The *concept* of context sufficiency is not ours. "Sufficient Context: A New
Lens on Retrieval Augmented Generation Systems" (arXiv:2411.06037, Google)
defines sufficient context as containing "all the necessary information to
provide a definitive answer", and shows large models often answer wrongly
instead of abstaining when it is absent -- guided abstention improves correct
answers by 2-10%.

What differs is the mechanism and the setting, verified against that work:

  * Their sufficiency signal is an LLM autorater -- a prompted Gemini 1.5 Pro
    with chain-of-thought and a 1-shot example, ~93% classification accuracy,
    one model call per query. This module reads state the optimizer has
    already computed: no model, no prompt, no marginal cost.
  * They evaluate query-context pairs *after* retrieval and do not use
    retriever-internal scores. Every signal here is retriever-internal --
    captured mass, shadow price, cutoff margin.
  * They address retrieval depth and abstention, explicitly not compression
    or truncation. This is a budgeted-compression signal: the failure it
    detects is an answer span severed by a token budget, which no amount of
    abstention or re-retrieval prevents.

AdaComp (arXiv:2409.01579) trains a Llama2-7B predictor to choose retrieval
depth k; EXIT (arXiv:2412.12559) learns contextual sentence extraction. Both
decide *which units to keep* and neither reports whether the kept set was
enough. The claim here is therefore narrow: a training-free sufficiency
signal derived from optimizer residue, in the compression setting, emitted as
an auditable receipt.

Notation, for candidate unit ``i``:

    u_i   utility (relevance), c_i   token cost,   d_i = u_i / c_i   density
    S_B   the set selected under budget B

Signals
-------
``captured_mass`` C_B
    Share of positive utility retained. High C_B with a small output means the
    evidence was localised and the selection found it -- the needle case, which
    used 56 of 2,000 tokens at retention 1.0.

``shadow_price`` λ_B
    max density among *excluded* candidates: the marginal relevance one more
    token would buy. The load-bearing signal. Low means the budget is not
    binding; high means useful evidence sits outside it, and at a hard limit
    that reads "expand or bypass", not "trust this compression".

``cutoff_ambiguity`` A_B
    How close the last-selected and best-excluded densities are, scaled by the
    MAD of the density distribution. Near 1 the selection is one perturbation
    away from a materially different answer.

``boundary_exposure`` E_B
    IDF-weighted share of anchors whose protected neighbourhood was not fully
    retained. Targets the observed failure directly: an anchor kept, its
    answer-bearing neighbour cut.

``query_coverage`` Q_B
    IDF-weighted share of the *original* query terms retained. Deliberately not
    the intent-expanded vocabulary: expansion inflates coverage and would make
    a selection that dropped every discriminating term look well covered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

_EPS = 1e-9


@dataclass(frozen=True)
class Candidate:
    """One unit the optimizer considered."""

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
    """Auditable verdict on whether a selection is likely to be sufficient."""

    captured_mass: float
    shadow_price: float
    cutoff_ambiguity: float
    boundary_exposure: float
    query_coverage: float
    verdict: str
    reasons: tuple[str, ...] = field(default=())

    @property
    def sufficient(self) -> bool:
        return self.verdict == "sufficient"

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_mass": round(self.captured_mass, 4),
            "shadow_price": round(self.shadow_price, 6),
            "cutoff_ambiguity": round(self.cutoff_ambiguity, 4),
            "boundary_exposure": round(self.boundary_exposure, 4),
            "query_coverage": round(self.query_coverage, 4),
            "verdict": self.verdict,
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
    """Median absolute deviation: a scale estimate that a few huge densities
    cannot inflate, which matters because one dominant unit is the normal
    shape for a localised answer."""
    if not values:
        return 0.0
    centre = _median(values)
    return _median([abs(v - centre) for v in values])


def captured_mass(candidates: Sequence[Candidate]) -> float:
    """C_B: share of positive utility retained."""
    total = sum(max(c.utility, 0.0) for c in candidates)
    if total <= _EPS:
        return 0.0
    kept = sum(max(c.utility, 0.0) for c in candidates if c.selected)
    return kept / (total + _EPS)


def shadow_price(candidates: Sequence[Candidate]) -> float:
    """λ_B: the best density the budget could not afford.

    Zero when nothing was excluded -- the budget was not binding, so the
    selection is complete rather than merely affordable.
    """
    excluded = [c for c in candidates if not c.selected]
    if not excluded:
        return 0.0
    return max((max(c.utility, 0.0) / c.cost) if c.cost > 0 else 0.0 for c in excluded)


def cutoff_ambiguity(candidates: Sequence[Candidate]) -> float:
    """A_B: how marginal the last inclusion was, in units of density spread."""
    selected = [c.density for c in candidates if c.selected]
    excluded = [c.density for c in candidates if not c.selected]
    if not selected or not excluded:
        return 0.0
    gap = abs(min(selected) - max(excluded))
    scale = _mad([c.density for c in candidates])
    return math.exp(-gap / (scale + _EPS))


def boundary_exposure(
    candidates: Sequence[Candidate],
    anchor_weights: dict[str, float] | None = None,
) -> float:
    """E_B: IDF-weighted share of anchors that lost required neighbours.

    An anchor is a high-relevance span or a query-term hit. Keeping the anchor
    while cutting its neighbourhood is the exact shape of the observed SQuAD
    failure: the model reaches the right topic and stops short of the answer.
    """
    kept = {c.unit_id for c in candidates if c.selected}
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
    if denominator <= _EPS:
        return 0.0
    return numerator / denominator


def query_coverage(
    retained_terms: Iterable[str],
    query_term_idf: dict[str, float],
) -> float:
    """Q_B: IDF-weighted share of ORIGINAL query terms retained.

    Uses the discriminative terms the user actually wrote. Intent expansion
    would let a selection that dropped every rare term still score highly by
    retaining common expansions of it.
    """
    total = sum(max(v, 0.0) for v in query_term_idf.values())
    if total <= _EPS:
        return 1.0
    kept = set(retained_terms)
    covered = sum(
        max(idf, 0.0) for term, idf in query_term_idf.items() if term in kept
    )
    return covered / (total + _EPS)


def certify(
    candidates: Sequence[Candidate],
    *,
    query_term_idf: dict[str, float] | None = None,
    retained_terms: Iterable[str] = (),
    anchor_weights: dict[str, float] | None = None,
    budget_exhausted: bool = True,
    shadow_price_limit: float = 0.0,
) -> SufficiencyCertificate:
    """Produce a fail-closed sufficiency verdict for one selection.

    ``shadow_price_limit`` is the calibrated threshold above which excluded
    evidence is considered materially valuable. It defaults to 0.0, meaning
    *any* positively scored exclusion at an exhausted budget is reported --
    the conservative reading, appropriate until the threshold is calibrated
    against measured failures on a given workload.
    """
    c_b = captured_mass(candidates)
    lambda_b = shadow_price(candidates)
    a_b = cutoff_ambiguity(candidates)
    e_b = boundary_exposure(candidates, anchor_weights)
    q_b = query_coverage(retained_terms, query_term_idf or {})

    reasons: list[str] = []
    # Order matters: the first two are the ones that fired on real failures.
    if budget_exhausted and lambda_b > shadow_price_limit:
        reasons.append(
            f"budget exhausted with excluded evidence still scoring "
            f"{lambda_b:.4f} per token"
        )
    if e_b > 0.0:
        reasons.append(
            f"{e_b:.0%} of anchor weight lost its required neighbouring context"
        )
    # Tolerance, not `< 1.0`: the ratio is divided by ``total + _EPS``, so full
    # coverage lands a hair under 1.0 and a naive comparison reports
    # "degraded: query coverage 100%" on a perfect selection. A certificate
    # that fires on a flawless case teaches its readers to ignore it, which is
    # exactly how a real degradation later gets waved through.
    if q_b < 1.0 - 1e-6:
        reasons.append(f"query coverage {q_b:.0%} of discriminative term weight")
    if a_b >= 0.5:
        reasons.append(
            f"cutoff ambiguity {a_b:.2f}: selection is marginal at this budget"
        )

    verdict = "sufficient" if not reasons else "degraded"
    return SufficiencyCertificate(
        captured_mass=c_b,
        shadow_price=lambda_b,
        cutoff_ambiguity=a_b,
        boundary_exposure=e_b,
        query_coverage=q_b,
        verdict=verdict,
        reasons=tuple(reasons),
    )
