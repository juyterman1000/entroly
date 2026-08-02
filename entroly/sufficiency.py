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

# Residual demand per unit captured, above which a selection reads degraded.
# Geometric mean of the measured gap between the worst HIT (0.0060) and the
# best MISS (0.0246). Calibrated on n=4 -- provisional, not authoritative.
RESIDUAL_RISK_LIMIT = 0.0121

# Share of discriminative query-term IDF weight that may be absent from every
# candidate before the selection is reported as `expand_required` rather than
# merely degraded.
#
# Set at 0.5 -- a majority of what the question discriminates on is missing from
# the corpus. This is a judgement, not a fit: the measured cases it must
# separate are unambiguous at either end (0.0 when every content term is
# present, 1.0 when the answer document was withheld entirely), and nothing in
# between has been measured. Re-fit before treating it as authoritative.
CORPUS_GAP_LIMIT = 0.5


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
    corpus_gap: float
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
            "corpus_gap": round(self.corpus_gap, 4),
            "residual_risk": round(
                self.shadow_price / max(self.captured_mass, _EPS), 6
            ),
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
    unattainable_terms: Iterable[str] = (),
) -> float:
    """Q_B: IDF-weighted share of ATTAINABLE query terms retained.

    Uses the discriminative terms the user actually wrote. Intent expansion
    would let a selection that dropped every rare term still score highly by
    retaining common expansions of it.

    Terms absent from every candidate are excluded from the denominator: no
    selection can retain what the corpus does not contain. Counting them made
    this metric unusable, because smoothed IDF gives a term in ZERO documents
    the HIGHEST weight, so ordinary question words dominated the denominator
    while being unattainable by construction.

    Measured on the auth fixture in benchmarks/sufficiency_baseline.py, query
    "how is the session token issued after login":

        term      idf     documents containing it
        how      2.6391   0
        the      2.6391   0
        issued   2.6391   0
        after    2.6391   0
        session  1.5404   1
        token    1.5404   1
        login    1.5404   1

    Coverage capped at 4.6212 / 15.1776 = 0.3045 for a selection that was the
    complete answer and excluded nothing (captured_mass 1.0, shadow_price 0.0,
    boundary_exposure 0.0). All 42 rows came back "degraded" on that basis.

    Absence is not discarded -- it is a different signal, reported by
    `corpus_gap`. Dropping it here without reporting it there would turn a
    fail-closed certificate fail-open on the case that matters most: a query
    whose answer is in no candidate at all.
    """
    unattainable = set(unattainable_terms)
    attainable = {
        t: max(v, 0.0) for t, v in query_term_idf.items() if t not in unattainable
    }
    total = sum(attainable.values())
    if total <= _EPS:
        # Nothing the query asked for exists in the corpus. Coverage is
        # undefined rather than perfect; `corpus_gap` carries the verdict.
        return 0.0
    kept = set(retained_terms)
    covered = sum(idf for term, idf in attainable.items() if term in kept)
    return covered / (total + _EPS)


def corpus_gap(
    query_term_idf: dict[str, float],
    unattainable_terms: Iterable[str] = (),
) -> float:
    """G_B: IDF-weighted share of discriminative query terms in NO candidate.

    Separates two failures that one coverage number conflates:

      * evidence exists and selection dropped it -> low `query_coverage`
      * evidence is in no candidate at all       -> high `corpus_gap`

    Only the second is fixable by retrieving more; selecting harder cannot
    recover a term that was never a candidate. That distinction is what makes
    `expand_required` a different verdict from `degraded`.

    Callers must filter question words before computing the inputs. "how",
    "the" and "after" are absent from most code corpora and would otherwise
    report a total evidence gap for every natural-language query.

    Deliberately UNWEIGHTED, unlike every other signal here. Weighting by IDF
    computed over this corpus is circular: a term absent from every candidate
    has df=0, which is exactly the input that makes smoothed IDF maximal, so
    absent terms are weighted highest *because* they are absent. Measured on
    the billing fixture, query "how is a credit card charged through the
    payment gateway":

        term      idf     df
        credit   2.6391   0
        charged  2.6391   0
        payment  2.6391   0
        card     1.5404   1
        gateway  1.5404   1

    IDF-weighted gap 0.7741 against an unweighted 0.6667 -- the weighting adds
    11 points of apparent gap purely from the absences it is measuring. An
    honest weighting needs document frequencies from a background corpus, which
    is not available here, so this counts terms.
    """
    if not query_term_idf:
        return 0.0
    unattainable = set(unattainable_terms) & set(query_term_idf)
    return len(unattainable) / len(query_term_idf)


def certify(
    candidates: Sequence[Candidate],
    *,
    query_term_idf: dict[str, float] | None = None,
    retained_terms: Iterable[str] = (),
    unattainable_terms: Iterable[str] = (),
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
    q_b = query_coverage(retained_terms, query_term_idf or {}, unattainable_terms)
    g_b = corpus_gap(query_term_idf or {}, unattainable_terms)

    # Residual demand per unit of captured evidence.
    #
    # Neither lambda_B nor C_B alone gives a usable threshold: lambda_B is on
    # the scorer's scale, C_B on a 0-1 scale, so any absolute cut on either is
    # workload-specific. Their ratio is dimensionally right -- evidence the
    # budget could not afford, per unit it did capture -- and separates
    # measured outcomes better than either alone.
    #
    # Measured on this repository against known retrieval outcomes:
    #
    #     truth   C_B     lambda_B   ratio
    #     HIT     0.230   0.00076    0.0033
    #     HIT     0.242   0.00145    0.0060
    #     MISS    0.147   0.00362    0.0246
    #     MISS    0.101   0.02819    0.2790
    #
    # A 4x gap between the worst HIT and the best MISS. RESIDUAL_RISK_LIMIT
    # sits at the geometric mean of that gap. Calibrated on n=4, which is far
    # too few to be authoritative: a defensible starting point, not a validated
    # constant. Any real workload should re-fit it.
    residual_risk = lambda_b / max(c_b, _EPS)

    reasons: list[str] = []
    if budget_exhausted and residual_risk > RESIDUAL_RISK_LIMIT:
        reasons.append(
            f"residual demand {residual_risk:.4f} per unit captured "
            f"(lambda={lambda_b:.5f}, captured={c_b:.3f})"
        )
    # The remaining signals are reported but do not by themselves condemn a
    # selection. On this workload each was constant -- E_B and A_B at 0.00, Q_B
    # at 1.00 across every case, hit and miss alike -- so letting them vote
    # reproduces the failure this replaces: every verdict "degraded", which is
    # indistinguishable from having no verdict at all.
    if e_b > 0.0:
        reasons.append(
            f"{e_b:.0%} of anchor weight lost its required neighbouring context"
        )
    if q_b < 0.5:
        reasons.append(f"query coverage {q_b:.0%}: discriminative terms dropped")

    # ── Verdict ─────────────────────────────────────────────────────────
    #
    # `expand_required` is a distinct state, not a flavour of `degraded`,
    # because the two call for opposite actions. Degraded means the evidence
    # was a candidate and selection dropped it: a smaller budget or better
    # ranking can fix it. expand_required means the discriminative terms are in
    # NO candidate, so selecting differently cannot recover them -- only
    # retrieving more can. Reporting both as "degraded" told a caller to tune
    # the thing that was not broken.
    #
    # Ordered most severe first, and expansion outranks selection quality: if
    # the evidence is not present, how well the present evidence was chosen is
    # not the finding worth surfacing.
    if g_b >= CORPUS_GAP_LIMIT:
        verdict = "expand_required"
        reasons.insert(
            0,
            f"{g_b:.0%} of discriminative query-term weight appears in no "
            f"candidate: the evidence was never retrieved",
        )
    elif reasons:
        verdict = "degraded"
    else:
        verdict = "sufficient"

    return SufficiencyCertificate(
        captured_mass=c_b,
        shadow_price=lambda_b,
        cutoff_ambiguity=a_b,
        boundary_exposure=e_b,
        query_coverage=q_b,
        corpus_gap=g_b,
        verdict=verdict,
        reasons=tuple(reasons),
    )


# ── Adapter: real selections, not synthetic shapes ──────────────────────────

# Question words carry no retrievable evidence, and code corpora almost never
# contain them. Left in, they are scored as maximally rare (df = 0 gives the
# highest smoothed IDF) and dominate both coverage and the corpus gap, so every
# natural-language query reports a near-total evidence gap regardless of what
# was selected. Length alone does not exclude them: "how", "the", "what",
# "does" and "after" all clear the len > 2 filter.
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


def _query_terms(query: str) -> list[str]:
    """Discriminative terms as the user wrote them, identifier-split.

    Deliberately not the intent-expanded vocabulary -- see ``query_coverage``.
    Identifiers are split so a prose query can match a symbol: the measured
    failure mode is "chunk oversized source files" never matching
    `chunk_oversized_source`.
    """
    import re

    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", query)
    return [
        t
        for t in re.split(r"[^A-Za-z0-9]+", split.lower())
        if len(t) > 2 and t not in _QUESTION_WORDS
    ]


def stem(word: str) -> str | None:
    """Reduce an English surface form to a matching stem, or None if already base.

    THIRD pinned copy of this rule. The others are
    `entroly_engine::bm25::morphological_stem` and
    `entroly_qccr::morphological_stem`; the three exist because entroly-engine
    and entroly-qccr cannot depend on each other (entroly-engine's Cargo.toml
    records why: it would force a regex-full/regex-lite choice on every
    consumer, WASM included) and no binding exposes either to Python. All three
    are pinned to identical behaviour by tests asserting the same cases.

    Needed here because attainability is decided in Python by substring test.
    Without it "charged" never meets "charge_card" and "retrying" never meets
    "retry_request", so present evidence is reported as absent from the corpus.
    """
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


def attainable(term: str, corpus_lowered: Sequence[str]) -> bool:
    """True when `term` (or its stem) occurs in any candidate."""
    if any(term in text for text in corpus_lowered):
        return True
    s = stem(term)
    return bool(s) and any(s in text for text in corpus_lowered)


def _idf(term: str, corpus_texts: Sequence[str]) -> float:
    """Smoothed IDF over the candidate corpus. Rare terms dominate coverage."""
    n = len(corpus_texts)
    if n == 0:
        return 1.0
    df = sum(1 for text in corpus_texts if term in text.lower())
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


def candidates_from_selection(
    all_fragments: Sequence[dict[str, Any]],
    selected_fragments: Sequence[dict[str, Any]],
    query: str,
    *,
    neighbourhood_radius: int = 1,
) -> list[Candidate]:
    """Build certificate inputs from an actual entroly selection.

    Anchors are the query's discriminative terms that a retained fragment
    contains. The neighbourhood is the adjacent chunks of the same logical
    source within ``neighbourhood_radius`` -- so an anchor whose neighbouring
    chunk was dropped registers as boundary exposure, which is exactly the
    severed-span failure this was built for.
    """
    from .qccr import logical_source

    def _key(fragment: dict[str, Any]) -> str:
        return str(fragment.get("id") or fragment.get("source") or id(fragment))

    selected_keys = {_key(f) for f in selected_fragments}
    terms = set(_query_terms(query))

    # Order fragments within each logical source so "adjacent" is meaningful.
    by_source: dict[str, list[dict[str, Any]]] = {}
    for fragment in all_fragments:
        by_source.setdefault(
            logical_source(str(fragment.get("source") or "")), []
        ).append(fragment)

    position: dict[str, tuple[str, int]] = {}
    for source, group in by_source.items():
        for index, fragment in enumerate(group):
            position[_key(fragment)] = (source, index)

    candidates: list[Candidate] = []
    for fragment in all_fragments:
        key = _key(fragment)
        content = str(fragment.get("content") or "")
        lowered = content.lower()

        utility = fragment.get("relevance")
        if not isinstance(utility, (int, float)):
            utility = fragment.get("relevance_score")
        if not isinstance(utility, (int, float)):
            utility = 0.0

        cost = fragment.get("token_count")
        if not isinstance(cost, (int, float)) or cost <= 0:
            cost = max(1, len(content) // 4)

        anchors = tuple(sorted(t for t in terms if t in lowered))

        source, index = position.get(key, ("", 0))
        group = by_source.get(source, [])
        neighbourhood = tuple(
            _key(group[j])
            for j in range(
                max(0, index - neighbourhood_radius),
                min(len(group), index + neighbourhood_radius + 1),
            )
        )

        candidates.append(
            Candidate(
                unit_id=key,
                utility=float(utility),
                cost=int(cost),
                selected=key in selected_keys,
                anchors=anchors,
                neighbourhood=neighbourhood,
            )
        )
    return candidates


# NOT USABLE YET -- see the note on `certify_selection` below.


def certify_selection(
    all_fragments: Sequence[dict[str, Any]],
    selected_fragments: Sequence[dict[str, Any]],
    query: str,
    *,
    token_budget: int,
    shadow_price_limit: float | None = None,
) -> SufficiencyCertificate:
    """Certify a real entroly selection.

    .. warning::
       **Not yet usable against the live engine.** Measured on this repository,
       every signal reads 0.000 for real selections, for two reasons that are
       properties of the engine rather than of this code:

       1. ``export_fragments()`` carries ``entropy_score``, ``recency_score``
          and ``frequency_score`` but **no per-candidate relevance**. Utility
          is computed inside the optimizer and discarded before anything is
          observable, so ``u_i`` is unavailable for candidates that were not
          selected -- and those are precisely the ones λ_B is about.
       2. The identifier namespaces do not intersect. Selected fragments are
          QCCR synthetics (``qccr::file:entroly/cli.py``); index fragments
          carry ``fragment_id`` with ``id`` unset. Membership of S_B therefore
          cannot be recovered by matching.

       The lesson is structural: an optimizer-derived certificate cannot be
       derived from outside the optimizer. Reconstructing residual state from
       the optimizer's inputs and outputs does not work, because the residue
       -- the utilities of the candidates it rejected -- is never emitted.

       Making this real requires the engine to emit, per candidate considered:
       ``(unit_id, utility, cost, selected)``. That is a small addition to the
       Rust selection path and is the actual prerequisite for this module.
       Until then the maths and the calibration hold only on the synthetic
       shapes in tests/test_sufficiency_certificate.py, which were fitted to
       measured needle and squad behaviour but are not the live path.
    """
    candidates = candidates_from_selection(all_fragments, selected_fragments, query)
    corpus = [str(f.get("content") or "") for f in all_fragments]
    terms = _query_terms(query)
    query_term_idf = {t: _idf(t, corpus) for t in terms}

    retained = {
        term
        for candidate in candidates
        if candidate.selected
        for term in candidate.anchors
    }

    delivered = sum(c.cost for c in candidates if c.selected)
    budget_exhausted = delivered >= int(token_budget) * 0.95

    return certify(
        candidates,
        query_term_idf=query_term_idf,
        retained_terms=retained,
        budget_exhausted=budget_exhausted,
        shadow_price_limit=(
            calibrated_shadow_price_limit(candidates)
            if shadow_price_limit is None
            else shadow_price_limit
        ),
    )


def calibrated_shadow_price_limit(candidates: Sequence[Candidate]) -> float:
    """Scale-free threshold for "materially valuable evidence was excluded".

    A fixed absolute limit cannot work: densities depend on the scorer's scale,
    which differs per corpus and per query. Anchoring on the *selected*
    distribution makes the threshold self-calibrating -- excluded evidence
    matters when it rivals what was actually kept.

    Set at the median selected density: an excluded unit denser than the
    typical retained one is, by the optimizer's own ranking, evidence the
    budget could not afford. Below that, the exclusion is the tail the
    optimizer was right to drop.
    """
    selected = [c.density for c in candidates if c.selected]
    if not selected:
        return 0.0
    return _median(selected)
