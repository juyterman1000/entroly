"""The certificate must separate measured successes from measured failures.

Calibrated against the two shapes in benchmarks/results:

    needle  10,807.6 input -> 56.4 delivered at a 2,000 budget, retention 1.0
            One localised span; the budget was never binding.

    squad     232.9 input ->  130.8 delivered at a   100 budget, retention 0.90
            Four regressions, all on *answerable* questions, one at only 8%
            compression. The model answered around the answer: anchor kept,
            answer-bearing neighbour cut.

A certificate that cannot tell those apart is worthless, and one that flags the
success is worse than worthless -- it teaches its readers to ignore it.
"""

from __future__ import annotations

import pytest

from entroly.sufficiency import (
    Candidate,
    captured_mass,
    certify,
    cutoff_ambiguity,
    query_coverage,
    shadow_price,
)


def _needle() -> list[Candidate]:
    """One dominant span among filler; budget not binding."""
    hit = Candidate("hit", utility=9.0, cost=56, selected=True,
                    anchors=("passphrase",), neighbourhood=("hit",))
    filler = [Candidate(f"f{i}", utility=0.0, cost=200, selected=False)
              for i in range(50)]
    return [hit, *filler]


def _squad_failure() -> list[Candidate]:
    """Anchor retained, answer-bearing neighbour excluded at a tight budget."""
    return [
        Candidate("anchor", utility=3.0, cost=90, selected=True,
                  anchors=("rhine",), neighbourhood=("anchor", "answer_span")),
        Candidate("answer_span", utility=2.6, cost=60, selected=False),
        Candidate("ctx", utility=2.4, cost=55, selected=False),
    ]


def test_localised_answer_certifies_sufficient() -> None:
    result = certify(
        _needle(),
        query_term_idf={"passphrase": 6.0},
        retained_terms=["passphrase"],
        budget_exhausted=True,
    )
    assert result.sufficient, f"needle should certify sufficient: {result.reasons}"
    assert result.reasons == (), "a flawless selection must raise nothing"


def test_severed_answer_span_certifies_degraded() -> None:
    result = certify(
        _squad_failure(),
        query_term_idf={"rhine": 4.0, "bridge": 5.2},
        retained_terms=["rhine"],
        budget_exhausted=True,
    )
    assert not result.sufficient
    assert len(result.reasons) >= 2, "the failure shape should trip several signals"


def test_full_query_coverage_is_not_reported_as_a_shortfall() -> None:
    """Guards a real epsilon bug.

    Q_B divides by ``total + _EPS``, so perfect coverage lands a hair under
    1.0. A naive ``q_b < 1.0`` reported 'degraded: query coverage 100%' on a
    flawless selection -- a certificate that fires on a perfect case is how a
    real degradation later gets ignored.
    """
    assert query_coverage(["a", "b"], {"a": 3.0, "b": 4.0}) >= 1.0 - 1e-6
    result = certify(
        _needle(),
        query_term_idf={"passphrase": 6.0},
        retained_terms=["passphrase"],
        budget_exhausted=True,
    )
    assert all("coverage" not in r for r in result.reasons)


# ── The individual signals ──────────────────────────────────────────────────


def test_shadow_price_is_zero_when_the_budget_did_not_bind() -> None:
    """λ_B distinguishes 'complete' from merely 'affordable'.

    Nothing excluded means the optimizer wanted nothing more, which is the
    needle case: 56 tokens of a 2,000 budget at retention 1.0.
    """
    complete = [Candidate("a", 5.0, 10, True), Candidate("b", 4.0, 10, True)]
    assert shadow_price(complete) == 0.0

    truncated = [Candidate("a", 5.0, 10, True), Candidate("b", 4.0, 10, False)]
    assert shadow_price(truncated) > 0.0


def test_shadow_price_ignores_worthless_exclusions() -> None:
    """Excluding zero-utility filler is not unmet demand."""
    filler_only = [Candidate("a", 5.0, 10, True),
                   Candidate("junk", 0.0, 500, False)]
    assert shadow_price(filler_only) == 0.0


def test_captured_mass_reflects_retained_utility_not_token_count() -> None:
    """A tiny output can still capture all the evidence."""
    candidates = _needle()
    assert captured_mass(candidates) >= 0.99
    assert captured_mass(_squad_failure()) < 0.5


def test_cutoff_ambiguity_rises_as_the_margin_shrinks() -> None:
    wide = [Candidate("in", 10.0, 10, True), Candidate("out", 0.1, 10, False)]
    narrow = [Candidate("in", 5.00, 10, True), Candidate("out", 4.99, 10, False)]
    assert cutoff_ambiguity(narrow) > cutoff_ambiguity(wide)


def test_query_coverage_uses_weights_not_counts() -> None:
    """Dropping the one discriminative term must dominate keeping common ones."""
    idf = {"the": 0.1, "rhine": 4.0, "bridge": 5.2}
    kept_common = query_coverage(["the"], idf)
    kept_rare = query_coverage(["bridge"], idf)
    assert kept_rare > kept_common * 4


def test_certificate_serialises_for_a_receipt() -> None:
    result = certify(_squad_failure(), query_term_idf={"rhine": 4.0},
                     retained_terms=[], budget_exhausted=True)
    payload = result.to_dict()
    for key in ("captured_mass", "shadow_price", "cutoff_ambiguity",
                "boundary_exposure", "query_coverage", "verdict", "reasons"):
        assert key in payload
    assert payload["verdict"] == "degraded"
    assert payload["reasons"], "a degraded verdict must say why"


# ── Corpus gap: "selection dropped it" vs "it was never retrieved" ──────────
#
# query_coverage counted query terms that appear in NO candidate. Smoothed IDF
# gives a term with df=0 the HIGHEST weight, so ordinary question words
# dominated the denominator while being unattainable by construction.
#
# Measured on the auth fixture of benchmarks/sufficiency_baseline.py, query
# "how is the session token issued after login": how/the/issued/after each in 0
# documents at idf 2.6391; session/token/login each in 1 document at idf 1.5404.
# Coverage capped at 0.3045 for a selection that WAS the complete answer, with
# captured_mass 1.0 and shadow_price 0.0. All 42 harness rows returned
# "degraded" on that alone -- a 100% false-insufficient rate.


def test_absent_terms_do_not_cap_coverage_of_a_complete_selection():
    from entroly.sufficiency import certify, Candidate

    cert = certify(
        [Candidate(unit_id="auth.py", utility=1.0, cost=24, selected=True,
                   anchors=("session", "token", "login"))],
        query_term_idf={
            "session": 1.5404, "token": 1.5404, "login": 1.5404,
            "issued": 2.6391,
        },
        retained_terms={"session", "token", "login"},
        unattainable_terms={"issued"},
        budget_exhausted=False,
    )
    assert cert.query_coverage == pytest.approx(1.0), (
        "a selection retaining every attainable query term must score full "
        f"coverage; got {cert.query_coverage}"
    )
    assert cert.verdict == "sufficient", cert.reasons


def test_evidence_absent_from_corpus_is_expand_required_not_degraded():
    """The two states call for opposite actions and must not be conflated."""
    from entroly.sufficiency import certify, Candidate

    cert = certify(
        [Candidate(unit_id="distractor.py", utility=0.4, cost=20, selected=True,
                   anchors=())],
        query_term_idf={"stripegateway": 2.64, "charge": 2.64, "card": 2.64},
        retained_terms=set(),
        unattainable_terms={"stripegateway", "charge", "card"},
        budget_exhausted=False,
    )
    assert cert.corpus_gap == pytest.approx(1.0)
    assert cert.verdict == "expand_required", (
        "when no discriminative term is in any candidate, selecting differently "
        f"cannot help; got {cert.verdict} / {cert.reasons}"
    )


def test_unanswerable_selection_never_reports_sufficient():
    """Fail-closed: dropping absent terms from coverage must not fail open."""
    from entroly.sufficiency import certify, Candidate

    cert = certify(
        [Candidate(unit_id="noise.py", utility=0.5, cost=30, selected=True)],
        query_term_idf={"timeouterror": 2.64, "retry": 2.64},
        retained_terms=set(),
        unattainable_terms={"timeouterror", "retry"},
        budget_exhausted=False,
    )
    assert not cert.sufficient, (
        "the answer is in no candidate; 'sufficient' here is the one verdict "
        f"that must never appear. got {cert.verdict}"
    )


def test_question_words_are_not_treated_as_evidence():
    from entroly.sufficiency import _query_terms

    terms = set(_query_terms("how is the session token issued after login"))
    assert "session" in terms and "token" in terms and "login" in terms
    for noise in ("how", "the", "after"):
        assert noise not in terms, (
            f"{noise!r} carries no retrievable evidence and is absent from most "
            "code corpora, where df=0 makes it maximally weighted"
        )
