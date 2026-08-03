from __future__ import annotations

import pytest

from entroly.sufficiency import (
    CalibrationPolicy,
    Candidate,
    _query_terms,
    captured_mass,
    certify,
    certify_selection,
    cutoff_ambiguity,
    query_coverage,
    shadow_price,
)


def _policy(**overrides) -> CalibrationPolicy:
    values = {
        "calibration_id": "heldout-repo-v1",
        "residual_risk_limit": 0.02,
        "minimum_query_coverage": 0.5,
        "maximum_boundary_exposure": 0.0,
    }
    values.update(overrides)
    return CalibrationPolicy(**values)


def _needle() -> list[Candidate]:
    return [
        Candidate(
            "hit",
            utility=9.0,
            cost=56,
            selected=True,
            anchors=("passphrase",),
            neighbourhood=("hit",),
        ),
        *[
            Candidate(f"f{index}", utility=0.0, cost=200, selected=False)
            for index in range(50)
        ],
    ]


def _squad_failure() -> list[Candidate]:
    return [
        Candidate(
            "anchor",
            utility=3.0,
            cost=90,
            selected=True,
            anchors=("rhine",),
            neighbourhood=("anchor", "answer_span"),
        ),
        Candidate("answer_span", utility=2.6, cost=60, selected=False),
        Candidate("ctx", utility=2.4, cost=55, selected=False),
    ]


def test_clean_selection_is_uncalibrated_by_default() -> None:
    result = certify(
        _needle(),
        query_term_idf={"passphrase": 6.0},
        retained_terms=["passphrase"],
        budget_exhausted=True,
    )
    assert result.verdict == "uncalibrated"
    assert not result.sufficient
    assert not result.calibrated
    assert result.calibration_id is None


def test_legacy_calibrated_boolean_cannot_authorize_sufficiency() -> None:
    result = certify(
        _needle(),
        query_term_idf={"passphrase": 6.0},
        retained_terms=["passphrase"],
        budget_exhausted=True,
        calibrated=True,
    )
    assert result.verdict == "uncalibrated"
    assert not result.sufficient
    assert any("CalibrationPolicy" in reason for reason in result.reasons)


def test_named_calibration_policy_can_certify_clean_selection() -> None:
    result = certify(
        _needle(),
        query_term_idf={"passphrase": 6.0},
        retained_terms=["passphrase"],
        budget_exhausted=True,
        calibration=_policy(),
    )
    assert result.verdict == "sufficient"
    assert result.sufficient
    assert result.calibrated
    assert result.calibration_id == "heldout-repo-v1"


def test_calibration_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="calibration_id"):
        _policy(calibration_id="")
    with pytest.raises(ValueError, match="residual_risk_limit"):
        _policy(residual_risk_limit=-1)
    with pytest.raises(ValueError, match="minimum_query_coverage"):
        _policy(minimum_query_coverage=1.1)


def test_severed_answer_span_certifies_degraded() -> None:
    result = certify(
        _squad_failure(),
        query_term_idf={"rhine": 4.0, "bridge": 5.2},
        retained_terms=["rhine"],
        budget_exhausted=True,
        calibration=_policy(),
    )
    assert result.verdict == "degraded"
    assert not result.sufficient
    assert len(result.reasons) >= 2


def test_unmeasured_boundary_cannot_pass_strict_policy() -> None:
    result = certify(
        _needle(),
        query_term_idf={"passphrase": 6.0},
        retained_terms=["passphrase"],
        budget_exhausted=False,
        boundary_exposure_measured=False,
        calibration=_policy(require_boundary_measurement=True),
    )
    assert result.verdict == "degraded"
    assert not result.sufficient
    assert not result.boundary_exposure_measured


def test_full_query_coverage_is_not_reported_as_a_shortfall() -> None:
    assert query_coverage(["a", "b"], {"a": 3.0, "b": 4.0}) >= 1.0 - 1e-6
    result = certify(
        _needle(),
        query_term_idf={"passphrase": 6.0},
        retained_terms=["passphrase"],
        budget_exhausted=True,
        calibration=_policy(),
    )
    assert all("coverage" not in reason for reason in result.reasons)


def test_shadow_price_is_zero_when_budget_did_not_bind() -> None:
    complete = [Candidate("a", 5.0, 10, True), Candidate("b", 4.0, 10, True)]
    truncated = [Candidate("a", 5.0, 10, True), Candidate("b", 4.0, 10, False)]
    assert shadow_price(complete) == 0.0
    assert shadow_price(truncated) > 0.0


def test_shadow_price_ignores_worthless_exclusions() -> None:
    candidates = [
        Candidate("a", 5.0, 10, True),
        Candidate("junk", 0.0, 500, False),
    ]
    assert shadow_price(candidates) == 0.0


def test_captured_mass_reflects_utility_not_token_count() -> None:
    assert captured_mass(_needle()) >= 0.99
    assert captured_mass(_squad_failure()) < 0.5


def test_cutoff_ambiguity_rises_as_margin_shrinks() -> None:
    wide = [Candidate("in", 10.0, 10, True), Candidate("out", 0.1, 10, False)]
    narrow = [Candidate("in", 5.0, 10, True), Candidate("out", 4.99, 10, False)]
    assert cutoff_ambiguity(narrow) > cutoff_ambiguity(wide)


def test_query_coverage_uses_weights_not_counts() -> None:
    weights = {"the": 0.1, "rhine": 4.0, "bridge": 5.2}
    assert query_coverage(["bridge"], weights) > query_coverage(["the"], weights) * 4


def test_certificate_serialises_calibration_and_measurement_state() -> None:
    payload = certify(
        _squad_failure(),
        query_term_idf={"rhine": 4.0},
        retained_terms=[],
        budget_exhausted=True,
        boundary_exposure_measured=False,
    ).to_dict()
    for key in (
        "captured_mass",
        "shadow_price",
        "cutoff_ambiguity",
        "boundary_exposure",
        "boundary_exposure_measured",
        "query_coverage",
        "corpus_gap",
        "verdict",
        "calibrated",
        "calibration_id",
        "reasons",
    ):
        assert key in payload
    assert payload["verdict"] == "degraded"
    assert not payload["calibrated"]


def test_absent_term_does_not_reduce_attainable_coverage_but_forces_expansion():
    certificate = certify(
        [
            Candidate(
                unit_id="auth.py",
                utility=1.0,
                cost=24,
                selected=True,
                anchors=("session", "token", "login"),
            )
        ],
        query_term_idf={
            "session": 1.5404,
            "token": 1.5404,
            "login": 1.5404,
            "issued": 2.6391,
        },
        retained_terms={"session", "token", "login"},
        unattainable_terms={"issued"},
        budget_exhausted=False,
        calibration=_policy(),
    )
    assert certificate.query_coverage == pytest.approx(1.0)
    assert certificate.verdict == "expand_required"
    assert not certificate.sufficient


def test_any_corpus_gap_is_expand_required() -> None:
    certificate = certify(
        [Candidate("answer", 1.0, 20, True, anchors=("card",))],
        query_term_idf={"card": 1.0, "charged": 2.0},
        retained_terms={"card"},
        unattainable_terms={"charged"},
        budget_exhausted=False,
        calibration=_policy(),
    )
    assert certificate.corpus_gap == pytest.approx(0.5)
    assert certificate.verdict == "expand_required"


def test_unanswerable_selection_never_reports_sufficient() -> None:
    certificate = certify(
        [Candidate("noise.py", 0.5, 30, True)],
        query_term_idf={"timeouterror": 2.64, "retry": 2.64},
        retained_terms=set(),
        unattainable_terms={"timeouterror", "retry"},
        budget_exhausted=False,
        calibration=_policy(),
    )
    assert certificate.verdict == "expand_required"
    assert not certificate.sufficient


def test_question_words_are_not_treated_as_evidence() -> None:
    terms = set(_query_terms("how is the session token issued after login"))
    assert {"session", "token", "issued", "login"}.issubset(terms)
    assert {"how", "the", "after"}.isdisjoint(terms)


def test_external_selection_adapter_refuses_invented_residual_state() -> None:
    with pytest.raises(RuntimeError, match="cannot reconstruct optimizer residual state"):
        certify_selection([], [], "query", token_budget=100)
