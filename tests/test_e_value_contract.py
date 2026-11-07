"""Finite, deterministic oracles for the e-value statistical contract."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from types import SimpleNamespace

import pytest

from entroly.e_value import (
    CompositeCertificate,
    EmpiricalCDF,
    EValueCalibrator,
    WorstCaseBound,
    build_certificate,
    e_mean,
    e_product,
    e_to_conservative_p,
    manifold_conditional_crc,
)
from entroly.eicv import EICVAnalyzer, EICVCalibration, EICVCertificate


INVALID_NUMBERS = (float("nan"), float("inf"), -float("inf"), True, False)
METADATA = {
    "evidence_method": "fixed-power-p-to-e-arithmetic-mean",
    "statistical_scope": (
        "single-null-test-under-exchangeability-not-answer-correctness"
    ),
    "production_authority": False,
    "e_product_scope": "diagnostic-only-independence-not-established",
}


@pytest.mark.parametrize(
    "population",
    [
        (0.5,) * 20,
        (0.0,) * 8 + (0.5,) * 7 + (1.0,) * 5,
        tuple(i / 19 for i in range(20)),
    ],
    ids=["all-tied", "partially-tied", "distinct-continuous-ranks"],
)
def test_exact_leave_one_out_exchangeable_ranks(population):
    """Every held-out index is equally likely; no simulation or fitted oracle."""
    size = len(population)
    p_values = []
    e_values = []
    for index, score in enumerate(population):
        remaining = population[:index] + population[index + 1 :]
        calibrator = EValueCalibrator.fit(remaining)
        expected_p = Fraction(sum(other >= score for other in population), size)
        actual_p = calibrator.p_value(score)
        actual_e = calibrator.e_value(score)
        assert actual_p == pytest.approx(float(expected_p), abs=1e-15)
        assert actual_e == pytest.approx(0.5 / math.sqrt(float(expected_p)))
        p_values.append(actual_p)
        e_values.append(actual_e)

    for numerator in range(size + 1):
        alpha = Fraction(numerator, size)
        count = sum(p <= float(alpha) + 1e-15 for p in p_values)
        assert Fraction(count, size) <= alpha
    assert math.fsum(e_values) / size <= 1.0


def test_all_nineteen_calibration_ties_are_conservative():
    calibrator = EValueCalibrator.fit([0.5] * 19)
    assert calibrator.p_value(0.5) == 1.0
    assert calibrator.e_value(0.5) == 0.5
    assert EmpiricalCDF([0.5] * 19)(0.5) == 1.0


@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
def test_empty_calibration_is_neutral(score):
    calibrator = EValueCalibrator.fit([])
    assert calibrator.p_value(score) == 1.0
    assert calibrator.e_value(score) == 1.0


def test_cap_and_epsilon_can_only_reduce_fixed_power_evidence():
    cdf = EmpiricalCDF([0.0] * 99)
    assert EValueCalibrator(cdf, max_evalue=2.0).e_value(1.0) == 2.0
    assert EValueCalibrator(cdf, eps=0.25).e_value(1.0) == 1.0
    assert EValueCalibrator(cdf).e_value(1.0) == 5.0


@pytest.mark.parametrize("n_calibration, expected_e", [(99, 5.0), (1599, 20.0)])
def test_correlated_layers_use_mean_for_rejection(n_calibration, expected_e):
    calibrator = EValueCalibrator.fit([0.0] * n_calibration)
    names = ("esg", "nli", "entropy")
    certificate = build_certificate(
        dict.fromkeys(names, 1.0), dict.fromkeys(names, calibrator)
    )
    assert certificate.e_mean == expected_e
    assert certificate.e_product == expected_e**3
    assert certificate.decision == (
        "hallucinated" if expected_e >= 20.0 else "abstain"
    )


def test_single_layer_rejects_at_exact_twenty_boundary():
    calibrator = EValueCalibrator.fit([0.0] * 1599)
    certificate = build_certificate({"score": 1.0}, {"score": calibrator})
    assert certificate.e_mean == 20.0
    assert certificate.decision == "hallucinated"


def test_rounding_cannot_promote_below_threshold_evidence_to_rejection():
    calibrator = EValueCalibrator.fit([0.0] * 1599, max_evalue=19.99999)
    certificate = build_certificate({"score": 1.0}, {"score": calibrator})
    bound = manifold_conditional_crc({"score": 1.0}, {"score": calibrator})
    assert certificate.decision == "abstain"
    assert bound.reject_at_alpha is False
    assert bound.reject(alpha=0.05) is False


def test_small_mean_cannot_certify_support_even_with_legacy_band():
    calibrator = EValueCalibrator.fit([1.0] * 19)
    certificate = build_certificate(
        {"score": 0.0}, {"score": calibrator}, abstain_band=(0.6, 3.0)
    )
    assert certificate.e_mean == 0.5
    assert certificate.decision == "abstain"
    assert build_certificate({}, {}).decision == "abstain"


def test_correlated_manifolds_preserve_bool_property_and_explicit_levels():
    calibrator = EValueCalibrator.fit([0.0] * 99)
    names = ("original", "paraphrase", "perturbation")
    bound = manifold_conditional_crc(
        dict.fromkeys(names, 1.0), dict.fromkeys(names, calibrator)
    )
    assert bound.mean_e == 5.0
    assert bound.joint_e == 125.0
    assert bound.reject_at_alpha is False
    assert bound.reject() is False
    assert bound.reject(alpha=0.2) is True


def test_manifold_mean_rejects_at_exact_threshold():
    calibrator = EValueCalibrator.fit([0.0] * 1599)
    bound = manifold_conditional_crc({"one": 1.0}, {"one": calibrator})
    assert bound.mean_e == 20.0
    assert bound.reject_at_alpha is True
    assert bound.reject(alpha=0.05) is True
    assert bound.reject(alpha=0.01) is False


def test_e_utilities_accept_zero_without_a_positive_product_floor():
    assert e_product([5.0, 0.0, 5.0]) == 0.0
    assert e_product([]) == 1.0
    assert e_product([0.5, 0.5]) == 0.25
    assert e_mean([0.0, 2.0]) == 1.0
    assert e_mean([]) == 1.0
    assert e_to_conservative_p(0.0) == 1.0
    assert e_to_conservative_p(0.5) == 1.0
    assert e_to_conservative_p(20.0) == 0.05


def test_diagnostic_product_survives_extreme_finite_factors():
    assert e_product([1e308, 1e308, 0.0]) == 0.0
    assert e_product([1e308, 1e308, 1e-308]) == pytest.approx(1e308)
    assert e_product([1e-308, 1e-308, 1e308]) == pytest.approx(1e-308, abs=0.0)
    assert math.isfinite(e_product([1e308, 1e308]))


@pytest.mark.parametrize("invalid", INVALID_NUMBERS)
def test_cdf_rejects_nonfinite_and_boolean_calibration(invalid):
    with pytest.raises((TypeError, ValueError)):
        EmpiricalCDF([0.0, invalid])


@pytest.mark.parametrize("invalid", INVALID_NUMBERS)
@pytest.mark.parametrize("method", ["__call__", "conformal_p", "quantile"])
def test_cdf_rejects_nonfinite_and_boolean_queries(invalid, method):
    with pytest.raises((TypeError, ValueError)):
        getattr(EmpiricalCDF([0.0, 1.0]), method)(invalid)


@pytest.mark.parametrize("invalid", INVALID_NUMBERS)
@pytest.mark.parametrize("method", ["p_value", "e_value"])
def test_calibrator_rejects_nonfinite_and_boolean_queries(invalid, method):
    with pytest.raises((TypeError, ValueError)):
        getattr(EValueCalibrator.fit([0.0]), method)(invalid)


@pytest.mark.parametrize("invalid", INVALID_NUMBERS + (0.0, -1.0))
@pytest.mark.parametrize("parameter", ["max_evalue", "eps"])
def test_calibrator_rejects_invalid_parameters(invalid, parameter):
    with pytest.raises((TypeError, ValueError)):
        EValueCalibrator(EmpiricalCDF([0.0]), **{parameter: invalid})


@pytest.mark.parametrize("invalid", INVALID_NUMBERS)
def test_calibrator_rejects_invalid_cdf(invalid):
    with pytest.raises((TypeError, ValueError)):
        EValueCalibrator(cdf=invalid)


@pytest.mark.parametrize("invalid", INVALID_NUMBERS + (-1.0,))
@pytest.mark.parametrize("utility", [e_product, e_mean, e_to_conservative_p])
def test_e_utilities_reject_invalid_evidence(invalid, utility):
    with pytest.raises((TypeError, ValueError)):
        utility(invalid if utility is e_to_conservative_p else [0.0, invalid])


@pytest.mark.parametrize("invalid", INVALID_NUMBERS + (0.0, -0.1, 1.1))
def test_alpha_is_validated_at_every_decision_entry_point(invalid):
    with pytest.raises((TypeError, ValueError)):
        build_certificate({}, {}, alpha=invalid)
    with pytest.raises((TypeError, ValueError)):
        manifold_conditional_crc({}, {}).reject(alpha=invalid)
    with pytest.raises((TypeError, ValueError)):
        EICVAnalyzer(alpha=invalid)


@pytest.mark.parametrize(
    "band", [(float("nan"), 3.0), (0.3, float("inf")), (False, 3.0), (3.0, 0.3)]
)
def test_legacy_abstain_band_is_still_validated(band):
    with pytest.raises((TypeError, ValueError)):
        build_certificate({}, {}, abstain_band=band)


def test_generic_certificate_serializes_statistical_scope():
    payload = build_certificate({}, {}).as_dict()
    for key, expected in METADATA.items():
        assert payload[key] == expected
    assert payload["production_authority"] is False


def test_uncalibrated_phi_rejection_has_no_statistical_authority():
    analyzer = EICVAnalyzer()
    analyzer._esg = SimpleNamespace(score=lambda evidence, claim: SimpleNamespace(
        tension=1.0,
        n_claim_atoms=1,
        n_ev_atoms=1,
        unsupported_fraction=1.0,
        contradiction_fraction=0.0,
    ))
    analyzer._sem = SimpleNamespace(analyze=lambda evidence, claim: SimpleNamespace(
        nli_bidir_score=1.0, h_sem_norm=1.0,
    ))
    certificate = analyzer.verify("evidence", "claim")
    assert certificate.phi == 0.0
    assert certificate.decision == "hallucinated"
    assert certificate.e_mean == 1.0
    assert certificate.conservative_p == 1.0
    assert certificate.as_dict()["production_authority"] is False
    assert certificate.as_dict()["decision_scope"] == (
        "phi-heuristic-plus-null-evidence-not-alpha-controlled"
    )


def test_calibration_existing_schema_round_trip_preserves_corrected_outputs(tmp_path):
    path = tmp_path / "calibration.json"
    bundle = EICVCalibration(
        calibrators={
            "ties": EValueCalibrator.fit([0.5] * 19),
            "ranks": EValueCalibrator.fit([i / 99 for i in range(99)]),
            "capped": EValueCalibrator.fit([0.0] * 99, max_evalue=2.0),
            "empty": EValueCalibrator.fit([]),
        },
        n_calibration=99,
        fit_at="2026-11-02T00:00:00Z",
    )
    bundle.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema"] == "eicv-calibration-v1"
    restored = EICVCalibration.load(path)
    assert restored.n_calibration == 99
    assert restored.fit_at == bundle.fit_at
    assert restored.calibrators["ties"].p_value(0.5) == 1.0
    assert restored.calibrators["ties"].e_value(0.5) == 0.5
    assert restored.calibrators["ranks"].e_value(1.0) == 5.0
    assert restored.calibrators["capped"].e_value(1.0) == 2.0
    assert restored.calibrators["empty"].e_value(1.0) == 1.0
    for name, original in bundle.calibrators.items():
        loaded = restored.calibrators[name]
        for score in (0.0, 0.5, 1.0):
            assert loaded.p_value(score) == original.p_value(score)
            assert loaded.e_value(score) == original.e_value(score)


def test_loading_legacy_score_bundle_uses_corrected_calibration(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({
            "schema": "eicv-calibration-v1",
            "n_calibration": 19,
            "fit_at": "legacy",
            "calibrators": {"esg": {"scores": [0.5] * 19, "max_evalue": 1e6}},
        }),
        encoding="utf-8",
    )
    calibrator = EICVCalibration.load(path).calibrators["esg"]
    assert calibrator.p_value(0.5) == 1.0
    assert calibrator.e_value(0.5) == 0.5


@pytest.mark.parametrize("n_calibration, expected_e", [(99, 5.0), (1599, 20.0)])
def test_real_analyzer_uses_mean_with_deterministic_layer_stubs(
    n_calibration, expected_e
):
    analyzer = EICVAnalyzer()
    analyzer._esg = SimpleNamespace(score=lambda evidence, claim: SimpleNamespace(
        tension=0.0,
        n_claim_atoms=1,
        n_ev_atoms=1,
        unsupported_fraction=0.0,
        contradiction_fraction=0.0,
    ))
    analyzer._sem = SimpleNamespace(analyze=lambda evidence, claim: SimpleNamespace(
        nli_bidir_score=0.0, h_sem_norm=0.0,
    ))
    analyzer.fit_calibrators([("evidence", "claim")] * n_calibration)
    analyzer._esg.score = lambda evidence, claim: SimpleNamespace(
        tension=0.5,
        n_claim_atoms=1,
        n_ev_atoms=1,
        unsupported_fraction=0.0,
        contradiction_fraction=0.0,
    )
    analyzer._sem.analyze = lambda evidence, claim: SimpleNamespace(
        nli_bidir_score=0.5, h_sem_norm=0.5,
    )
    # Phi stays in the abstention zone while all three layers are top-ranked.
    certificate = analyzer.verify("evidence", "claim")
    assert certificate.phi == 0.5
    assert certificate.e_mean == expected_e
    assert certificate.e_product == expected_e**3
    assert certificate.conservative_p == 1.0 / expected_e
    assert certificate.decision == (
        "hallucinated" if expected_e >= 20.0 else "abstain"
    )
    payload = certificate.as_dict()
    for key, expected in METADATA.items():
        assert payload[key] == expected
    assert payload["production_authority"] is False
    assert payload["decision_scope"] == (
        "phi-heuristic-plus-null-evidence-not-alpha-controlled"
    )


@pytest.mark.parametrize(
    "report_type, fields",
    [
        (
            EICVCertificate,
            {
                "phi": 0.51,
                "hallucination_score": 0.49,
                "decision": "abstain",
                "layer_scores": {"esg_tension": 0.49, "custom": 0.123456789},
                "e_product": 125.5,
                "e_mean": 3.125,
                "conservative_p": 0.32,
                "alpha": 0.025,
                "worst_manifold_e": 4.25,
                "worst_manifold_name": "perturbation-a",
                "worst_manifold_p": 0.11764705882352941,
                "n_claim_atoms": 7,
                "n_ev_atoms": 11,
                "unsupported_fraction": 0.125,
                "contradiction_fraction": 0.0625,
                "profile": "qa",
                "schema_version": "eicv-cert-v1",
                "elapsed_ms": 12.75,
            },
        ),
        (
            WorstCaseBound,
            {
                "worst_e": 5.125,
                "joint_e": 25.25,
                "mean_e": 4.875,
                "worst_p": 0.009518143961927425,
                "per_manifold_e": {"original": 4.625, "variant": 5.125},
                "per_manifold_p": {"original": 0.011687, "variant": 0.009518},
            },
        ),
        (
            CompositeCertificate,
            {
                "layers": [],
                "e_product": 125.25,
                "e_mean": 4.999999,
                "decision": "abstain",
                "alpha": 0.025,
            },
        ),
    ],
    ids=["eicv-certificate", "manifold-bound", "composite-certificate"],
)
@pytest.mark.parametrize("spoof_metadata", [False, True])
def test_report_constructor_round_trip_preserves_fields_and_safe_metadata(
    report_type, fields, spoof_metadata
):
    metadata = {
        "evidence_method": "untrusted-method",
        "statistical_scope": "answer-correctness",
        "e_product_scope": "formal-rejection-authority",
        "production_authority": True,
    } if spoof_metadata else {}
    if spoof_metadata and report_type is EICVCertificate:
        metadata["decision_scope"] = "alpha-controlled-answer-correctness"
    report = report_type(**fields, **metadata)
    payload = report.as_dict()
    restored = report_type(**payload)
    for name, expected in fields.items():
        assert getattr(restored, name) == expected
    for name, expected in METADATA.items():
        assert payload[name] == expected
        assert getattr(restored, name) == expected
    assert payload["production_authority"] is False
    assert restored.production_authority is False
    assert restored.as_dict() == payload
    if report_type is EICVCertificate:
        assert payload["decision_scope"] == (
            "phi-heuristic-plus-null-evidence-not-alpha-controlled"
        )


def test_analyzer_near_threshold_nested_evidence_keeps_full_precision():
    analyzer = EICVAnalyzer()
    analyzer._esg = SimpleNamespace(score=lambda evidence, claim: SimpleNamespace(
        tension=0.5,
        n_claim_atoms=1,
        n_ev_atoms=1,
        unsupported_fraction=0.0,
        contradiction_fraction=0.0,
    ))
    analyzer._sem = SimpleNamespace(analyze=lambda evidence, claim: SimpleNamespace(
        nli_bidir_score=0.5, h_sem_norm=0.5,
    ))
    calibrator = EValueCalibrator.fit([0.0] * 1598)
    analyzer.calibration = EICVCalibration(
        calibrators=dict.fromkeys(
            ("esg_tension", "nli_bidir_score", "h_sem_norm"), calibrator
        ),
        n_calibration=1598,
    )
    certificate = analyzer.verify("evidence", "claim")
    expected_e = 0.5 * math.sqrt(1599)
    expected_p = 2.0 / math.sqrt(1599)
    assert certificate.phi == 0.5
    assert certificate.e_mean == pytest.approx(expected_e, rel=1e-14)
    assert certificate.e_mean < 20.0
    assert certificate.conservative_p == pytest.approx(expected_p, rel=1e-14)
    assert certificate.conservative_p > certificate.alpha
    assert certificate.decision == "abstain"
    assert certificate.layer_scores["e_mean"] == certificate.e_mean
    assert certificate.layer_scores["conservative_p"] == certificate.conservative_p
    assert certificate.layer_scores["conservative_p"] > 0.05
    payload = certificate.as_dict()
    assert payload["layer_scores"]["e_mean"] == payload["e_mean"]
    assert payload["layer_scores"]["conservative_p"] == payload["conservative_p"]
    assert payload["layer_scores"]["conservative_p"] > 0.05


@pytest.mark.parametrize("invalid", INVALID_NUMBERS)
def test_unknown_manifold_still_rejects_invalid_score(invalid):
    with pytest.raises(ValueError):
        manifold_conditional_crc({"uncalibrated-manifold": invalid}, {})
