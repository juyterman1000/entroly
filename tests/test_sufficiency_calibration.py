from __future__ import annotations

import pytest

from entroly.sufficiency_calibration import (
    CalibrationObservation,
    certify_with_profile,
    evaluate_holdout,
    fit_profile,
    risk_coverage_curve,
    validate_profile,
)


def observations(
    n: int,
    failures: set[int] | None = None,
    *,
    split: str = "calibration",
    prefix: str = "sample",
):
    failures = failures or set()
    return [
        CalibrationObservation(
            sample_id=f"{prefix}-{i}",
            dataset=f"dataset-{i % 2}",
            model=f"model-{i % 2}",
            split=split,
            residual_risk=i / 10_000,
            failed=i in failures,
        )
        for i in range(n)
    ]


def safe_metrics(residual: float = 0.0):
    return {
        "source_span_integrity": True,
        "boundary_exposure": 0.0,
        "query_coverage": 1.0,
        "residual_risk": residual,
    }


def test_small_profile_cannot_promote_semantic_scope() -> None:
    profile = fit_profile(
        observations(20),
        min_samples=100,
        min_accepted=10,
    )
    assert not profile.calibration_ready
    assert not profile.semantic_validated
    certificate = certify_with_profile(
        safe_metrics(),
        profile,
    )
    assert certificate.verdict.value == "uncertain"
    assert certificate.scope.value == "candidate_units"


def test_calibration_alone_never_promotes_semantic_scope() -> None:
    profile = fit_profile(
        observations(200),
        target_failure_rate=0.05,
        min_samples=100,
        min_accepted=100,
    )
    assert profile.calibration_ready
    assert not profile.semantic_validated
    certificate = certify_with_profile(
        safe_metrics(profile.threshold),
        profile,
    )
    assert certificate.verdict.value == "uncertain"
    assert certificate.scope.value == "candidate_units"


def test_disjoint_holdout_promotes_only_below_threshold() -> None:
    profile = fit_profile(
        observations(300),
        target_failure_rate=0.05,
        min_samples=200,
        min_accepted=150,
    )
    report = evaluate_holdout(
        profile,
        observations(
            300,
            split="holdout",
            prefix="holdout",
        ),
        min_samples=200,
        min_accepted=100,
    )
    assert report.validated
    validated = validate_profile(profile, report)
    assert validated.semantic_validated

    good = certify_with_profile(
        safe_metrics(validated.threshold),
        validated,
    )
    bad = certify_with_profile(
        safe_metrics(validated.threshold + 1.0),
        validated,
    )
    assert good.verdict.value == "sufficient"
    assert good.scope.value == "semantic"
    assert bad.verdict.value == "uncertain"


def test_holdout_overlap_fails_closed() -> None:
    calibration = observations(250)
    profile = fit_profile(
        calibration,
        target_failure_rate=0.05,
        min_samples=200,
        min_accepted=100,
    )
    overlapping = [
        CalibrationObservation(
            sample_id=item.sample_id,
            dataset=item.dataset,
            model=item.model,
            split="holdout",
            residual_risk=item.residual_risk,
            failed=False,
        )
        for item in calibration
    ]
    report = evaluate_holdout(
        profile,
        overlapping,
        min_samples=200,
        min_accepted=100,
    )
    assert not report.disjoint
    assert not report.validated
    assert "overlaps" in " ".join(report.reasons)


def test_duplicate_dataset_sample_ids_are_rejected() -> None:
    item = observations(1)[0]
    with pytest.raises(ValueError, match="must be unique"):
        fit_profile(
            [item, item],
            min_samples=1,
            min_accepted=1,
        )


def test_equal_risk_group_is_never_split_by_sample_order() -> None:
    base = [
        CalibrationObservation(
            sample_id=f"base-{i}",
            dataset=f"dataset-{i % 2}",
            model=f"model-{i % 2}",
            split="calibration",
            residual_risk=0.1,
            failed=False,
        )
        for i in range(200)
    ]
    tied = [
        CalibrationObservation(
            sample_id=f"tied-{i:02d}",
            dataset=f"dataset-{i % 2}",
            model=f"model-{i % 2}",
            split="calibration",
            residual_risk=0.2,
            failed=i < 10,
        )
        for i in range(20)
    ]
    profile = fit_profile(
        base + tied,
        target_failure_rate=0.03,
        min_samples=200,
        min_accepted=100,
    )
    assert profile.calibration_ready
    assert profile.threshold == 0.1
    assert profile.accepted_samples == 200
    points = risk_coverage_curve(base + tied)
    assert [
        point["threshold"]
        for point in points
    ] == [0.1, 0.2]


def test_holdout_report_cannot_validate_a_different_profile() -> None:
    first = fit_profile(
        observations(250, prefix="first"),
        target_failure_rate=0.05,
        min_samples=200,
        min_accepted=100,
    )
    second = fit_profile(
        observations(250, prefix="second"),
        target_failure_rate=0.05,
        min_samples=200,
        min_accepted=100,
    )
    report = evaluate_holdout(
        first,
        observations(
            250,
            split="holdout",
            prefix="holdout",
        ),
        min_samples=200,
        min_accepted=100,
    )
    with pytest.raises(
        ValueError,
        match="different calibration profile",
    ):
        validate_profile(second, report)
