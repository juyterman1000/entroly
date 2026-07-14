from __future__ import annotations

import copy

import pytest

from benchmarks.neural_evidence_frontier import (
    SCHEMA_VERSION,
    _calibrate_override,
    _gated_metrics,
    _guarded_metrics,
    _mcnemar_exact,
    _metrics,
    _wilson_upper,
    verify_report,
)


def _row(
    index: int,
    *,
    lexical_top: int,
    neural_top: int,
    gold: int,
    margin: float,
    partition: str = "calibration",
) -> dict:
    lexical_order = [lexical_top, 1 - lexical_top]
    neural_order = [neural_top, 1 - neural_top]
    return {
        "trial_index": index,
        "gold_index": gold,
        "lexical_order": lexical_order,
        "neural_order": neural_order,
        "lexical_top": lexical_top,
        "neural_top": neural_top,
        "lexical_correct": lexical_top == gold,
        "neural_correct": neural_top == gold,
        "neural_margin": margin,
        "partition": partition,
        "gated_top": lexical_top,
        "gated_correct": lexical_top == gold,
    }


def test_wilson_bound_is_conservative_and_shrinks_with_evidence() -> None:
    assert _wilson_upper(0, 10) > 0.20
    assert _wilson_upper(0, 100) < 0.04
    assert _wilson_upper(10, 100) > 0.10


def test_calibration_requires_enough_low_risk_noninferior_overrides() -> None:
    clean = [
        _row(index, lexical_top=0, neural_top=1, gold=1, margin=0.2)
        for index in range(50)
    ]

    calibration = _calibrate_override(
        clean, max_error_upper=0.10, minimum_overrides=40
    )

    assert calibration["eligible"] is True
    assert calibration["threshold"] == pytest.approx(0.2)
    assert calibration["override_errors"] == 0

    risky = [
        _row(
            index,
            lexical_top=0,
            neural_top=1,
            gold=0 if index < 10 else 1,
            margin=0.2,
        )
        for index in range(50)
    ]
    rejected = _calibrate_override(
        risky, max_error_upper=0.10, minimum_overrides=40
    )
    assert rejected["eligible"] is False


def test_mcnemar_exact_is_paired_and_symmetric() -> None:
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(12, 1) == _mcnemar_exact(1, 12)
    assert _mcnemar_exact(12, 1) < 0.05


def test_verifier_recomputes_orders_metrics_and_headline_gate() -> None:
    rows = [
        _row(
            0,
            lexical_top=0,
            neural_top=1,
            gold=1,
            margin=0.4,
            partition="test",
        ),
        _row(
            1,
            lexical_top=1,
            neural_top=1,
            gold=1,
            margin=0.3,
            partition="test",
        ),
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {"candidates_per_trial": 2},
        "test_metrics": {
            "lexical_bm25": _metrics(rows, "lexical"),
            "local_transformer": _metrics(rows, "neural"),
            "risk_gated": _gated_metrics(rows),
            "dual_channel_guard": _guarded_metrics(
                rows, candidates_per_trial=2
            ),
            "paired": {
                "transformer_only_correct": 1,
                "lexical_only_correct": 0,
                "mcnemar_exact_p": 1.0,
            },
        },
        "headline_eligible": False,
        "trials": rows,
    }

    verify_report(report)

    tampered = copy.deepcopy(report)
    tampered["test_metrics"]["local_transformer"]["top1_correct"] = 0
    with pytest.raises(ValueError, match="metrics"):
        verify_report(tampered)
