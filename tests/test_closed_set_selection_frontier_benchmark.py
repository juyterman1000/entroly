from __future__ import annotations

import copy

import pytest

from benchmarks.closed_set_selection_frontier import run_matrix, verify_report


def test_small_matrix_proves_optimality_and_detects_rank_order_gap():
    report = run_matrix(max_nodes=5)
    control = report["rank_order_atomic_control"]
    production = report["certified_closed_set_selector"]

    assert report["matrix"]["cases"] > 0
    assert control["suboptimal_cases"] > 0
    assert production["strict_improvements_vs_rank_order"] > 0
    assert production["regressions_vs_rank_order"] == 0
    assert production["optimal_cases"] == report["matrix"]["cases"]
    assert production["aggregate_regret"] == 0.0
    assert production["budget_violations"] == 0
    assert production["partial_dependency_closures"] == 0
    assert production["invalid_certificates"] == 0


def test_report_verifier_rejects_a_claimed_regression():
    report = run_matrix(max_nodes=2)
    tampered = copy.deepcopy(report)
    tampered["certified_closed_set_selector"]["regressions_vs_rank_order"] = 1

    with pytest.raises(ValueError, match="regressed against rank order"):
        verify_report(tampered)
