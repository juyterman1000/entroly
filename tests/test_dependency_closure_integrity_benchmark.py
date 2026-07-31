from __future__ import annotations

import copy

import pytest

from benchmarks.dependency_closure_integrity import run_matrix, verify_report


def test_small_matrix_proves_atomic_closure_and_detects_legacy_defect():
    report = run_matrix(max_nodes=4)

    assert report["matrix"]["cases"] > 0
    assert report["legacy_partial_add_control"]["partial_closure_cases"] > 0
    assert report["legacy_partial_add_control"]["budget_violations"] == 0
    assert report["atomic_dependency_closure"]["partial_closure_cases"] == 0
    assert report["atomic_dependency_closure"]["budget_violations"] == 0
    assert report["atomic_dependency_closure"]["closure_integrity_pass"] is True
    assert report["atomic_dependency_closure"]["hard_budget_pass"] is True


def test_report_verifier_rejects_a_partial_atomic_closure():
    report = run_matrix(max_nodes=2)
    tampered = copy.deepcopy(report)
    tampered["atomic_dependency_closure"]["partial_closure_cases"] = 1

    with pytest.raises(ValueError, match="partial dependency closure"):
        verify_report(tampered)
