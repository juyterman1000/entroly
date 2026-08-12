from __future__ import annotations

import copy
import hashlib
import json

import pytest

from benchmarks.closed_set_selection_frontier import (
    DEFAULT_OUTPUT,
    run_matrix,
    verify_report,
)


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


def test_committed_report_is_self_verifying_and_checksum_bound():
    raw = DEFAULT_OUTPUT.read_bytes()
    report = json.loads(raw)

    verify_report(report)

    sidecar = DEFAULT_OUTPUT.with_suffix(DEFAULT_OUTPUT.suffix + ".sha256")
    assert (
        sidecar.read_text(encoding="ascii").strip()
        == hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    )
    assert report["matrix"]["cases"] == 47862
    assert report["rank_order_atomic_control"]["suboptimal_cases"] == 378
    assert (
        report["certified_closed_set_selector"][
            "strict_improvements_vs_rank_order"
        ]
        == 378
    )
    assert report["certified_closed_set_selector"]["regressions_vs_rank_order"] == 0
    assert report["certified_closed_set_selector"]["optimal_cases"] == 47862
    assert report["certified_closed_set_selector"]["aggregate_regret"] == 0.0
    assert report["certified_closed_set_selector"]["invalid_certificates"] == 0
