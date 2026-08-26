from __future__ import annotations

import copy

import pytest

from benchmarks.work_graph_performance import run_benchmark, verify_report
from entroly.work_graph import WorkGraph, WorkGraphUnavailableError


def _require_native() -> None:
    try:
        WorkGraph("native-probe")
    except WorkGraphUnavailableError as exc:
        pytest.skip(str(exc))


def test_work_graph_performance_gate_covers_real_pyo3_delivery() -> None:
    _require_native()
    report = run_benchmark(files=200, events=40, polls=20)
    verify_report(report)
    assert report["state"]["events"] == 41
    assert report["state"]["passive_event_growth"] == 0
    assert report["state"]["unfinished"] == 1


def test_performance_report_verifier_rejects_forged_pass() -> None:
    report = {
        "schema_version": "entroly.work-graph-performance.v1",
        "passed": True,
        "failures": [],
        "state": {"passive_event_growth": 0},
    }
    forged = copy.deepcopy(report)
    forged["state"]["passive_event_growth"] = 1
    with pytest.raises(ValueError, match="amplified"):
        verify_report(forged)
