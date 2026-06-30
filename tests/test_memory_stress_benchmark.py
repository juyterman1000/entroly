from __future__ import annotations

from benchmarks.memory_stress_test import run_benchmark


def test_memory_stress_benchmark_passes() -> None:
    report = run_benchmark()

    assert report.passed
    assert report.score == 1.0
    assert all(s.passed for s in report.scenarios)
