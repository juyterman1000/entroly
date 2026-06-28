from __future__ import annotations

from benchmarks.compression_proxy_scoreboard import run_scoreboard


def test_compression_proxy_scoreboard_passes() -> None:
    report = run_scoreboard()

    assert report.passed
    assert report.mean_savings_ratio >= 0.70
    assert all(s.passed for s in report.scenarios)
    assert all(s.evidence_preserved for s in report.scenarios)
    assert all(s.receipt_emitted for s in report.scenarios)
