from __future__ import annotations

import pytest

from scripts.onboarding_self_dogfood import _validate_local_report


def _report() -> dict:
    return {
        "queries": [
            {
                "baseline_tokens": 100,
                "selected_tokens": 25,
                "tokens_saved": 75,
            }
        ],
        "files_indexed": 1,
        "repo_tokens_indexed": 100,
        "baseline_tokens_per_query": 100,
        "total_tokens_saved": 75,
        "average_reduction_pct": 75.0,
        "latency_ms": {"min": 1.0, "p95": 2.0, "max": 3.0},
    }


def test_local_report_accepts_structured_latency_contract() -> None:
    _validate_local_report(_report(), label="simulate")


@pytest.mark.parametrize(
    ("latency", "message"),
    [
        (1.0, "must be an object"),
        ({"min": 1.0, "max": 3.0}, "missing required fields"),
        ({"min": 1.0, "p95": -1.0, "max": 3.0}, "must be finite and non-negative"),
        ({"min": 2.0, "p95": 3.0, "max": 1.0}, "min <= p95 <= max"),
    ],
)
def test_local_report_rejects_invalid_latency_contract(
    latency: object, message: str
) -> None:
    report = _report()
    report["latency_ms"] = latency

    with pytest.raises(AssertionError, match=message):
        _validate_local_report(report, label="simulate")
