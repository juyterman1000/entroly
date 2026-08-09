from __future__ import annotations

from benchmarks.verified_code_context import run_benchmark, verify


def test_verified_code_context_preregistered_matrix() -> None:
    payload = run_benchmark()
    assert verify(payload)
    assert payload["errors"] == []
