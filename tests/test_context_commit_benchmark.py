from __future__ import annotations

from benchmarks.context_commit_conformance import run_benchmark


def test_context_commit_conformance_benchmark_falsification_gate():
    result = run_benchmark(cases=3)
    aggregate = result["aggregate"]

    assert aggregate["cases"] >= 3
    assert aggregate["valid_commit_rate"] == 1.0
    assert aggregate["deterministic_replay_rate"] == 1.0
    assert aggregate["exact_context_replay_rate"] == 1.0
    assert aggregate["exact_omission_recovery_rate"] == 1.0
    assert aggregate["tamper_detection_rate"] == 1.0
    assert aggregate["omitted_chunks_verified"] > 0
    assert aggregate["tamper_trials"] >= 18
