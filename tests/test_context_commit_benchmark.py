from __future__ import annotations

import json
from pathlib import Path

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


def test_committed_context_commit_claims_match_readme():
    root = Path(__file__).resolve().parents[1]
    result = json.loads(
        (root / "benchmarks/results/context_commit_conformance.json").read_text(
            encoding="utf-8"
        )
    )
    readme = (root / "README.md").read_text(encoding="utf-8")
    aggregate = result["aggregate"]

    assert f"**{aggregate['cases']} / {aggregate['cases']}**" in readme
    recovered = aggregate["omitted_chunks_verified"]
    assert f"**{recovered} / {recovered}**" in readme
    tampered = aggregate["tamper_trials"]
    assert f"**{tampered} / {tampered}**" in readme
