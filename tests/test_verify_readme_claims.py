"""Regression tests for scripts/verify_readme_claims.py.

The gate must fail on evidence that cannot carry a headline, and must not fire
on evidence that is honestly labelled — a gate that cries wolf gets disabled.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_readme_claims.py"
SPEC = importlib.util.spec_from_file_location("verify_readme_claims", SCRIPT)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


# ── citation scanning ────────────────────────────────────────────────────────


def test_citations_only_returns_hits_inside_the_first_screen():
    lines = [
        "intro",
        "see benchmarks/results/alpha.json",
        *["filler"] * 20,
        "see benchmarks/results/beta.json",
    ]
    found = gate.citations(lines, limit=5)
    assert found == [(2, "benchmarks/results/alpha.json")]


def test_citations_finds_multiple_artifacts_on_one_line():
    lines = ["a benchmarks/results/one.json and benchmarks/results/two.json"]
    assert len(gate.citations(lines, limit=10)) == 2


# ── experimental labelling ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "marker",
    ["experimental", "not statistically conclusive", "pilot", "preliminary"],
)
def test_nearby_experimental_label_is_detected(marker):
    lines = ["filler"] * 5 + [f"this result is {marker}"] + ["cite"] * 3
    assert gate.has_nearby_label(lines, line_number=9)


def test_label_far_away_does_not_count():
    lines = ["experimental"] + ["filler"] * 60 + ["cite"]
    assert not gate.has_nearby_label(lines, line_number=61)


def test_absent_label_is_not_invented():
    lines = ["a plain confident claim with no caveat"] * 5
    assert not gate.has_nearby_label(lines, line_number=3)


# ── benchmark → module mapping ───────────────────────────────────────────────


def test_modules_measured_by_finds_both_import_forms(tmp_path):
    bench = tmp_path / "b.py"
    bench.write_text(
        "import entroly.alpha\n"
        "from entroly.beta import Thing\n"
        "from entroly.sub.gamma import (A, B)\n"
        "import os\n",
        encoding="utf-8",
    )
    assert gate.modules_measured_by(bench) == {
        "entroly.alpha", "entroly.beta", "entroly.sub.gamma",
    }


def test_benchmark_module_for_returns_none_when_absent(tmp_path):
    assert gate.benchmark_module_for(tmp_path / "no_such_benchmark.json") is None


def test_portable_text_sha256_is_independent_of_checkout_newlines(tmp_path):
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")
    assert gate.portable_text_sha256(lf) == gate.portable_text_sha256(crlf)


def test_evidence_metadata_requires_pinned_reproducible_checks(tmp_path):
    artifact = tmp_path / "benchmarks" / "results" / "demo.json"
    benchmark = tmp_path / "benchmarks" / "demo.py"
    artifact.parent.mkdir(parents=True)
    benchmark.write_text("# deterministic benchmark\n", encoding="utf-8")
    payload = {
        "headline_eligible": True,
        "claim_scope": "One deterministic fixture.",
        "sample_size": {"cases": 1},
        "reproduction_command": (
            "python -m benchmarks.demo verify benchmarks/results/demo.json"
        ),
        "implementation": {"commit": "a" * 40},
        "limitations": ["One fixture only."],
        "benchmark_module": "benchmarks/demo.py",
        "harness_sha256": gate.portable_text_sha256(benchmark),
    }
    encoded = (json.dumps(payload) + "\n").encode()
    artifact.write_bytes(encoded)
    artifact.with_suffix(".json.sha256").write_text(
        hashlib.sha256(encoded).hexdigest() + "  demo.json\n",
        encoding="ascii",
    )

    original_root = gate.REPO_ROOT
    gate.REPO_ROOT = tmp_path
    try:
        assert gate.evidence_metadata_failures(
            "benchmarks/results/demo.json", artifact, payload
        ) == []
        artifact.write_text('{"tampered": true}\n', encoding="utf-8")
        assert "missing or mismatched .sha256 sidecar" in gate.evidence_metadata_failures(
            "benchmarks/results/demo.json", artifact, payload
        )
    finally:
        gate.REPO_ROOT = original_root


# ── the real repository ──────────────────────────────────────────────────────


def test_headline_ineligible_artifact_is_still_flagged_when_unlabelled(tmp_path):
    """Gate 1: an ineligible artifact above the fold with no caveat must fail."""
    artifact_dir = tmp_path / "benchmarks" / "results"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "demo.json").write_text(
        json.dumps({"headline_eligible": False, "calibration": {"reason": "gate not met"}}),
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Product\n\nBest in class: benchmarks/results/demo.json\n", encoding="utf-8"
    )

    original_root = gate.REPO_ROOT
    gate.REPO_ROOT = tmp_path
    try:
        lines = readme.read_text(encoding="utf-8").splitlines()
        cited = gate.citations(lines, limit=130)
        assert cited, "artifact should have been cited"
        line_number, relative = cited[0]
        payload = json.loads((tmp_path / relative).read_text(encoding="utf-8"))
        assert payload["headline_eligible"] is False
        assert not gate.has_nearby_label(lines, line_number)
    finally:
        gate.REPO_ROOT = original_root


def test_gate_passes_on_the_current_readme():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=SCRIPT.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "README claim gate OK" in completed.stdout
