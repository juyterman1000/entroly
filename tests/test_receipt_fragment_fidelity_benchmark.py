"""Regression tests for benchmarks/receipt_fragment_fidelity.py.

The measurement must be immutable: it describes entroly at one pinned commit,
so nothing added to the repository afterwards — documentation, tests, or the
benchmark's own result artifacts — may move a published number.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from benchmarks import receipt_fragment_fidelity as fidelity


def test_baseline_ref_is_a_pinned_full_sha():
    """A branch name or short sha could silently repoint the corpus."""
    assert len(fidelity.BASELINE_REF) == 40
    assert all(c in "0123456789abcdef" for c in fidelity.BASELINE_REF)


def test_full_suite_ci_jobs_fetch_the_pinned_baseline():
    """Jobs running this test suite must not use checkout's shallow default."""
    workflow = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")
    for job in ("integration", "python-fallback"):
        match = re.search(
            rf"(?ms)^  {re.escape(job)}:\n.*?(?=^  [A-Za-z0-9_-]+:\n|\Z)",
            workflow,
        )
        assert match, f"missing {job} CI job"
        assert "fetch-depth: 0" in match.group(), (
            f"{job} cannot resolve pinned evidence baseline {fidelity.BASELINE_REF}"
        )


def test_corpus_is_read_from_the_baseline_not_the_working_tree():
    """The benchmark's own artifacts postdate the baseline, so they cannot appear."""
    included, _ = fidelity.build_corpus()
    paths = {str(record["path"]) for record in included}

    for self_artifact in (
        "benchmarks/results/receipt_fragment_fidelity_prefix.json",
        "benchmarks/results/receipt_fragment_fidelity_sdk_prefix.json",
        "benchmarks/receipt_fragment_fidelity.py",
        "tests/test_receipt_fragment_fidelity_benchmark.py",
    ):
        assert self_artifact not in paths, f"{self_artifact} leaked into its own corpus"


def test_blob_reader_returns_content_matching_recorded_hashes():
    included, _ = fidelity.build_corpus()
    sample = [str(r["path"]) for r in included[:25]]
    blobs = fidelity.read_baseline_blobs(fidelity.BASELINE_REF, sample)

    assert set(blobs) == set(sample)
    by_path = {str(r["path"]): r for r in included}
    for path, raw in blobs.items():
        assert hashlib.sha256(raw).hexdigest() == by_path[path]["sha256"]


def test_baseline_blobs_are_lf_normalised():
    """Git stores LF, so results are identical on a CRLF (Windows) checkout."""
    blobs = fidelity.read_baseline_blobs(fidelity.BASELINE_REF, ["entroly/esg.py"])
    raw = blobs["entroly/esg.py"]
    assert raw.count(b"\r\n") == 0
    assert raw.count(b"\n") > 0


def test_read_baseline_blobs_handles_empty_request():
    assert fidelity.read_baseline_blobs(fidelity.BASELINE_REF, []) == {}


def test_excluded_files_all_carry_a_reason():
    _, excluded = fidelity.build_corpus()
    for record in excluded:
        assert record["reason"]
        assert record["path"]


def test_measure_file_counts_are_internally_consistent():
    text = "def f():\n    # a comment\n    return 1\n"
    counts = fidelity.measure_file("fixture.py", text.encode("utf-8"), text)
    assert counts["fragments"] >= 1
    assert 0 <= counts["verbatim"] <= counts["fragments"]
    assert 0 <= counts["byte_span_exact"] <= counts["fragments"]
    assert counts["source_digest_valid"] == counts["fragments"]
    assert counts["fragment_digest_valid"] == counts["fragments"]


def test_public_sdk_probe_uses_the_exact_receipt_contract():
    report = fidelity.sdk_probe()
    totals = report["totals"]
    recovered = totals["recovered_fragments"]

    assert recovered > 0
    assert report["headline_eligible"] is True
    assert totals["source_digest_valid"] == recovered
    assert totals["fragment_digest_valid"] == recovered
    assert totals["recovered_bytes_match_source_span"] == recovered
    assert totals["recovery_verified_exact"] == recovered


def test_rate_handles_zero_denominator():
    assert fidelity._rate(0, 0) == 0.0
    assert fidelity._rate(5, 10) == 0.5


def test_public_artifact_writer_emits_a_matching_checksum(tmp_path):
    target = tmp_path / "result.json"
    fidelity.write_artifact(target, {"schema_version": "test.v1", "value": 1})

    assert fidelity.artifact_checksum_matches(target)
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    assert fidelity.artifact_checksum_matches(target)
    target.write_text('{"tampered":true}\n', encoding="utf-8")
    assert not fidelity.artifact_checksum_matches(target)
