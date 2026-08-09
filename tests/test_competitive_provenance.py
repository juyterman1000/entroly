from __future__ import annotations

from benchmarks.competitive_provenance import validate_manifest


GOOD_SHA = "a" * 64


def _valid() -> dict:
    return {
        "schema_version": "entroly.competitive-run.v1",
        "subjects": [
            {"label": "candidate", "version": "1.2.3", "commit": "a" * 40, "executable_sha256": GOOD_SHA, "status": "valid"},
            {"label": "baseline-a", "version": "4.5.6", "commit": "b" * 40, "executable_sha256": "b" * 64, "status": "valid"},
        ],
        "workload_sha256": "c" * 64,
        "raw_results_sha256": "d" * 64,
        "verdict": "supported",
    }


def test_complete_pinned_manifest_is_publishable() -> None:
    assert validate_manifest(_valid()) == []


def test_missing_exact_identity_is_rejected() -> None:
    payload = _valid()
    payload["subjects"][1]["commit"] = "main"
    payload["subjects"][1]["executable_sha256"] = ""
    failures = validate_manifest(payload)
    assert any("full 40-character" in failure for failure in failures)
    assert any("executable_sha256" in failure for failure in failures)


def test_void_arm_forbids_directional_verdict() -> None:
    payload = _valid()
    payload["subjects"][1]["status"] = "void"
    assert any("directional verdict" in failure for failure in validate_manifest(payload))
    payload["verdict"] = "void"
    assert validate_manifest(payload) == []
