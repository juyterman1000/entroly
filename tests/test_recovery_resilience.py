from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from benchmarks import recovery_resilience as resilience


ROOT = Path(__file__).resolve().parents[1]


def _adapter(
    system: str,
    *,
    workers: int,
    entries_per_worker: int,
    seed: int,
    missing: int = 0,
) -> dict[str, object]:
    expected = workers * entries_per_worker
    rows = [
        {
            "worker_id": index // entries_per_worker,
            "entry_index": index % entries_per_worker,
            "payload_sha256": f"payload-{index}",
            "recovered_sha256": f"payload-{index}",
            "exact": True,
            "store_latency_ms": 1.0,
            "retrieve_latency_ms": 0.5,
            "retrieval_error": None,
        }
        for index in range(expected - missing)
    ]
    return {
        "system": system,
        "participant": {
            "package": system,
            "version": "1.0.59" if system == "entroly" else "0.31.0",
        },
        "configuration": {
            "workers": workers,
            "entries_per_worker": entries_per_worker,
            "seed": seed,
        },
        "worker_runs": [
            {"worker_id": index, "exit_code": 0, "errors": []}
            for index in range(workers)
        ],
        "recovery_open_error": None,
        "rows": rows,
        "state_files": [{"name": "state", "bytes": 100}],
    }


def test_protocol_freezes_broad_evidence_program_and_forbids_aggregate_claim() -> None:
    protocol = resilience._protocol()
    dimension_ids = {item["id"] for item in protocol["dimensions"]}

    assert protocol["claim_policy"]["aggregate_score_allowed"] is False
    assert protocol["claim_policy"]["failures_remain_in_sample"] is True
    assert protocol["claim_policy"]["negative_results_are_published"] is True
    assert {
        "active_context_quality",
        "recovery_resilience",
        "end_to_end_model_recovery",
        "compression_latency",
        "provider_protocol_conformance",
        "interruption_recovery",
        "security_and_secret_handling",
        "packaging_and_first_run",
        "operator_ux_and_diagnostics",
        "provider_observed_cost",
    } <= dimension_ids


def test_verified_development_report_never_allows_public_claim() -> None:
    protocol = resilience._protocol()
    config = resilience._phase_config(protocol, "development")
    adapters = [
        _adapter("entroly", **config),
        _adapter("external_adapter", **config),
    ]

    report = resilience.analyze(
        protocol=protocol,
        phase="development",
        adapters=adapters,
    )

    resilience.verify_report(report)
    assert report["claim_gate"]["public_leadership_claim_allowed"] is False
    assert report["aggregates"]["entroly"]["passed"] is True
    assert report["aggregates"]["external_adapter"]["passed"] is True


def test_holdout_claim_requires_entroly_pass_and_external_adapter_failure() -> None:
    protocol = resilience._protocol()
    config = resilience._phase_config(protocol, "holdout")
    report = resilience.analyze(
        protocol=protocol,
        phase="holdout",
        adapters=[
            _adapter("entroly", **config),
            _adapter("external_adapter", **config, missing=1),
        ],
    )

    assert report["claim_gate"]["public_leadership_claim_allowed"] is True
    assert report["claim_gate"]["universal_superiority_claim_allowed"] is False


def test_verifier_rejects_payload_tampering() -> None:
    protocol = resilience._protocol()
    config = resilience._phase_config(protocol, "development")
    report = resilience.analyze(
        protocol=protocol,
        phase="development",
        adapters=[
            _adapter("entroly", **config),
            _adapter("external_adapter", **config),
        ],
    )
    tampered = copy.deepcopy(report)
    tampered["aggregates"]["entroly"]["exact_entries"] -= 1

    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        resilience.verify_report(tampered)


def test_adapter_preserves_virtualenv_launcher_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    try:
        launcher.symlink_to(Path(sys.executable))
    except OSError:
        pytest.skip("platform does not permit symlink creation")

    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0
        stdout = json.dumps({"system": "external_adapter"})
        stderr = ""

    def fake_run(command: list[str], **kwargs: object) -> _Completed:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(resilience.subprocess, "run", fake_run)

    result = resilience._invoke_adapter(
        str(launcher),
        "external_adapter",
        {"workers": 1, "entries_per_worker": 1, "seed": 7},
        timeout=1.0,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[0] == str(launcher.absolute())
    assert command[0] != str(launcher.resolve())
    assert result["system"] == "external_adapter"


def test_committed_holdout_is_current_verified_and_scoped_in_evidence_policy() -> None:
    report = json.loads(
        (
            ROOT
            / "benchmarks/results/recovery_resilience_holdout_revalidation_v5.json"
        ).read_text(encoding="utf-8")
    )
    resilience.verify_report(report)
    current_implementation = resilience._canonical_source_sha256(
        (
            Path(resilience.__file__).resolve(),
            ROOT / "entroly/compression_retrieval_store.py",
        )
    )
    evidence = (ROOT / "docs/public-evidence.md").read_text(encoding="utf-8")

    assert (
        report["participants"]["entroly"]["runtime"]["implementation_sha256"]
        == current_implementation
    )
    assert report["aggregates"]["entroly"]["exact_entries"] == 66
    # v4 is a parity run: both systems satisfy the recovery-integrity gate, so
    # no public leadership claim is permitted. The v3 competitor failure was a
    # transient store lock a clean re-run did not reproduce.
    assert report["aggregates"]["external_adapter"]["exact_entries"] == 66
    assert report["claim_gate"]["public_leadership_claim_allowed"] is False
    external_adapter_errors = [
        error["message"]
        for worker in report["adapters"]["external_adapter"]["worker_runs"]
        for error in worker["errors"]
    ]
    assert external_adapter_errors == []
    assert "**66/66** exact entries for Entroly" in evidence
    assert "**66/66** for the External Baseline A 0.31.0 comparison" in evidence
    assert "parity, not leadership" in evidence
    assert "does not establish universal recovery superiority" in evidence
