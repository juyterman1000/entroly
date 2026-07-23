from __future__ import annotations

import copy
import json
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
        _adapter("headroom", **config),
    ]

    report = resilience.analyze(
        protocol=protocol,
        phase="development",
        adapters=adapters,
    )

    resilience.verify_report(report)
    assert report["claim_gate"]["public_leadership_claim_allowed"] is False
    assert report["aggregates"]["entroly"]["passed"] is True
    assert report["aggregates"]["headroom"]["passed"] is True


def test_holdout_claim_requires_entroly_pass_and_headroom_failure() -> None:
    protocol = resilience._protocol()
    config = resilience._phase_config(protocol, "holdout")
    report = resilience.analyze(
        protocol=protocol,
        phase="holdout",
        adapters=[
            _adapter("entroly", **config),
            _adapter("headroom", **config, missing=1),
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
            _adapter("headroom", **config),
        ],
    )
    tampered = copy.deepcopy(report)
    tampered["aggregates"]["entroly"]["exact_entries"] -= 1

    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        resilience.verify_report(tampered)


def test_committed_holdout_is_current_verified_and_scoped_in_readme() -> None:
    report = json.loads(
        (
            ROOT
            / "benchmarks/results/recovery_resilience_holdout_revalidation_v4.json"
        ).read_text(encoding="utf-8")
    )
    resilience.verify_report(report)
    current_implementation = resilience._canonical_source_sha256(
        (
            Path(resilience.__file__).resolve(),
            ROOT / "entroly/compression_retrieval_store.py",
        )
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        report["participants"]["entroly"]["runtime"]["implementation_sha256"]
        == current_implementation
    )
    assert report["aggregates"]["entroly"]["exact_entries"] == 66
    # v4 is a parity run: both systems satisfy the recovery-integrity gate, so
    # no public leadership claim is permitted. The v3 competitor failure was a
    # transient store lock a clean re-run did not reproduce.
    assert report["aggregates"]["headroom"]["exact_entries"] == 66
    assert report["claim_gate"]["public_leadership_claim_allowed"] is False
    headroom_errors = [
        error["message"]
        for worker in report["adapters"]["headroom"]["worker_runs"]
        for error in worker["errors"]
    ]
    assert headroom_errors == []
    assert "parity, not leadership" in readme
    assert "does not establish universal recovery superiority" in readme
