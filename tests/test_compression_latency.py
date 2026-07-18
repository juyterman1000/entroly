from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("tiktoken")

from benchmarks import compression_latency as latency


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "benchmarks/results/compression_latency_development.json"
HOLDOUT = ROOT / "benchmarks/results/compression_latency_holdout.json"


def _development_report() -> dict[str, object]:
    return json.loads(DEVELOPMENT.read_text(encoding="utf-8"))


def _holdout_report() -> dict[str, object]:
    return json.loads(HOLDOUT.read_text(encoding="utf-8"))


def test_protocol_freezes_quality_gates_and_separate_latency_modes() -> None:
    protocol = latency._protocol()

    assert protocol["participants"]["entroly"]["version"] == "1.0.59"
    assert protocol["participants"]["headroom"]["version"] == "0.31.0"
    assert protocol["quality_gates"]["evidence_recall"] == 1.0
    assert protocol["quality_gates"]["maximum_token_inflation"] == 0.0
    assert protocol["claim_policy"]["aggregate_product_score_allowed"] is False
    assert protocol["claim_policy"]["failures_remain_in_sample"] is True
    assert protocol["claim_policy"]["universal_superiority_claim_allowed"] is False
    assert set(protocol["definitions"]) == {
        "warm_call_latency",
        "cold_latency",
        "overall_speedup",
        "bootstrap",
    }


def test_development_artifact_verifies_but_cannot_enable_public_claim() -> None:
    report = _development_report()

    latency.verify_report(report)

    assert report["quality_gates"] == {"entroly": True, "headroom": True}
    assert report["claim_gate"]["any_latency_claim_allowed"] is False
    assert report["mode_analysis"]["warm"]["entroly_faster_claim_allowed"] is False
    assert report["mode_analysis"]["cold"]["entroly_faster_claim_allowed"] is False


def test_development_artifact_preserves_historical_implementation_identity() -> None:
    report = _development_report()
    implementation = report["participants"]["entroly"]["runtime"][
        "implementation_sha256"
    ]

    # Published evidence is a snapshot of the measured implementation, not a
    # claim about every later source revision. Its sealed payload and raw-cell
    # participant metadata remain verifiable after unrelated benchmark tooling
    # changes; new runs bind themselves to current source before publication.
    assert len(implementation) == 64
    assert int(implementation, 16) >= 0
    assert report["participants"]["entroly"]["version"] == report["protocol"][
        "participants"
    ]["entroly"]["version"]


def test_verifier_rejects_raw_latency_tampering() -> None:
    report = _development_report()
    tampered = copy.deepcopy(report)
    tampered["raw"]["warm_cells"][0]["call_latency_ms"][0] = 0.000001

    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        latency.verify_report(tampered)


def test_missing_participant_metadata_is_retained_as_failed_gate() -> None:
    report = _development_report()
    warm = copy.deepcopy(report["raw"]["warm_cells"])
    cold = copy.deepcopy(report["raw"]["cold_cells"])
    for cell in [*warm, *cold]:
        if cell["system"] == "headroom":
            cell.pop("participant", None)

    analyzed = latency.analyze(
        protocol=report["protocol"],
        phase="development",
        warm_cells=warm,
        cold_cells=cold,
    )

    assert analyzed["participant_gates"]["headroom"]["passed"] is False
    assert analyzed["quality_gates"]["headroom"] is False
    assert analyzed["claim_gate"]["any_latency_claim_allowed"] is False


def test_duplicate_cold_replicate_fails_closed() -> None:
    report = _development_report()
    warm = copy.deepcopy(report["raw"]["warm_cells"])
    cold = copy.deepcopy(report["raw"]["cold_cells"])
    target_system = cold[0]["system"]
    target_scenario = cold[0]["scenario_id"]
    matching = [
        cell
        for cell in cold
        if cell["system"] == target_system
        and cell["scenario_id"] == target_scenario
    ]
    matching[1]["replicate"] = matching[0]["replicate"]

    with pytest.raises(ValueError, match="replicate ids"):
        latency.analyze(
            protocol=report["protocol"],
            phase="development",
            warm_cells=warm,
            cold_cells=cold,
        )


def test_holdout_verifies_historical_source_and_passes_scoped_claims() -> None:
    report = _holdout_report()

    latency.verify_report(report)

    implementation = report["participants"]["entroly"]["runtime"][
        "implementation_sha256"
    ]
    assert len(implementation) == 64
    assert int(implementation, 16) >= 0
    assert report["participants"]["entroly"]["version"] == report["protocol"][
        "participants"
    ]["entroly"]["version"]
    assert report["quality_gates"] == {"entroly": True, "headroom": True}
    assert report["mode_analysis"]["warm"]["ci95_lower"] > 1.0
    assert report["mode_analysis"]["cold"]["ci95_lower"] > 1.0
    assert report["mode_analysis"]["warm"]["entroly_faster_claim_allowed"] is True
    assert report["mode_analysis"]["cold"]["entroly_faster_claim_allowed"] is True
    assert report["claim_gate"]["aggregate_product_score_allowed"] is False
    assert report["claim_gate"]["universal_superiority_claim_allowed"] is False
