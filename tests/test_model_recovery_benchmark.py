from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.model_recovery import (
    _canonical_sha256,
    _limitations,
    build_fixtures,
    verify_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return json.loads(
        (ROOT / "benchmarks" / "model_recovery_protocol_v5.json").read_text(
            encoding="utf-8"
        )
    )


def test_fixture_generation_is_deterministic_and_phase_separated() -> None:
    protocol = _protocol()
    development = build_fixtures(protocol, "development")
    repeated = build_fixtures(protocol, "development")
    holdout = build_fixtures(protocol, "holdout")

    assert development == repeated
    assert len(development) == protocol["phases"]["development"]["trials"]
    assert len(holdout) == protocol["phases"]["holdout"]["trials"]
    assert {row["content_sha256"] for row in development}.isdisjoint(
        {row["content_sha256"] for row in holdout}
    )
    assert all(
        row["expected_answer"] in row["content"] for row in development + holdout
    )


def test_limitations_name_the_actual_model() -> None:
    limitations = _limitations("qwen2.5:1.5b")

    assert limitations[0].startswith("Local qwen2.5:1.5b short-answer")
    assert "7B/32K" not in " ".join(limitations)


def test_verifier_rejects_payload_tampering() -> None:
    report = json.loads(
        (
            ROOT / "benchmarks" / "results" / "model_recovery_v7_development.json"
        ).read_text(encoding="utf-8")
    )
    verify_report(report)

    tampered = json.loads(json.dumps(report))
    tampered["rows"][0]["active_response"]["prediction"] = "forged"
    tampered["payload_sha256"] = _canonical_sha256(
        {key: value for key, value in tampered.items() if key != "payload_sha256"}
    )
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        verify_report(tampered)

    canonical_tamper = json.loads(json.dumps(report))
    canonical_tamper["rows"][0]["active_response"]["canonical_prediction"] = (
        "forged-canonical"
    )
    canonical_tamper["payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in canonical_tamper.items()
            if key != "payload_sha256"
        }
    )
    with pytest.raises(ValueError, match="canonical prediction mismatch"):
        verify_report(canonical_tamper)

    limitation_tamper = json.loads(json.dumps(report))
    limitation_tamper["limitations"][0] = "Local 7B/32K stale label."
    limitation_tamper["payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in limitation_tamper.items()
            if key != "payload_sha256"
        }
    )
    with pytest.raises(ValueError, match="model-scope limitations mismatch"):
        verify_report(limitation_tamper)
