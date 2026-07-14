from __future__ import annotations

import hashlib
import copy
import os

import pytest

from benchmarks.compression_gauntlet import (
    SCHEMA_VERSION,
    _implementation_sha256,
    _subprocess_env,
    analyze,
    build_scenarios,
    render_markdown,
    render_svg,
    run_adapter,
    verify_report,
)


def test_implementation_fingerprint_is_line_ending_independent(tmp_path) -> None:
    windows = tmp_path / "windows" / "impl.py"
    linux = tmp_path / "linux" / "impl.py"
    windows.parent.mkdir()
    linux.parent.mkdir()
    windows.write_bytes(b"first = 1\r\nsecond = 2\r\n")
    linux.write_bytes(b"first = 1\nsecond = 2\n")

    assert _implementation_sha256((windows,)) == _implementation_sha256((linux,))


def _payload():
    scenarios = build_scenarios()
    return {
        "schema_version": SCHEMA_VERSION,
        "model": "gpt-4o",
        "budget_tokens": 1_200,
        "runs": 2,
        "warmups": 0,
        "scenarios": [scenario.public_record() for scenario in scenarios],
    }


def test_fixtures_are_deterministic_and_preregister_evidence() -> None:
    first = build_scenarios()
    second = build_scenarios()

    assert [scenario.scenario_id for scenario in first] == [
        "cargo-build-failure",
        "json-incident-middle",
        "sre-incident-tail",
        "code-search-middle",
    ]
    assert [scenario.content for scenario in first] == [scenario.content for scenario in second]
    assert all(
        needle in scenario.content
        for scenario in first
        for needle in scenario.evidence_needles
    )


def test_adapter_environment_does_not_inherit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))

    environment = _subprocess_env()

    assert "OPENAI_API_KEY" not in environment
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_entroly_adapter_preserves_every_preregistered_needle() -> None:
    payload = _payload()
    report = run_adapter("entroly", payload)

    assert report["version"]
    assert len(report["results"]) == 4
    by_id = {scenario.scenario_id: scenario for scenario in build_scenarios()}
    for result in report["results"]:
        assert result["deterministic"] is True
        assert result["output_sha256"] == hashlib.sha256(
            result["output_text"].encode("utf-8")
        ).hexdigest()
        assert all(
            needle in result["output_text"]
            for needle in by_id[result["scenario_id"]].evidence_needles
        )


def test_single_participant_is_not_a_competitive_claim() -> None:
    scenarios = build_scenarios()
    adapter = run_adapter("entroly", _payload())
    report = analyze(
        scenarios=scenarios,
        adapters=[adapter],
        protocol={"model": "gpt-4o", "budget_tokens": 1_200, "runs": 2, "warmups": 0},
    )

    assert report["aggregates"]["entroly"]["passed"] is True
    assert report["suite_winner"] is None
    assert report["headline_eligible"] is False
    assert "NO COMPETITIVE CLAIM" in render_markdown(report)


def test_evidence_loss_disqualifies_smaller_output() -> None:
    scenarios = build_scenarios()
    entroly = run_adapter("entroly", _payload())
    lossy = {
        **entroly,
        "system": "lossy",
        "package": "fixture",
        "version": "1",
        "results": [
            {
                **result,
                "output_text": "tiny",
                "output_sha256": hashlib.sha256(b"tiny").hexdigest(),
            }
            for result in entroly["results"]
        ],
    }

    report = analyze(
        scenarios=scenarios,
        adapters=[entroly, lossy],
        protocol={"model": "gpt-4o", "budget_tokens": 1_200, "runs": 2, "warmups": 0},
    )

    assert report["aggregates"]["lossy"]["weighted_savings_ratio"] > 0.99
    assert report["aggregates"]["lossy"]["passed"] is False
    assert report["suite_winner"] is None


def test_artifact_verifier_rejects_tampered_output() -> None:
    scenarios = build_scenarios()
    adapter = run_adapter("entroly", _payload())
    report = analyze(
        scenarios=scenarios,
        adapters=[adapter],
        protocol={"model": "gpt-4o", "budget_tokens": 1_200, "runs": 2, "warmups": 0},
    )
    verify_report(report)
    tampered = copy.deepcopy(report)
    tampered["results"][0]["output_text"] += "tampered"

    with pytest.raises(ValueError, match="output_sha256"):
        verify_report(tampered)


def test_artifact_verifier_rejects_stale_entroly_version() -> None:
    scenarios = build_scenarios()
    adapter = run_adapter("entroly", _payload())
    report = analyze(
        scenarios=scenarios,
        adapters=[adapter],
        protocol={"model": "gpt-4o", "budget_tokens": 1_200, "runs": 2, "warmups": 0},
    )
    report["participants"]["entroly"]["version"] = "stale-version"

    with pytest.raises(ValueError, match="version changed"):
        verify_report(report)


def test_social_card_keeps_scope_caveat_attached() -> None:
    scenarios = build_scenarios()
    first = run_adapter("entroly", _payload())
    second = {**first, "system": "competitor", "package": "fixture", "version": "1"}
    report = analyze(
        scenarios=scenarios,
        adapters=[first, second],
        protocol={"model": "gpt-4o", "budget_tokens": 1_200, "runs": 2, "warmups": 0},
    )

    svg = render_svg(report)

    assert "Same inputs. Same evidence. Fewer tokens." in svg
    assert "Synthetic, no-model result" in svg
    assert "not downstream answer quality or ML superiority" in svg


def test_tied_participants_do_not_create_a_winner() -> None:
    scenarios = build_scenarios()
    first = run_adapter("entroly", _payload())
    second = {**first, "system": "same", "package": "fixture", "version": "1"}
    report = analyze(
        scenarios=scenarios,
        adapters=[first, second],
        protocol={"model": "gpt-4o", "budget_tokens": 1_200, "runs": 2, "warmups": 0},
    )

    assert report["suite_winner"] is None
    assert report["headline_eligible"] is False
