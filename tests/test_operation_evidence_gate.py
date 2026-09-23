from __future__ import annotations

import json

import pytest

from entroly.operation_evidence_gate import (
    build_operation_evidence_gate,
    score_operation_evidence_gate,
)
from entroly.sufficiency import Candidate


@pytest.fixture
def trusted_source(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "auth.py"
    source.write_text("class AuthManager: pass", encoding="utf-8")
    return root, source


def _allowing_gate(root, **overrides):
    values = {
        "task": "Verify authentication behavior",
        "candidates": [Candidate("auth", 1.0, 20, True)],
        "obligations": ["authentication"],
        "coverage": {"authentication": ["auth"]},
        "budget": 20,
        "answer": "`AuthManager` validates authentication.",
        "selected_sources": [
            {"source_path": "auth.py", "text": "class AuthManager: pass"}
        ],
        "omitted_sources": [],
        "trusted_source_root": root,
        "skill_id": "auth-check",
        "skill_evidence_validated": True,
    }
    values.update(overrides)
    return build_operation_evidence_gate(**values)


def test_all_surfaces_pass_preflight_without_authorizing_operation(trusted_source):
    root, _ = trusted_source
    result = _allowing_gate(root)
    assert result.decision == "pass"
    assert result.preflight_passed
    assert not result.autonomous_execution_allowed
    assert result.reasons == ()
    assert result.exact_budget_cost == 20
    assert score_operation_evidence_gate(result) == 1.0


def test_lexical_match_does_not_verify_surrounding_claim(trusted_source):
    root, _ = trusted_source
    result = _allowing_gate(root)
    # The source only declares an empty class; it does not establish the
    # answer's assertion that authentication is validated.
    assert result.decision == "pass"
    assert result.reference_coverage == 1.0
    assert not result.autonomous_execution_allowed


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"budget": 19}, "obligation:insufficient_budget"),
        ({"answer": "`ImaginaryManager` validates authentication."},
         "answer:unsourced_reference"),
        ({"skill_evidence_validated": False}, "skill:stale_or_invalid_evidence"),
    ],
)
def test_concrete_failure_blocks(trusted_source, overrides, reason):
    root, _ = trusted_source
    result = _allowing_gate(root, **overrides)
    assert result.decision == "block"
    assert reason in result.reasons
    assert not result.autonomous_execution_allowed


def test_external_selected_source_blocks_even_when_text_matches(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("class AuthManager: pass", encoding="utf-8")
    result = _allowing_gate(
        root,
        selected_sources=[
            {"source_path": str(outside), "text": "class AuthManager: pass"}
        ],
    )
    assert result.decision == "block"
    assert "source:outside_policy" in result.reasons


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"context_update": "auth.py changed", "omitted_fragment": "old auth rule",
          "retained_context": "authentication overview"},
         "context:reverification_required"),
        ({"skill_evidence_validated": None}, "skill:validation_missing"),
        ({"answer": "Authentication is configured."},
         "answer:no_measurable_reference"),
    ],
)
def test_incomplete_evidence_holds(trusted_source, overrides, reason):
    root, _ = trusted_source
    result = _allowing_gate(root, **overrides)
    assert result.decision == "hold"
    assert reason in result.reasons
    assert not result.autonomous_execution_allowed


def test_receipt_is_deterministic_and_sensitive_to_evidence(trusted_source):
    root, _ = trusted_source
    first = _allowing_gate(root)
    second = _allowing_gate(root)
    changed = _allowing_gate(
        root,
        selected_sources=[
            {"source_path": "auth.py", "text": "class AuthManager: updated"}
        ],
    )
    assert first.evidence_sha256 == second.evidence_sha256
    assert first.evidence_sha256 != changed.evidence_sha256
    json.dumps(first.to_dict(), sort_keys=True)


def test_provider_and_model_names_are_not_gate_inputs(trusted_source):
    root, _ = trusted_source
    result = _allowing_gate(root)
    payload = result.to_dict()
    assert "provider" not in payload
    assert "model" not in payload


def test_skill_evidence_without_skill_is_rejected(trusted_source):
    root, _ = trusted_source
    with pytest.raises(ValueError):
        _allowing_gate(root, skill_id=None, skill_evidence_validated=True)
