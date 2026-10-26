"""Belief writes are validated at the vault boundary.

`vault_write_belief` (server.py) passes agent-supplied values straight into
BeliefArtifact -> VaultManager.write_belief. Confidence 7.0, -3.0 or NaN, an
unknown status and an empty entity were all written, after which
`coverage_index()` reported `average_confidence: NaN`.
"""

from __future__ import annotations

import json
import math

import pytest

from entroly.vault import BeliefArtifact, VaultConfig, VaultManager


def _vault(tmp_path) -> VaultManager:
    return VaultManager(VaultConfig(base_path=str(tmp_path / "vault")))


def _belief(**overrides) -> BeliefArtifact:
    fields = dict(entity="auth::session", title="t", body="b", confidence=0.7,
                  status="inferred", sources=["src/auth.py:1"])
    fields.update(overrides)
    return BeliefArtifact(**fields)


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"confidence": 7.0}, "confidence"),
        ({"confidence": -3.0}, "confidence"),
        ({"confidence": float("nan")}, "confidence"),
        ({"confidence": float("inf")}, "confidence"),
        ({"status": "definitely-true"}, "status"),
        ({"entity": ""}, "entity"),
        ({"entity": "   "}, "entity"),
    ],
)
def test_invalid_belief_is_refused_with_a_reason(tmp_path, overrides, field) -> None:
    vault = _vault(tmp_path)

    result = vault.write_belief(_belief(**overrides))

    assert result["status"] == "not_written"
    assert field in result["reason"]
    assert vault.list_beliefs() == []


def test_valid_belief_is_still_written(tmp_path) -> None:
    vault = _vault(tmp_path)

    assert vault.write_belief(_belief(confidence=1.0, status="verified"))["status"] == "written"
    assert vault.write_belief(_belief(entity="auth::token", confidence=0.0))["status"] == "written"
    assert vault.coverage_index()["total_beliefs"] == 2


def test_coverage_never_reports_nan_for_a_legacy_belief(tmp_path) -> None:
    vault = _vault(tmp_path)
    vault.write_belief(_belief(confidence=0.8))
    legacy = tmp_path / "vault" / "beliefs" / "legacy.md"
    legacy.write_text(
        "---\nclaim_id: legacy-1\nentity: legacy\nstatus: inferred\nconfidence: nan\n"
        "sources:\n  - src/legacy.py\n---\n\n# legacy\n",
        encoding="utf-8",
    )

    coverage = vault.coverage_index()

    assert coverage["total_beliefs"] == 2
    assert math.isfinite(coverage["average_confidence"])
    assert 0.0 <= coverage["average_confidence"] <= 1.0
    json.loads(json.dumps(coverage), parse_constant=lambda token: pytest.fail(f"non-JSON {token}"))
