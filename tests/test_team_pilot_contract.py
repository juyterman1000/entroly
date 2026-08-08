from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_team_pilot_contract_requires_matched_quality_and_provider_evidence() -> None:
    contract = json.loads(
        (ROOT / "docs" / "team-pilot-contract.json").read_text(encoding="utf-8")
    )
    assert contract["schema_version"] == "entroly.team-pilot.v1"
    assert contract["universal_claim"] is False
    assert "task_quality_score" in contract["required_per_trial_evidence"]
    assert "provider_input_tokens_when_available" in contract["required_per_trial_evidence"]
    assert "entroly_receipt_id_for_entroly_arm" in contract["required_per_trial_evidence"]
    assert contract["accounting_rules"]["missing_provider_usage"] == "unknown_not_zero"


def test_public_pilot_intake_warns_against_confidential_data() -> None:
    template = (ROOT / ".github" / "ISSUE_TEMPLATE" / "team-pilot.yml").read_text(
        encoding="utf-8"
    )
    guide = (ROOT / "docs" / "team-pilot.md").read_text(encoding="utf-8")
    for phrase in ("Do not include source code", "provider keys", "confidential"):
        assert phrase in template
        assert phrase in guide
