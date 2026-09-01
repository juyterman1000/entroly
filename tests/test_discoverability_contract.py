from __future__ import annotations

import json
from pathlib import Path

from scripts import verify_discoverability


ROOT = Path(__file__).resolve().parents[1]


def test_public_discoverability_contract_passes() -> None:
    assert verify_discoverability.collect_failures() == []


def test_registry_keeps_rankings_external_and_tasks_primary() -> None:
    registry = json.loads(
        (ROOT / "docs/discoverability-registry.json").read_text(encoding="utf-8")
    )
    contract = registry["measurement_contract"]
    assert contract["primary_outcome"] == "cost per successful, evidence-supported task"
    assert any("No universal best-tool" in item for item in contract["boundaries"])
    assert all(
        channel["status"]
        in {"requires_site_owner_connection", "baseline_pending"}
        for channel in registry["observation_channels"]
    )
