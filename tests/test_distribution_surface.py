from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "distribution" / "targets.json"
CHECKER = ROOT / "scripts" / "check_distribution_surface.py"


def test_distribution_surface_check_passes_offline() -> None:
    completed = subprocess.run(
        [sys.executable, str(CHECKER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "distribution surface check passed" in completed.stdout


def test_high_priority_targets_are_actionable_without_implied_submission() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    high_priority = [
        target for target in registry["targets"] if target["priority"] == 1
    ]

    assert high_priority
    assert {target["status"] for target in high_priority} == {"prepared"}
    for target in high_priority:
        assert target["proof_url"] is None
        assert target["next_action"].strip()
        assert target["required_assets"]


def test_published_targets_have_public_proof_urls() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    published = [
        target for target in registry["targets"] if target["status"] == "published"
    ]

    assert published
    for target in published:
        assert target["proof_url"].startswith("https://")
