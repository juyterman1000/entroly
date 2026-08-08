from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_platform_readiness_matrix_is_self_verifying() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/verify_platform_readiness.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(
        (ROOT / "docs" / "platform-readiness.json").read_text(encoding="utf-8")
    )
    assert {item["id"] for item in payload["platforms"]} == {
        "linux-x86_64",
        "windows-x86_64",
        "macos-arm64",
    }


def test_supply_chain_workflow_generates_both_sbom_formats_and_attests_releases() -> None:
    workflow = (ROOT / ".github" / "workflows" / "supply-chain.yml").read_text(
        encoding="utf-8"
    )
    assert "format: spdx-json" in workflow
    assert "format: cyclonedx-json" in workflow
    assert "uses: actions/attest@v4" in workflow
    assert "if: github.event_name == 'release'" in workflow
    assert "subject-path: \"dist/*\"" in workflow
    assert "sbom-path: \"sbom/entroly.spdx.json\"" in workflow
