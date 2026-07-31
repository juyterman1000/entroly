from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_capability_coverage.py"
SPEC = importlib.util.spec_from_file_location("verify_capability_coverage", SCRIPT)
assert SPEC and SPEC.loader
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


def test_repository_capability_coverage_is_complete_and_resolvable():
    assert coverage.validate() == []


def test_validator_rejects_missing_proof_and_unknown_surface(tmp_path, monkeypatch):
    manifest = tmp_path / "capabilities.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "entroly.capability-coverage.v1",
                "capabilities": [
                    {
                        "id": "unsafe_claim",
                        "status": "production",
                        "implementation": ["missing.py"],
                        "tests": [],
                        "docs": [],
                        "package_surfaces": ["telepathy"],
                        "public_entrypoints": [],
                        "limitations": ""
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(coverage, "REPO_ROOT", tmp_path)

    failures = coverage.validate(manifest)

    assert any("missing implementation" in failure for failure in failures)
    assert any("unknown package surfaces" in failure for failure in failures)
    assert any("has no tests" in failure for failure in failures)
    assert any("missing limitation" in failure for failure in failures)
