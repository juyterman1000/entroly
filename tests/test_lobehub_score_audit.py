from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_lobehub_score.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_lobehub_score", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_has_complete_local_lobehub_evidence() -> None:
    module = _load_module()
    report = module.collect(ROOT, protocol_validated=True)
    evidence = report["evidence"]

    for criterion in (
        "deployment",
        "deployMoreThanManual",
        "license",
        "readme",
        "tools",
        "prompts",
        "resources",
    ):
        assert evidence[criterion]["check"], criterion

    assert evidence["validated"]["check"] is True
    assert evidence["validated"]["external"] is True
    assert report["local_readiness"]["score"] == 96
    assert report["local_readiness"]["grade_if_lobehub_confirmed_same_flags"] == "A"
    assert report["local_readiness"]["warning"].startswith(
        "This is repository/local-protocol"
    )


def test_claimed_point_remains_external_not_fabricated() -> None:
    module = _load_module()
    report = module.collect(ROOT, protocol_validated=False)

    assert report["evidence"]["claimed"] == {
        "check": False,
        "source": "Only LobeHub can confirm ownership; its detail score currently does not receive claimed state.",
        "classification": "external implementation/index state",
        "external": True,
    }
    assert report["evidence"]["validated"]["check"] is False
    assert report["local_readiness"]["grade_if_lobehub_confirmed_same_flags"] == "F"
