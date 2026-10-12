"""The public-catalog gate must not hide new TDQS warnings."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_public_mcp_tdqs.py"
SPEC = importlib.util.spec_from_file_location("check_public_mcp_tdqs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _report():
    return {
        "server": {"name": "entroly", "toolCount": 2},
        "tools": [
            {"name": "work_record_review", "contextSignals": {"annotationValues": {"readOnly": False}}},
            {"name": "read_source_file", "contextSignals": {"annotationValues": {"readOnly": True}}},
        ],
        "shadowCandidates": [{"tool": "work_record_review", "cheaperSibling": "read_source_file"}],
        "findings": [{
            "rule": "shadow-candidate", "severity": "warning", "tool": "work_record_review",
            "message": "work_record_review may be shadowed by read_source_file",
        }],
    }


def test_known_cost_only_candidate_is_documented() -> None:
    assert GATE.check_report(_report()) == []


def test_new_warning_still_fails_the_gate() -> None:
    report = _report()
    report["findings"].append({
        "rule": "missing-annotations", "severity": "warning",
        "tool": "read_source_file", "message": "missing annotations",
    })
    assert GATE.check_report(report) == ["missing annotations"]


def test_review_must_remain_state_changing_and_reader_read_only() -> None:
    report = _report()
    report["tools"][0]["contextSignals"]["annotationValues"]["readOnly"] = True
    assert GATE.check_report(report)


def test_different_shadow_pair_is_not_excused() -> None:
    report = _report()
    report["shadowCandidates"][0]["cheaperSibling"] = "other_tool"
    assert GATE.check_report(report)


def test_empty_catalog_is_not_accepted() -> None:
    report = {"server": {"name": "entroly", "toolCount": 0}, "tools": [], "findings": [], "shadowCandidates": []}
    assert GATE.check_report(report)
