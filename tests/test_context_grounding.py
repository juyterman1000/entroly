"""Tests for Spec 5: Context grounding gate — filter ungrounded fragments."""

from __future__ import annotations

from pathlib import Path

from entroly.context_grounding import (
    ContextGroundingGate,
    GroundingReport,
    GroundingResult,
    GROUNDING_THRESHOLD,
)


def _write(root: Path, name: str, content: str) -> None:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_grounding_gate_passes_grounded_fragments(tmp_path: Path) -> None:
    _write(tmp_path, "auth.py", "def authenticate(user, password):\n    return True\n")
    gate = ContextGroundingGate(repo_root=str(tmp_path))

    fragments = [
        {"id": "f1", "content": "authenticate(user, password)"},
    ]
    filtered, report = gate.filter_fragments(fragments)

    assert report.total == 1
    assert report.passed == 1
    assert report.demoted == 0
    assert len(filtered) == 1


def test_grounding_gate_demotes_ungrounded_fragments(tmp_path: Path) -> None:
    _write(tmp_path, "utils.py", "def helper():\n    return 1\n")
    gate = ContextGroundingGate(repo_root=str(tmp_path))

    fragments = [
        {"id": "f1", "content": "helper()"},
        {"id": "f2", "content": "nonexistent_fabricated_api_method()"},
    ]
    filtered, report = gate.filter_fragments(fragments)

    assert report.total == 2
    assert len(filtered) == 2
    assert filtered[0]["id"] == "f1"


def test_grounding_gate_empty_fragments_returns_empty(tmp_path: Path) -> None:
    gate = ContextGroundingGate(repo_root=str(tmp_path))
    filtered, report = gate.filter_fragments([])
    assert report.total == 0
    assert report.passed == 0
    assert report.demoted == 0
    assert filtered == []


def test_grounding_gate_caches_manifest(tmp_path: Path) -> None:
    _write(tmp_path, "mod.py", "def foo():\n    pass\n")
    gate = ContextGroundingGate(repo_root=str(tmp_path))

    gate.filter_fragments([{"id": "f1", "content": "foo()"}])
    first_built = gate._manifest_built_at

    gate.filter_fragments([{"id": "f2", "content": "foo()"}])
    assert gate._manifest_built_at == first_built


def test_grounding_result_dataclass() -> None:
    r = GroundingResult(
        fragment_id="test",
        h_score=0.3,
        grounded=True,
        n_unresolved=0,
    )
    assert r.grounded is True
    assert r.h_score == 0.3


def test_grounding_report_counts() -> None:
    report = GroundingReport(
        total=5,
        passed=3,
        demoted=2,
        results=(),
    )
    assert report.total == 5
    assert report.passed == 3
    assert report.demoted == 2


def test_score_fragment_without_manifest_returns_zero(tmp_path: Path) -> None:
    gate = ContextGroundingGate(repo_root=str(tmp_path))
    h_score, n_unresolved = gate.score_fragment("some_function()")
    assert h_score == 0.0
    assert n_unresolved == 0
