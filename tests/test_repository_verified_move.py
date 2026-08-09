from __future__ import annotations

from pathlib import Path

import pytest

from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.service import RepositoryIntelligenceService
from entroly.repository_intelligence.verified_move import (
    apply_verified_file_move_plan,
    build_verified_file_move_plan,
)
from entroly.repository_intelligence.verified_refactor import (
    verify_refactor_apply_commitment,
    verify_refactor_plan_commitment,
)


def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _project(root: Path) -> RepositoryIntelligenceService:
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/old.py", "def execute():\n    return 1\n")
    _write(
        root,
        "app.py",
        "from pkg.old import execute\ndef run():\n    return execute()\n",
    )
    return RepositoryIntelligenceService(root)


def test_file_move_rewrites_exact_import_and_rebuilds_resolved_graph(
    tmp_path: Path,
) -> None:
    service = _project(tmp_path)
    summary = service.summary()
    assert service._index is not None
    plan = build_verified_file_move_plan(
        tmp_path,
        service._index,
        "pkg/old.py",
        "pkg/new.py",
        index_digest=str(summary["index_digest"]),
    )
    assert plan["safe_to_apply"] is True
    assert plan["changes"][0]["old_identifier"] == "pkg.old"
    assert plan["changes"][0]["new_identifier"] == "pkg.new"
    assert verify_refactor_plan_commitment(plan)
    with pytest.raises(ValueError, match="explicit acknowledgement"):
        apply_verified_file_move_plan(
            tmp_path,
            service._index,
            plan,
            index_digest=str(summary["index_digest"]),
            expected_plan_sha256=plan["receipt"]["plan_sha256"],
        )
    applied = apply_verified_file_move_plan(
        tmp_path,
        service._index,
        plan,
        index_digest=str(summary["index_digest"]),
        expected_plan_sha256=plan["receipt"]["plan_sha256"],
        acknowledge_incomplete=True,
    )
    assert verify_refactor_apply_commitment(applied)
    assert not (tmp_path / "pkg/old.py").exists()
    assert (tmp_path / "pkg/new.py").exists()
    assert "from pkg.new import execute" in (tmp_path / "app.py").read_text(
        encoding="utf-8"
    )
    rebuilt = build_repository_index(tmp_path)
    assert rebuilt.file_dependencies["app.py"] == ("pkg/new.py",)
    assert any(edge.callee_id.startswith("pkg/new.py::execute") for edge in rebuilt.call_edges)


def test_file_move_blocks_dotted_unaliased_import(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/old.py", "VALUE = 1\n")
    _write(tmp_path, "app.py", "import pkg.old\nprint(pkg.old.VALUE)\n")
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    assert service._index is not None
    plan = build_verified_file_move_plan(
        tmp_path,
        service._index,
        "pkg/old.py",
        "pkg/new.py",
        index_digest=str(summary["index_digest"]),
    )
    assert plan["safe_to_apply"] is False
    assert any(
        item["kind"] == "dotted-import-binding-change"
        for item in plan["blockers"]
    )
    assert plan["changes"] == []


def test_file_move_blocks_relative_import_when_package_changes(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/__init__.py", "")
    _write(tmp_path, "pkg/helper.py", "VALUE = 1\n")
    _write(tmp_path, "pkg/old.py", "from .helper import VALUE\n")
    (tmp_path / "other").mkdir()
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    assert service._index is not None
    plan = build_verified_file_move_plan(
        tmp_path,
        service._index,
        "pkg/old.py",
        "other/new.py",
        index_digest=str(summary["index_digest"]),
    )
    assert plan["safe_to_apply"] is False
    assert any(
        item["kind"] == "relative-import-package-change"
        for item in plan["blockers"]
    )


def test_file_move_refuses_source_changed_after_preview(tmp_path: Path) -> None:
    service = _project(tmp_path)
    summary = service.summary()
    assert service._index is not None
    plan = build_verified_file_move_plan(
        tmp_path,
        service._index,
        "pkg/old.py",
        "pkg/new.py",
        index_digest=str(summary["index_digest"]),
    )
    _write(tmp_path, "pkg/old.py", "def execute():\n    return 2\n")
    with pytest.raises(ValueError, match="source preimage"):
        apply_verified_file_move_plan(
            tmp_path,
            service._index,
            plan,
            index_digest=str(summary["index_digest"]),
            expected_plan_sha256=plan["receipt"]["plan_sha256"],
            acknowledge_incomplete=True,
        )
