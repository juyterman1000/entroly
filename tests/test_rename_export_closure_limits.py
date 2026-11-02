"""Limits must leave explicit uncertainty instead of a closure claim."""
from pathlib import Path

from entroly.repository_intelligence import rename_export_closure
from entroly.repository_intelligence.service import RepositoryIntelligenceService
from entroly.repository_intelligence.verified_refactor import build_verified_rename_plan


def _plan(root: Path):
    service = RepositoryIntelligenceService(root)
    summary = service.summary()
    return build_verified_rename_plan(root, service._index, "execute", "perform",
                                      index_digest=summary["index_digest"])


def test_file_limit_reports_unexamined_exports(tmp_path: Path, monkeypatch):
    (tmp_path / "source.py").write_bytes(b"def execute():\n    return 1\n")
    (tmp_path / "facade.py").write_bytes(b"from source import execute\n")
    monkeypatch.setattr(rename_export_closure, "MAX_EXPORT_FILES", 1)
    plan = _plan(tmp_path)
    report = plan["risk"]["python_export_closure"]
    assert report["status"] == "unknown"
    assert report["files_examined"] == 1
    assert report["unknown_reasons"]["file-limit"] == 1
    assert plan["receipt"]["omissions_by_reason"]["python-export-closure:file-limit"] == 1


def test_module_and_package_name_collision_stops_propagation(tmp_path: Path):
    (tmp_path / "source.py").write_bytes(b"def execute():\n    return 1\n")
    (tmp_path / "facade.py").write_bytes(b"from source import execute\n")
    (tmp_path / "facade").mkdir()
    (tmp_path / "facade/__init__.py").write_bytes(b"def execute():\n    return 2\n")
    (tmp_path / "caller.py").write_bytes(b"from facade import execute\ncallback = execute\n")
    service = RepositoryIntelligenceService(tmp_path)
    summary = service.summary()
    plan = build_verified_rename_plan(tmp_path, service._index,
        "source.py::execute::function", "perform", index_digest=summary["index_digest"])
    report = plan["risk"]["python_export_closure"]
    assert report["status"] == "unknown"
    assert report["unknown_reasons"]["ambiguous-module-name"] == 1
    assert not any(change["path"] == "caller.py" for change in plan["changes"])
