"""Regression coverage for source-atomic workspace index reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest

from entroly.auto_index import auto_index, reconcile_index
from entroly.config import EntrolyConfig
from entroly.server import EntrolyEngine


def _native_engine(tmp_path: Path, *, persistent: bool = False) -> EntrolyEngine:
    engine = EntrolyEngine(
        EntrolyConfig(
            use_persistent_index=persistent,
            checkpoint_dir=tmp_path / "checkpoints",
        )
    )
    if not engine._use_rust:
        pytest.skip("source-atomic reconciliation requires the native engine")
    if not hasattr(engine._rust, "remove_sources"):
        pytest.skip("installed entroly-core predates source-atomic removal")
    return engine


def _file_fragments(engine: EntrolyEngine, source: str) -> list[dict]:
    return [
        dict(fragment)
        for fragment in engine._rust.export_fragments()
        if fragment.get("source") == source
    ]


def test_reconcile_replaces_modified_source_without_stale_copy(tmp_path: Path):
    source_file = tmp_path / "auth.py"
    source_file.write_text("def authenticate(): return 'v1'\n", encoding="utf-8")
    engine = _native_engine(tmp_path)

    auto_index(engine, project_dir=str(tmp_path), force=True)
    source_file.write_text("def authenticate(): return 'v2'\n", encoding="utf-8")

    receipt = reconcile_index(engine, str(tmp_path))

    fragments = _file_fragments(engine, "file:auth.py")
    assert receipt["status"] == "updated"
    assert receipt["files_replaced"] == 1
    assert len(fragments) == 1
    assert "'v2'" in fragments[0]["content"]
    assert "'v1'" not in fragments[0]["content"]


def test_reconcile_rolls_back_failed_replacement(tmp_path: Path, monkeypatch):
    source_file = tmp_path / "auth.py"
    source_file.write_text("def authenticate(): return 'v1'\n", encoding="utf-8")
    engine = _native_engine(tmp_path)
    auto_index(engine, project_dir=str(tmp_path), force=True)
    source_file.write_text("def authenticate(): return 'v2'\n", encoding="utf-8")

    def fail_ingest(*args, **kwargs):
        raise RuntimeError("injected ingest failure")

    monkeypatch.setattr(engine, "ingest_fragment", fail_ingest)
    receipt = reconcile_index(engine, str(tmp_path))

    assert receipt["status"] == "partial"
    assert receipt["rolled_back"] is True
    assert receipt["files_replaced"] == 0
    assert receipt["persistence"]["status"] == "rolled_back"
    fragments = _file_fragments(engine, "file:auth.py")
    assert len(fragments) == 1
    assert "'v1'" in fragments[0]["content"]


def test_forced_reindex_rolls_back_when_persistence_fails(tmp_path: Path, monkeypatch):
    source_file = tmp_path / "service.py"
    source_file.write_text("VERSION = 'old'\n", encoding="utf-8")
    engine = _native_engine(tmp_path, persistent=True)
    auto_index(engine, project_dir=str(tmp_path), force=True)
    source_file.write_text("VERSION = 'new'\n", encoding="utf-8")

    def fail_persist():
        raise OSError("injected disk failure")

    monkeypatch.setattr(engine, "persist_index", fail_persist)
    receipt = auto_index(engine, project_dir=str(tmp_path), force=True)

    assert receipt["status"] == "error"
    assert receipt["force_removal"]["status"] == "rolled_back"
    fragments = _file_fragments(engine, "file:service.py")
    assert len(fragments) == 1
    assert "'old'" in fragments[0]["content"]


def test_reconcile_rebinds_existing_callers_to_replacement(tmp_path: Path):
    provider = tmp_path / "provider.py"
    caller = tmp_path / "caller.py"
    provider.write_text("def provide(): return 'v1'\n", encoding="utf-8")
    caller.write_text("def consume(): return provide()\n", encoding="utf-8")
    engine = _native_engine(tmp_path)
    auto_index(engine, project_dir=str(tmp_path), force=True)
    assert dict(engine._rust.dep_graph_stats())["edges"] >= 1

    provider.write_text("def provide(): return 'v2'\n", encoding="utf-8")
    receipt = reconcile_index(engine, str(tmp_path))

    assert receipt["dependency_refresh"]["status"] == "rebuilt"
    assert dict(engine._rust.dep_graph_stats())["edges"] >= 1


def test_reconcile_removes_deleted_source(tmp_path: Path):
    source_file = tmp_path / "obsolete.py"
    source_file.write_text("def obsolete(): return True\n", encoding="utf-8")
    engine = _native_engine(tmp_path)
    auto_index(engine, project_dir=str(tmp_path), force=True)
    assert _file_fragments(engine, "file:obsolete.py")

    source_file.unlink()
    receipt = reconcile_index(engine, str(tmp_path))

    assert receipt["files_removed"] == 1
    assert not _file_fragments(engine, "file:obsolete.py")


def test_warm_start_reconciles_and_persists_changed_workspace(tmp_path: Path):
    source_file = tmp_path / "service.py"
    source_file.write_text("VERSION = 'old'\n", encoding="utf-8")
    first = _native_engine(tmp_path, persistent=True)
    indexed = auto_index(first, project_dir=str(tmp_path), force=True)
    assert indexed["persistence"]["status"] == "persisted"

    source_file.write_text("VERSION = 'new'\n", encoding="utf-8")
    second = _native_engine(tmp_path, persistent=True)
    second.wait_until_warm()
    warm = auto_index(second, project_dir=str(tmp_path))

    assert warm["status"] == "skipped"  # backward-compatible outer contract
    assert warm["reconciliation"]["files_replaced"] == 1
    assert warm["reconciliation"]["persistence"]["status"] == "persisted"
    fragments = _file_fragments(second, "file:service.py")
    assert len(fragments) == 1
    assert "'new'" in fragments[0]["content"]

    # A third process must not resurrect the stale pre-reconciliation bytes.
    third = _native_engine(tmp_path, persistent=True)
    third.wait_until_warm()
    restored = _file_fragments(third, "file:service.py")
    assert len(restored) == 1
    assert "'new'" in restored[0]["content"]


def test_remove_sources_is_exact_and_preserves_non_file_context(tmp_path: Path):
    engine = _native_engine(tmp_path)
    engine.ingest_fragment("old file", "file:target.py", 4)
    engine.ingest_fragment("session decision", "memory:decision", 4)
    target_id = _file_fragments(engine, "file:target.py")[0]["fragment_id"]
    engine._fragment_selection_counts[target_id] = 3
    engine._prefetch.record_access("target.py", turn=1)
    engine._prefetch._pending_predictions["target.py"] = {"other.py"}

    result = engine.remove_sources(["file:target.py"])

    assert result["removed_fragments"] == 1
    assert target_id not in engine._fragment_selection_counts
    assert all(
        path not in {"target.py", "file:target.py"}
        for path, _turn in engine._prefetch._recent_accesses
    )
    assert "target.py" not in engine._prefetch._pending_predictions
    remaining = [dict(fragment) for fragment in engine._rust.export_fragments()]
    assert [fragment["source"] for fragment in remaining] == ["memory:decision"]
