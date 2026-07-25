"""Regression coverage for source-atomic workspace index reconciliation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

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


class _FakeReconcileEngine:
    """Deterministic orchestration double; exact matches simulate native dedup."""

    def __init__(self, checkpoint_dir: Path):
        self.config = SimpleNamespace(checkpoint_dir=checkpoint_dir)
        self._use_rust = False
        self._fragments: dict[str, SimpleNamespace] = {}
        self._next_id = 1
        self.dependency_rebuilds = 0
        self.persistence_calls = 0

    def seed(self, source: str, content: str, fragment_id: str | None = None) -> str:
        fragment_id = fragment_id or f"fragment-{self._next_id}"
        self._next_id += 1
        self._fragments[fragment_id] = SimpleNamespace(
            fragment_id=fragment_id,
            source=source,
            content=content,
            token_count=max(1, len(content) // 4),
        )
        return fragment_id

    def wait_until_warm(self):
        return True

    def snapshot_index_state(self):
        return copy.deepcopy(self._fragments)

    def restore_index_state(self, snapshot):
        self._fragments = copy.deepcopy(snapshot)

    def remove_sources(self, sources):
        requested = set(sources)
        removed_sources = []
        for fragment_id, fragment in list(self._fragments.items()):
            if fragment.source in requested:
                removed_sources.append(fragment.source)
                del self._fragments[fragment_id]
        return {
            "status": "removed",
            "removed_fragments": len(removed_sources),
            "removed_sources": sorted(set(removed_sources)),
            "missing_sources": sorted(requested - set(removed_sources)),
        }

    def ingest_fragment(self, content, source, token_count, is_pinned=False):
        del is_pinned
        for fragment in self._fragments.values():
            if fragment.content == content:
                return {
                    "status": "duplicate",
                    "duplicate_of": fragment.fragment_id,
                    "tokens_saved": token_count,
                }
        fragment_id = self.seed(source, content)
        return {
            "status": "ingested",
            "fragment_id": fragment_id,
            "token_count": token_count,
        }

    def rebuild_dependencies(self):
        self.dependency_rebuilds += 1
        return {"status": "rebuilt"}

    def persist_index(self):
        self.persistence_calls += 1
        return {"status": "persisted"}


def _fake_file_sources(engine: _FakeReconcileEngine) -> set[str]:
    return {fragment.source for fragment in engine._fragments.values()}


_DUP_BODY = (
    '"""A substantial module so duplicate detection has real signal."""\n\n'
    + "\n".join(
        f"def handler_{i}(request):\n"
        f"    value = compute_step_{i}(request.payload, index={i})\n"
        f"    return normalize(value, scale={i * 7 + 3})\n"
        for i in range(40)
    )
    + "\n"
)


def _seed_duplicate_workspace(tmp_path: Path):
    original = tmp_path / "original.py"
    twin = tmp_path / "twin.py"
    original.write_text(_DUP_BODY, encoding="utf-8")
    twin.write_text(_DUP_BODY, encoding="utf-8")
    engine = _FakeReconcileEngine(tmp_path / "state")
    representative_id = engine.seed("file:original.py", _DUP_BODY)
    first = reconcile_index(engine, str(tmp_path))
    assert first["files_duplicate"] == 1
    assert first["files_added"] == 0
    assert first["dependency_refresh"]["status"] != "rebuilt"
    return engine, original, twin, representative_id


def test_duplicate_ledger_skips_only_with_live_representative(tmp_path: Path):
    engine, _original, _twin, _representative_id = _seed_duplicate_workspace(
        tmp_path
    )

    second = reconcile_index(engine, str(tmp_path))

    assert second["status"] == "current"
    assert second["files_duplicate"] == 0
    assert second["files_duplicate_skipped"] == 1
    assert engine.dependency_rebuilds == 0


def test_duplicate_ledger_promotes_twin_when_representative_deleted(tmp_path: Path):
    engine, original, _twin, _representative_id = _seed_duplicate_workspace(
        tmp_path
    )
    original.unlink()

    receipt = reconcile_index(engine, str(tmp_path))

    assert receipt["status"] == "updated"
    assert receipt["files_duplicate_skipped"] == 0
    assert receipt["files_added"] == 1
    assert receipt["files_removed"] == 1
    assert _fake_file_sources(engine) == {"file:twin.py"}


def test_duplicate_ledger_promotes_twin_when_representative_changes(tmp_path: Path):
    engine, original, _twin, _representative_id = _seed_duplicate_workspace(
        tmp_path
    )
    original.write_text(
        "def changed_representative(): return 'new content'\n",
        encoding="utf-8",
    )

    receipt = reconcile_index(engine, str(tmp_path))

    assert receipt["files_duplicate_skipped"] == 0
    assert receipt["files_added"] == 1
    assert receipt["files_replaced"] == 1
    assert _fake_file_sources(engine) == {
        "file:original.py",
        "file:twin.py",
    }


def test_duplicate_ledger_survives_restart_with_same_fragment_identity(tmp_path: Path):
    engine, _original, _twin, representative_id = _seed_duplicate_workspace(
        tmp_path
    )
    restarted = _FakeReconcileEngine(engine.config.checkpoint_dir)
    restarted.seed("file:original.py", _DUP_BODY, fragment_id=representative_id)

    receipt = reconcile_index(restarted, str(tmp_path))

    assert receipt["status"] == "current"
    assert receipt["files_duplicate_skipped"] == 1
    assert _fake_file_sources(restarted) == {"file:original.py"}


@pytest.mark.parametrize(
    "ledger_text",
    [
        "{not-json",
        json.dumps({"file:twin.py": "0" * 64}),
    ],
)
def test_corrupt_or_legacy_duplicate_ledger_fails_open(
    tmp_path: Path,
    ledger_text: str,
):
    original = tmp_path / "original.py"
    twin = tmp_path / "twin.py"
    original.write_text(_DUP_BODY, encoding="utf-8")
    twin.write_text(_DUP_BODY, encoding="utf-8")
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "reconcile_duplicates.json").write_text(
        ledger_text,
        encoding="utf-8",
    )
    engine = _FakeReconcileEngine(state_dir)
    engine.seed("file:original.py", _DUP_BODY)

    receipt = reconcile_index(engine, str(tmp_path))

    assert receipt["files_duplicate_skipped"] == 0
    assert receipt["files_duplicate"] == 1
    persisted = json.loads(
        (state_dir / "reconcile_duplicates.json").read_text(encoding="utf-8")
    )
    assert persisted["schema_version"] == 2
    assert "file:twin.py" in persisted["entries"]


def test_reconcile_excludes_nested_entroly_state_directory(tmp_path: Path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "internal.json").write_text('{"private": true}', encoding="utf-8")
    (tmp_path / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    engine = _FakeReconcileEngine(state_dir)

    receipt = reconcile_index(engine, str(tmp_path))

    assert receipt["files_scanned"] == 1
    assert _fake_file_sources(engine) == {"file:app.py"}


def test_native_duplicate_is_not_counted_as_index_mutation(tmp_path: Path):
    (tmp_path / "original.py").write_text(_DUP_BODY, encoding="utf-8")
    (tmp_path / "twin.py").write_text(_DUP_BODY, encoding="utf-8")
    engine = _native_engine(tmp_path, persistent=True)
    auto_index(engine, project_dir=str(tmp_path), force=True)

    sources = {
        fragment.get("source")
        for fragment in engine._rust.export_fragments()
    } & {"file:original.py", "file:twin.py"}
    if len(sources) != 1:
        pytest.skip("installed entroly-core did not dedup identical files")

    first = reconcile_index(engine, str(tmp_path))
    second = reconcile_index(engine, str(tmp_path))

    assert first["files_duplicate"] == 1
    assert first["files_added"] == 0
    assert first["dependency_refresh"]["status"] != "rebuilt"
    assert second["files_duplicate_skipped"] == 1
    assert second["dependency_refresh"]["status"] != "rebuilt"


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
