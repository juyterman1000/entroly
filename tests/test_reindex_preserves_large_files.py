"""Re-indexing an unchanged repository must be a no-op, including large files.

Indexing and reconcile were two ingest paths with opposite rules for the same
file. Indexing chunks an oversized source (`path`, `path#1`, ...) so it stays
retrievable; reconcile read the same file, got `too_large`, and deleted it as
"unavailable" -- so the first re-index removed exactly what indexing had
deliberately kept, and the re-ingest could not restore it because that path
never chunked either.

Measured on this repository: pass 1 indexed cli.py as 33 chunks, proxy.py as 32,
auto_index.py as 9; pass 2 dropped chunk 0 of all 18 chunked files and the
corpus fell from 1222 to 1204. Those files were then permanently missing their
first 8 KB -- imports, module docstring, top-level configuration.

This is an integration test on purpose. The bug lived in the disagreement
between two code paths, so testing either one alone would have missed it.
"""

from __future__ import annotations

import pytest

from entroly.auto_index import MAX_FILE_BYTES, auto_index


def _fragment_sources(engine) -> list[str]:
    """Inspect indexed sources without assuming the native backend exists."""
    if engine._use_rust:
        fragments = engine._rust.export_fragments()
        return [str(fragment.get("source") or "") for fragment in fragments]
    return [
        str(getattr(fragment, "source", "") or "")
        for fragment in engine._fragments.values()
    ]


@pytest.fixture()
def indexed_project(tmp_path, monkeypatch):
    """A project holding one source file well above the per-file cap."""
    project = tmp_path / "proj"
    project.mkdir()
    # Distinct lines so chunking is content-addressable, not repetitive text
    # that a duplicate detector could legitimately collapse.
    big = "\n".join(f"def fn_{i}():\n    return {i}  # payload {i}" for i in range(6000))
    (project / "big_module.py").write_text(big, encoding="utf-8")
    (project / "small.py").write_text("def tiny():\n    return 1\n", encoding="utf-8")
    assert len(big.encode("utf-8")) > MAX_FILE_BYTES, "fixture must exceed the cap"

    monkeypatch.setenv("ENTROLY_SOURCE", str(project))
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    from entroly.server import EntrolyConfig, EntrolyEngine

    engine = EntrolyEngine(config=EntrolyConfig())
    engine._ensure_index_loaded()
    return engine, str(project)


def test_large_file_is_chunked_rather_than_skipped(indexed_project):
    engine, project = indexed_project
    auto_index(engine, project)
    chunks = [s for s in _fragment_sources(engine)
              if s == "file:big_module.py" or s.startswith("file:big_module.py#")]
    assert len(chunks) > 1, "an oversized source must be chunked, not dropped"
    assert "file:big_module.py" in chunks, "chunk 0 carries the head of the file"


def test_reindexing_unchanged_repo_preserves_every_chunk(indexed_project):
    engine, project = indexed_project
    auto_index(engine, project)
    first = sorted(_fragment_sources(engine))
    second_result = auto_index(engine, project)
    second = sorted(_fragment_sources(engine))

    assert second_result["status"] == "skipped"
    assert second_result["reconciliation"]["status"] == "current"
    lost = sorted(set(first) - set(second))
    assert not lost, f"re-indexing an unchanged repo deleted evidence: {lost}"
    assert first == second

    # And it must stay stable, not merely survive one extra pass.
    third_result = auto_index(engine, project)
    assert third_result["reconciliation"]["status"] == "current"
    assert sorted(_fragment_sources(engine)) == first


def test_editing_a_large_file_does_not_strand_stale_chunks(indexed_project, tmp_path):
    engine, project_dir = indexed_project
    auto_index(engine, project_dir)
    before = _fragment_sources(engine)
    assert any(s.startswith("file:big_module.py#") for s in before)

    # Shrink the file below the cap: the old chunk group must not linger and
    # keep answering queries with text that no longer exists on disk.
    project = tmp_path / "proj"
    (project / "big_module.py").write_text("def only():\n    return 0\n", encoding="utf-8")
    auto_index(engine, project_dir)

    after = _fragment_sources(engine)
    stale = [s for s in after if s.startswith("file:big_module.py#")]
    assert not stale, f"stale chunks survived the edit: {stale}"
    assert "file:big_module.py" in after
