"""`files_indexed` must be a count of files, and must respect its own cap.

The batch handed to ingest holds one entry per *chunk*: an oversized source file
is split and appended as ``file:path``, ``file:path#1``, ``file:path#2``... so
counting batch entries counts chunks. ``auto_index`` did exactly that, first via
the Rust ``ingested`` count and then via ``len(batch)``.

Measured on this repository before the fix, driving it through the command a new
user is told to run first:

    entroly verify-claims --max-files 10   ->   10 files
    entroly verify-claims --max-files 20   ->   60 files
    entroly verify-claims --max-files 120  ->  189 files

The number is printed to users, returned by ``simulate``/``perf`` ``--json`` and
``verify-claims``, and published as the headline of the CI dogfood evidence
artifact, so it overstated coverage on every surface at once. It is also a claim
that can be checked against the filesystem, which is the standard CLAUDE.md sets
for benchmark honesty.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import entroly.auto_index as auto_index  # noqa: E402
from entroly.auto_index import _logical_rel_path  # noqa: E402
from entroly.server import EntrolyEngine  # noqa: E402


def test_chunk_parts_collapse_to_one_file():
    """The helper the count depends on, checked on the shapes it will see."""
    assert _logical_rel_path("file:pkg/mod.py") == "file:pkg/mod.py"
    assert _logical_rel_path("file:pkg/mod.py#1") == "file:pkg/mod.py"
    assert _logical_rel_path("file:pkg/mod.py#12") == "file:pkg/mod.py"
    # A '#' that is not a chunk marker belongs to the path and must survive.
    assert _logical_rel_path("file:pkg/c#sharp.py") == "file:pkg/c#sharp.py"
    assert _logical_rel_path("file:pkg/a#b.py#2") == "file:pkg/a#b.py"


@pytest.fixture
def repo_with_oversized_sources(tmp_path, monkeypatch):
    """A project whose files land in the band that gets chunked.

    Chunking is the mechanism that produced the overcount, so a fixture of small
    files would pass against the broken implementation. The size has to sit
    between the soft per-file budget (192 KiB, above which a source file is
    chunked) and the hard ceiling (500 KiB, above which it is skipped outright).
    An earlier version of this fixture used ~574 KiB files, which were skipped
    rather than chunked -- the "was anything chunked?" assertion at the end of
    this module is what caught it.
    """
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)

    # Every function body is unique, within and across files. SimHash dedup
    # drops near-duplicates, so repeating one block would collapse a file's
    # chunks back into a single fragment, and byte-identical files would
    # collapse into one file -- either one silently undoes what this fixture
    # exists to produce. Both were observed before the bodies were varied.
    for i in range(3):
        big_body = "\n".join(
            f"def process_{i}_{n}(records):\n"
            f"    total_{n} = {n * 7 + i}\n"
            f"    for record in records:\n"
            f"        total_{n} += record.value_{n} * {n % 97 + 3}\n"
            f"    return total_{n} - {n * 13 % 101}\n"
            for n in range(1800)
        )
        path = project / "src" / f"large_{i}.py"
        path.write_text(big_body, encoding="utf-8")
        size = path.stat().st_size
        assert 192 * 1024 < size < 500 * 1024, (
            f"fixture file is {size} bytes; it must exceed the 192 KiB soft "
            "budget to be chunked but stay under the 500 KiB hard ceiling or "
            "it is skipped instead"
        )
    for i in range(2):
        (project / "src" / f"small_{i}.py").write_text(
            f"def tiny_{i}():\n    return {i}\n", encoding="utf-8"
        )

    monkeypatch.chdir(project)
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    return project


def _index(max_files: int, state_dir: Path, monkeypatch) -> dict:
    """Index once with a *fresh* state directory.

    Reusing one ``ENTROLY_DIR`` across calls makes the second call reload the
    persisted index and return the cache path instead of indexing, so the run
    silently measures the previous run's index rather than this cap.
    """
    monkeypatch.setenv("ENTROLY_DIR", str(state_dir))
    previous = auto_index.MAX_FILES
    auto_index.MAX_FILES = max_files
    try:
        return auto_index.auto_index(EntrolyEngine())
    finally:
        auto_index.MAX_FILES = previous


def test_files_indexed_never_exceeds_the_cap(
    repo_with_oversized_sources, tmp_path, monkeypatch
):
    """The cap is the user's explicit request; reporting past it is a false claim."""
    for cap in (1, 2, 3, 5):
        result = _index(cap, tmp_path / f"state_cap_{cap}", monkeypatch)
        assert result["files_indexed"] <= cap, (
            f"asked for at most {cap} files but reported "
            f"{result['files_indexed']}; chunks of one file are being counted "
            f"as separate files (fragments_ingested="
            f"{result.get('fragments_ingested')})"
        )


def test_reloading_a_persisted_index_reports_files_not_chunks(
    repo_with_oversized_sources, tmp_path, monkeypatch
):
    """The cache path counts too, and it counted chunks.

    When a persisted index is already loaded, ``auto_index`` short-circuits and
    derives ``files_indexed`` from the fragments in the engine rather than from a
    fresh batch. That branch counted raw fragment sources, so a chunked file
    contributed one "file" per chunk: measured at 68 files for a 5-file project.

    It is reached by pointing a second engine at the same ``ENTROLY_DIR``, which
    is the normal case -- every command after the first index.
    """
    state = tmp_path / "shared_state"

    first = _index(20, state, monkeypatch)
    assert first.get("status") != "skipped", (
        "the first pass should build the index, not reload one"
    )

    second = _index(20, state, monkeypatch)
    if second.get("status") != "skipped":
        pytest.skip(
            "the second pass rebuilt instead of reloading; cache path not exercised"
        )

    assert second["files_indexed"] == first["files_indexed"], (
        f"the same index reports {first['files_indexed']} files when built and "
        f"{second['files_indexed']} when reloaded; the reload path is counting "
        "chunks as files"
    )


def test_files_indexed_matches_the_files_actually_in_the_index(
    repo_with_oversized_sources,
):
    """Cross-check the reported count against the engine's own contents.

    Scoped to distinct files on purpose. Whether a byte-identical duplicate
    should count is a separate, pre-existing disagreement between the two ingest
    paths: the pure-Python path counts only ``status == "ingested"`` and so
    excludes duplicates, while the Rust ``batch_ingest`` path returns aggregate
    counts only and cannot attribute duplicates back to a file. That is not what
    this test is about, and mixing the two would make a failure ambiguous.
    """
    previous = auto_index.MAX_FILES
    auto_index.MAX_FILES = 5
    engine = EntrolyEngine()
    try:
        result = auto_index.auto_index(engine)
    finally:
        auto_index.MAX_FILES = previous

    if engine._use_rust:
        fragments = engine._rust.export_fragments()
        sources = {f.get("source", "") for f in fragments if f.get("source")}
    else:
        sources = {
            getattr(f, "source", "") for f in engine._fragments.values()
        }
        sources = {s for s in sources if s}

    distinct_files = {_logical_rel_path(s) for s in sources}

    assert result["files_indexed"] == len(distinct_files), (
        f"reported {result['files_indexed']} files but the index holds "
        f"{len(distinct_files)} distinct source files "
        f"({len(sources)} fragment sources)"
    )
    # And the fixture must actually have exercised chunking, or this proves
    # nothing about the defect.
    assert len(sources) > len(distinct_files), (
        "no file was chunked, so this fixture cannot distinguish a chunk count "
        "from a file count; raise the file size"
    )
