"""
Regression tests for engine isolation and auto_index API contract.

These tests lock in two architectural fixes that landed in v0.18.0:

  1. `EntrolyEngine(EntrolyConfig(use_persistent_index=False))` must be
     fully ephemeral — no load from, no write to, the shared warm-start
     index at `~/.entroly/checkpoints/<id>/index.json.gz`.

     Prior to this fix, every `EntrolyEngine()` silently auto-loaded ~400
     fragments from the user-home cache, contaminating tests and probes,
     and silently overwrote that file with whatever fragments the instance
     had — including from short-lived test runs that had no business
     polluting the global warm-start state.

  2. `auto_index()` must return a unified dict shape regardless of whether
     it freshly indexed or skipped (because the engine already had
     fragments). Callers should not have to know which path ran to read
     `files_indexed` / `total_tokens` / `duration_s`.

     Prior to this fix, the skip path returned only {status, reason,
     existing_fragments}, causing `KeyError` in every caller that read
     `files_indexed` on the skip path.

Both fixes are silent-correctness defects: code "worked" in the happy path
and broke subtly in others. Regression tests are the only defense.
"""
from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path

import pytest

from entroly.auto_index import auto_index
from entroly.config import EntrolyConfig
from entroly.server import EntrolyEngine


# ── 1. Engine isolation (use_persistent_index=False) ─────────────────


@pytest.fixture
def ephemeral_engine(tmp_path: Path) -> EntrolyEngine:
    """A fully ephemeral engine that cannot touch the user-home index."""
    return EntrolyEngine(EntrolyConfig(
        use_persistent_index=False,
        checkpoint_dir=tmp_path / "ckpt",
    ))


def test_ephemeral_engine_starts_empty(ephemeral_engine: EntrolyEngine):
    """An engine with use_persistent_index=False must start with zero
    fragments regardless of what's in the user-home warm-start index."""
    assert ephemeral_engine._rust.fragment_count() == 0


def test_ephemeral_engine_does_not_create_index_file(tmp_path: Path):
    """Ingest enough fragments to trigger auto-checkpoint, then assert that
    no `index.json.gz` was ever written into the engine's checkpoint dir.

    This is the load-bearing invariant: ephemeral engines MUST NOT corrupt
    the warm-start index of any other caller, even transitively via
    auto-checkpoint.
    """
    ckpt = tmp_path / "ckpt"
    engine = EntrolyEngine(EntrolyConfig(
        use_persistent_index=False,
        checkpoint_dir=ckpt,
        auto_checkpoint_interval=1,  # force checkpoint after every ingest
    ))
    # Auto-checkpoint fires inside ingest_fragment when should_auto_checkpoint
    # returns True. Drive enough operations to cross that threshold.
    for i in range(10):
        engine.ingest_fragment(
            content=f"def fn_{i}(): return {i}",
            source=f"test://probe-{i}.py",
            token_count=10,
            is_pinned=False,
        )

    # Manually trigger checkpoint to be thorough — proves the save path
    # is gated, not just that the threshold wasn't crossed.
    engine.checkpoint()

    index_file = ckpt / "index.json.gz"
    assert not index_file.exists(), (
        f"Ephemeral engine wrote {index_file} — this would corrupt the "
        "shared warm-start index for any other engine pointed at this dir."
    )


def test_two_ephemeral_engines_are_isolated(tmp_path: Path):
    """Two ephemeral engines pointing at the same checkpoint_dir must not
    see each other's fragments. (They share a path but neither uses it.)"""
    shared_ckpt = tmp_path / "shared_ckpt"
    a = EntrolyEngine(EntrolyConfig(
        use_persistent_index=False,
        checkpoint_dir=shared_ckpt,
    ))
    a.ingest_fragment(content="engine A only", source="a://only.py", token_count=5)
    a.checkpoint()

    b = EntrolyEngine(EntrolyConfig(
        use_persistent_index=False,
        checkpoint_dir=shared_ckpt,
    ))
    assert b._rust.fragment_count() == 0, (
        "Engine B should not see Engine A's fragments — both opted out "
        "of the shared index."
    )


def test_persistent_engine_writes_index_file(tmp_path: Path):
    """Positive control: the default (use_persistent_index=True) DOES write
    the index file AND the written content actually contains the ingested
    fragment. If this fails, the persistence feature regressed and the
    negative tests above would pass trivially (no save means no leak).

    Verifying contents (not just file existence) catches the subtler case
    where save() writes an empty file or a stale one.
    """
    ckpt = tmp_path / "ckpt"
    engine = EntrolyEngine(EntrolyConfig(
        use_persistent_index=True,
        checkpoint_dir=ckpt,
        auto_checkpoint_interval=1,
    ))
    unique_marker = "def fn_marker_5a3f8c(): return 42"
    engine.ingest_fragment(
        content=unique_marker,
        source="test://probe.py",
        token_count=10,
    )
    engine.checkpoint()

    index_file = ckpt / "index.json.gz"
    assert index_file.exists(), "Persistent engine must write index.json.gz"

    # Verify gzip is well-formed and contains our marker
    with gzip.open(index_file, "rt", encoding="utf-8") as f:
        body = f.read()
    assert "fn_marker_5a3f8c" in body, (
        "Saved index does not contain the just-ingested fragment — save path "
        "is writing stale or empty data."
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX-only permission semantics")
def test_persistent_index_not_world_readable(tmp_path: Path):
    """The persisted index can contain code content from the ingested
    workspace. On POSIX, ensure it's not world-readable.

    No-op on Windows where file modes don't map to the same security model.
    """
    ckpt = tmp_path / "ckpt"
    engine = EntrolyEngine(EntrolyConfig(
        use_persistent_index=True,
        checkpoint_dir=ckpt,
        auto_checkpoint_interval=1,
    ))
    engine.ingest_fragment(
        content="SECRET_TOKEN = 'never-leak'",
        source="test://secrets.py",
        token_count=5,
    )
    engine.checkpoint()
    mode = (ckpt / "index.json.gz").stat().st_mode & 0o777
    assert mode & 0o044 == 0, (
        f"index.json.gz is world-readable (mode={oct(mode)}); it may "
        "contain ingested code content — restrict to owner."
    )


def test_persistent_engine_loads_index_file(tmp_path: Path):
    """Positive control: writing then re-opening with the same checkpoint
    dir warm-starts. If this breaks, the persistence feature regressed."""
    ckpt = tmp_path / "ckpt"
    a = EntrolyEngine(EntrolyConfig(
        use_persistent_index=True,
        checkpoint_dir=ckpt,
        auto_checkpoint_interval=1,
    ))
    a.ingest_fragment(
        content="def persistent_one(): return 1",
        source="test://persistent.py",
        token_count=15,
    )
    a.checkpoint()
    expected_count = a._rust.fragment_count()
    assert expected_count > 0  # sanity

    b = EntrolyEngine(EntrolyConfig(
        use_persistent_index=True,
        checkpoint_dir=ckpt,
    ))
    assert b._rust.fragment_count() == expected_count, (
        "Engine B should warm-start from Engine A's persisted index."
    )


# ── 2. auto_index unified return shape ───────────────────────────────


REQUIRED_KEYS = {
    "status",
    "files_indexed",
    "total_tokens",
    "beliefs_attached",
    "duration_s",
    "discovery_method",
    "project_dir",
}


def test_auto_index_fresh_path_returns_required_keys(tmp_path: Path):
    """Fresh-index path must return the unified key set."""
    engine = EntrolyEngine(EntrolyConfig(
        use_persistent_index=False,
        checkpoint_dir=tmp_path / "ckpt",
    ))
    # Create one real file to index
    (tmp_path / "foo.py").write_text("def hello(): return 'world'\n")
    result = auto_index(engine, project_dir=str(tmp_path), force=True)
    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Fresh auto_index missing keys: {missing}"
    assert result["status"] == "indexed"
    assert isinstance(result["files_indexed"], int)
    assert isinstance(result["total_tokens"], int)


def test_auto_index_skip_path_returns_required_keys(tmp_path: Path):
    """Skip path (engine already populated) must return the SAME key set.

    This is the regression that broke `verify_claims.py` for v0.18.0 —
    callers were reading `files_indexed` directly and got KeyError on the
    skip path. The fix unified the schema.
    """
    engine = EntrolyEngine(EntrolyConfig(
        use_persistent_index=False,
        checkpoint_dir=tmp_path / "ckpt",
    ))
    # Pre-populate so auto_index will hit the skip path
    engine.ingest_fragment(
        content="def already_loaded(): pass",
        source="test://existing.py",
        token_count=10,
    )
    assert engine._rust.fragment_count() > 0

    result = auto_index(engine, project_dir=str(tmp_path))
    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Skip auto_index missing keys: {missing}"
    assert result["status"] == "skipped"
    assert result["reason"] == "persistent_index_loaded"

    # Honest stats on the skip path: files = distinct sources, tokens = sum
    assert result["files_indexed"] == 1, (
        "Skip path should report 1 file (one distinct source path)"
    )
    assert result["total_tokens"] == 10
    assert result["duration_s"] == 0.0
    assert result["discovery_method"] == "cache"
    # Skip-specific diagnostic still present for back-compat:
    assert result["existing_fragments"] == 1


def test_auto_index_force_overrides_skip(tmp_path: Path):
    """force=True bypasses the skip path even when fragments already exist."""
    engine = EntrolyEngine(EntrolyConfig(
        use_persistent_index=False,
        checkpoint_dir=tmp_path / "ckpt",
    ))
    engine.ingest_fragment(
        content="def x(): pass",
        source="test://x.py",
        token_count=5,
    )
    (tmp_path / "real.py").write_text("def real(): return 42\n")
    result = auto_index(engine, project_dir=str(tmp_path), force=True)
    assert result["status"] == "indexed"  # not "skipped"
