"""Warm-start must not block engine construction / MCP handshake.

The persistent index loads LAZILY on first use (optimize/recall), on the
calling thread — never at construction — so the engine constructs instantly
and there is no background thread to race the Rust &mut borrow. Guards the fix
for the ~6.5s blocking cold start.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from entroly.config import EntrolyConfig
from entroly.server import EntrolyEngine


def _needs_rust(engine):
    if not getattr(engine, "_use_rust", False):
        pytest.skip("persistent warm-start index is native-engine only")


def test_ephemeral_engine_is_warm_immediately():
    e = EntrolyEngine(EntrolyConfig(use_persistent_index=False))
    assert e.wait_until_warm(0) is True          # nothing to load
    e.optimize_context(1000, "hello world")       # works without an index


def test_construction_does_not_load_index(tmp_path: Path):
    """Construction must be instant — the index is NOT loaded until first use."""
    e = EntrolyEngine(EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True))
    _needs_rust(e)
    assert e._index_loaded is False               # deferred, not loaded at construct
    e.wait_until_warm()                            # triggers the lazy load now
    assert e._index_loaded is True


def test_persistent_engine_constructs_fast_and_warm_restores_index(tmp_path: Path):
    # 1) Build + persist a real index.
    e1 = EntrolyEngine(EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True))
    _needs_rust(e1)
    for i in range(40):
        e1.ingest_fragment(f"def func_{i}(x):\n    return x * {i}", source=f"mod{i}.py")
    n = e1._rust.fragment_count()
    assert n > 0
    e1._rust.persist_index(str(Path(tmp_path) / "index.json.gz"))

    # 2) A new engine must construct WITHOUT loading the index (instant).
    t = time.perf_counter()
    e2 = EntrolyEngine(EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True))
    construct_ms = (time.perf_counter() - t) * 1000
    assert construct_ms < 1500, f"construct blocked {construct_ms:.0f}ms — index not deferred"

    # 3) First request triggers the lazy load and restores the persisted index.
    e2.optimize_context(1000, "func")             # lazy-loads on first call, no crash
    assert e2.wait_until_warm() is True
    assert e2._rust.fragment_count() == n


def test_persisted_index_roundtrip_preserves_native_selection(tmp_path: Path):
    """Cold and warm native engines must select identically from one snapshot."""
    cfg = EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True)
    cold = EntrolyEngine(cfg)
    _needs_rust(cold)
    cold._rust.set_exploration_rate(0.0)

    for i in range(96):
        cold.ingest_fragment(
            f"def helper_{i}(value):\n    return value + {i}  # generic pipeline helper",
            source=f"src/helpers/module_{i:03d}.py",
        )
    cold.ingest_fragment(
        "def refresh_access_token(token, expiry):\n"
        "    if expiry <= now():\n"
        "        return rotate_refresh_token(token)\n"
        "    return token",
        source="src/auth/token_refresh.py",
    )

    index_path = Path(tmp_path) / "index.json.gz"
    cold._rust.persist_index(str(index_path))
    query = "refresh access token expiry rotate token"
    cold_recall = [dict(item) for item in cold._rust.recall(query, 12)]
    cold_optimized = dict(cold._rust.optimize(700, query))

    warm = EntrolyEngine(cfg)
    _needs_rust(warm)
    warm._rust.set_exploration_rate(0.0)
    assert warm.wait_until_warm() is True
    warm_recall = [dict(item) for item in warm._rust.recall(query, 12)]
    warm_optimized = dict(warm._rust.optimize(700, query))

    assert warm._rust.fragment_count() == cold._rust.fragment_count()
    assert warm_recall == cold_recall
    assert warm_optimized.get("total_tokens") == cold_optimized.get("total_tokens")
    assert [dict(item) for item in warm_optimized.get("selected", [])] == [
        dict(item) for item in cold_optimized.get("selected", [])
    ]


def test_missing_index_loads_clean_no_crash(tmp_path: Path):
    # Empty checkpoint dir → first use finds no index, proceeds cleanly.
    e = EntrolyEngine(EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True))
    e.optimize_context(1000, "anything")          # works in native + pure-Python
    assert e.wait_until_warm() is True
