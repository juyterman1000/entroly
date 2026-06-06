"""Warm-start must not block engine construction / MCP handshake.

The persistent index is loaded on a background thread (the Rust load_index
releases the GIL), so the engine constructs instantly and a request that
arrives mid-load safely serializes on the engine lock instead of racing the
Rust &mut borrow. Guards the fix for the ~6.5s blocking cold start.
"""
from __future__ import annotations

import time
from pathlib import Path

from entroly.config import EntrolyConfig
from entroly.server import EntrolyEngine


def test_ephemeral_engine_is_warm_immediately():
    e = EntrolyEngine(EntrolyConfig(use_persistent_index=False))
    assert e.wait_until_warm(0) is True          # no background load needed
    e.optimize_context(1000, "hello world")       # works without an index


def test_persistent_engine_constructs_fast_and_warm_restores_index(tmp_path: Path):
    # 1) Build + persist a real index.
    e1 = EntrolyEngine(EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True))
    e1.wait_until_warm(15)
    for i in range(40):
        e1.ingest_fragment(f"def func_{i}(x):\n    return x * {i}", source=f"mod{i}.py")
    n = e1._rust.fragment_count()
    assert n > 0
    e1._rust.persist_index(str(Path(tmp_path) / "index.json.gz"))

    # 2) A new engine warm-loads it — construction must NOT block on the load.
    t = time.perf_counter()
    e2 = EntrolyEngine(EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True))
    construct_ms = (time.perf_counter() - t) * 1000
    assert construct_ms < 1500, f"construct blocked {construct_ms:.0f}ms — warm not backgrounded"

    # 3) A request issued during/after warm must not crash (it serializes on
    #    the engine lock rather than racing the background &mut borrow).
    e2.optimize_context(1000, "func")

    # 4) Warm completes and restores exactly the persisted fragments.
    assert e2.wait_until_warm(15) is True
    assert e2._rust.fragment_count() == n


def test_persistent_engine_has_lock_and_serialized_mutators(tmp_path: Path):
    e = EntrolyEngine(EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True))
    assert hasattr(e, "_rust_lock")
    assert e.wait_until_warm(15) is True
    # functools.wraps keeps the public name even though the call is wrapped.
    assert e.optimize_context.__name__ == "optimize_context"


def test_missing_index_warms_clean_no_crash(tmp_path: Path):
    # Empty checkpoint dir → background load finds no index, warms cleanly.
    e = EntrolyEngine(EntrolyConfig(checkpoint_dir=tmp_path, use_persistent_index=True))
    assert e.wait_until_warm(15) is True
    e.optimize_context(1000, "anything")
