"""Library-safety contract: Entroly never owns the host process GC policy."""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

from entroly.config import EntrolyConfig
from entroly.engine import EntrolyEngine


def _restore_gc(enabled: bool) -> None:
    if enabled:
        gc.enable()
    else:
        gc.disable()


@pytest.mark.parametrize("enabled", [True, False])
def test_engine_construction_and_hot_paths_preserve_host_gc_policy(
    tmp_path: Path, enabled: bool
) -> None:
    original = gc.isenabled()
    try:
        _restore_gc(enabled)
        engine = EntrolyEngine(
            EntrolyConfig(
                checkpoint_dir=tmp_path / ("enabled" if enabled else "disabled"),
                use_persistent_index=False,
            )
        )
        assert gc.isenabled() is enabled

        result = engine.ingest_fragment(
            "def host_gc_policy_probe(): return 1",
            "gc_policy.py",
            8,
            False,
        )
        assert result.get("status") in {"ingested", "duplicate"}
        assert gc.isenabled() is enabled

        engine.advance_turn()
        assert gc.isenabled() is enabled
    finally:
        _restore_gc(original)


def test_engine_source_contains_no_process_global_gc_controls() -> None:
    source = (Path(__file__).resolve().parents[1] / "entroly" / "engine.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("gc.disable(", "gc.enable(", "gc.freeze(", "gc.collect("):
        assert forbidden not in source
