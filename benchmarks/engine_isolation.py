"""Enforced isolation for benchmark runs that call the live engine.

`EntrolyEngine` warm-starts from a persistent index under `ENTROLY_DIR`. In a
developer checkout that index holds this repository's own fragments, so a
benchmark that calls `optimize()` without isolating the directory does not
measure what it passed in. Observed directly:

    optimize([auth.py, billing.py, crypto.py], query="fix the login bug")

    without isolation -> ['auth.py',
                          'file:entroly-wasm/bin/entroly-wasm.js',
                          'file:entroly/npm-alias/index.d.ts']
    with isolation    -> ['auth.py', 'crypto.py', 'billing.py']

Two of the three "selected" fragments were never supplied, and `crypto.py` --
the file the query actually needed -- was displaced. The resulting numbers look
entirely plausible, which is what makes the failure dangerous.

This module makes isolation a precondition that fails closed, rather than a
convention a future runner can forget.
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["isolated_engine_dir", "assert_engine_isolated", "EngineNotIsolatedError"]

_ENV_VAR = "ENTROLY_DIR"


class EngineNotIsolatedError(RuntimeError):
    """Raised when the engine would warm-start from a shared index."""


@contextmanager
def isolated_engine_dir(
    prefix: str = "entroly-bench-", *, reload_modules: bool = False
) -> Iterator[Path]:
    """Run the enclosed block against a private, empty engine directory.

    `reload_modules` drops already-imported `entroly.*` modules so they re-read
    ENTROLY_DIR. It defaults to **off** because it is destructive: modules
    reimported this way are new objects, so any caller still holding a class,
    singleton, or store from the previous import silently ends up with two
    incompatible copies. Enabling it by default corrupted eight unrelated
    tests in the same process.

    A benchmark runner should isolate by running in its own process, which
    needs no module surgery. Reach for `reload_modules=True` only when nothing
    else in the process holds engine state.
    """
    previous = os.environ.get(_ENV_VAR)
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        os.environ[_ENV_VAR] = tmp
        if reload_modules:
            _drop_engine_modules()
        try:
            yield Path(tmp)
        finally:
            if previous is None:
                os.environ.pop(_ENV_VAR, None)
            else:
                os.environ[_ENV_VAR] = previous
            if reload_modules:
                _drop_engine_modules()


def assert_engine_isolated() -> Path:
    """Fail unless ENTROLY_DIR points at a private, empty directory.

    Called by a runner before the first engine call. An index that already
    holds fragments means the run would be contaminated, and a contaminated
    run is worse than no run: it produces a number nobody can invalidate by
    inspection.
    """
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        raise EngineNotIsolatedError(
            f"{_ENV_VAR} is unset, so the engine would warm-start from the "
            "default index and select fragments that were never passed in. "
            "Wrap the run in benchmarks.engine_isolation.isolated_engine_dir()."
        )

    path = Path(raw)
    index_files = []
    if path.exists():
        index_files = [
            entry
            for entry in path.rglob("*")
            if entry.is_file() and entry.suffix in {".json", ".gz"}
        ]
    if index_files:
        raise EngineNotIsolatedError(
            f"{_ENV_VAR}={path} already contains {len(index_files)} index "
            "file(s); a warm start would mix foreign fragments into the "
            "selection. Use a fresh directory per run."
        )
    return path


def _drop_engine_modules() -> None:
    """Forget imported engine modules so they re-read ENTROLY_DIR."""
    doomed = [
        name
        for name in sys.modules
        if name == "entroly" or name.startswith("entroly.")
    ]
    for name in doomed:
        sys.modules.pop(name, None)
