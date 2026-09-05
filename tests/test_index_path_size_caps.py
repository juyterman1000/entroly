"""The per-file size cap must treat a top-level directory like a nested one.

`_max_bytes_for_path` grants source directories a larger budget (192 KiB rather
than 50 KiB) because, as its docstring puts it, skipping real implementation
files "creates false economy: the optimizer saves tokens but cannot select the
code that actually answers the query".

Both marker lists are written with surrounding slashes -- ``/src/``, ``/dist/``
-- so that ``mysrc/`` does not match. But the paths passed in are
workspace-relative and have no leading slash, so the markers only ever matched
*nested* directories. Measured before the fix:

    src/big.py          ->  50 KiB      pkg/src/big.py  -> 192 KiB
    lib/big.py          ->  50 KiB      server/app.py   ->  50 KiB

and the exclusion list failed the same way in reverse: ``dist/src/x.js`` matched
``/src/`` but not ``/dist/``, so a generated file received the *larger* budget.

A top-level ``src/`` is the most common project layout there is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from entroly.auto_index import (  # noqa: E402
    MAX_FILE_BYTES,
    SOURCE_FILE_SOFT_MAX_BYTES,
    _max_bytes_for_path,
)


@pytest.fixture(autouse=True)
def _default_caps(monkeypatch):
    """Pin the env these assertions are about.

    ``_max_bytes_for_path`` returns ``MAX_FILE_BYTES`` immediately when
    ``ENTROLY_MAX_FILE_BYTES`` is set -- an explicit override, and correct
    behaviour. These tests are about the *default* rule, so they have to say so.

    Not hypothetical: importing ``benchmarks/contextbench_floor.py`` used to set
    that variable at module import, and ``tests/test_contextbench_runner_safety``
    imports it. In a full-suite run every test after that point saw a
    500,000-byte cap. These tests passed alone and failed in the suite, which is
    how the leak was found.
    """
    monkeypatch.delenv("ENTROLY_MAX_FILE_BYTES", raising=False)
    monkeypatch.delenv("ENTROLY_MAX_SOURCE_FILE_BYTES", raising=False)


@pytest.mark.parametrize(
    "top_level,nested",
    [
        ("src/big.py", "pkg/src/big.py"),
        ("lib/big.py", "pkg/lib/big.py"),
        ("core/big.py", "pkg/core/big.py"),
        ("server/app.py", "pkg/server/app.py"),
        ("worker/job.py", "pkg/worker/job.py"),
        ("services/api.py", "pkg/services/api.py"),
        ("packages/a/index.js", "repo/packages/a/index.js"),
    ],
)
def test_depth_does_not_change_the_budget(top_level, nested):
    """The same directory name must grant the same budget at any depth."""
    assert _max_bytes_for_path(top_level) == _max_bytes_for_path(nested), (
        f"{top_level} gets {_max_bytes_for_path(top_level)} bytes but "
        f"{nested} gets {_max_bytes_for_path(nested)}; the marker only matches "
        "when the directory is nested"
    )
    assert _max_bytes_for_path(top_level) == SOURCE_FILE_SOFT_MAX_BYTES


@pytest.mark.parametrize(
    "path",
    [
        "dist/src/x.js",
        "build/src/x.js",
        "generated/src/x.py",
        "src/generated/x.py",
        "pkg/dist/x.js",
        "src/__snapshots__/x.js",
    ],
)
def test_generated_output_never_gets_the_larger_budget(path):
    """The exclusion list has to win over the source-directory list.

    ``dist/src/x.js`` is the case that inverted: it matched ``/src/`` because the
    marker appears mid-path, but missed ``/dist/`` because that one was at the
    start.
    """
    assert _max_bytes_for_path(path) == MAX_FILE_BYTES, (
        f"{path} was granted {_max_bytes_for_path(path)} bytes; generated "
        "output must stay on the conservative cap"
    )


@pytest.mark.parametrize("path", ["mysrc/big.py", "resources/big.py", "srcs/big.py"])
def test_the_slashes_still_do_their_job(path):
    """Normalising must not turn the markers into bare substring matches."""
    assert _max_bytes_for_path(path) == MAX_FILE_BYTES, (
        f"{path} matched a source-directory marker it should not; the "
        "surrounding slashes exist to prevent exactly this"
    )


def test_non_source_files_are_unaffected():
    for path in ("src/README.md", "src/data.csv", "lib/notes.txt"):
        assert _max_bytes_for_path(path) == MAX_FILE_BYTES
