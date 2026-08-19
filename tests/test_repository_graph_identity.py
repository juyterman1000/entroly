"""Repository intelligence and the Work Graph must address the same artifacts.

Section 4.1 of the master implementation prompt requires the repository to be
graph-addressable as ``Repository CONTAINS File DEFINES Symbol`` with a stable
identity strategy. Two implementations of that shape existed -- the Rust Work
Graph with `stable_node_id`, and `entroly/repository_intelligence` with readable
`path::qualified::kind` keys -- and no way to move between them.

These tests pin the join: an artifact named by repository intelligence resolves
to exactly the node id the graph would use, computed by the engine rather than
recomputed in Python.
"""

from __future__ import annotations

import pytest

from entroly.repository_intelligence.graph_identity import (
    contains_edge_id,
    defines_edge_id,
    file_node_id,
    file_record_node_id,
    imports_edge_id,
    repository_node_id,
    symbol_node_id,
    symbol_record_node_id,
)
from entroly.repository_intelligence.models import FileRecord, Symbol
from entroly.work_graph import WorkGraphUnavailableError, stable_node_id

REPO = "repo:demo"


def _skip_without_native() -> None:
    try:
        stable_node_id("file", "repo:probe", "probe.py")
    except WorkGraphUnavailableError as exc:  # pragma: no cover - environment
        pytest.skip(f"native work graph unavailable: {exc}")


def _record() -> FileRecord:
    return FileRecord(
        path="src/app.py",
        language="python",
        sha256="0" * 64,
        byte_length=120,
        line_count=8,
        is_test=False,
    )


def _symbol() -> Symbol:
    return Symbol(
        symbol_id="src/app.py::App.handler::function",
        path="src/app.py",
        name="handler",
        qualified_name="App.handler",
        kind="function",
        line_start=3,
        line_end=6,
        language="python",
    )


def test_file_record_resolves_to_the_graph_file_node() -> None:
    """A FileRecord and a NodeKind::File for the same path are one node."""
    _skip_without_native()

    assert file_record_node_id(REPO, _record()) == stable_node_id(
        "file", REPO, "src/app.py"
    )


def test_symbol_resolves_to_the_graph_symbol_node() -> None:
    """The readable local key is the key the engine hashes, unchanged."""
    _skip_without_native()

    symbol = _symbol()

    assert symbol_record_node_id(REPO, symbol) == stable_node_id(
        "symbol", REPO, "src/app.py::App.handler::function"
    )


def test_local_symbol_id_is_left_alone() -> None:
    """Joining must not renumber what dozens of modules and caches rely on."""
    symbol = _symbol()

    assert symbol.symbol_id == "src/app.py::App.handler::function"


def test_a_file_and_its_symbol_are_different_nodes() -> None:
    _skip_without_native()

    assert file_node_id(REPO, "src/app.py") != symbol_node_id(REPO, "src/app.py")


def test_the_same_path_in_two_repositories_is_two_nodes() -> None:
    _skip_without_native()

    assert file_node_id("repo:a", "src/app.py") != file_node_id("repo:b", "src/app.py")


def test_section_four_one_edges_are_derivable() -> None:
    """Repository CONTAINS File DEFINES Symbol, and File IMPORTS File."""
    _skip_without_native()

    contains = contains_edge_id(REPO, "src/app.py")
    defines = defines_edge_id(REPO, "src/app.py", "src/app.py::App.handler::function")
    imports = imports_edge_id(REPO, "src/app.py", "src/lib.py")

    assert contains.startswith("edge:")
    assert defines.startswith("edge:")
    assert imports.startswith("edge:")
    assert len({contains, defines, imports}) == 3


def test_repository_node_is_addressable() -> None:
    _skip_without_native()

    assert repository_node_id(REPO) == stable_node_id("repository", REPO, REPO)


def test_identity_is_derived_not_reimplemented() -> None:
    """The Python layer must not carry its own copy of the hash.

    A second implementation would pass every test above on the day it was
    written and drift silently afterwards, which is the failure this module
    exists to end.
    """
    from entroly.repository_intelligence import graph_identity

    source = graph_identity.__doc__ or ""
    assert "not reimplemented" in source or "deliberately not reimplemented" in source

    import inspect

    body = inspect.getsource(graph_identity)
    assert "hashlib" not in body
    assert "sha256" not in body.replace("``sha256(", "").replace("sha256`", "")
