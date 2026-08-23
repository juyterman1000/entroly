"""The repository becomes reachable as graph nodes and edges.

Section 4.1 of the master implementation prompt requires
``Repository CONTAINS File DEFINES Symbol`` and ``File IMPORTS File``. Section
4.2 requires that materialization stay bounded and store references rather than
copies. These tests hold both halves: the projection produces the right shape,
and it refuses to grow without limit or to smuggle source text into graph state.

The last test applies the projection to a real `WorkGraph` and reads the nodes
back, because a payload that merely looks correct proves nothing about whether
the engine accepts it.
"""

from __future__ import annotations

import pytest

from entroly.repository_intelligence.graph_identity import (
    file_node_id,
    repository_node_id,
    symbol_node_id,
)
from entroly.repository_intelligence.graph_projection import (
    apply_repository_scope,
    project_repository_scope,
)
from entroly.repository_intelligence.models import FileRecord, Symbol
from entroly.work_graph import WorkGraph, WorkGraphUnavailableError, stable_node_id

REPO = "repo:demo"


def _skip_without_native() -> None:
    try:
        stable_node_id("file", "repo:probe", "probe.py")
    except WorkGraphUnavailableError as exc:  # pragma: no cover - environment
        pytest.skip(f"native work graph unavailable: {exc}")


def _file(path: str, is_test: bool = False) -> FileRecord:
    return FileRecord(
        path=path,
        language="python",
        sha256="a" * 64,
        byte_length=200,
        line_count=12,
        is_test=is_test,
    )


def _symbol(path: str, name: str) -> Symbol:
    return Symbol(
        symbol_id=f"{path}::{name}::function",
        path=path,
        name=name,
        qualified_name=name,
        kind="function",
        line_start=1,
        line_end=4,
        language="python",
    )


def _project(**kwargs):
    _skip_without_native()
    return project_repository_scope(REPO, observed_at_ms=1_000, **kwargs)


def test_projection_emits_the_section_four_one_shape() -> None:
    payload = _project(
        files=[_file("src/app.py")],
        symbols={"src/app.py": [_symbol("src/app.py", "handler")]},
        imports=[("src/app.py", "src/lib.py")],
    )
    ops = payload["operations"]

    kinds = [op.get("edge", {}).get("kind") for op in ops if op["op"] == "upsert_edge"]
    node_kinds = [op["node"]["kind"] for op in ops if op["op"] == "upsert_node"]

    # repository, the in-scope file, its symbol, and a boundary file for the
    # import target -- the engine rejects an edge to a node it does not know, so
    # the target must exist for `imports` to be assertable at all.
    assert node_kinds == ["repository", "file", "symbol", "file"]
    assert kinds == ["contains", "defines", "contains", "imports"]

    boundary = [
        op["node"] for op in ops
        if op["op"] == "upsert_node" and op["node"]["attributes"].get("boundary")
    ]
    assert len(boundary) == 1
    assert boundary[0]["attributes"]["path"] == "src/lib.py"
    # A boundary node carries no digest: it was never read, only referenced.
    assert "sha256" not in boundary[0]["attributes"]


def test_nodes_use_canonical_identity() -> None:
    payload = _project(
        files=[_file("src/app.py")],
        symbols={"src/app.py": [_symbol("src/app.py", "handler")]},
    )
    ids = [op["node"]["node_id"] for op in payload["operations"] if op["op"] == "upsert_node"]

    assert ids == [
        repository_node_id(REPO),
        file_node_id(REPO, "src/app.py"),
        symbol_node_id(REPO, "src/app.py::handler::function"),
    ]


def test_source_text_never_enters_graph_state() -> None:
    """A commitment, not a copy.

    Embedding content would make each observation grow the persisted graph
    without bound, and the digest already gives exact recovery.
    """
    payload = _project(files=[_file("src/app.py")])
    file_node = next(
        op["node"] for op in payload["operations"]
        if op["op"] == "upsert_node" and op["node"]["kind"] == "file"
    )

    assert file_node["attributes"]["sha256"] == "a" * 64
    assert "content" not in file_node["attributes"]
    assert "source" not in file_node["attributes"]
    assert "text" not in file_node["attributes"]


def test_everything_projected_is_observed_never_verified() -> None:
    """A directory listing is not a verified claim."""
    payload = _project(
        files=[_file("src/app.py")],
        symbols={"src/app.py": [_symbol("src/app.py", "handler")]},
        imports=[("src/app.py", "src/lib.py")],
    )

    for op in payload["operations"]:
        entity = op.get("node") or op.get("edge")
        assert entity["trust"] == "observed"


def test_file_cap_is_reported_not_silent() -> None:
    payload = _project(files=[_file(f"src/f{i}.py") for i in range(10)], max_files=3)

    assert payload["projection"]["files_projected"] == 3
    assert payload["projection"]["files_dropped"] == 7
    repository = payload["operations"][0]["node"]
    assert repository["attributes"]["active_scope_projection"] == payload["projection"]


def test_symbol_cap_is_reported_not_silent() -> None:
    payload = _project(
        files=[_file("src/app.py")],
        symbols={"src/app.py": [_symbol("src/app.py", f"f{i}") for i in range(10)]},
        max_symbols_per_file=2,
    )

    assert payload["projection"]["symbols_dropped"] == 8


def test_operation_cap_marks_the_result_truncated() -> None:
    payload = _project(
        files=[_file(f"src/f{i}.py") for i in range(50)],
        max_operations=10,
    )

    assert payload["projection"]["truncated"] is True
    assert payload["projection"]["operation_count"] <= 12


def test_edge_ids_are_the_engine_values_not_python_ones() -> None:
    """The engine requires `edge_id`, and it must be the canonical one.

    `WorkEdge.edge_id` has no serde default, so an event omitting it is
    rejected. Supplying it is therefore mandatory -- but it is still derived by
    calling the engine, never recomputed in Python, so the graph and the
    projection cannot disagree about what an edge is called.
    """
    from entroly.work_graph import stable_edge_id

    payload = _project(files=[_file("src/app.py")])
    edges = [op["edge"] for op in payload["operations"] if op["op"] == "upsert_edge"]

    assert edges
    for edge in edges:
        assert edge["edge_id"] == stable_edge_id(
            edge["from_node"], edge["kind"], edge["to_node"]
        )


def test_projection_is_accepted_by_a_real_graph() -> None:
    """End to end: the engine takes it, and the nodes come back addressable."""
    _skip_without_native()

    graph = WorkGraph(REPO)
    projection = apply_repository_scope(
        graph,
        REPO,
        files=[_file("src/app.py")],
        symbols={"src/app.py": [_symbol("src/app.py", "handler")]},
        imports=[("src/app.py", "src/lib.py")],
        observed_at_ms=1_000,
    )

    assert projection["truncated"] is False
    assert projection["boundary_files"] == 1

    # `export_state` is the persisted document and carries events only --
    # materialized nodes are derived and deliberately never serialized. The
    # materialized view is what proves the engine accepted and rebuilt them.
    snapshot = graph.snapshot()
    node_ids = {node["node_id"] for node in snapshot["nodes"]}
    edge_kinds = {edge["kind"] for edge in snapshot["edges"]}

    assert file_node_id(REPO, "src/app.py") in node_ids
    assert symbol_node_id(REPO, "src/app.py::handler::function") in node_ids
    assert file_node_id(REPO, "src/lib.py") in node_ids
    assert {"contains", "defines", "imports"} <= edge_kinds
    assert graph.event_count == 1
    assert graph.graph_commitment
