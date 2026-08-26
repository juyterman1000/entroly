"""Regression tests for the machine-generated ownership/migration matrix."""

from __future__ import annotations

from scripts import ownership_matrix as matrix


def _row(path: str):
    return next(row for row in matrix.build_rows() if row.path == path)


def test_engine_dependency_closure_follows_private_helpers() -> None:
    ctx = matrix.build_context()
    assert "coordination_index" in ctx["engine_graph"]["work_graph"]
    assert "coordination_index" in ctx["core_reachable"]
    assert "coordination_index" in ctx["wasm_reachable"]
    assert "rnr" in ctx["engine_graph"]["eicv"]
    assert "rnr" in ctx["core_reachable"]
    assert "rnr" in ctx["wasm_reachable"]


def test_private_engine_helpers_are_not_reported_dead_or_partial() -> None:
    coordination = _row("entroly-engine/src/coordination_index.rs")
    rnr = _row("entroly-engine/src/rnr.rs")
    assert coordination.migration_status == "canonical"
    assert coordination.python_surface == "entroly-core"
    assert coordination.wasm_node_surface == "entroly-wasm"
    assert rnr.migration_status == "canonical"
    assert rnr.python_surface == "entroly-core"
    assert rnr.wasm_node_surface == "entroly-wasm"


def test_unexposed_queue_means_no_transitive_delivery_path() -> None:
    ctx = matrix.build_context()
    for row in matrix.build_rows():
        if row.migration_status != "unexposed":
            continue
        module = row.rust_module_if_shared
        assert module not in ctx["core_reachable"]
        assert module not in ctx["wasm_reachable"]
