#!/usr/bin/env python3
"""Guarded one-shot repair for ownership-matrix Rust reachability.

The old matrix treated direct `entroly_engine::module` imports in binding crates
as if they were the whole product reachability graph. That mislabels private
helpers such as coordination_index (used by work_graph) as dead and rnr (used by
eicv) as npm-only. This transform adds engine-internal dependency closure and a
persistent regression test, then lets the canonical generator rebuild the doc.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ownership_matrix.py"
TEST = ROOT / "tests/test_ownership_matrix.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly 1 anchor, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = SCRIPT.read_text(encoding="utf-8")
    if "def engine_dependency_graph()" in text:
        print("ownership reachability repair already applied")
        return 0

    anchor = '''def crate_references(crate: str) -> set[str]:
    """Engine module names referenced from `crate`'s sources."""
    src = REPO_ROOT / crate / "src"
    found: set[str] = set()
    if not src.is_dir():
        return found
    for path in src.rglob("*.rs"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in re.finditer(r"entroly_engine::([a-z0-9_]+)", text):
            found.add(match.group(1))
    return found


'''
    replacement = anchor + '''def engine_dependency_graph() -> dict[str, set[str]]:
    """Direct top-level `crate::module` dependencies inside entroly-engine.

    This is deliberately structural rather than semantic: it answers which
    canonical Rust modules are reachable from another canonical Rust module.
    Public re-exports in lib.rs are not edges by themselves; executable module
    references are. The graph is later closed transitively from each delivery
    crate's direct engine roots.
    """
    src = REPO_ROOT / "entroly-engine" / "src"
    modules = rust_modules("entroly-engine")
    graph: dict[str, set[str]] = {module: set() for module in modules}
    if not src.is_dir():
        return graph
    for path in src.glob("*.rs"):
        module = path.stem
        if module == "lib" or module not in graph:
            continue
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in re.finditer(r"\\bcrate::([a-z0-9_]+)", source):
            dependency = match.group(1)
            if dependency in modules and dependency != module:
                graph[module].add(dependency)
    return graph


def transitive_engine_references(
    roots: set[str], graph: dict[str, set[str]]
) -> set[str]:
    """Return engine modules transitively reachable from host-crate roots."""
    seen: set[str] = set()
    queue = [module for module in roots if module in graph]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        queue.extend(graph.get(module, ()))
    return seen


'''
    text = replace_once(text, anchor, replacement, "crate reference helper")

    old_rust = '''    if p.startswith("entroly-engine/src/") and p.endswith(".rs"):
        module = name[:-3]
        in_core = module in ctx["core_refs"]
        in_wasm = module in ctx["wasm_refs"]
        if module == "lib":
            status = "canonical"
            note = "crate root"
        elif not in_core and not in_wasm:
            status = "unexposed"
            note = "reachable from no binding; dead unless another engine module uses it"
        elif in_core and in_wasm:
            status = "canonical"
            note = ""
        else:
            status = "partial-parity"
            note = "exposed to %s only" % ("Python" if in_core else "npm")
        return Row(
            p, "shared semantic owner", "rust", "semantic", RUST_SEMANTIC_OWNER,
            module,
            "entroly-core" if in_core else "-",
            "entroly-wasm" if in_wasm else "-",
            "cargo test --lib", "-", status,
            "high" if status == "partial-parity" else "medium", note,
        )
'''
    new_rust = '''    if p.startswith("entroly-engine/src/") and p.endswith(".rs"):
        module = name[:-3]
        core_direct = module in ctx["core_refs"]
        wasm_direct = module in ctx["wasm_refs"]
        in_core = module in ctx["core_reachable"]
        in_wasm = module in ctx["wasm_reachable"]
        engine_users = sorted(
            owner for owner, dependencies in ctx["engine_graph"].items()
            if module in dependencies
        )
        if module == "lib":
            status = "canonical"
            note = "crate root"
        elif not in_core and not in_wasm:
            status = "unexposed"
            if engine_users:
                note = (
                    "engine-internal dependency of " + ", ".join(engine_users)
                    + "; none of those owners is reachable from a delivery root"
                )
            else:
                note = "reachable from no Python/npm delivery root and no engine semantic owner"
        elif in_core and in_wasm:
            status = "canonical"
            paths: list[str] = []
            if not core_direct:
                paths.append("Python transitively")
            if not wasm_direct:
                paths.append("npm transitively")
            note = (
                "shared through engine dependency closure"
                + (" (" + ", ".join(paths) + ")" if paths else "")
                if paths else ""
            )
        else:
            status = "partial-parity"
            note = "reachable from %s delivery only" % ("Python" if in_core else "npm")
        return Row(
            p, "shared semantic owner", "rust", "semantic", RUST_SEMANTIC_OWNER,
            module,
            "entroly-core" if in_core else "-",
            "entroly-wasm" if in_wasm else "-",
            "cargo test --lib", "-", status,
            "high" if status == "partial-parity" else "medium", note,
        )
'''
    text = replace_once(text, old_rust, new_rust, "engine classification")

    old_build = '''def build_rows() -> list[Row]:
    ctx = {
        "core_refs": crate_references("entroly-core"),
        "wasm_refs": crate_references("entroly-wasm"),
        "reachable": reachable_python_modules(),
    }
    return [classify(path, ctx) for path in tracked_files()]
'''
    new_build = '''def build_context() -> dict:
    graph = engine_dependency_graph()
    core_refs = crate_references("entroly-core")
    wasm_refs = crate_references("entroly-wasm")
    return {
        "core_refs": core_refs,
        "wasm_refs": wasm_refs,
        "core_reachable": transitive_engine_references(core_refs, graph),
        "wasm_reachable": transitive_engine_references(wasm_refs, graph),
        "engine_graph": graph,
        "reachable": reachable_python_modules(),
    }


def build_rows() -> list[Row]:
    ctx = build_context()
    return [classify(path, ctx) for path in tracked_files()]
'''
    text = replace_once(text, old_build, new_build, "context builder")

    text = replace_once(
        text,
        '    lines.append(f"* **partial parity: {len(partial)}** — engine modules exposed to one runtime only.")\n'
        '    lines.append(f"* **unexposed engine modules: {len(unexposed)}** — reachable from no binding.")\n',
        '    lines.append(f"* **partial parity: {len(partial)}** — engine modules reachable from one delivery runtime only.")\n'
        '    lines.append(f"* **unexposed engine modules: {len(unexposed)}** — reachable from no Python/npm delivery root after engine dependency closure.")\n',
        "queue wording",
    )

    SCRIPT.write_text(text, encoding="utf-8")

    TEST.write_text('''"""Regression tests for the machine-generated ownership/migration matrix."""\n\nfrom __future__ import annotations\n\nfrom scripts import ownership_matrix as matrix\n\n\ndef _row(path: str):\n    return next(row for row in matrix.build_rows() if row.path == path)\n\n\ndef test_engine_dependency_closure_follows_private_helpers() -> None:\n    ctx = matrix.build_context()\n    assert "coordination_index" in ctx["engine_graph"]["work_graph"]\n    assert "coordination_index" in ctx["core_reachable"]\n    assert "coordination_index" in ctx["wasm_reachable"]\n    assert "rnr" in ctx["engine_graph"]["eicv"]\n    assert "rnr" in ctx["core_reachable"]\n    assert "rnr" in ctx["wasm_reachable"]\n\n\ndef test_private_engine_helpers_are_not_reported_dead_or_partial() -> None:\n    coordination = _row("entroly-engine/src/coordination_index.rs")\n    rnr = _row("entroly-engine/src/rnr.rs")\n    assert coordination.migration_status == "canonical"\n    assert coordination.python_surface == "entroly-core"\n    assert coordination.wasm_node_surface == "entroly-wasm"\n    assert rnr.migration_status == "canonical"\n    assert rnr.python_surface == "entroly-core"\n    assert rnr.wasm_node_surface == "entroly-wasm"\n\n\ndef test_unexposed_queue_means_no_transitive_delivery_path() -> None:\n    ctx = matrix.build_context()\n    for row in matrix.build_rows():\n        if row.migration_status != "unexposed":\n            continue\n        module = row.rust_module_if_shared\n        assert module not in ctx["core_reachable"]\n        assert module not in ctx["wasm_reachable"]\n''', encoding="utf-8")
    print("applied guarded ownership reachability repair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
