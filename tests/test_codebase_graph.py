"""Regression tests for scripts/codebase_graph.py.

The graph tool is used to decide what is safe to change and what is reachable
by a user, so its edge extraction and reachability must be correct rather than
approximately right. Every test builds a small synthetic package on disk and
asserts against a hand-computed expected graph.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "codebase_graph.py"
SPEC = importlib.util.spec_from_file_location("codebase_graph", SCRIPT)
assert SPEC and SPEC.loader
codebase_graph = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(codebase_graph)


def _write_pkg(root: Path, files: dict[str, str]) -> Path:
    """Materialise a synthetic package and return its root."""
    pkg = root / "pkg"
    for rel, body in files.items():
        target = pkg / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return pkg


# ── module naming ────────────────────────────────────────────────────────────


def test_module_name_strips_init_and_suffix(tmp_path):
    pkg = _write_pkg(tmp_path, {"__init__.py": "", "sub/__init__.py": "", "sub/leaf.py": ""})
    assert codebase_graph.module_name(pkg / "__init__.py", pkg) == "pkg"
    assert codebase_graph.module_name(pkg / "sub" / "__init__.py", pkg) == "pkg.sub"
    assert codebase_graph.module_name(pkg / "sub" / "leaf.py", pkg) == "pkg.sub.leaf"


def test_discover_skips_build_and_cache_directories(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        {
            "__init__.py": "",
            "real.py": "",
            "__pycache__/cached.py": "",
            "target/generated.py": "",
            "node_modules/dep.py": "",
        },
    )
    found = set(codebase_graph.discover_modules(pkg))
    assert found == {"pkg", "pkg.real"}


# ── import extraction ────────────────────────────────────────────────────────


def test_absolute_relative_and_nested_imports_are_all_found(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        {
            "__init__.py": "",
            "alpha.py": "",
            "beta.py": "",
            "sub/__init__.py": "",
            "sub/deep.py": "",
            "user.py": (
                "import pkg.alpha\n"
                "from pkg import beta\n"
                "from .sub import deep\n"
                "def lazy():\n"
                "    from pkg.sub.deep import thing\n"
                "    return thing\n"
            ),
        },
    )
    known = set(codebase_graph.discover_modules(pkg))
    internal, native = codebase_graph.module_imports(pkg / "user.py", "pkg.user", known)

    assert internal == {"pkg.alpha", "pkg.beta", "pkg.sub.deep"}
    assert native == set()


def test_from_package_import_symbol_resolves_to_parent_module(tmp_path):
    """`from pkg.alpha import CONSTANT` is an edge to pkg.alpha, not a new node."""
    pkg = _write_pkg(
        tmp_path,
        {"__init__.py": "", "alpha.py": "", "user.py": "from pkg.alpha import CONSTANT\n"},
    )
    known = set(codebase_graph.discover_modules(pkg))
    internal, _ = codebase_graph.module_imports(pkg / "user.py", "pkg.user", known)
    assert internal == {"pkg.alpha"}


def test_native_extension_imports_are_reported_separately(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        {"__init__.py": "", "user.py": "import entroly_core\nfrom entroly_qccr import x\n"},
    )
    known = set(codebase_graph.discover_modules(pkg))
    internal, native = codebase_graph.module_imports(pkg / "user.py", "pkg.user", known)
    assert internal == set()
    assert native == {"entroly_core", "entroly_qccr"}


def test_external_and_self_imports_are_not_edges(tmp_path):
    pkg = _write_pkg(
        tmp_path,
        {"__init__.py": "", "user.py": "import os\nimport json\nimport pkg.user\n"},
    )
    known = set(codebase_graph.discover_modules(pkg))
    internal, _ = codebase_graph.module_imports(pkg / "user.py", "pkg.user", known)
    assert internal == set()


def test_syntax_error_does_not_abort_the_graph(tmp_path):
    pkg = _write_pkg(tmp_path, {"__init__.py": "", "broken.py": "def (:\n"})
    known = set(codebase_graph.discover_modules(pkg))
    internal, native = codebase_graph.module_imports(pkg / "broken.py", "pkg.broken", known)
    assert internal == set() and native == set()


# ── graph algorithms ─────────────────────────────────────────────────────────


def test_cycles_are_detected_and_acyclic_graphs_report_none():
    cyclic = {"a": ["b"], "b": ["c"], "c": ["a"], "d": []}
    assert codebase_graph.strongly_connected(cyclic) == [["a", "b", "c"]]

    acyclic = {"a": ["b"], "b": ["c"], "c": []}
    assert codebase_graph.strongly_connected(acyclic) == []


def test_reachability_follows_transitive_edges_only():
    adjacency = {"root": ["mid"], "mid": ["leaf"], "leaf": [], "island": ["lonely"], "lonely": []}
    assert codebase_graph.reachable(["root"], adjacency) == {"root", "mid", "leaf"}


def test_reachability_terminates_on_cyclic_graphs():
    adjacency = {"a": ["b"], "b": ["a"]}
    assert codebase_graph.reachable(["a"], adjacency) == {"a", "b"}


def test_reaching_a_submodule_reaches_its_parent_packages():
    """Importing pkg.sub.leaf executes pkg/__init__.py and pkg/sub/__init__.py."""
    adjacency = {"pkg": [], "pkg.sub": [], "pkg.sub.leaf": [], "entry": ["pkg.sub.leaf"]}
    assert codebase_graph.reachable(["entry"], adjacency) == {
        "entry", "pkg.sub.leaf", "pkg.sub", "pkg",
    }


def test_parent_closure_ignores_packages_absent_from_the_graph():
    adjacency = {"pkg.sub.leaf": [], "entry": ["pkg.sub.leaf"]}
    assert codebase_graph.reachable(["entry"], adjacency) == {"entry", "pkg.sub.leaf"}


def test_pagerank_ranks_the_most_imported_module_highest():
    adjacency = {"hub": [], "a": ["hub"], "b": ["hub"], "c": ["hub"], "d": ["a"]}
    ranks = codebase_graph.pagerank(adjacency)
    assert max(ranks, key=ranks.__getitem__) == "hub"
    assert pytest.approx(sum(ranks.values()), abs=1e-6) == 1.0


def test_pagerank_handles_empty_graph():
    assert codebase_graph.pagerank({}) == {}


# ── entry points ─────────────────────────────────────────────────────────────


def test_entry_points_include_declared_console_scripts():
    """The console scripts in pyproject are the only ways into the package."""
    entries = codebase_graph.shipped_entry_points("entroly")

    assert "entroly" in entries
    assert "entroly.__main__" in entries
    assert "entroly.sdk" in entries
    # Declared in [project.scripts]; notably cli.py is NOT an entry point.
    assert "entroly.docker_launcher_safe" in entries


# ── end-to-end on the real package ───────────────────────────────────────────


def test_analyse_reports_a_connected_graph_of_the_real_package():
    pkg = Path(__file__).resolve().parent.parent / "entroly"
    report = codebase_graph.analyse(pkg)

    assert report["modules"] > 100
    assert report["edges"] > 100
    assert report["reachable"] <= report["modules"]
    # Every reachable module must appear in the adjacency map.
    assert set(report["adjacency"]).issuperset(report["unreachable"])
    # Hubs are sorted by descending score.
    scores = [score for _, score in report["hubs"]]
    assert scores == sorted(scores, reverse=True)


def test_unreachable_set_is_disjoint_from_entry_points():
    pkg = Path(__file__).resolve().parent.parent / "entroly"
    report = codebase_graph.analyse(pkg)
    assert not set(report["entry_points"]) & set(report["unreachable"])
