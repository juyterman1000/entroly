#!/usr/bin/env python3
"""Treat the Entroly package as a directed module graph and report its shape.

Nodes are Python modules under a package root; edges are static imports
(absolute and relative, including imports nested inside functions). The graph
answers questions that are expensive to answer by reading files:

* which modules are the real architectural hubs (PageRank over the import graph)
* which modules a user can actually reach from a shipped entry point
* where the import cycles are
* which modules cross the PyO3 boundary into the Rust core

The reachability report is the load-bearing one. ``[project.scripts]`` in
pyproject.toml, ``python -m entroly`` and ``import entroly`` are the only ways
into the package from outside. A module that no entry point reaches is not on
any path a user can trigger through the installed package, however many tests
import it directly.

Usage::

    python scripts/codebase_graph.py                  # human summary
    python scripts/codebase_graph.py --json out.json  # machine-readable
    python scripts/codebase_graph.py --check          # exit 1 if drift
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    tomllib = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIR_PARTS = {"__pycache__", "target", "node_modules", ".venv", "build", "dist"}
NATIVE_PREFIXES = ("entroly_core", "entroly_qccr")


# ── graph construction ───────────────────────────────────────────────────────


def module_name(path: Path, pkg_root: Path) -> str:
    rel = path.relative_to(pkg_root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_modules(pkg_root: Path) -> dict[str, Path]:
    return {
        module_name(f, pkg_root): f
        for f in sorted(pkg_root.rglob("*.py"))
        if not SKIP_DIR_PARTS.intersection(f.parts)
    }


def _resolve(target: str, known: set[str]) -> str | None:
    """Map a dotted import target onto the nearest known module.

    ``from entroly.pkg import thing`` may name either a submodule or a symbol
    inside ``entroly.pkg``; fall back to the parent when the full path is not a
    module of its own.
    """
    if target in known:
        return target
    parent = target.rsplit(".", 1)[0]
    return parent if parent in known else None


def module_imports(path: Path, self_mod: str, known: set[str]) -> tuple[set[str], set[str]]:
    """Return ``(internal_module_deps, native_extension_deps)`` for one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set(), set()

    internal: set[str] = set()
    native: set[str] = set()
    own_parts = self_mod.split(".")
    container = own_parts if path.name == "__init__.py" else own_parts[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(NATIVE_PREFIXES):
                    native.add(alias.name.split(".")[0])
                elif (hit := _resolve(alias.name, known)) is not None:
                    internal.add(hit)

        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = container[: len(container) - node.level + 1]
                target = ".".join([*base, node.module] if node.module else base)
            else:
                target = node.module or ""

            if target.startswith(NATIVE_PREFIXES):
                native.add(target.split(".")[0])
                continue
            if not target:
                continue

            # Resolve per imported name, not on `target` alone: in
            # `from pkg import beta` the real dependency is `pkg.beta` when that
            # submodule exists. _resolve falls back to the parent package when
            # the name is a symbol rather than a module, so both forms are
            # handled by the same pass.
            resolved = {
                hit
                for alias in node.names
                if (hit := _resolve(f"{target}.{alias.name}", known)) is not None
            }
            if resolved:
                internal |= resolved
            elif (hit := _resolve(target, known)) is not None:
                internal.add(hit)

    internal.discard(self_mod)
    return internal, native


def build_graph(pkg_root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, int]]:
    modules = discover_modules(pkg_root)
    known = set(modules)
    adjacency: dict[str, list[str]] = {}
    native: dict[str, list[str]] = {}
    line_counts: dict[str, int] = {}

    for name, path in modules.items():
        deps, nat = module_imports(path, name, known)
        adjacency[name] = sorted(deps)
        if nat:
            native[name] = sorted(nat)
        line_counts[name] = len(path.read_text(encoding="utf-8", errors="replace").splitlines())

    return adjacency, native, line_counts


# ── graph analysis ───────────────────────────────────────────────────────────


def shipped_entry_points(pkg: str) -> list[str]:
    """Every module a user can reach without importing a private path."""
    entries = {pkg, f"{pkg}.__main__", f"{pkg}.sdk"}
    pyproject = REPO_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return sorted(entries)

    text = pyproject.read_text(encoding="utf-8")
    if tomllib is not None:
        targets = tomllib.loads(text).get("project", {}).get("scripts", {}).values()
    else:  # 3.10: read just the [project.scripts] table we need
        block = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
        targets = re.findall(r'=\s*"([^"]+)"', block.group(1)) if block else []

    entries.update(target.split(":", 1)[0] for target in targets)
    return sorted(entries)


def _ancestors(module: str, known: set[str]) -> list[str]:
    """Parent packages of ``module`` that exist in the graph.

    Importing ``pkg.sub`` executes ``pkg/__init__.py``, so a reachable submodule
    always makes its parent packages reachable too. Modelled here rather than as
    graph edges, which would manufacture cycles between every package and its
    own children.
    """
    parts = module.split(".")
    return [
        candidate
        for i in range(1, len(parts))
        if (candidate := ".".join(parts[:i])) in known
    ]


def reachable(starts: list[str], adjacency: dict[str, list[str]]) -> set[str]:
    known = set(adjacency)
    seen: set[str] = set()
    queue = deque(starts)
    while queue:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(dep for dep in adjacency.get(node, ()) if dep not in seen)
        queue.extend(pkg for pkg in _ancestors(node, known) if pkg not in seen)
    return seen


def strongly_connected(adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Iterative Tarjan; returns only components larger than one node."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: dict[str, bool] = defaultdict(bool)
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in adjacency:
        if root in index:
            continue
        work = [(root, iter(adjacency.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True

        while work:
            node, children = work[-1]
            descended = False
            for child in children:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack[child] = True
                    work.append((child, iter(adjacency.get(child, ()))))
                    descended = True
                    break
                if on_stack[child]:
                    low[node] = min(low[node], index[child])
            if descended:
                continue

            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1:
                    components.append(sorted(component))

    return sorted(components, key=len, reverse=True)


def pagerank(adjacency: dict[str, list[str]], damping: float = 0.85, iterations: int = 60) -> dict[str, float]:
    nodes = list(adjacency)
    count = len(nodes)
    if not count:
        return {}
    rank = dict.fromkeys(nodes, 1.0 / count)
    inbound: dict[str, list[str]] = defaultdict(list)
    for source, targets in adjacency.items():
        for target in targets:
            inbound[target].append(source)
    out_degree = {node: len(adjacency[node]) for node in nodes}

    for _ in range(iterations):
        dangling = sum(rank[n] for n in nodes if not out_degree[n])
        rank = {
            node: (1 - damping) / count
            + damping
            * (
                sum(rank[src] / out_degree[src] for src in inbound.get(node, ()) if out_degree[src])
                + dangling / count
            )
            for node in nodes
        }
    return rank


def analyse(pkg_root: Path) -> dict[str, object]:
    pkg = pkg_root.name
    adjacency, native, lines = build_graph(pkg_root)
    entries = [e for e in shipped_entry_points(pkg) if e in adjacency]
    live = reachable(entries, adjacency)
    unreached = sorted(set(adjacency) - live)
    ranks = pagerank(adjacency)
    in_degree: dict[str, int] = defaultdict(int)
    for targets in adjacency.values():
        for target in targets:
            in_degree[target] += 1

    return {
        "package": pkg,
        "modules": len(adjacency),
        "edges": sum(len(v) for v in adjacency.values()),
        "lines": sum(lines.values()),
        "entry_points": entries,
        "reachable": len(live),
        "unreachable": unreached,
        "unreachable_lines": sum(lines.get(m, 0) for m in unreached),
        "cycles": strongly_connected(adjacency),
        "native_boundary": native,
        "hubs": sorted(ranks.items(), key=lambda kv: -kv[1])[:20],
        "most_imported": sorted(in_degree.items(), key=lambda kv: -kv[1])[:20],
        "widest_fanout": sorted(((k, len(v)) for k, v in adjacency.items()), key=lambda kv: -kv[1])[:15],
        "adjacency": adjacency,
    }


# ── reporting ────────────────────────────────────────────────────────────────


def render(report: dict[str, object]) -> str:
    out: list[str] = []
    w = out.append
    w(f"package            : {report['package']}")
    w(f"modules / edges    : {report['modules']} / {report['edges']}")
    w(f"total python lines : {report['lines']:,}")
    w(f"entry points       : {', '.join(report['entry_points'])}")  # type: ignore[arg-type]
    w(f"reachable modules  : {report['reachable']}/{report['modules']}")
    w(
        f"UNREACHABLE        : {len(report['unreachable'])} modules, "  # type: ignore[arg-type]
        f"{report['unreachable_lines']:,} lines"
    )
    w("")
    w("architectural hubs (PageRank over imports):")
    for name, score in report["hubs"][:10]:  # type: ignore[index]
        w(f"  {score:.5f}  {name}")
    w("")
    w(f"import cycles: {len(report['cycles'])}")  # type: ignore[arg-type]
    for cycle in report["cycles"][:5]:  # type: ignore[index]
        w(f"  [{len(cycle)}] {', '.join(cycle[:6])}{' …' if len(cycle) > 6 else ''}")
    w("")
    w(f"PyO3 / native boundary: {len(report['native_boundary'])} modules import the Rust core")  # type: ignore[arg-type]
    w("")
    w("modules NOT reachable from any shipped entry point:")
    for name in report["unreachable"]:  # type: ignore[index]
        w(f"  {name}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(REPO_ROOT / "entroly"))
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any module is unreachable from a shipped entry point",
    )
    args = parser.parse_args()

    report = analyse(Path(args.package).resolve())
    if args.json:
        args.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(render(report))

    if args.check and report["unreachable"]:
        print(
            f"\nFAIL: {len(report['unreachable'])} modules are unreachable "  # type: ignore[arg-type]
            "from every shipped entry point.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
