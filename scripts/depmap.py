"""Build a per-file dependency table for every Rust / Python / JS source file.

Walks the source tree, extracts internal imports, resolves them to file paths,
and reports orphans (zero in-edges, not reachable from any entrypoint, not
in a known dynamic-discovery directory).

Run from the repo root:
    python scripts/depmap.py
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDE = {".venv", "target", "node_modules", "pkg", ".cache", "__pycache__",
           ".entroly", "dist", "build", ".git", ".github"}

# ── Source enumeration ──────────────────────────────────────────────


def collect_sources() -> dict[str, list[Path]]:
    sources: dict[str, list[Path]] = {"rust": [], "python": [], "js": []}
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        # exclude any path containing one of the excluded dirs
        rel_parts = p.relative_to(ROOT).parts
        if any(seg in EXCLUDE for seg in rel_parts):
            continue
        if p.suffix == ".rs":
            sources["rust"].append(p)
        elif p.suffix == ".py":
            sources["python"].append(p)
        elif p.suffix == ".js":
            sources["js"].append(p)
    return sources


# ── Import extractors ───────────────────────────────────────────────

RUST_MOD = re.compile(r"^\s*(?:pub\s+)?mod\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;", re.M)
RUST_USE_CRATE = re.compile(r"^\s*use\s+crate::([a-zA-Z_][a-zA-Z0-9_]*)", re.M)
RUST_USE_SUPER = re.compile(r"^\s*use\s+super::([a-zA-Z_][a-zA-Z0-9_]*)", re.M)

PY_FROM = re.compile(r"^\s*from\s+([\w.]+)\s+import", re.M)
PY_IMPORT = re.compile(r"^\s*import\s+([\w.]+)", re.M)
PY_REL_FROM = re.compile(r"^\s*from\s+(\.+)([\w.]*)\s+import", re.M)

JS_REQUIRE = re.compile(r"""require\(\s*['"](\.[^'"]+)['"]\s*\)""")
JS_IMPORT = re.compile(r"""import\s+[^'";]+from\s+['"](\.[^'"]+)['"]""")

INTERNAL_PY_ROOTS = {"entroly", "entroly_core", "entroly_wasm", "bench",
                     "scripts", "tests"}


def extract_rust(p: Path) -> set[Path]:
    """Resolve `mod X;` and `use crate::X` references for a Rust file."""
    text = p.read_text(encoding="utf-8", errors="ignore")
    out: set[Path] = set()
    src_dir = _rust_src_root(p)
    if src_dir is None:
        return out

    self_dir = p.parent

    # mod X; — resolves relative to this file
    for m in RUST_MOD.findall(text):
        candidates = [
            self_dir / f"{m}.rs",
            self_dir / m / "mod.rs",
        ]
        # If file is lib.rs, mods are in the same dir (already covered).
        # If file is foo.rs, child mods can also live in foo/
        if p.stem != "lib" and p.stem != "mod":
            candidates += [
                self_dir / p.stem / f"{m}.rs",
                self_dir / p.stem / m / "mod.rs",
            ]
        for c in candidates:
            if c.exists():
                out.add(c.resolve())
                break

    # use crate::X — X is a top-level module in src/
    for u in RUST_USE_CRATE.findall(text):
        candidates = [src_dir / f"{u}.rs", src_dir / u / "mod.rs"]
        for c in candidates:
            if c.exists():
                out.add(c.resolve())
                break

    # use super::X — sibling of self_dir
    for u in RUST_USE_SUPER.findall(text):
        parent = self_dir.parent
        candidates = [parent / f"{u}.rs", parent / u / "mod.rs"]
        for c in candidates:
            if c.exists():
                out.add(c.resolve())
                break

    return out


def _rust_src_root(p: Path) -> Path | None:
    for ancestor in p.parents:
        if ancestor.name == "src" and (ancestor.parent / "Cargo.toml").exists():
            return ancestor
    return None


def extract_python(p: Path, all_py: set[Path]) -> set[Path]:
    """Resolve internal Python imports."""
    text = p.read_text(encoding="utf-8", errors="ignore")
    out: set[Path] = set()

    def resolve_dotted(mod: str) -> Path | None:
        # entroly.foo.bar -> entroly/foo/bar.py or entroly/foo/bar/__init__.py
        parts = mod.split(".")
        if not parts or parts[0] not in INTERNAL_PY_ROOTS:
            return None
        candidates = [
            ROOT / Path(*parts).with_suffix(".py"),
            ROOT / Path(*parts) / "__init__.py",
        ]
        for c in candidates:
            if c.exists() and c.resolve() in all_py:
                return c.resolve()
        return None

    for mod in PY_FROM.findall(text):
        if mod.startswith("."):
            continue  # handled below
        r = resolve_dotted(mod)
        if r:
            out.add(r)

    for mod in PY_IMPORT.findall(text):
        if mod.startswith("."):
            continue
        # could be "import entroly.foo as bar" — first token already captured
        r = resolve_dotted(mod)
        if r:
            out.add(r)

    # Relative: from .foo import / from ..foo import
    for dots, mod in PY_REL_FROM.findall(text):
        levels = len(dots)  # 1 == from . , 2 == from ..
        base = p.parent
        for _ in range(levels - 1):
            base = base.parent
        if mod:
            sub = base / Path(*mod.split("."))
            candidates = [sub.with_suffix(".py"), sub / "__init__.py"]
            for c in candidates:
                if c.exists() and c.resolve() in all_py:
                    out.add(c.resolve())
                    break
        else:
            # bare `from . import X` — without parsing the imported name,
            # mark the package __init__ as touched
            init = base / "__init__.py"
            if init.exists() and init.resolve() in all_py:
                out.add(init.resolve())
    return out


def extract_js(p: Path, all_js: set[Path]) -> set[Path]:
    """Resolve relative require/import paths."""
    text = p.read_text(encoding="utf-8", errors="ignore")
    out: set[Path] = set()
    raw = list(JS_REQUIRE.findall(text)) + list(JS_IMPORT.findall(text))
    for spec in raw:
        # spec like "./js/foo" or "../bar"
        target = (p.parent / spec).resolve()
        candidates = [
            target,
            target.with_suffix(".js"),
            Path(str(target) + ".js"),
            target / "index.js",
        ]
        for c in candidates:
            if c.exists() and c.resolve() in all_js:
                out.add(c.resolve())
                break
    return out


# ── Build graph + reachability ──────────────────────────────────────


def build_graph(sources: dict[str, list[Path]]) -> tuple[dict[Path, set[Path]], dict[Path, set[Path]]]:
    out_edges: dict[Path, set[Path]] = defaultdict(set)
    all_py = {p.resolve() for p in sources["python"]}
    all_js = {p.resolve() for p in sources["js"]}

    for p in sources["rust"]:
        out_edges[p.resolve()] = extract_rust(p)
    for p in sources["python"]:
        out_edges[p.resolve()] = extract_python(p, all_py)
    for p in sources["js"]:
        out_edges[p.resolve()] = extract_js(p, all_js)

    in_edges: dict[Path, set[Path]] = defaultdict(set)
    for src, dsts in out_edges.items():
        for d in dsts:
            in_edges[d].add(src)
    return out_edges, in_edges


def find_entrypoints(sources: dict[str, list[Path]]) -> set[Path]:
    eps: set[Path] = set()
    # Rust roots
    for p in sources["rust"]:
        if p.name == "lib.rs":
            eps.add(p.resolve())
    # Python: every test, every bench, every script, plus declared entrypoints
    for p in sources["python"]:
        rel = p.relative_to(ROOT).as_posix()
        if (rel.startswith("tests/") or "/tests/" in rel
                or rel.startswith("bench/") or "/bench/" in rel
                or rel.startswith("scripts/") or rel.startswith("examples/")):
            eps.add(p.resolve())
        if p.name in {"__init__.py", "cli.py", "server.py", "proxy.py",
                      "_docker_launcher.py", "__main__.py"}:
            eps.add(p.resolve())
        # any root-level python file
        if p.parent == ROOT:
            eps.add(p.resolve())
    # JS: index.js and all test_*.js
    for p in sources["js"]:
        if p.name == "index.js" or p.name.startswith("test_") or p.name == "cli.js":
            eps.add(p.resolve())
    return eps


def reachable(eps: set[Path], out_edges: dict[Path, set[Path]]) -> set[Path]:
    seen: set[Path] = set(eps)
    q: deque[Path] = deque(eps)
    while q:
        cur = q.popleft()
        for nxt in out_edges.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


# ── Status classifier ───────────────────────────────────────────────

DYNAMIC_DIRS = {"entroly/integrations"}


def classify(p: Path, in_deg: int, reach: bool, eps: set[Path]) -> str:
    if p in eps:
        return "entrypoint"
    rel = p.relative_to(ROOT).as_posix()
    if any(rel.startswith(d) for d in DYNAMIC_DIRS):
        return "dynamic" if not reach else "live"
    if reach:
        return "live"
    if in_deg > 0:
        # has callers but not from entrypoints — likely cycle or test-only
        return "live (non-entry)"
    return "orphan"


# ── Render ──────────────────────────────────────────────────────────


def loc(p: Path) -> int:
    try:
        return sum(1 for _ in p.read_text(encoding="utf-8", errors="ignore").splitlines())
    except OSError:
        return 0


def main() -> None:
    sources = collect_sources()
    print(f"# Source counts (after exclusions)")
    for k, v in sources.items():
        print(f"  {k:7s}: {len(v):4d} files")
    print(f"  TOTAL : {sum(len(v) for v in sources.values()):4d} files\n")

    out_edges, in_edges = build_graph(sources)
    eps = find_entrypoints(sources)
    reach = reachable(eps, out_edges)

    rows = []
    for kind, files in sources.items():
        for p in sorted(files):
            r = p.resolve()
            rows.append({
                "kind": kind,
                "path": p.relative_to(ROOT).as_posix(),
                "loc": loc(p),
                "out": len(out_edges.get(r, ())),
                "in": len(in_edges.get(r, ())),
                "reach": r in reach,
                "status": classify(r, len(in_edges.get(r, ())), r in reach, eps),
            })

    # Print table grouped by directory
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        d = "/".join(row["path"].split("/")[:-1]) or "."
        by_dir[d].append(row)

    print("# Per-file dependency table\n")
    for d in sorted(by_dir):
        files = by_dir[d]
        print(f"\n## {d}/ ({len(files)} files)\n")
        print("| File | LOC | out | in | reach | status |")
        print("|---|---:|---:|---:|:---:|---|")
        for row in sorted(files, key=lambda r: r["path"]):
            name = row["path"].split("/")[-1]
            mark = "Y" if row["reach"] else "-"
            print(f"| {name} | {row['loc']} | {row['out']} | {row['in']} | {mark} | {row['status']} |")

    # Orphans summary
    orphans = [r for r in rows if r["status"] == "orphan"]
    print(f"\n# Orphans ({len(orphans)})\n")
    if not orphans:
        print("None — every file is reachable from at least one entrypoint.")
    else:
        for r in sorted(orphans, key=lambda r: r["path"]):
            print(f"- `{r['path']}` ({r['loc']} LOC, in={r['in']}, out={r['out']})")


if __name__ == "__main__":
    main()
