"""Generate the section 5 ownership/migration matrix from the actual branch.

The master implementation prompt (section 5) requires a machine-checkable
inventory classifying every production-relevant file into one of nine outcomes,
and section 22 requires a report proving no production-relevant file is
"forgotten". Section 5 also says to treat any existing map as useful evidence
rather than unquestionable truth -- which turned out to matter: at the time this
script was written `docs/repo_file_map.md` carried 280 rows of which 174 pointed
at files that no longer existed, and omitted 861 of 917 tracked Python modules.

So this reads the branch instead of a document.

What it will and will not decide
--------------------------------
Path and import structure are facts, and this script reports them as facts:
which crate a file belongs to, whether a Python module sits on the native
boundary, whether a module is reachable from a console entry point.

Whether a given Python module *should* become Rust is a judgement, and a
heuristic that pretends otherwise would manufacture exactly the false
completeness the audit gate exists to catch. Modules that carry computation but
show no clear orchestration or native signal are therefore classified
``review-required`` and listed as an actionable queue. ``--check`` reports them;
it fails only on ``unknown``, which means the classifier had no rule at all.

Usage
-----
    python scripts/ownership_matrix.py                 # write docs/OWNERSHIP_MATRIX.md
    python scripts/ownership_matrix.py --json out.json
    python scripts/ownership_matrix.py --check         # non-zero if anything is unknown
    python scripts/ownership_matrix.py --summary       # counts only
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Section 5's nine outcomes, verbatim in meaning.
RUST_SEMANTIC_OWNER = "rust-semantic-owner"
PYO3_BINDING = "pyo3-binding"
WASM_BINDING = "wasm-binding"
PYTHON_HOST = "python-host-orchestration"
NODE_HOST = "node-host-orchestration"
COMPAT_SHIM = "compatibility-shim"
LEGACY_DUPLICATE = "legacy-duplicate"
TESTS_DOCS_PACKAGING = "tests-fixtures-docs-packaging"
GENERATED = "generated-build-artifact"
REVIEW_REQUIRED = "review-required"
UNKNOWN = "unknown"

PRODUCTION_OUTCOMES = {
    RUST_SEMANTIC_OWNER,
    PYO3_BINDING,
    WASM_BINDING,
    PYTHON_HOST,
    NODE_HOST,
    COMPAT_SHIM,
    LEGACY_DUPLICATE,
    REVIEW_REQUIRED,
}

# Console entry points, from [project.scripts]. Reachability is computed from
# these rather than from cli.py -- see CLAUDE.md, which records that cli.py is
# not itself an entry point.
ENTRY_MODULES = (
    "entroly.docker_launcher_safe",
    "entroly.memory_cli",
    "entroly.compression_mcp",
    "entroly.__main__",
    "entroly.sdk",
    "entroly",
)

GENERATED_PATTERNS = (
    re.compile(r"(^|/)target/"),
    re.compile(r"(^|/)node_modules/"),
    re.compile(r"(^|/)dist/"),
    re.compile(r"(^|/)pkg/"),
    re.compile(r"(^|/)__pycache__/"),
    re.compile(r"\.egg-info(/|$)"),
    re.compile(r"\.min\.(js|css)$"),
    re.compile(r"\.sha256$"),
    re.compile(r"(^|/)vendor/"),
    re.compile(r"(^|/)\.venv/"),
)

# Filesystem, network, process and protocol surfaces. A Python module touching
# these is host orchestration by definition -- it is the glue section 3 says
# should stay in Python.
ORCHESTRATION_IMPORTS = {
    "os", "sys", "pathlib", "shutil", "subprocess", "tempfile", "glob",
    "socket", "http", "urllib", "asyncio", "threading", "multiprocessing",
    "sqlite3", "logging", "argparse", "signal", "webbrowser", "platform",
    "click", "typer", "rich", "httpx", "requests", "aiohttp", "flask",
    "fastapi", "uvicorn", "starlette", "mcp", "anthropic", "openai",
    "watchdog", "psutil", "yaml", "toml", "tomllib", "configparser",
}

# Pure-computation signals. Present without orchestration imports, a module is
# a candidate for Rust ownership and goes to the review queue.
SEMANTIC_IMPORTS = {"math", "statistics", "heapq", "bisect", "itertools", "array", "decimal", "fractions"}

# Engine primitives may deliberately remain internal until a calibrated consumer
# needs them. Internal is an explicit ownership decision, not a delivery failure.
ENGINE_INTERNAL_PRIMITIVES: dict[str, str] = {
    "simhash_wide": (
        "versioned internal SimHash estimator primitive; retained for calibrated "
        "future consumers, with no public Python/npm contract today"
    ),
}


@dataclass
class Row:
    """The thirteen fields section 5 asks for."""

    path: str
    current_role: str
    runtime: str
    semantic_or_orchestration: str
    canonical_owner: str
    rust_module_if_shared: str
    python_surface: str
    wasm_node_surface: str
    tests: str
    public_entrypoints: str
    migration_status: str
    compatibility_risk: str
    notes: str = ""


def repository_files() -> list[str]:
    """Return tracked files plus non-ignored files created by the current work.

    The ownership gate must see a new delivery file before it is staged; otherwise
    a clean report can be generated while the proposed product surface is absent
    from the inventory.  Repository-local Codex state is intentionally outside
    the product audit and is never added to the matrix.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(
        line
        for line in out.splitlines()
        if line.strip() and not line.replace("\\", "/").startswith(".codex/")
    )


def is_generated(path: str) -> bool:
    return any(pattern.search(path) for pattern in GENERATED_PATTERNS)


def rust_modules(crate: str) -> set[str]:
    src = REPO_ROOT / crate / "src"
    if not src.is_dir():
        return set()
    return {p.stem for p in src.glob("*.rs") if p.stem != "lib"}


def crate_references(crate: str) -> set[str]:
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


def engine_dependency_graph() -> dict[str, set[str]]:
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
        for match in re.finditer(r"\bcrate::([a-z0-9_]+)", source):
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


def python_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import -- internal, not a signal
                continue
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def reachable_python_modules() -> set[str]:
    """Modules reachable from the declared console entry points."""
    seen: set[str] = set()
    queue = [m for m in ENTRY_MODULES]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        rel = module.replace(".", "/")
        for candidate in (REPO_ROOT / f"{rel}.py", REPO_ROOT / rel / "__init__.py"):
            if not candidate.is_file():
                continue
            try:
                tree = ast.parse(candidate.read_text(encoding="utf-8", errors="replace"))
            except (OSError, SyntaxError):
                continue
            pkg = module if candidate.name == "__init__.py" else module.rsplit(".", 1)[0]
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("entroly"):
                            queue.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = pkg
                        for _ in range(node.level - 1):
                            base = base.rsplit(".", 1)[0] if "." in base else base
                        target = f"{base}.{node.module}" if node.module else base
                        queue.append(target)
                    elif node.module and node.module.startswith("entroly"):
                        queue.append(node.module)
    return seen


def classify(path: str, ctx: dict) -> Row:
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]

    if is_generated(p):
        return Row(p, "generated or vendored artifact", "-", "neither", GENERATED,
                   "-", "-", "-", "-", "-", "excluded", "none",
                   "excluded from semantic migration per section 4.4")

    # ---- Rust ------------------------------------------------------------
    if p.startswith("entroly-engine/src/") and p.endswith(".rs"):
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
        elif module in ENGINE_INTERNAL_PRIMITIVES:
            status = "internal"
            note = ENGINE_INTERNAL_PRIMITIVES[module]
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

    if p.startswith("entroly-core/src/") and p.endswith(".rs"):
        return Row(p, "PyO3 conversion/export", "rust", "orchestration", PYO3_BINDING,
                   "-", "entroly_core", "-", "cargo test --lib / pytest", "import entroly_core",
                   "canonical", "medium", "must stay transport-only")

    if p.startswith("entroly-wasm/src/") and p.endswith(".rs"):
        return Row(p, "wasm-bindgen conversion/export", "rust", "orchestration", WASM_BINDING,
                   "-", "-", "entroly-wasm", "wasm-pack test", "npm package",
                   "canonical", "medium", "must stay transport-only")

    if p.startswith("entroly-qccr/src/") and p.endswith(".rs"):
        return Row(p, "QCCR semantic owner", "rust", "semantic", RUST_SEMANTIC_OWNER,
                   "qccr", "-", "entroly-wasm", "cargo test --lib", "-",
                   "canonical", "medium", "")

    # ---- Tests, docs, packaging -----------------------------------------
    if p.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py"):
        return Row(p, "test", "python", "neither", TESTS_DOCS_PACKAGING,
                   "-", "-", "-", "self", "-", "n/a", "none",
                   "maps to the behaviour it protects")

    if p.startswith(("docs/", "benchmarks/", "examples/", ".github/")) or name.endswith(
        (".md", ".yml", ".yaml", ".toml", ".cfg", ".txt", ".svg", ".html", ".css", ".jsonl", ".lock")
    ):
        return Row(p, "documentation, packaging or CI", "-", "neither", TESTS_DOCS_PACKAGING,
                   "-", "-", "-", "-", "-", "n/a", "low", "")

    # ---- Node ------------------------------------------------------------
    if name.endswith((".js", ".ts", ".mjs", ".cjs")):
        return Row(p, "Node environment/integration glue", "node", "orchestration", NODE_HOST,
                   "-", "-", "npm", "npm test", "npm package",
                   "canonical", "medium", "")

    # ---- Python ----------------------------------------------------------
    if name.endswith(".py"):
        full = REPO_ROOT / p
        imports = python_imports(full)
        module_name = p[:-3].replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        reachable = module_name in ctx["reachable"]
        native = "entroly_core" in imports
        orchestration = bool(imports & ORCHESTRATION_IMPORTS)

        if p.startswith("scripts/"):
            return Row(p, "developer/release tooling", "python", "orchestration", PYTHON_HOST,
                       "-", "-", "-", "-", "-", "canonical", "low",
                       "not shipped in the wheel")

        if native:
            return Row(p, "Python surface over the native engine", "python", "orchestration",
                       PYTHON_HOST, "-", module_name, "-",
                       "pytest", "yes" if reachable else "-",
                       "canonical", "high",
                       "on the native boundary; must keep a pure-Python fallback")

        if orchestration:
            return Row(p, "host orchestration", "python", "orchestration", PYTHON_HOST,
                       "-", module_name, "-", "pytest", "yes" if reachable else "-",
                       "canonical", "medium" if reachable else "low", "")

        if imports & SEMANTIC_IMPORTS:
            return Row(p, "computation with no host or native signal", "python", "semantic?",
                       REVIEW_REQUIRED, "-", module_name, "-", "pytest",
                       "yes" if reachable else "-", "review-required",
                       "high" if reachable else "medium",
                       "candidate Rust owner; classification needs a human decision")

        return Row(p, "python module", "python", "orchestration", PYTHON_HOST,
                   "-", module_name, "-", "pytest", "yes" if reachable else "-",
                   "canonical", "low",
                   "no orchestration, native or computation signal; treated as glue")

    if name.endswith((".json", ".jsonc", ".sql", ".xml", ".cff", ".ipynb", ".tape")):
        return Row(p, "configuration, fixture or site data", "-", "neither", TESTS_DOCS_PACKAGING,
                   "-", "-", "-", "-", "-", "n/a", "low", "")

    # Binary or archive blobs committed as source. Section 22 question 10 asks
    # whether generated artifacts are being treated as source; these are the
    # answer, so they are reported rather than quietly bucketed as packaging.
    if name.endswith((".tar.gz", ".tgz", ".zip", ".whl", ".mcpb", ".wasm", ".so", ".pyd", ".dll")):
        return Row(p, "committed build artifact", "-", "neither", GENERATED,
                   "-", "-", "-", "-", "-", "excluded", "medium",
                   "binary artifact tracked in git; should be produced by the build, not committed")

    # Shell, container and packaging recipes: host-side operational glue.
    if (
        name.endswith((".sh", ".bash", ".ps1", ".rb", ".bat"))
        or name.startswith("Dockerfile")
        or "/git-hooks/" in p
        or p.startswith(".githooks/")
    ):
        return Row(p, "build, install or CI operational script", "-", "orchestration",
                   PYTHON_HOST if name.endswith((".sh", ".bash")) else TESTS_DOCS_PACKAGING,
                   "-", "-", "-", "-", "-", "canonical", "low",
                   "operational glue; no shared semantics")

    # Repository support files: ignore rules, licences, notices.
    if name in {"LICENSE", "NOTICE", "COPYING", "AUTHORS"} or name.startswith(".") or "ignore" in name:
        return Row(p, "repository support or licence file", "-", "neither", TESTS_DOCS_PACKAGING,
                   "-", "-", "-", "-", "-", "n/a", "none", "")

    return Row(p, "unclassified", "-", "unknown", UNKNOWN,
               "-", "-", "-", "-", "-", "unknown", "unknown",
               "no classification rule matched")


def build_context() -> dict:
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
    return [classify(path, ctx) for path in repository_files()]


def render_markdown(rows: list[Row]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.canonical_owner] = counts.get(row.canonical_owner, 0) + 1

    lines: list[str] = []
    lines.append("# Ownership / Migration Matrix")
    lines.append("")
    lines.append("Generated by `scripts/ownership_matrix.py` from tracked and non-ignored")
    lines.append("repository files; local `.codex/` task state is excluded.")
    lines.append("Regenerate after any move; `--check` fails on unclassified files.")
    lines.append("")
    lines.append("Section 5 of the master implementation prompt requires this inventory and")
    lines.append("says to treat existing maps as evidence rather than truth. It is machine-")
    lines.append("generated for that reason: `docs/repo_file_map.md` had drifted to 174 dead")
    lines.append("rows out of 280 and omitted 861 of 917 tracked Python modules.")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append("| Outcome | Files |")
    lines.append("|---|---:|")
    for key in sorted(counts, key=lambda k: (-counts[k], k)):
        lines.append(f"| `{key}` | {counts[key]} |")
    lines.append(f"| **total** | **{len(rows)}** |")
    lines.append("")

    review = [r for r in rows if r.canonical_owner == REVIEW_REQUIRED]
    unknown = [r for r in rows if r.canonical_owner == UNKNOWN]
    partial = [r for r in rows if r.migration_status == "partial-parity"]
    unexposed = [r for r in rows if r.migration_status == "unexposed"]

    lines.append("## Actionable queues")
    lines.append("")
    lines.append(f"* **unknown ownership: {len(unknown)}** — must be zero for the section 24 gate.")
    lines.append(f"* **review-required: {len(review)}** — computation with no host or native signal.")
    lines.append(f"* **partial parity: {len(partial)}** — engine modules reachable from one delivery runtime only.")
    lines.append(f"* **unexposed engine modules: {len(unexposed)}** — reachable from no Python/npm delivery root after engine dependency closure.")
    lines.append("")
    for label, bucket in (("Unknown", unknown), ("Review required", review),
                          ("Partial parity", partial), ("Unexposed", unexposed)):
        if not bucket:
            continue
        lines.append(f"### {label}")
        lines.append("")
        for row in bucket:
            note = f" — {row.notes}" if row.notes else ""
            lines.append(f"* `{row.path}`{note}")
        lines.append("")

    lines.append("## Full matrix")
    lines.append("")
    header = ["path", "current_role", "runtime", "semantic_or_orchestration",
              "canonical_owner", "rust_module_if_shared", "python_surface",
              "wasm_node_surface", "tests", "public_entrypoints",
              "migration_status", "compatibility_risk", "notes"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        values = [getattr(row, key) for key in header]
        cells = [str(v).replace("|", "\\|") for v in values]
        cells[0] = f"`{cells[0]}`"
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", metavar="PATH", help="also write the matrix as JSON")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any file has unknown ownership")
    parser.add_argument("--summary", action="store_true", help="print counts only")
    parser.add_argument("--out", default="docs/OWNERSHIP_MATRIX.md",
                        help="markdown destination (default: docs/OWNERSHIP_MATRIX.md)")
    args = parser.parse_args()

    rows = build_rows()
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.canonical_owner] = counts.get(row.canonical_owner, 0) + 1

    if args.summary or args.check:
        for key in sorted(counts, key=lambda k: (-counts[k], k)):
            print(f"{counts[key]:6d}  {key}")
        print(f"{len(rows):6d}  total")

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json}")

    if not args.summary and not args.check:
        out = REPO_ROOT / args.out
        out.write_text(render_markdown(rows), encoding="utf-8")
        print(f"wrote {args.out} ({len(rows)} files)")

    if args.check:
        unknown = [r.path for r in rows if r.canonical_owner == UNKNOWN]
        if unknown:
            print(f"\nFAIL: {len(unknown)} file(s) with unknown ownership:", file=sys.stderr)
            for path in unknown[:40]:
                print(f"  {path}", file=sys.stderr)
            return 1
        review = sum(1 for r in rows if r.canonical_owner == REVIEW_REQUIRED)
        print(f"\nOK: every repository file is classified ({review} awaiting human review)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
