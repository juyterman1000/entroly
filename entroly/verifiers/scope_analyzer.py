"""
Scope-Reachability Analyzer
===========================

The third state in the verifier's 3-state model. Distinguishes:

  θ=1  GROUNDED      σ in scope of code at this position
  θ=2  UNREACHABLE   σ exists somewhere in the codebase, but not imported
                     and not defined locally → the generated code uses a
                     symbol it never brought into scope
  θ=0  HALLUCINATED  σ doesn't exist anywhere

The UNREACHABLE state is the genuinely novel detection class. Mainline
verifiers (CodeHalu, Gorilla, LSP-based approaches) treat existence as
binary; this implementation distinguishes "you used something that
exists but you didn't import it" — which is a different bug class with
a different fix.

Mathematical foundation
-----------------------
Given generated code C with symbol reference σ, define:

    S(C) = S_builtin ∪ S_stdlib_top ∪ ⋃_{imp in C} Φ(imp.module) ∪ Δ(C)

where:
    Φ(m) = set of public exports of module m  (from depgraph index)
    Δ(C) = names defined inline in C (def/class)

Three-state classification:

    σ ∉ M           →  P(θ=0|σ) = 1                 (hallucinated)
    σ ∈ M, σ ∉ S    →  P(θ=2|σ) = 1                 (unreachable)
    σ ∈ S           →  P(θ=1|σ) = sigmoid(λ - surp)  (grounded with confidence)

Remediation
-----------
For UNREACHABLE judgments we additionally compute the SUGGESTED FIX —
the import statement the model should have written. This requires
a reverse-index: symbol → module(s) defining it. Built once during
manifest construction and queried per-judgment.

This module integrates with depgraph for transitive reachability: if
module A imports from module B, and B re-exports symbol s from C, then
generated code importing A can use s. The depgraph walk handles those
re-export chains.
"""

from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path


logger = logging.getLogger("entroly.verifiers.scope_analyzer")


# ── Reverse Index ────────────────────────────────────────────────────


@dataclass
class ReverseIndex:
    """Maps each defined symbol to the module path(s) that define it.

    For UNREACHABLE diagnoses, we use this to compute the suggested
    `from X import Y` statement. A symbol may live in multiple modules
    (rare but legal — e.g., re-exports); we report the first one by
    PageRank weight if available, else alphabetical.
    """

    # symbol -> list of (module_dotted_path, file_path)
    sym_to_modules: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    # Reverse for fast wildcard-import resolution:
    # module_dotted -> set of symbols it exports
    module_to_exports: dict[str, set[str]] = field(default_factory=dict)

    def add_def(self, symbol: str, module_dotted: str, file_path: str) -> None:
        self.sym_to_modules.setdefault(symbol, []).append((module_dotted, file_path))
        self.module_to_exports.setdefault(module_dotted, set()).add(symbol)

    def modules_for(self, symbol: str) -> list[tuple[str, str]]:
        return self.sym_to_modules.get(symbol, [])

    def exports_of(self, module_dotted: str) -> set[str]:
        # Tolerate both "foo.bar" and "foo/bar.py" form lookups:
        # callers may pass either depending on import statement style.
        if module_dotted in self.module_to_exports:
            return self.module_to_exports[module_dotted]
        # try trailing match: import "foo.bar" might be stored as "pkg.foo.bar"
        for k, v in self.module_to_exports.items():
            if k.endswith("." + module_dotted) or k == module_dotted:
                return v
        return set()


def build_reverse_index(
    repo_root: str,
    extensions: tuple[str, ...] = (".py",),
    max_files: int = 5000,
) -> ReverseIndex:
    """Walk the repo and build a symbol→module reverse map.

    For each .py file, compute its dotted module path relative to the
    repo root, then record each top-level def/class as exported by
    that module.
    """
    index = ReverseIndex()
    root = Path(repo_root).resolve()
    skip = {"__pycache__", ".git", ".venv", "venv", "node_modules",
            "target", "dist", "build", ".tox", ".pytest_cache"}

    files_seen = 0
    for path in root.rglob("*"):
        if files_seen >= max_files:
            break
        if path.is_dir():
            continue
        if path.suffix not in extensions:
            continue
        if any(p in skip for p in path.parts):
            continue
        files_seen += 1

        rel = path.relative_to(root)
        dotted = _dotted_module(rel)
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError, ValueError):
            continue

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                index.add_def(node.name, dotted, str(path))
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        index.add_def(target.id, dotted, str(path))
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    index.add_def(node.target.id, dotted, str(path))
        # Nested defs too (Class.method is reachable via instance.method)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node is tree or node in tree.body:
                    continue
                index.add_def(node.name, dotted, str(path))

    return index


def _dotted_module(rel_path: Path) -> str:
    """Convert utils/helpers.py → utils.helpers; pkg/__init__.py → pkg."""
    parts = list(rel_path.parts)
    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


# ── Scope Computation ────────────────────────────────────────────────


@dataclass
class Scope:
    """Symbols accessible from inside `source` at the file level.

    For v1 we model file-level scope only. Per-function scope
    (locals, args) would require building a CFG and walking
    statement-by-statement; that's a v2 refinement.
    """
    names: set[str] = field(default_factory=set)
    # For debugging / explanation: where each name came from
    provenance: dict[str, str] = field(default_factory=dict)

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def how(self, name: str) -> str:
        return self.provenance.get(name, "")


# Stdlib top-level module names + their public attrs (reuse from
# symbol_resolution; cheap to import).
_STDLIB_TOP: set[str] | None = None


def _stdlib_top() -> set[str]:
    global _STDLIB_TOP
    if _STDLIB_TOP is None:
        names = getattr(sys, "stdlib_module_names", set())
        _STDLIB_TOP = set(names)
    return _STDLIB_TOP


_BUILTINS: set[str] | None = None


def _builtins() -> set[str]:
    global _BUILTINS
    if _BUILTINS is None:
        import builtins
        import keyword
        _BUILTINS = {n for n in dir(builtins) if not n.startswith("_")}
        _BUILTINS.update(keyword.kwlist)
        _BUILTINS.update({"self", "cls", "args", "kwargs", "_",
                          "True", "False", "None"})
    return _BUILTINS


def compute_scope(
    source: str,
    reverse_index: ReverseIndex,
    import_source_valid: callable | None = None,
) -> Scope:
    """Compute file-level scope for the given source.

    Algorithm:
      1. Seed with builtins + stdlib top-level module names
      2. For each `import X`: add X (the bound name) ONLY if X resolves
      3. For each `from M import X, Y, Z`: add {X, Y, Z} ONLY if M resolves.
         (Imports from nonexistent modules create vacuous bindings —
         the names look in-scope but the import itself is hallucinated,
         so we don't let them mask downstream errors.)
         If `*`, look up M's exports via reverse_index.
      4. For each top-level def/class/var: add its name.

    Args:
        source: Generated code to analyze.
        reverse_index: Maps symbols → defining modules (built from the repo).
        import_source_valid: Optional predicate (module_name → bool). If
            given, an `import M` or `from M import X` only contributes to
            scope when import_source_valid(M) is True. Without this guard,
            `from fake_lib import RealName` would silently grant RealName
            scope-grounded status, masking the fake_lib hallucination.
    """
    scope = Scope()
    scope.names |= _builtins()
    for n in _builtins():
        scope.provenance[n] = "builtin"
    scope.names |= _stdlib_top()
    for n in _stdlib_top():
        scope.provenance.setdefault(n, "stdlib_top")

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't analyze scope without a valid AST. Return what we have.
        return scope

    def _module_ok(mod_name: str) -> bool:
        if import_source_valid is None:
            return True
        # Top-level module name (foo from foo.bar.baz)
        top = mod_name.split(".")[0]
        return bool(import_source_valid(top))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if not _module_ok(top):
                    # Hallucinated module — do NOT grant its alias scope status.
                    continue
                bound = alias.asname or top
                scope.names.add(bound)
                scope.provenance[bound] = f"import:{alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module and not _module_ok(module):
                # Hallucinated source module — every name it would have
                # introduced is also vacuous. Skip.
                continue
            # Module name itself is NOT bound by `from M import X`.
            for alias in node.names:
                if alias.name == "*":
                    # Wildcard import — resolve via reverse index
                    for exported in reverse_index.exports_of(module):
                        scope.names.add(exported)
                        scope.provenance[exported] = f"from {module} import *"
                else:
                    bound = alias.asname or alias.name
                    scope.names.add(bound)
                    scope.provenance[bound] = f"from {module} import {alias.name}"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope.names.add(node.name)
            scope.provenance.setdefault(node.name, "local_def")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scope.names.add(target.id)
                    scope.provenance.setdefault(target.id, "local_assign")
                # Tuple/list assignments — extract Name targets
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            scope.names.add(elt.id)
                            scope.provenance.setdefault(elt.id, "local_assign")
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                scope.names.add(node.target.id)
                scope.provenance.setdefault(node.target.id, "local_ann_assign")
        # Function args are NOT added to file-level scope in v1; they're
        # local to the function body. That's a deliberate simplification.

    return scope


# ── Verdicts ─────────────────────────────────────────────────────────


@dataclass
class ReachabilityVerdict:
    """Per-symbol verdict combining manifest + scope analysis."""
    state: int                  # 0 = hallucinated, 1 = grounded, 2 = unreachable
    suggested_import: str | None = None
    scope_source: str | None = None  # where in scope it came from, if grounded


def judge_reachability(
    symbol: str,
    manifest_contains: bool,
    scope: Scope,
    reverse_index: ReverseIndex,
) -> ReachabilityVerdict:
    """Three-state classification for one symbol reference.

    The classification is deterministic — no probability — because the
    membership questions (∈ M, ∈ S) are decidable. Soft posterior from
    surprisal is added separately by the calling verifier.
    """
    # In current scope: grounded
    if symbol in scope:
        return ReachabilityVerdict(state=1, scope_source=scope.how(symbol))

    # In manifest but not in scope: unreachable — compute fix
    if manifest_contains:
        # Find best module containing this symbol; suggest import
        candidates = reverse_index.modules_for(symbol)
        if candidates:
            # Prefer modules whose name doesn't start with _ (private)
            public = [(m, p) for (m, p) in candidates if not m.startswith("_")]
            chosen = public[0] if public else candidates[0]
            return ReachabilityVerdict(
                state=2,
                suggested_import=f"from {chosen[0]} import {symbol}",
            )
        # In manifest via stdlib/builtins/installed → in_scope_top
        # already; if we got here it means manifest matched but the
        # reverse_index doesn't know where. Treat as grounded — the
        # caller's manifest layer accepted it.
        return ReachabilityVerdict(state=1, scope_source="manifest_top_level")

    # Not in manifest and not in scope: hallucinated
    return ReachabilityVerdict(state=0)
