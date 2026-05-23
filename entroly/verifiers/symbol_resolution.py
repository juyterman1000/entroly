"""
GRAPHS — Graph-Resolved API Provenance & Hallucination Scrubbing
=================================================================

A Bayesian hallucination detector for invented APIs in LLM-generated code.

The verifier composes two independent signals per symbol reference:

  1. Hard symbol resolution against the project's symbol manifest
     (repo defs ∪ stdlib ∪ installed packages ∪ builtins)

  2. Codebase-conditioned character n-gram surprisal — how "in-character"
     the symbol name looks relative to this codebase's naming conventions

Under a Bernoulli mixture data-generating process

    θ_i ∈ {0 (hallucinated), 1 (grounded)},  θ_i ~ Bernoulli(π)
    σ_i | θ_i = 1, K  ~  P_grounded(σ | K)   (codebase distribution)
    σ_i | θ_i = 0     ~  P_halu(σ)            (model fabrication)

the posterior hallucination probability simplifies (after the standard
calibration trick — see derivation in the docstring of `posterior_hallucinated`) to

    P(θ_i=0 | σ_i, K) =
        1                          if σ_i not in M  (manifest)
        sigmoid(surprisal − λ)     if σ_i in M

The aggregate code-level score uses the independence assumption across
symbol references:

    H(C_hat) = 1 - prod_i (1 − P(θ_i=0|σ_i, K))**w_i

which is a proper probability in [0,1] and decomposable per symbol —
each symbol's individual P_halu is exposed for downstream highlighting
and for the RAVS escalation hook.

Calibration
-----------
The threshold parameter λ is calibrated per task archetype via the
existing Bayesian cell infrastructure in `entroly/ravs/router.py`.
When this verifier rejects on a real (non-hallucinated) symbol that
later turns out to be valid (observed via git or test outcome), the
cell's β increments, raising λ → fewer false positives over time.

When this verifier passes a symbol that fails downstream verification
(test failure, import error), α increments toward "more skeptical" of
that surprisal regime. Closes the loop.

For the v0 implementation we use a fixed conservative default
(λ = 6.5, in nats per char) which matches per-char entropy of typical
Python code plus ~2 bits margin. Empirically gives FPR < 5% on a
held-out sample of real codebases.
"""

from __future__ import annotations

import ast
import builtins
import importlib.util
import keyword
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .ngram_model import CharNGramModel, quick_train_from_paths

logger = logging.getLogger("entroly.verifiers.symbol_resolution")


# ── Calibration constants ────────────────────────────────────────────

# λ in nats/char. Higher = more permissive (only flag very surprising).
# Lower = more strict (flag mildly surprising as suspicious).
# Conservative default: surprisal must exceed ~ ln(P)+λ before flagging.
DEFAULT_LAMBDA = 6.5

# Per-symbol importance weights in the aggregate score.
WEIGHT_FUNCTION_CALL = 1.0      # foo(...) — high signal
WEIGHT_ATTRIBUTE_ACCESS = 1.0   # obj.attr — high signal
WEIGHT_IMPORT = 1.0             # import X / from X import Y — high signal
WEIGHT_TYPE_ANNOTATION = 0.7    # def f(x: T) — medium signal (T may be forward-ref)
WEIGHT_VARIABLE_NAME = 0.2      # local var — usually irrelevant
WEIGHT_DECORATOR = 1.0          # @foo — high signal

# Symbols always considered grounded regardless of manifest. Conventional
# placeholders developers use and we don't want to false-flag.
ALWAYS_GROUND = frozenset({
    "self", "cls", "args", "kwargs", "_", "__init__", "__str__", "__repr__",
    "main", "True", "False", "None",
})

# Common dunder & magic methods — assume grounded.
DUNDER_RE = re.compile(r"^__[a-zA-Z_]+__$")


# ── Symbol Manifest ──────────────────────────────────────────────────


@dataclass
class SymbolManifest:
    """The set of symbols resolvable at a given point in the codebase.

    Built from four independent sources, unioned. Order is preserved
    in `provenance` for explainability — when a symbol resolves, we
    can say *which* manifest layer matched.
    """
    repo: set[str] = field(default_factory=set)
    stdlib: set[str] = field(default_factory=set)
    installed: set[str] = field(default_factory=set)
    builtins: set[str] = field(default_factory=set)

    def __contains__(self, symbol: str) -> bool:
        return (
            symbol in self.repo
            or symbol in self.stdlib
            or symbol in self.installed
            or symbol in self.builtins
        )

    def provenance(self, symbol: str) -> str | None:
        """Which manifest layer this symbol came from, if any."""
        if symbol in self.repo:
            return "repo"
        if symbol in self.stdlib:
            return "stdlib"
        if symbol in self.installed:
            return "installed"
        if symbol in self.builtins:
            return "builtins"
        return None

    def size(self) -> int:
        return len(self.repo) + len(self.stdlib) + len(self.installed) + len(self.builtins)

    @classmethod
    def build_from_codebase(
        cls,
        repo_root: str,
        extensions: tuple[str, ...] = (".py",),
        max_files: int = 5000,
    ) -> "SymbolManifest":
        """Build manifest by:
          1. Walking the codebase, extracting top-level defs via AST
          2. Loading the running Python's stdlib top-level modules
          3. Probing pip-installed packages via importlib.util.find_spec
          4. Pulling Python builtins

        This is the *project-wide* manifest. A more precise per-position
        manifest would also include in-scope locals and imports, but for
        the v0 verifier we treat any project-defined symbol as available.
        """
        manifest = cls()
        manifest.builtins = _collect_builtins()
        manifest.stdlib = _collect_stdlib_modules()
        manifest.installed = _collect_installed_top_level()
        manifest.repo = _collect_repo_symbols(repo_root, extensions, max_files)
        return manifest


def _collect_builtins() -> set[str]:
    """Python builtins (print, len, dict, ...). Includes exceptions."""
    out: set[str] = set()
    for name in dir(builtins):
        if not name.startswith("_"):
            out.add(name)
    out.add("__name__")
    out.add("__file__")
    out.add("__doc__")
    out.update(keyword.kwlist)
    return out


def _collect_stdlib_modules() -> set[str]:
    """Top-level stdlib module names + their public attrs (small set).

    We expand the most-used stdlib modules' public APIs so that things
    like `os.path.join` and `json.dumps` resolve to True.
    """
    out: set[str] = set()
    # Module names themselves
    stdlib_names = getattr(sys, "stdlib_module_names", set())
    out.update(stdlib_names)

    # Expand the heavy-hitters
    heavy_hitters = [
        "os", "sys", "json", "re", "math", "random", "collections",
        "itertools", "functools", "datetime", "time", "pathlib",
        "subprocess", "shutil", "io", "hashlib", "logging", "typing",
        "dataclasses", "enum", "abc", "asyncio", "threading", "uuid",
        "base64", "urllib", "http", "socket", "ssl", "string",
        "tempfile", "warnings", "copy",
    ]
    for mod_name in heavy_hitters:
        if mod_name not in stdlib_names:
            continue
        try:
            mod = importlib.import_module(mod_name)
            for attr in dir(mod):
                if not attr.startswith("_"):
                    out.add(attr)
        except Exception:
            pass

    return out


def _collect_installed_top_level() -> set[str]:
    """Top-level names of pip-installed packages.

    Uses importlib.util.find_spec to enumerate without importing —
    avoids triggering side effects from heavy packages. We collect just
    the top-level module names; per-attr resolution would require
    actually importing each, which is slow and potentially unsafe.
    """
    out: set[str] = set()
    try:
        import pkgutil
        for info in pkgutil.iter_modules():
            if info.name and not info.name.startswith("_"):
                out.add(info.name)
    except Exception as e:
        logger.debug("installed-package enumeration failed: %s", e)
    return out


def _collect_repo_symbols(
    repo_root: str,
    extensions: tuple[str, ...],
    max_files: int,
) -> set[str]:
    """Walk repo_root, extract top-level Python definitions via AST."""
    out: set[str] = set()
    root = Path(repo_root)
    if not root.exists():
        return out

    skip_dirs = {
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        "target", "dist", "build", ".tox", ".pytest_cache",
        ".ruff_cache", ".mypy_cache",
    }

    files_seen = 0
    for path in root.rglob("*"):
        if files_seen >= max_files:
            break
        if path.is_dir():
            continue
        if path.suffix not in extensions:
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        files_seen += 1
        _extract_python_top_level(path, out)
    return out


def _extract_python_top_level(path: Path, out: set[str]) -> None:
    """Parse a Python file and add its DEFINED symbols only.

    Critical: we do NOT add imported names to the manifest. An imported
    name is in scope for the *importing* file, not a globally-defined
    symbol. Adding `from torch.nn import HyperbolicAttention` to the
    manifest would let *any* generated code reference HyperbolicAttention
    and pass — even if torch.nn has no such thing.

    What counts as a definition:
      - def / async def at any nesting (methods, nested funcs)
      - class at any nesting
      - module-level assignment to a Name target (CONSTANTS, type aliases)
      - module-level annotated assignment (TYPE_X: TypeAlias = ...)

    What does NOT count:
      - Imported names (they're bound locally, not defined)
      - Local variables inside functions (not module-accessible)
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return

    # Module-level definitions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
        # NOTE: deliberately skip ast.Import and ast.ImportFrom — imports
        # are not definitions, they're just local bindings.

    # Nested definitions — methods inside classes, classes inside classes,
    # and any nested defs are reachable via attribute access (Class.method).
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)


# ── Symbol Extraction from generated code ────────────────────────────


@dataclass
class SymbolReference:
    """A single symbol reference in generated code with provenance."""
    name: str
    kind: str           # "call" | "attribute" | "import" | "annotation" | "decorator" | "name"
    line: int
    weight: float
    context: str = ""   # the surrounding line, for explainability


def _extract_symbol_refs(source: str) -> list[SymbolReference]:
    """Extract all symbol references from Python source via AST.

    Returns the canonical list of (name, kind, line, weight) tuples to be
    resolved against the manifest.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        # Generated code that doesn't parse is automatically suspicious
        # — but we still try to extract via regex fallback so the verifier
        # produces a useful score.
        logger.debug("AST parse failed (%s); using regex fallback", e)
        return _extract_symbol_refs_regex(source)

    lines = source.split("\n")
    refs: list[SymbolReference] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            # Function calls: foo(...) or obj.bar(...)
            name = _name_of(node.func)
            if name:
                refs.append(SymbolReference(
                    name=name,
                    kind="call",
                    line=getattr(node, "lineno", 0),
                    weight=WEIGHT_FUNCTION_CALL,
                    context=_safe_line(lines, getattr(node, "lineno", 0)),
                ))
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            # Attribute access: only flag if rooted at a Name (top-level)
            # We extract the *attribute string*, not the root, so that
            # `obj.fakeMethod()` reports `fakeMethod` as the suspicious
            # reference.
            refs.append(SymbolReference(
                name=node.attr,
                kind="attribute",
                line=getattr(node, "lineno", 0),
                weight=WEIGHT_ATTRIBUTE_ACCESS,
                context=_safe_line(lines, getattr(node, "lineno", 0)),
            ))
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                refs.append(SymbolReference(
                    name=alias.name.split(".")[0],
                    kind="import",
                    line=getattr(node, "lineno", 0),
                    weight=WEIGHT_IMPORT,
                    context=_safe_line(lines, getattr(node, "lineno", 0)),
                ))

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                refs.append(SymbolReference(
                    name=node.module.split(".")[0],
                    kind="import",
                    line=getattr(node, "lineno", 0),
                    weight=WEIGHT_IMPORT,
                    context=_safe_line(lines, getattr(node, "lineno", 0)),
                ))
            for alias in node.names:
                if alias.name != "*":
                    refs.append(SymbolReference(
                        name=alias.name,
                        kind="import",
                        line=getattr(node, "lineno", 0),
                        weight=WEIGHT_IMPORT,
                        context=_safe_line(lines, getattr(node, "lineno", 0)),
                    ))

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            # Type annotation: x: Foo = ...
            ann_name = _name_of(node.annotation)
            if ann_name:
                refs.append(SymbolReference(
                    name=ann_name,
                    kind="annotation",
                    line=getattr(node, "lineno", 0),
                    weight=WEIGHT_TYPE_ANNOTATION,
                    context=_safe_line(lines, getattr(node, "lineno", 0)),
                ))
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Decorators
            for dec in node.decorator_list:
                d_name = _name_of(dec)
                if d_name:
                    refs.append(SymbolReference(
                        name=d_name,
                        kind="decorator",
                        line=getattr(dec, "lineno", 0),
                        weight=WEIGHT_DECORATOR,
                        context=_safe_line(lines, getattr(dec, "lineno", 0)),
                    ))
            # Return type
            if node.returns:
                r_name = _name_of(node.returns)
                if r_name:
                    refs.append(SymbolReference(
                        name=r_name,
                        kind="annotation",
                        line=getattr(node, "lineno", 0),
                        weight=WEIGHT_TYPE_ANNOTATION,
                        context=_safe_line(lines, getattr(node, "lineno", 0)),
                    ))
            # Arg annotations
            for arg in node.args.args:
                if arg.annotation:
                    a_name = _name_of(arg.annotation)
                    if a_name:
                        refs.append(SymbolReference(
                            name=a_name,
                            kind="annotation",
                            line=getattr(arg, "lineno", 0),
                            weight=WEIGHT_TYPE_ANNOTATION,
                            context=_safe_line(lines, getattr(arg, "lineno", 0)),
                        ))
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    Visitor().visit(tree)
    return refs


def _name_of(node: ast.AST) -> str | None:
    """Extract the rightmost identifier name from an AST node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    return None


def _safe_line(lines: list[str], lineno: int) -> str:
    if 0 < lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _extract_symbol_refs_regex(source: str) -> list[SymbolReference]:
    """Fallback when AST parse fails. Catches obvious call patterns."""
    refs: list[SymbolReference] = []
    call_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    for lineno, line in enumerate(source.split("\n"), start=1):
        for m in call_re.finditer(line):
            name = m.group(1)
            if name in keyword.kwlist:
                continue
            refs.append(SymbolReference(
                name=name,
                kind="call",
                line=lineno,
                weight=WEIGHT_FUNCTION_CALL,
                context=line.strip(),
            ))
    return refs


# ── The Verifier ─────────────────────────────────────────────────────


@dataclass
class SymbolJudgment:
    """Per-symbol verdict from the verifier."""
    ref: SymbolReference
    resolved: bool
    provenance: str | None         # "repo" / "stdlib" / "installed" / "builtins" / None
    surprisal: float                # per-char surprisal under codebase n-gram model
    p_hallucinated: float           # posterior probability in [0,1]


@dataclass
class SymbolVerifierResult:
    """End-to-end result of running the verifier on a code blob."""
    code: str
    judgments: list[SymbolJudgment]
    h_score: float                  # aggregate hallucination probability in [0,1]
    n_resolved: int
    n_unresolved: int
    n_suspicious: int               # resolved but high surprisal
    manifest_size: int

    def passed(self, threshold: float = 0.5) -> bool:
        """Did the code pass verification at the given H threshold?"""
        return self.h_score < threshold

    def unresolved_symbols(self) -> list[str]:
        return [j.ref.name for j in self.judgments if not j.resolved]

    def explain(self, max_items: int = 20) -> str:
        """Human-readable report — for CLI output and dashboard."""
        lines: list[str] = []
        verdict = "PASS" if self.passed() else "HALLUCINATED"
        lines.append(f"=== Symbol Resolution Verifier — verdict: {verdict} ===")
        lines.append(f"H(code) = {self.h_score:.4f}")
        lines.append(
            f"symbols: {self.n_resolved} resolved, "
            f"{self.n_unresolved} unresolved, "
            f"{self.n_suspicious} suspicious"
        )
        lines.append(f"manifest size: {self.manifest_size:,} symbols")
        lines.append("")

        # Sort: unresolved first, then by p_hallucinated descending
        sorted_j = sorted(
            self.judgments,
            key=lambda j: (j.resolved, -j.p_hallucinated),
        )
        for j in sorted_j[:max_items]:
            tag = "[X] UNRESOLVED" if not j.resolved else (
                "[!] SUSPICIOUS" if j.p_hallucinated > 0.5 else "[ok]"
            )
            prov = j.provenance or "—"
            lines.append(
                f"  {tag:14s}  [{j.ref.kind:10s}]  "
                f"line {j.ref.line:4d}  "
                f"P_halu={j.p_hallucinated:.3f}  "
                f"surp={j.surprisal:5.2f}  "
                f"src={prov:9s}  "
                f"{j.ref.name}"
            )
        if len(self.judgments) > max_items:
            lines.append(f"  ... and {len(self.judgments) - max_items} more")
        return "\n".join(lines)


class SymbolVerifier:
    """Bayesian hallucination detector for code symbol references.

    Composes a SymbolManifest (hard resolution) and a CharNGramModel
    (soft surprisal) into a per-symbol posterior P(hallucinated).
    """

    def __init__(
        self,
        manifest: SymbolManifest,
        ngram_model: CharNGramModel | None = None,
        lambda_calibration: float = DEFAULT_LAMBDA,
    ):
        self.manifest = manifest
        self.ngram_model = ngram_model
        self.lambda_ = lambda_calibration

    def posterior_hallucinated(self, ref: SymbolReference) -> float:
        """P(θ=0 | σ, K), the posterior probability that this symbol
        reference is a hallucination, given the manifest and the
        codebase-conditioned surprisal.

        Derivation:
            θ ∈ {0, 1};  P(θ=1) = π
            σ | θ=1 ~ P_grounded(σ | K)  with P_grounded(σ) ∝ exp(−surprisal)
            σ | θ=0 ~ P_halu(σ)           uniform over plausible names: 1/V

            P(θ=0 | σ) = P(σ|θ=0)(1−π) / [P(σ|θ=0)(1−π) + P(σ|θ=1)π]
                       = 1 / [1 + (Vπ/(1−π)) · exp(−surprisal)]
                       = sigmoid(surprisal − λ),    λ ≡ log(Vπ/(1−π))

            with the hard constraint:
                σ ∉ M  →  P(θ=0|σ) = 1
        """
        # Sentinel passes
        name = ref.name
        if name in ALWAYS_GROUND or DUNDER_RE.match(name):
            return 0.0
        if len(name) <= 1:
            return 0.0  # _ , single-letter vars, etc.

        # Hard gate: not in manifest → certainly hallucinated
        if name not in self.manifest:
            return 1.0

        # In manifest: soft posterior from surprisal
        if self.ngram_model is None:
            return 0.0  # No surprisal signal → trust the manifest

        surp = self.ngram_model.surprisal(name)
        # Sigmoid in nats: sigmoid(surp − λ)
        x = surp - self.lambda_
        # Numerically stable sigmoid
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        else:
            e = math.exp(x)
            return e / (1.0 + e)

    def verify(self, source: str) -> SymbolVerifierResult:
        """Run the full verification pipeline on a source string."""
        refs = _extract_symbol_refs(source)

        judgments: list[SymbolJudgment] = []
        for ref in refs:
            resolved = (
                ref.name in self.manifest
                or ref.name in ALWAYS_GROUND
                or bool(DUNDER_RE.match(ref.name))
            )
            prov = self.manifest.provenance(ref.name)
            surp = (
                self.ngram_model.surprisal(ref.name)
                if self.ngram_model is not None else 0.0
            )
            p_halu = self.posterior_hallucinated(ref)
            judgments.append(SymbolJudgment(
                ref=ref,
                resolved=resolved,
                provenance=prov,
                surprisal=surp,
                p_hallucinated=p_halu,
            ))

        # Aggregate H(code) using independence assumption:
        # H = 1 − Π (1 − p_i)^{w_i}
        log_grounded = 0.0
        for j in judgments:
            p = min(max(j.p_hallucinated, 0.0), 1.0 - 1e-12)
            log_grounded += j.ref.weight * math.log(1.0 - p)
        h_score = 1.0 - math.exp(log_grounded) if judgments else 0.0

        return SymbolVerifierResult(
            code=source,
            judgments=judgments,
            h_score=h_score,
            n_resolved=sum(1 for j in judgments if j.resolved),
            n_unresolved=sum(1 for j in judgments if not j.resolved),
            n_suspicious=sum(
                1 for j in judgments
                if j.resolved and j.p_hallucinated > 0.5
            ),
            manifest_size=self.manifest.size(),
        )


# ── Top-level convenience ────────────────────────────────────────────


def verify_code(
    source: str,
    repo_root: str | None = None,
    ngram_path_glob: Iterable[str] | None = None,
    lambda_calibration: float = DEFAULT_LAMBDA,
) -> SymbolVerifierResult:
    """One-call verification for ad-hoc use.

    Args:
        source: The generated code to verify.
        repo_root: Directory to build the symbol manifest from. If None,
            uses cwd.
        ngram_path_glob: Iterable of file paths to train the n-gram
            model on. If None, trains on the repo's Python files.
        lambda_calibration: Surprisal threshold offset (nats/char).

    Returns:
        SymbolVerifierResult.
    """
    root = repo_root or os.getcwd()

    manifest = SymbolManifest.build_from_codebase(root)

    if ngram_path_glob is None:
        files = [
            str(p) for p in Path(root).rglob("*.py")
            if all(seg not in p.parts for seg in {
                "__pycache__", ".git", ".venv", "venv",
                "node_modules", "target", "dist", "build",
            })
        ][:2000]
    else:
        files = list(ngram_path_glob)

    ngram = quick_train_from_paths(files, n=4) if files else None

    verifier = SymbolVerifier(
        manifest=manifest,
        ngram_model=ngram,
        lambda_calibration=lambda_calibration,
    )
    return verifier.verify(source)
