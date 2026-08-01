"""Dependency-aware repository intelligence and context bundling.

The implementation is dependency-free and deterministic. Python receives AST
analysis; Rust, JavaScript, and TypeScript receive conservative import/symbol
extraction. Context bundles contain whole source lines with exact line ranges
and never claim semantic completeness.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
import re
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .text_features import is_instruction_path, query_terms, text_terms

_SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", "target", "dist", "build", ".mypy_cache",
}
_SUPPORTED_SUFFIXES = {
    ".py", ".rs", ".js", ".jsx", ".ts", ".tsx",
    ".md", ".mdx", ".rst", ".txt",
}
_NON_PY_SYMBOL_RE = re.compile(
    r"(?m)^\s*(?:pub\s+)?(?:async\s+)?(?:fn|struct|enum|trait|type|class|"
    r"interface|function|const)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
_JS_IMPORT_RE = re.compile(
    r"(?:from\s+|require\s*\(\s*)['\"](?P<path>\.{1,2}/[^'\"]+)['\"]"
)
_RUST_MOD_RE = re.compile(r"(?m)^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")
_RUST_USE_RE = re.compile(r"(?m)^\s*use\s+crate::([A-Za-z_][A-Za-z0-9_:]*)")


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: str
    line: int
    end_line: int


@dataclass(frozen=True)
class FileIntelligence:
    path: str
    language: str
    lines: int
    bytes: int
    sha256: str
    symbols: tuple[Symbol, ...]
    imports: tuple[str, ...]
    parse_error: str | None = None
    instruction_file: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = [asdict(symbol) for symbol in self.symbols]
        return payload


@dataclass(frozen=True)
class ImpactNode:
    path: str
    distance: int
    reason: str
    score: float


@dataclass(frozen=True)
class ImpactReport:
    changed: tuple[str, ...]
    impacted: tuple[ImpactNode, ...]
    unresolved: tuple[str, ...]
    dependency_edges: int
    repository_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed": list(self.changed),
            "impacted": [asdict(node) for node in self.impacted],
            "unresolved": list(self.unresolved),
            "dependency_edges": self.dependency_edges,
            "repository_fingerprint": self.repository_fingerprint,
        }


@dataclass(frozen=True)
class RepositoryOverview:
    files: int
    languages: tuple[tuple[str, int], ...]
    dependency_edges: int
    parse_errors: tuple[str, ...]
    instruction_files: tuple[str, ...]
    top_important: tuple[tuple[str, float], ...]
    repository_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "languages": dict(self.languages),
            "dependency_edges": self.dependency_edges,
            "parse_errors": list(self.parse_errors),
            "instruction_files": list(self.instruction_files),
            "top_important": [
                {"path": path, "score": score} for path, score in self.top_important
            ],
            "repository_fingerprint": self.repository_fingerprint,
        }


@dataclass(frozen=True)
class CodeSmell:
    path: str
    kind: str
    severity: str
    line: int
    score: float
    message: str


@dataclass(frozen=True)
class CodeSmellReport:
    findings: tuple[CodeSmell, ...]
    files_scanned: int
    repository_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [asdict(finding) for finding in self.findings],
            "files_scanned": self.files_scanned,
            "repository_fingerprint": self.repository_fingerprint,
        }


@dataclass(frozen=True)
class ContextExcerpt:
    path: str
    start_line: int
    end_line: int
    text: str
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RepositoryContextBundle:
    query: str
    requested_budget: int
    emitted_tokens: int
    excerpts: tuple[ContextExcerpt, ...]
    impacted_files: tuple[str, ...]
    repository_fingerprint: str
    truncated: bool

    def render(self) -> str:
        blocks = []
        for excerpt in self.excerpts:
            blocks.append(
                f"### {excerpt.path}:{excerpt.start_line}-{excerpt.end_line}\n"
                f"{excerpt.text.rstrip()}"
            )
        return "\n\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "requested_budget": self.requested_budget,
            "emitted_tokens": self.emitted_tokens,
            "excerpts": [asdict(excerpt) for excerpt in self.excerpts],
            "impacted_files": list(self.impacted_files),
            "repository_fingerprint": self.repository_fingerprint,
            "truncated": self.truncated,
        }


def _language(path: Path) -> str:
    return {
        ".py": "python",
        ".rs": "rust",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".md": "markdown",
        ".mdx": "markdown",
        ".rst": "text",
        ".txt": "text",
    }.get(path.suffix.lower(), "text")


def _safe_relative(root: Path, path: str | Path) -> Path | None:
    candidate = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _python_analysis(text: str) -> tuple[list[Symbol], list[str], str | None]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], [], f"SyntaxError:{exc.lineno}:{exc.msg}"[:300]
    symbols: list[Symbol] = []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(
                Symbol(
                    node.name,
                    "class" if isinstance(node, ast.ClassDef) else "function",
                    int(node.lineno),
                    int(getattr(node, "end_lineno", node.lineno)),
                )
            )
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * int(node.level)
            imports.append(prefix + (node.module or ""))
    symbols.sort(key=lambda symbol: (symbol.line, symbol.kind, symbol.name))
    return symbols, sorted(set(imports)), None


def _non_python_analysis(text: str, language: str) -> tuple[list[Symbol], list[str], str | None]:
    symbols = [
        Symbol(match.group(1), "symbol", text.count("\n", 0, match.start()) + 1,
               text.count("\n", 0, match.start()) + 1)
        for match in _NON_PY_SYMBOL_RE.finditer(text)
    ]
    imports: set[str] = set()
    if language in {"javascript", "typescript"}:
        imports.update(match.group("path") for match in _JS_IMPORT_RE.finditer(text))
    elif language == "rust":
        imports.update(f"mod:{match.group(1)}" for match in _RUST_MOD_RE.finditer(text))
        imports.update(f"crate:{match.group(1)}" for match in _RUST_USE_RE.finditer(text))
    return symbols, sorted(imports), None


def _resolve_python_import(path: Path, module: str, root: Path) -> list[str]:
    resolved: list[str] = []
    if module.startswith("."):
        level = len(module) - len(module.lstrip("."))
        remainder = module[level:]
        base = path.parent
        for _ in range(max(0, level - 1)):
            base = base.parent
        candidate_base = base / Path(*([part for part in remainder.split(".") if part]))
    else:
        candidate_base = root / Path(*module.split("."))
    for candidate in (candidate_base.with_suffix(".py"), candidate_base / "__init__.py"):
        if candidate.is_file():
            resolved.append(candidate.relative_to(root).as_posix())
    return resolved


def _resolve_js_import(path: Path, target: str, root: Path) -> list[str]:
    base = (path.parent / target).resolve()
    try:
        base.relative_to(root)
    except ValueError:
        return []
    candidates = [base]
    for suffix in (".js", ".jsx", ".ts", ".tsx"):
        candidates.append(base.with_suffix(suffix))
    for suffix in (".js", ".ts"):
        candidates.append(base / f"index{suffix}")
    return [candidate.relative_to(root).as_posix() for candidate in candidates if candidate.is_file()]


def _resolve_rust_import(path: Path, target: str, root: Path) -> list[str]:
    if target.startswith("mod:"):
        name = target.split(":", 1)[1]
        candidates = [path.parent / f"{name}.rs", path.parent / name / "mod.rs"]
    elif target.startswith("crate:"):
        parts = target.split(":", 1)[1].split("::")
        crate_root = path
        while crate_root.parent != root and crate_root.name != "src":
            crate_root = crate_root.parent
        src = crate_root if crate_root.name == "src" else path.parent
        candidates = [src.joinpath(*parts).with_suffix(".rs"), src.joinpath(*parts, "mod.rs")]
    else:
        return []
    return [candidate.relative_to(root).as_posix() for candidate in candidates if candidate.is_file()]


def _severity(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _max_control_depth(node: ast.AST, depth: int = 0) -> int:
    control = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
    next_depth = depth + 1 if isinstance(node, control) else depth
    child_depths = [_max_control_depth(child, next_depth) for child in ast.iter_child_nodes(node)]
    return max([next_depth, *child_depths])


def _python_smells(path: str, text: str) -> list[CodeSmell]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    findings: list[CodeSmell] = []
    line_count = text.count("\n") + (1 if text else 0)
    if line_count >= 1_000:
        score = min(1.0, line_count / 2_000)
        findings.append(CodeSmell(path, "large_file", _severity(score), 1, round(score, 6), f"file has {line_count} lines"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        end_line = int(getattr(node, "end_lineno", node.lineno))
        length = end_line - int(node.lineno) + 1
        name = getattr(node, "name", "<anonymous>")
        if isinstance(node, ast.ClassDef):
            if length >= 400:
                score = min(1.0, length / 800)
                findings.append(CodeSmell(path, "large_class", _severity(score), int(node.lineno), round(score, 6), f"class {name} spans {length} lines"))
            continue
        branches = sum(
            isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match, ast.BoolOp, ast.comprehension))
            for child in ast.walk(node)
        )
        depth = _max_control_depth(node)
        if length >= 80:
            score = min(1.0, length / 180)
            findings.append(CodeSmell(path, "long_function", _severity(score), int(node.lineno), round(score, 6), f"function {name} spans {length} lines"))
        if branches >= 15:
            score = min(1.0, branches / 35)
            findings.append(CodeSmell(path, "high_branching", _severity(score), int(node.lineno), round(score, 6), f"function {name} has {branches} branch points"))
        if depth >= 5:
            score = min(1.0, depth / 9)
            findings.append(CodeSmell(path, "deep_nesting", _severity(score), int(node.lineno), round(score, 6), f"function {name} reaches control depth {depth}"))
    return findings


def _generic_smells(path: str, text: str) -> list[CodeSmell]:
    lines = text.splitlines()
    findings: list[CodeSmell] = []
    if len(lines) >= 1_200:
        score = min(1.0, len(lines) / 2_400)
        findings.append(CodeSmell(path, "large_file", _severity(score), 1, round(score, 6), f"file has {len(lines)} lines"))
    longest = max((len(line) for line in lines), default=0)
    if longest >= 300:
        line_number = next(index for index, line in enumerate(lines, 1) if len(line) == longest)
        score = min(1.0, longest / 800)
        findings.append(CodeSmell(path, "very_long_line", _severity(score), line_number, round(score, 6), f"line has {longest} characters"))
    return findings


class RepositoryIntelligence:
    """Scanned repository with deterministic dependency and symbol indexes."""

    def __init__(
        self,
        root: str | Path,
        files: dict[str, FileIntelligence],
        dependencies: dict[str, tuple[str, ...]],
    ) -> None:
        self.root = Path(root).resolve()
        self.files = dict(files)
        self.dependencies = dict(dependencies)
        reverse: dict[str, set[str]] = defaultdict(set)
        for source, targets in dependencies.items():
            for target in targets:
                reverse[target].add(source)
        self.reverse_dependencies = {
            path: tuple(sorted(dependents)) for path, dependents in reverse.items()
        }
        digest = hashlib.sha256()
        for path, facts in sorted(self.files.items()):
            digest.update(path.encode("utf-8"))
            digest.update(facts.sha256.encode("ascii"))
            for target in self.dependencies.get(path, ()):
                digest.update(target.encode("utf-8"))
        self.fingerprint = digest.hexdigest()

    @classmethod
    def scan(
        cls,
        root: str | Path,
        *,
        max_file_bytes: int = 2_000_000,
    ) -> "RepositoryIntelligence":
        root_path = Path(root).resolve()
        if not root_path.is_dir():
            raise ValueError("root must be a directory")
        files: dict[str, FileIntelligence] = {}
        for directory, dirnames, filenames in os.walk(root_path):
            dirnames[:] = sorted(name for name in dirnames if name not in _SKIP_DIRS)
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size > max_file_bytes:
                    continue
                try:
                    raw = path.read_bytes()
                    text = raw.decode("utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                language = _language(path)
                if language == "python":
                    symbols, imports, parse_error = _python_analysis(text)
                else:
                    symbols, imports, parse_error = _non_python_analysis(text, language)
                relative = path.relative_to(root_path).as_posix()
                files[relative] = FileIntelligence(
                    path=relative,
                    language=language,
                    lines=text.count("\n") + (1 if text else 0),
                    bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    symbols=tuple(symbols),
                    imports=tuple(imports),
                    parse_error=parse_error,
                    instruction_file=is_instruction_path(relative),
                )

        dependencies: dict[str, tuple[str, ...]] = {}
        for relative, facts in files.items():
            path = root_path / relative
            resolved: set[str] = set()
            for imported in facts.imports:
                if facts.language == "python":
                    resolved.update(_resolve_python_import(path, imported, root_path))
                elif facts.language in {"javascript", "typescript"}:
                    resolved.update(_resolve_js_import(path, imported, root_path))
                elif facts.language == "rust":
                    resolved.update(_resolve_rust_import(path, imported, root_path))
            dependencies[relative] = tuple(sorted(target for target in resolved if target in files))
        return cls(root_path, files, dependencies)

    def importance_scores(
        self,
        *,
        damping: float = 0.85,
        iterations: int = 50,
        tolerance: float = 1e-12,
    ) -> dict[str, float]:
        """Return deterministic PageRank-style foundational-file importance."""
        if not 0.0 < damping < 1.0:
            raise ValueError("damping must be inside (0, 1)")
        if iterations <= 0 or tolerance <= 0:
            raise ValueError("iterations and tolerance must be positive")
        nodes = sorted(self.files)
        if not nodes:
            return {}
        count = len(nodes)
        ranks = {node: 1.0 / count for node in nodes}
        for _ in range(iterations):
            dangling = sum(ranks[node] for node in nodes if not self.dependencies.get(node))
            next_ranks = {
                node: (1.0 - damping) / count + damping * dangling / count
                for node in nodes
            }
            for source in nodes:
                targets = self.dependencies.get(source, ())
                if not targets:
                    continue
                share = damping * ranks[source] / len(targets)
                for target in targets:
                    if target in next_ranks:
                        next_ranks[target] += share
            delta = sum(abs(next_ranks[node] - ranks[node]) for node in nodes)
            ranks = next_ranks
            if delta <= tolerance:
                break
        total = sum(ranks.values()) or 1.0
        return {node: round(ranks[node] / total, 12) for node in nodes}

    def overview(self, *, top_n: int = 20) -> RepositoryOverview:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        languages = Counter(facts.language for facts in self.files.values())
        parse_errors = tuple(sorted(path for path, facts in self.files.items() if facts.parse_error))
        instruction_files = tuple(sorted(path for path, facts in self.files.items() if facts.instruction_file))
        importance = self.importance_scores()
        top = tuple(sorted(importance.items(), key=lambda item: (-item[1], item[0]))[:top_n])
        return RepositoryOverview(
            files=len(self.files),
            languages=tuple(sorted(languages.items())),
            dependency_edges=sum(len(targets) for targets in self.dependencies.values()),
            parse_errors=parse_errors,
            instruction_files=instruction_files,
            top_important=top,
            repository_fingerprint=self.fingerprint,
        )

    def smell_report(self, *, max_findings: int = 200) -> CodeSmellReport:
        if max_findings <= 0:
            raise ValueError("max_findings must be positive")
        findings: list[CodeSmell] = []
        for path, facts in sorted(self.files.items()):
            try:
                text = (self.root / path).read_text(encoding="utf-8")
            except OSError:
                continue
            if facts.language == "python":
                findings.extend(_python_smells(path, text))
            elif facts.language in {"rust", "javascript", "typescript"}:
                findings.extend(_generic_smells(path, text))
        findings.sort(key=lambda finding: (-finding.score, finding.path, finding.line, finding.kind))
        return CodeSmellReport(tuple(findings[:max_findings]), len(self.files), self.fingerprint)

    def impact_report(
        self,
        changed_paths: Sequence[str | Path],
        *,
        max_depth: int = 4,
        max_files: int = 200,
    ) -> ImpactReport:
        if max_depth < 0 or max_files <= 0:
            raise ValueError("max_depth must be non-negative and max_files positive")
        changed: list[str] = []
        unresolved: list[str] = []
        for raw in changed_paths:
            safe = _safe_relative(self.root, raw)
            if safe is None:
                unresolved.append(str(raw))
                continue
            relative = safe.relative_to(self.root).as_posix()
            if relative not in self.files:
                unresolved.append(relative)
            elif relative not in changed:
                changed.append(relative)

        distances: dict[str, int] = {path: 0 for path in changed}
        reasons: dict[str, str] = {path: "changed" for path in changed}
        queue = deque(changed)
        while queue:
            current = queue.popleft()
            distance = distances[current]
            if distance >= max_depth:
                continue
            for dependent in self.reverse_dependencies.get(current, ()):
                if dependent not in distances or distance + 1 < distances[dependent]:
                    distances[dependent] = distance + 1
                    reasons[dependent] = f"depends_on:{current}"
                    queue.append(dependent)

        importance = self.importance_scores()
        max_importance = max(importance.values(), default=1.0) or 1.0
        nodes: list[ImpactNode] = []
        for path, distance in distances.items():
            facts = self.files[path]
            test_boost = 0.35 if path.startswith("tests/") or "/test" in path else 0.0
            symbol_boost = min(0.25, len(facts.symbols) / 100)
            importance_boost = 0.3 * importance.get(path, 0.0) / max_importance
            score = 1.0 / (1 + distance) + test_boost + symbol_boost + importance_boost
            nodes.append(ImpactNode(path, distance, reasons[path], round(score, 6)))
        nodes.sort(key=lambda node: (-node.score, node.distance, node.path))
        nodes = nodes[:max_files]
        edge_count = sum(len(targets) for targets in self.dependencies.values())
        return ImpactReport(
            tuple(changed), tuple(nodes), tuple(sorted(unresolved)), edge_count, self.fingerprint
        )

    def _candidate_excerpts(
        self,
        path: str,
        query_terms: set[str],
        impact_score: float,
    ) -> list[ContextExcerpt]:
        file_path = self.root / path
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        facts = self.files[path]
        candidates: list[ContextExcerpt] = []
        if facts.instruction_file:
            text = "\n".join(lines)
            return [
                ContextExcerpt(
                    path,
                    1,
                    len(lines),
                    text,
                    round(impact_score + 1.0, 6),
                    (f"impact={impact_score:.3f}", "instruction_file_full_fidelity"),
                )
            ] if lines else []
        for symbol in facts.symbols:
            start = max(1, symbol.line - 2)
            end = min(len(lines), max(symbol.end_line, symbol.line + 8))
            text = "\n".join(lines[start - 1 : end])
            words = text_terms(text)
            overlap = len(query_terms & words) / max(1, len(query_terms))
            score = impact_score + overlap + (0.15 if symbol.kind == "class" else 0.05)
            reasons = [f"impact={impact_score:.3f}", f"symbol={symbol.name}"]
            if overlap:
                reasons.append(f"query_overlap={overlap:.3f}")
            candidates.append(
                ContextExcerpt(path, start, end, text, round(score, 6), tuple(reasons))
            )
        if not candidates:
            end = min(len(lines), 40)
            if end:
                text = "\n".join(lines[:end])
                words = text_terms(text)
                overlap = len(query_terms & words) / max(1, len(query_terms))
                candidates.append(
                    ContextExcerpt(
                        path, 1, end, text, round(impact_score + overlap, 6),
                        (f"impact={impact_score:.3f}", "file_head"),
                    )
                )
        return candidates

    def context_bundle(
        self,
        *,
        query: str,
        changed_paths: Sequence[str | Path],
        budget_tokens: int,
        max_depth: int = 3,
    ) -> RepositoryContextBundle:
        if budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        impact = self.impact_report(changed_paths, max_depth=max_depth)
        query_words = query_terms(query)
        candidates: list[ContextExcerpt] = []
        for node in impact.impacted:
            candidates.extend(self._candidate_excerpts(node.path, query_words, node.score))
        candidates.sort(key=lambda item: (-item.score, item.path, item.start_line, item.end_line))

        selected: list[ContextExcerpt] = []
        seen_ranges: set[tuple[str, int, int]] = set()
        used = 0
        for candidate in candidates:
            key = (candidate.path, candidate.start_line, candidate.end_line)
            if key in seen_ranges:
                continue
            rendered = (
                f"### {candidate.path}:{candidate.start_line}-{candidate.end_line}\n"
                f"{candidate.text.rstrip()}"
            )
            cost = max(1, math.ceil(len(rendered) / 4))
            if used + cost > budget_tokens:
                continue
            selected.append(candidate)
            seen_ranges.add(key)
            used += cost
        return RepositoryContextBundle(
            query=query,
            requested_budget=budget_tokens,
            emitted_tokens=used,
            excerpts=tuple(selected),
            impacted_files=tuple(node.path for node in impact.impacted),
            repository_fingerprint=self.fingerprint,
            truncated=len(selected) < len(candidates),
        )


__all__ = [
    "CodeSmell",
    "CodeSmellReport",
    "ContextExcerpt",
    "FileIntelligence",
    "ImpactNode",
    "ImpactReport",
    "RepositoryContextBundle",
    "RepositoryIntelligence",
    "RepositoryOverview",
    "Symbol",
]
