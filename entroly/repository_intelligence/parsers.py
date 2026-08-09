"""Bounded source discovery and deterministic symbol extraction."""
from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .models import FileRecord, RepositoryLimits, Symbol, normalize_relative
from ..tree_sitter_support import (
    LANGUAGE_BY_SUFFIX,
    StructuralSpan,
    extract_structural_spans,
)

SOURCE_SUFFIXES = frozenset(LANGUAGE_BY_SUFFIX)
IGNORED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        ".tox", ".venv", "venv", "node_modules", "target", "dist", "build",
        "vendor", "__pycache__",
    }
)
TEST_PARTS = frozenset({"test", "tests", "testing", "spec", "specs", "__tests__"})
CALL_KEYWORDS = frozenset(
    {
        "if", "for", "while", "match", "switch", "catch", "return",
        "sizeof", "typeof", "function", "fn", "new",
    }
)

RUST_DEF = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?"
    r"(fn|struct|enum|trait|type|const|static|mod)\s+([A-Za-z_][A-Za-z0-9_]*)"
)
RUST_USE = re.compile(r"^\s*(?:pub\s+)?use\s+([^;]+);")
JS_DEF = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?"
    r"(function|class)\s+([A-Za-z_$][\w$]*)|"
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
    r"(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>"
)
JS_IMPORT = re.compile(r"(?:from\s+|require\s*\(\s*)['\"]([^'\"]+)['\"]")
CALL = re.compile(r"(?<![\w$])([A-Za-z_$][\w$]*)\s*\(")


@dataclass
class ParsedFile:
    record: FileRecord
    symbols: list[Symbol]
    imports: set[str]
    import_aliases: dict[str, str]
    calls: list[tuple[str | None, str, int]]


def module_name(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _language(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "unknown")


def _is_test_path(path: str) -> bool:
    parsed = Path(path)
    parts = {part.lower() for part in parsed.parts[:-1]}
    stem = parsed.stem.lower()
    return bool(parts & TEST_PARTS) or stem.startswith("test_") or any(
        stem.endswith(suffix) for suffix in ("_test", ".test", ".spec")
    )


def _symbol_id(path: str, qualified: str, kind: str) -> str:
    return f"{path}::{qualified}::{kind}"


def _import_target(module: str | None, level: int, path: str) -> str:
    if level == 0:
        return module or ""
    current = module_name(path).split(".")
    if Path(path).name != "__init__.py" and current:
        current.pop()
    current = current[: max(0, len(current) - level + 1)]
    if module:
        current.extend(part for part in module.split(".") if part)
    return ".".join(current)


class PythonVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.symbols: list[Symbol] = []
        self.imports: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.calls: list[tuple[str | None, str, int]] = []
        self.scope: list[Symbol] = []

    def _symbol(self, node: ast.AST, name: str, kind: str) -> Symbol:
        qualified = ".".join([*(item.name for item in self.scope), name])
        symbol = Symbol(
            _symbol_id(self.path, qualified, kind), self.path, name, qualified, kind,
            max(1, int(getattr(node, "lineno", 1))),
            max(1, int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))),
            "python", self.scope[-1].symbol_id if self.scope else None,
        )
        self.symbols.append(symbol)
        return symbol

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        symbol = self._symbol(node, node.name, "class")
        self.scope.append(symbol)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.scope and self.scope[-1].kind == "class" else "function"
        if node.name.startswith("test_"):
            kind = "test"
        symbol = self._symbol(node, node.name, kind)
        self.scope.append(symbol)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(alias.name)
            self.aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        target = _import_target(node.module, node.level, self.path)
        if target:
            self.imports.add(target)
        for alias in node.names:
            if alias.name != "*":
                local = alias.asname or alias.name
                self.aliases[local] = f"{target}.{alias.name}" if target else alias.name

    def visit_Call(self, node: ast.Call) -> None:
        owner: str | None = None
        name: str | None = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if isinstance(node.func.value, ast.Name):
                owner = node.func.value.id
        if name:
            target = f"{owner}.{name}" if owner else name
            caller = self.scope[-1].symbol_id if self.scope else None
            self.calls.append((caller, target, max(1, int(getattr(node, "lineno", 1)))))
        self.generic_visit(node)


def _record(
    path: str,
    language: str,
    raw: bytes,
    text: str,
    *,
    is_test: bool,
    imports: set[str],
    error: str | None = None,
) -> FileRecord:
    return FileRecord(
        path, language, hashlib.sha256(raw).hexdigest(), len(raw),
        max(1, text.count("\n") + 1), is_test, tuple(sorted(imports)), error,
    )


def _parse_python(path: str, text: str, raw: bytes) -> ParsedFile:
    try:
        tree = ast.parse(text, filename=path, type_comments=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError) as exc:
        record = _record(
            path,
            "python",
            raw,
            text,
            is_test=_is_test_path(path),
            imports=set(),
            error=f"{type(exc).__name__}: {exc}",
        )
        return ParsedFile(record, [], set(), {}, [])
    visitor = PythonVisitor(path)
    visitor.visit(tree)
    is_test = _is_test_path(path) or any(item.kind == "test" for item in visitor.symbols)
    return ParsedFile(
        _record(path, "python", raw, text, is_test=is_test, imports=visitor.imports),
        visitor.symbols, visitor.imports, visitor.aliases, visitor.calls,
    )


def _parse_conservative(path: str, text: str, raw: bytes, language: str) -> ParsedFile:
    symbols: list[Symbol] = []
    imports: set[str] = set()
    calls: list[tuple[str | None, str, int]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        current: Symbol | None = None
        if language == "rust":
            match = RUST_DEF.match(line)
            if match:
                kind, name = match.groups()
                current = Symbol(
                    _symbol_id(path, name, kind), path, name, name, kind,
                    line_number, line_number, language,
                )
                symbols.append(current)
            use = RUST_USE.match(line)
            if use:
                imports.add(use.group(1).strip().split("::{", 1)[0])
        else:
            match = JS_DEF.match(line)
            if match:
                kind = match.group(1) or "function"
                name = match.group(2) or match.group(3)
                current = Symbol(
                    _symbol_id(path, name, kind), path, name, name, kind,
                    line_number, line_number, language,
                )
                symbols.append(current)
            imports.update(JS_IMPORT.findall(line))
        for name in CALL.findall(line):
            if name not in CALL_KEYWORDS:
                calls.append((current.symbol_id if current else None, name, line_number))
    return ParsedFile(
        _record(path, language, raw, text, is_test=_is_test_path(path), imports=imports),
        symbols, imports, {}, calls,
    )


def _parse_parser_backed(path: str, text: str, raw: bytes, language: str) -> ParsedFile:
    """Use exact parser spans while retaining conservative dependency signals."""
    conservative = _parse_conservative(path, text, raw, language)
    spans = extract_structural_spans(text, path)
    if not spans:
        return conservative

    symbols: list[Symbol] = []
    active: list[tuple[Symbol, StructuralSpan]] = []
    for span in sorted(spans, key=lambda item: (item.start_byte, -item.end_byte)):
        while active and span.start_byte >= active[-1][1].end_byte:
            active.pop()
        parent = (
            active[-1][0]
            if active and span.end_byte <= active[-1][1].end_byte
            else None
        )
        qualified = f"{parent.qualified_name}.{span.name}" if parent else span.name
        kind = "test" if span.name.startswith(("test_", "test")) else span.kind
        # Preserve the v1 index identity contract used by call-edge resolution.
        if language == "rust" and kind == "function":
            kind = "fn"
        symbol = Symbol(
            _symbol_id(path, qualified, kind), path, span.name, qualified, kind,
            span.start_line, span.end_line, language,
            parent.symbol_id if parent else None,
        )
        symbols.append(symbol)
        active.append((symbol, span))
    return ParsedFile(
        conservative.record,
        symbols,
        conservative.imports,
        conservative.import_aliases,
        conservative.calls,
    )


def scan_repository(
    root: Path,
    limits: RepositoryLimits,
) -> tuple[dict[str, ParsedFile], list[str]]:
    parsed: dict[str, ParsedFile] = {}
    diagnostics: list[str] = []
    total = 0
    symbol_count = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            candidate = Path(directory) / filename
            if candidate.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            relative_hint = normalize_relative(candidate.relative_to(root))
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
                size = resolved.stat().st_size
            except (OSError, RuntimeError, ValueError):
                diagnostics.append(f"skipped unsafe or unreadable path: {relative_hint}")
                continue
            if size > limits.max_file_bytes:
                diagnostics.append(f"skipped oversized file: {relative_hint} ({size} bytes)")
                continue
            if len(parsed) >= limits.max_files or total + size > limits.max_total_bytes:
                diagnostics.append("repository limits reached; index truncated")
                return parsed, diagnostics
            try:
                raw = resolved.read_bytes()
                text = raw.decode("utf-8", errors="surrogateescape")
            except OSError as exc:
                diagnostics.append(f"failed to read {relative_hint}: {type(exc).__name__}")
                continue
            language = _language(resolved)
            if language == "python":
                item = _parse_python(relative_hint, text, raw)
            else:
                item = _parse_parser_backed(relative_hint, text, raw, language)
            remaining = max(0, limits.max_symbols - symbol_count)
            if len(item.symbols) > remaining:
                item.symbols[:] = item.symbols[:remaining]
                diagnostics.append("symbol limit reached; remaining symbols omitted")
            symbol_count += len(item.symbols)
            parsed[relative_hint] = item
            total += size
    return dict(sorted(parsed.items())), diagnostics
