"""Bounded source discovery and deterministic symbol extraction."""
from __future__ import annotations

import ast
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import FileRecord, RepositoryLimits, Symbol, normalize_relative
from .registry_frontend import extract_registry_facts
from ..tree_sitter_support import (
    StructuralCall,
    StructuralSpan,
    extract_structural_calls,
    extract_structural_spans,
    language_for_path,
    language_for_source,
)

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
NON_SOURCE_SUFFIXES = frozenset({
    ".7z", ".a", ".avi", ".bmp", ".class", ".dll", ".dylib", ".exe",
    ".gif", ".gz", ".ico", ".jar", ".jpeg", ".jpg", ".lockb", ".mov",
    ".mp3", ".mp4", ".o", ".obj", ".pdf", ".png", ".pyc", ".so",
    ".tar", ".tiff", ".ttf", ".wav", ".webm", ".webp", ".woff",
    ".woff2", ".xz", ".zip",
})

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
# A name introduced by one of these keywords is being declared, not called.
# `CALL` cannot tell "fn run(&self)" from "run()", so without this every
# declaration became a call to itself. Matched against the text preceding the
# name so it also catches declarations nested mid-line, as in
# "impl Worker { fn run(&self) { helper(); } }", which no line-anchored
# declaration pattern sees.
DECLARATION_PREFIX = re.compile(
    r"(?:^|[^\w$])(?:fn|function|class|struct|enum|trait|impl|mod|def|sub)\s+$"
)


@dataclass
class ParsedFile:
    record: FileRecord
    symbols: list[Symbol]
    imports: set[str]
    import_aliases: dict[str, str]
    calls: list["ParsedCall"]


@dataclass(frozen=True)
class ParsedCall:
    caller_id: str | None
    target: str
    line: int
    start_byte: int = 0
    end_byte: int = 0
    evidence_sha256: str = ""
    parse_backend: str = "conservative"
    receiver_type: str = ""
    receiver_binding: str = ""


def module_name(path: str) -> str:
    parts = list(Path(path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _language(path: Path, text: str = "") -> str:
    return language_for_source(str(path), text) or language_for_path(str(path)) or "unknown"


def _is_test_path(path: str) -> bool:
    parsed = Path(path)
    parts = {part.lower() for part in parsed.parts[:-1]}
    stem = parsed.stem.lower()
    return bool(parts & TEST_PARTS) or stem.startswith("test_") or any(
        stem.endswith(suffix) for suffix in ("_test", ".test", ".spec")
    )


def _looks_like_source(path: str, text: str) -> bool:
    """Conservative fallback for a language newer than the local registry."""
    if not text.strip() or "\x00" in text:
        return False
    first = text.splitlines()[0].strip() if text.splitlines() else ""
    if first.startswith("#!"):
        return True
    suffix = Path(path).suffix.lower()
    if suffix in {".md", ".rst", ".txt", ".csv"}:
        return False
    sample = text[:16_384]
    structural = sum(sample.count(token) for token in ("{", "}", "(", ")", "=>"))
    keywords = sum(
        1
        for token in (
            "fn ", "def ", "func ", "function ", "class ", "struct ",
            "enum ", "import ", "use ", "package ", "module ", "return ",
        )
        if token in sample
    )
    return structural >= 4 or keywords >= 2


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
    def __init__(self, path: str, text: str, raw: bytes) -> None:
        self.path = path
        self.text = text
        self.raw = raw
        self._line_offsets = [0]
        running = 0
        for line in text.splitlines(keepends=True):
            running += len(line.encode("utf-8", errors="surrogateescape"))
            self._line_offsets.append(running)
        self.symbols: list[Symbol] = []
        self.imports: set[str] = set()
        self.aliases: dict[str, str] = {}
        self.calls: list[ParsedCall] = []
        self.scope: list[Symbol] = []
        self.type_scopes: list[dict[str, tuple[str, str]]] = [{}]

    @staticmethod
    def _annotation_name(node: ast.AST | None) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            owner = PythonVisitor._annotation_name(node.value)
            return f"{owner}.{node.attr}" if owner else node.attr
        if isinstance(node, ast.Subscript):
            base = PythonVisitor._annotation_name(node.value)
            if base in {"Optional", "typing.Optional"}:
                return PythonVisitor._annotation_name(node.slice)
            return base
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.strip(" '\"")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = PythonVisitor._annotation_name(node.left)
            right = PythonVisitor._annotation_name(node.right)
            non_none = [item for item in (left, right) if item not in {"", "None"}]
            return non_none[0] if len(non_none) == 1 else ""
        return ""

    def _lookup_type(self, name: str) -> tuple[str, str]:
        for scope in reversed(self.type_scopes):
            if name in scope:
                return scope[name]
        return "", ""

    def _expression_type(self, node: ast.AST) -> tuple[str, str]:
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id, "constructor-assignment"
            if isinstance(node.func, ast.Attribute):
                owner = self._annotation_name(node.func.value)
                if owner:
                    return f"{owner}.{node.func.attr}", "constructor-assignment"
        if isinstance(node, ast.Name):
            inferred, _binding = self._lookup_type(node.id)
            if inferred:
                return inferred, "assignment-propagation"
        return "", ""

    def _byte_range(self, node: ast.AST) -> tuple[int, int]:
        line = max(1, int(getattr(node, "lineno", 1)))
        end_line = max(line, int(getattr(node, "end_lineno", line)))
        start = self._line_offsets[min(line - 1, len(self._line_offsets) - 1)]
        start += max(0, int(getattr(node, "col_offset", 0)))
        end = self._line_offsets[min(end_line - 1, len(self._line_offsets) - 1)]
        end += max(0, int(getattr(node, "end_col_offset", 0)))
        return min(start, len(self.raw)), min(max(start, end), len(self.raw))

    def _symbol(self, node: ast.AST, name: str, kind: str) -> Symbol:
        qualified = ".".join([*(item.name for item in self.scope), name])
        start, end = self._byte_range(node)
        signature_end = end
        body = getattr(node, "body", None)
        if body:
            signature_end = self._byte_range(body[0])[0]
        signature = self.raw[start:signature_end].decode(
            "utf-8", errors="surrogateescape"
        ).strip()
        if not signature or len(signature) > 600:
            signature = self.raw[start:end].decode(
                "utf-8", errors="surrogateescape"
            ).splitlines()[0].strip()
        symbol = Symbol(
            _symbol_id(self.path, qualified, kind), self.path, name, qualified, kind,
            max(1, int(getattr(node, "lineno", 1))),
            max(1, int(getattr(node, "end_lineno", getattr(node, "lineno", 1)))),
            "python", self.scope[-1].symbol_id if self.scope else None,
            signature, start, end, "python-ast",
            hashlib.sha256(self.raw[start:end]).hexdigest(),
        )
        self.symbols.append(symbol)
        return symbol

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        symbol = self._symbol(node, node.name, "class")
        self.scope.append(symbol)
        self.type_scopes.append({})
        self.generic_visit(node)
        self.type_scopes.pop()
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        kind = "method" if self.scope and self.scope[-1].kind == "class" else "function"
        if node.name.startswith("test_"):
            kind = "test"
        symbol = self._symbol(node, node.name, kind)
        local_types: dict[str, tuple[str, str]] = {}
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for argument in arguments:
            annotation = self._annotation_name(argument.annotation)
            if annotation:
                local_types[argument.arg] = (annotation, "annotation")
        if self.scope and self.scope[-1].kind == "class" and arguments:
            first = arguments[0].arg
            if first in {"self", "cls"}:
                local_types[first] = (self.scope[-1].qualified_name, "implicit-self")
        self.scope.append(symbol)
        self.type_scopes.append(local_types)
        self.generic_visit(node)
        self.type_scopes.pop()
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

    def visit_Assign(self, node: ast.Assign) -> None:
        inferred, binding = self._expression_type(node.value)
        if inferred:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.type_scopes[-1][target.id] = (inferred, binding)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            inferred = self._annotation_name(node.annotation)
            binding = "annotation"
            if not inferred and node.value is not None:
                inferred, binding = self._expression_type(node.value)
            if inferred:
                self.type_scopes[-1][node.target.id] = (inferred, binding)
        self.generic_visit(node)

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
            receiver_type, receiver_binding = (
                self._lookup_type(owner) if owner else ("", "")
            )
            start, end = self._byte_range(node)
            self.calls.append(ParsedCall(
                caller,
                target,
                max(1, int(getattr(node, "lineno", 1))),
                start,
                end,
                hashlib.sha256(self.raw[start:end]).hexdigest(),
                "python-ast",
                receiver_type,
                receiver_binding,
            ))
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
    visitor = PythonVisitor(path, text, raw)
    visitor.visit(tree)
    is_test = _is_test_path(path) or any(item.kind == "test" for item in visitor.symbols)
    return ParsedFile(
        _record(path, "python", raw, text, is_test=is_test, imports=visitor.imports),
        visitor.symbols, visitor.imports, visitor.aliases, visitor.calls,
    )


def _parse_conservative(path: str, text: str, raw: bytes, language: str) -> ParsedFile:
    symbols: list[Symbol] = []
    imports: set[str] = set()
    calls: list[ParsedCall] = []
    js_family = language in {"javascript", "typescript", "tsx"}
    # Keeping the terminators lets us advance an exact byte cursor alongside the
    # line loop. `text` came from raw.decode("utf-8", "surrogateescape"), so
    # re-encoding a prefix the same way reproduces the original byte length --
    # including for astral characters and undecodable bytes. Splitting `raw` on
    # b"\n" instead would desync here, because str.splitlines() also breaks on
    # \r, \x0b, \x0c, \x1c-\x1e and U+2028/U+2029.
    byte_cursor = 0
    for line_number, terminated in enumerate(text.splitlines(keepends=True), 1):
        stripped = terminated.splitlines()
        line = stripped[0] if stripped else ""
        line_start = byte_cursor
        byte_cursor += len(terminated.encode("utf-8", "surrogateescape"))
        current: Symbol | None = None
        # Where this line declares a name, rather than calls one. `CALL` also
        # matches the "helper(" inside "function helper()", which turned every
        # single-line declaration into a call to itself.
        declared_name_span: tuple[int, int] | None = None
        if language == "rust":
            match = RUST_DEF.match(line)
            if match:
                kind, name = match.groups()
                declared_name_span = match.span(2)
                current = Symbol(
                    _symbol_id(path, name, kind), path, name, name, kind,
                    line_number, line_number, language, None, line.strip(),
                    0, 0, "conservative",
                )
                symbols.append(current)
            use = RUST_USE.match(line)
            if use:
                imports.add(use.group(1).strip().split("::{", 1)[0])
        elif js_family:
            match = JS_DEF.match(line)
            if match:
                kind = match.group(1) or "function"
                name = match.group(2) or match.group(3)
                declared_name_span = match.span(2 if match.group(2) else 3)
                current = Symbol(
                    _symbol_id(path, name, kind), path, name, name, kind,
                    line_number, line_number, language, None, line.strip(),
                    0, 0, "conservative",
                )
                symbols.append(current)
            imports.update(JS_IMPORT.findall(line))
        if language == "rust" or js_family:
            for call_match in CALL.finditer(line):
                name = call_match.group(1)
                if name in CALL_KEYWORDS:
                    continue
                # Suppress by position, not by name. Skipping every occurrence
                # matching the enclosing symbol would also drop real recursion
                # written on one line, as in "fn f() { f() }" -- there only the
                # first occurrence is a declaration.
                prefix_text = line[:call_match.start(1)]
                if call_match.span(1) == declared_name_span or DECLARATION_PREFIX.search(
                    prefix_text
                ):
                    continue
                # Every emitted edge must carry evidence a caller can recover
                # and re-hash. Leaving these at the ParsedCall defaults shipped
                # a (0, 0) span with an empty digest, so the edge asserted a
                # call while validating against zero bytes.
                prefix = line[:call_match.start(1)].encode("utf-8", "surrogateescape")
                token = name.encode("utf-8", "surrogateescape")
                start_byte = line_start + len(prefix)
                end_byte = start_byte + len(token)
                calls.append(ParsedCall(
                    current.symbol_id if current else None,
                    name,
                    line_number,
                    start_byte,
                    end_byte,
                    hashlib.sha256(raw[start_byte:end_byte]).hexdigest(),
                ))
    return ParsedFile(
        _record(path, language, raw, text, is_test=_is_test_path(path), imports=imports),
        symbols, imports, {}, calls,
    )


def _parse_parser_backed(path: str, text: str, raw: bytes, language: str) -> ParsedFile:
    """Use normalized parser spans/facts while retaining safe fallback signals."""
    conservative = _parse_conservative(path, text, raw, language)
    spans = extract_structural_spans(text, path)
    if spans is None:
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
        if language == "rust" and kind == "function":
            kind = "fn"
        symbol = Symbol(
            _symbol_id(path, qualified, kind), path, span.name, qualified, kind,
            span.start_line, span.end_line, language,
            parent.symbol_id if parent else None,
            span.signature,
            span.start_byte,
            span.end_byte,
            "tree-sitter",
            hashlib.sha256(raw[span.start_byte:span.end_byte]).hexdigest(),
        )
        symbols.append(symbol)
        active.append((symbol, span))

    parser_calls = extract_structural_calls(text, path)
    calls = conservative.calls
    if parser_calls is not None:
        calls = _attribute_structural_calls(parser_calls, symbols)

    # Registry import extraction is syntax evidence, not binding evidence.  It
    # may strengthen file dependency/impact analysis, but import_aliases stays
    # untouched so call resolution cannot invent a callee from parser syntax.
    imports = set(conservative.imports)
    facts = extract_registry_facts(text, language)
    if facts is not None and facts.complete:
        imports.update(item.source for item in facts.imports if item.source)

    is_test = _is_test_path(path) or any(symbol.kind == "test" for symbol in symbols)
    record = _record(
        path,
        language,
        raw,
        text,
        is_test=is_test,
        imports=imports,
    )
    return ParsedFile(
        record,
        symbols,
        imports,
        conservative.import_aliases,
        calls,
    )


def _attribute_structural_calls(
    calls: list[StructuralCall],
    symbols: list[Symbol],
) -> list[ParsedCall]:
    """Attach each parser-observed call to its narrowest enclosing symbol."""
    result: list[ParsedCall] = []
    for call in calls:
        enclosing = [
            symbol
            for symbol in symbols
            if symbol.start_byte <= call.start_byte < symbol.end_byte
        ]
        owner = min(
            enclosing,
            key=lambda symbol: (symbol.end_byte - symbol.start_byte, symbol.symbol_id),
            default=None,
        )
        result.append(ParsedCall(
            owner.symbol_id if owner else None,
            call.target,
            call.start_line,
            call.start_byte,
            call.end_byte,
            call.evidence_sha256,
            "tree-sitter",
        ))
    return result


def scan_repository(
    root: Path,
    limits: RepositoryLimits,
    *,
    load_cached: Callable[[str, str], ParsedFile | None] | None = None,
    store_cached: Callable[[str, str, ParsedFile], None] | None = None,
) -> tuple[dict[str, ParsedFile], list[str]]:
    parsed: dict[str, ParsedFile] = {}
    diagnostics: list[str] = []
    total = 0
    symbol_count = 0
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            candidate = Path(directory) / filename
            if candidate.suffix.lower() in NON_SOURCE_SUFFIXES:
                continue
            relative_hint = normalize_relative(candidate.relative_to(root))
            path_language = language_for_path(relative_hint)
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
                size = resolved.stat().st_size
            except (OSError, RuntimeError, ValueError):
                diagnostics.append(f"skipped unsafe or unreadable path: {relative_hint}")
                continue
            if size > limits.max_file_bytes:
                if path_language:
                    diagnostics.append(
                        f"skipped oversized file: {relative_hint} ({size} bytes)"
                    )
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
            if "\x00" in text:
                continue
            language = language_for_source(relative_hint, text) or path_language
            if language is None:
                if not _looks_like_source(relative_hint, text):
                    continue
                language = "unknown"
            source_sha256 = hashlib.sha256(raw).hexdigest()
            item = load_cached(relative_hint, source_sha256) if load_cached else None
            if item is None:
                if language == "python":
                    item = _parse_python(relative_hint, text, raw)
                else:
                    item = _parse_parser_backed(relative_hint, text, raw, language)
                if store_cached:
                    store_cached(relative_hint, source_sha256, item)
            remaining = max(0, limits.max_symbols - symbol_count)
            if len(item.symbols) > remaining:
                item.symbols[:] = item.symbols[:remaining]
                diagnostics.append("symbol limit reached; remaining symbols omitted")
            symbol_count += len(item.symbols)
            parsed[relative_hint] = item
            total += size
    return dict(sorted(parsed.items())), diagnostics


def scan_repository_scope(
    root: Path,
    paths: set[str],
    limits: RepositoryLimits,
    *,
    load_cached: Callable[[str, str], ParsedFile | None] | None = None,
    store_cached: Callable[[str, str, ParsedFile], None] | None = None,
) -> tuple[dict[str, ParsedFile], tuple[str, ...], list[str]]:
    """Parse only an active path set while cataloguing dependency targets.

    The catalog walk reads no source bytes for out-of-scope files. It exists so
    import resolution can still emit one-hop boundary paths without turning a
    one-file edit into a whole-repository parse/cache reload.
    """
    selected = {normalize_relative(path) for path in paths if normalize_relative(path)}
    parsed: dict[str, ParsedFile] = {}
    catalog: list[str] = []
    diagnostics: list[str] = []
    total = 0
    symbol_count = 0
    catalog_full = False

    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            candidate = Path(directory) / filename
            if candidate.suffix.lower() in NON_SOURCE_SUFFIXES:
                continue
            relative_hint = normalize_relative(candidate.relative_to(root))
            path_language = language_for_path(relative_hint)
            if path_language is None and relative_hint not in selected:
                continue
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
                size = resolved.stat().st_size
            except (OSError, RuntimeError, ValueError):
                if relative_hint in selected:
                    diagnostics.append(f"skipped unsafe or unreadable path: {relative_hint}")
                continue
            if size > limits.max_file_bytes:
                if relative_hint in selected:
                    diagnostics.append(
                        f"skipped oversized file: {relative_hint} ({size} bytes)"
                    )
                continue
            if len(catalog) >= limits.max_files:
                diagnostics.append("repository catalog limit reached; dependency scope truncated")
                catalog_full = True
                break
            catalog.append(relative_hint)
            if relative_hint not in selected:
                continue
            if total + size > limits.max_total_bytes:
                diagnostics.append("active repository scope byte limit reached")
                continue
            try:
                raw = resolved.read_bytes()
                text = raw.decode("utf-8", errors="surrogateescape")
            except OSError as exc:
                diagnostics.append(f"failed to read {relative_hint}: {type(exc).__name__}")
                continue
            if "\x00" in text:
                continue
            language = language_for_source(relative_hint, text) or path_language
            if language is None:
                if not _looks_like_source(relative_hint, text):
                    continue
                language = "unknown"
            source_sha256 = hashlib.sha256(raw).hexdigest()
            item = load_cached(relative_hint, source_sha256) if load_cached else None
            if item is None:
                if language == "python":
                    item = _parse_python(relative_hint, text, raw)
                else:
                    item = _parse_parser_backed(relative_hint, text, raw, language)
                if store_cached:
                    store_cached(relative_hint, source_sha256, item)
            remaining = max(0, limits.max_symbols - symbol_count)
            if len(item.symbols) > remaining:
                item.symbols[:] = item.symbols[:remaining]
                diagnostics.append("symbol limit reached; remaining symbols omitted")
            symbol_count += len(item.symbols)
            parsed[relative_hint] = item
            total += size
        if catalog_full:
            break

    missing = selected - set(parsed)
    for path in sorted(missing):
        diagnostics.append(f"active source path not indexed: {path}")
    return dict(sorted(parsed.items())), tuple(sorted(catalog)), diagnostics
