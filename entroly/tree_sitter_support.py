"""Optional, local-first parser-backed structural extraction.

The language pack is an accelerator, never a correctness dependency.  Entroly
does not download parser binaries as a side effect of reading a file: language
pack 1.x parsers are used only when already cached, unless the operator opts in
with ``ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD=1``.  Any import, parse, or grammar
failure returns ``None`` so callers can use their exact deterministic fallback.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


LANGUAGE_BY_SUFFIX: dict[str, str] = {
    ".py": "python", ".pyi": "python", ".pyw": "python",
    ".rs": "rust", ".js": "javascript", ".jsx": "javascript",
    ".mjs": "javascript", ".cjs": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".go": "go", ".java": "java", ".kt": "kotlin",
    ".kts": "kotlin", ".cs": "c_sharp", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".swift": "swift", ".rb": "ruby", ".php": "php", ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash", ".fish": "fish",
    ".lua": "lua", ".ex": "elixir", ".exs": "elixir", ".erl": "erlang",
    ".hrl": "erlang", ".hs": "haskell", ".dart": "dart", ".r": "r",
    ".vue": "vue", ".svelte": "svelte", ".html": "html", ".css": "css",
    ".scss": "scss", ".sql": "sql", ".proto": "proto", ".sol": "solidity",
    ".zig": "zig", ".jl": "julia", ".groovy": "groovy",
}

_DECLARATION_TYPES = frozenset({
    "function_definition", "function_declaration", "function_item",
    "function_signature", "method_definition", "method_declaration",
    "method", "constructor_declaration", "class_definition",
    "class_declaration", "class_specifier", "struct_item", "struct_specifier",
    "struct_declaration", "enum_item", "enum_declaration", "enum_specifier",
    "trait_item", "trait_declaration", "interface_declaration",
    "interface_definition", "impl_item", "implementation_definition",
    "type_alias_declaration", "type_item", "module", "module_declaration",
    "namespace_definition", "object_declaration", "protocol_declaration",
    "record_declaration", "union_specifier", "macro_definition",
})

_KIND_HINTS = (
    ("method", "method"), ("function", "function"), ("constructor", "constructor"),
    ("class", "class"), ("struct", "struct"), ("enum", "enum"),
    ("trait", "trait"), ("interface", "interface"), ("protocol", "interface"),
    ("impl", "implementation"), ("namespace", "namespace"),
    ("module", "module"), ("type", "type"), ("macro", "macro"),
)


@dataclass(frozen=True)
class StructuralSpan:
    """One exact source span backed by a concrete parser node."""

    name: str
    kind: str
    start_line: int
    end_line: int
    source: str
    signature: str
    indent: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class StructuralCall:
    """One parser-observed call site with byte-exact source evidence."""

    target: str
    start_line: int
    start_byte: int
    end_byte: int
    evidence_sha256: str


def language_for_path(file_path: str) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(Path(file_path).suffix.lower())


def _downloads_allowed() -> bool:
    return os.getenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _get_local_parser(language: str) -> Any | None:
    try:
        import tree_sitter_language_pack as pack
    except (ImportError, OSError):
        return None

    # Pack 1.x downloads a missing parser on get_parser().  Avoid that surprise.
    downloaded = getattr(pack, "downloaded_languages", None)
    if callable(downloaded) and not _downloads_allowed():
        try:
            cached = {str(item).lower().replace("-", "_") for item in downloaded()}
        except Exception:
            return None
        if language.lower().replace("-", "_") not in cached:
            return None
    try:
        return pack.get_parser(language)
    except Exception:
        return None


def _walk(root: Any, max_nodes: int) -> Iterable[Any]:
    stack = [root]
    seen = 0
    while stack and seen < max_nodes:
        node = stack.pop()
        seen += 1
        yield node
        children = getattr(node, "children", ())
        stack.extend(reversed(children))


def _first_identifier(node: Any) -> Any | None:
    stack = [node]
    while stack:
        current = stack.pop()
        if getattr(current, "type", "") in {
            "identifier", "type_identifier", "field_identifier", "property_identifier",
            "constant", "operator_name",
        }:
            return current
        stack.extend(reversed(getattr(current, "children", ())))
    return None


def _name_node(node: Any) -> Any | None:
    field = getattr(node, "child_by_field_name", None)
    if callable(field):
        named = field("name")
        if named is not None:
            return named
        for name in ("declarator", "type"):
            candidate = field(name)
            if candidate is not None:
                identifier = _first_identifier(candidate)
                if identifier is not None:
                    return identifier
    return _first_identifier(node)


def _kind(node_type: str) -> str:
    for hint, kind in _KIND_HINTS:
        if hint in node_type:
            return kind
    return "declaration"


_CALL_TYPES = frozenset({
    "call", "call_expression", "function_call", "function_call_expression",
    "invocation_expression", "method_invocation", "method_call_expression",
})


def _parse_source(
    source: str,
    file_path: str,
    max_bytes: int,
) -> tuple[bytes, Any] | None:
    language = language_for_path(file_path)
    if not language or not source.strip():
        return None
    try:
        raw = source.encode("utf-8", errors="surrogateescape")
    except UnicodeEncodeError:
        return None
    if len(raw) > max_bytes:
        return None
    parser = _get_local_parser(language)
    if parser is None:
        return None
    try:
        return raw, parser.parse(raw)
    except Exception:
        return None


def _spans_from_tree(raw: bytes, tree: Any, max_nodes: int) -> list[StructuralSpan]:
    spans: list[StructuralSpan] = []
    seen: set[tuple[int, int, str]] = set()
    for node in _walk(tree.root_node, max_nodes):
        node_type = str(getattr(node, "type", ""))
        if node_type not in _DECLARATION_TYPES or bool(getattr(node, "is_error", False)):
            continue
        start = int(getattr(node, "start_byte", 0))
        end = int(getattr(node, "end_byte", 0))
        if end <= start or end > len(raw):
            continue
        name_node = _name_node(node)
        if name_node is None:
            continue
        name = raw[int(name_node.start_byte):int(name_node.end_byte)].decode(
            "utf-8", errors="surrogateescape"
        ).strip()
        if not name or len(name) > 256 or "\n" in name:
            continue
        key = (start, end, node_type)
        if key in seen:
            continue
        seen.add(key)
        body = None
        field = getattr(node, "child_by_field_name", None)
        if callable(field):
            body = field("body")
        signature_end = int(getattr(body, "start_byte", end)) if body is not None else end
        signature_bytes = raw[start:max(start, min(signature_end, end))]
        signature = signature_bytes.decode("utf-8", errors="surrogateescape").strip()
        if not signature or len(signature) > 600:
            signature = raw[start:end].decode(
                "utf-8", errors="surrogateescape"
            ).splitlines()[0].strip()
        block_source = raw[start:end].decode("utf-8", errors="surrogateescape")
        block_lines = block_source.splitlines()
        first_line = block_lines[0] if block_lines else ""
        spans.append(StructuralSpan(
            name=name,
            kind=_kind(node_type),
            start_line=int(node.start_point[0]) + 1,
            end_line=int(node.end_point[0]) + 1,
            source=block_source,
            signature=signature,
            indent=len(first_line) - len(first_line.lstrip()),
            start_byte=start,
            end_byte=end,
        ))
    return spans


def _call_target(node: Any, raw: bytes) -> str:
    field = getattr(node, "child_by_field_name", None)
    candidate = None
    if callable(field):
        for name in ("function", "method", "name"):
            candidate = field(name)
            if candidate is not None:
                break
    if candidate is None:
        children = list(getattr(node, "named_children", ()))
        candidate = children[0] if children else None
    if candidate is None:
        return ""
    value = raw[int(candidate.start_byte):int(candidate.end_byte)].decode(
        "utf-8", errors="surrogateescape"
    ).strip()
    if not value or len(value) > 512 or "\n" in value:
        return ""
    return value


def extract_structural_calls(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> list[StructuralCall] | None:
    """Return parser-backed call sites, or ``None`` when safely unavailable."""
    parsed = _parse_source(source, file_path, max_bytes)
    if parsed is None:
        return None
    raw, tree = parsed
    calls: list[StructuralCall] = []
    seen: set[tuple[int, int, str]] = set()
    for node in _walk(tree.root_node, max_nodes):
        node_type = str(getattr(node, "type", ""))
        if node_type not in _CALL_TYPES or bool(getattr(node, "is_error", False)):
            continue
        start = int(getattr(node, "start_byte", 0))
        end = int(getattr(node, "end_byte", 0))
        target = _call_target(node, raw)
        key = (start, end, target)
        if not target or end <= start or end > len(raw) or key in seen:
            continue
        seen.add(key)
        evidence = raw[start:end]
        calls.append(StructuralCall(
            target=target,
            start_line=int(node.start_point[0]) + 1,
            start_byte=start,
            end_byte=end,
            evidence_sha256=hashlib.sha256(evidence).hexdigest(),
        ))
    return calls


def extract_structural_spans(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> list[StructuralSpan] | None:
    """Return parser-backed exact spans, or ``None`` when safely unavailable."""
    parsed = _parse_source(source, file_path, max_bytes)
    if parsed is None:
        return None
    raw, tree = parsed
    spans = _spans_from_tree(raw, tree, max_nodes)
    return spans or None
