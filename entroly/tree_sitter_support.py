"""Universal, bounded parser-backed structural extraction.

Repository intelligence remains available without the optional language registry.
When the registry is installed it is used for path/shebang detection and normalized
structure across its available grammars. Missing grammars are never acquired as a
side effect of reading code unless ``ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD=1`` is set;
``ENTROLY_AIR_GAP=1`` always disables acquisition. Any parser failure fails open to
deterministic fallbacks, and bounded traversal never presents a truncated tree as
complete.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterable, TypeVar


# Small offline fallback only. The installed language registry is authoritative
# when available, so new grammars do not require Entroly releases.
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
    ".asm": "asm", ".s": "asm", ".ada": "ada", ".adb": "ada",
    ".ads": "ada", ".fs": "fsharp", ".fsx": "fsharp", ".ml": "ocaml",
    ".mli": "ocaml", ".nim": "nim", ".v": "v", ".c3": "c3",
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
    "procedure_declaration", "subprogram_body", "subroutine",
})
_DECLARATION_HINTS = frozenset({
    "function", "method", "constructor", "class", "struct", "enum", "trait",
    "interface", "protocol", "impl", "namespace", "module", "record", "union",
    "macro", "procedure", "subroutine", "routine", "type_alias",
})

# Grammars whose declaration nodes the generic heuristic cannot match, keyed by
# language so the names cannot leak across grammars. These were read from the
# real parse trees, not guessed: Haskell names a definition `function` (bare,
# with a `name` field), Perl a sub `subroutine_declaration_statement` (ends in
# `_statement`, so the suffix rule skips it). Both carry a `name` field, so the
# existing name/kind extraction handles them unchanged. The names are
# deliberately NOT added to the global set: JavaScript and TypeScript both emit
# a bare `function` node for function expressions, and matching those globally
# would invent named declarations from anonymous expressions.
_LANGUAGE_DECLARATION_TYPES: dict[str, frozenset[str]] = {
    "haskell": frozenset({"function", "data_type"}),
    "perl": frozenset({"subroutine_declaration_statement"}),
    # Go names a type in `type_spec` under `type_declaration`; the suffix rule
    # sees only the wrapper and dropped every `type` in the file.
    "go": frozenset({"type_spec"}),
    # Solidity's contract/library/interface are `*_declaration` but the suffix
    # rule's hint list has no "contract"; its functions already matched.
    "solidity": frozenset({"contract_declaration", "library_declaration", "interface_declaration"}),
    # Erlang defines a function as one or more `function_clause`s, each naming
    # the function; nothing in the generic set matched them.
    "erlang": frozenset({"function_clause"}),
    # Zig names a function in `FnProto` positionally (a bare IDENTIFIER child);
    # the suffix rule matches none of its PascalCase node types.
    "zig": frozenset({"FnProto"}),
    # Protobuf `message` names via a `message_name` child, no `name` field.
    "proto": frozenset({"message", "enum"}),
    # SQL names a created object in an `object_reference` child, no `name` field.
    "sql": frozenset({"create_function", "create_procedure", "create_table", "create_view"}),
}

# Node types to NOT treat as declarations for a language, overriding the generic
# rule. R has no standalone function declaration: a function is an anonymous
# value bound by assignment, so the `function_definition` node itself carries no
# name and the generic rule emitted the keyword `function`. The binding is
# matched by predicate below instead.
_LANGUAGE_SUPPRESS_TYPES: dict[str, frozenset[str]] = {
    "r": frozenset({"function_definition"}),
}


def _r_binds_function(node: Any) -> bool:
    """An R assignment whose right-hand side is a function definition.

    `helper <- function(x) x + 1` parses as a `binary_operator` with the name
    on the left, an assignment operator, and the function on the right. This is
    the only place an R function acquires a name, so it is the declaration.
    """
    if str(getattr(node, "type", "")) != "binary_operator":
        return False
    children = list(getattr(node, "children", ()))
    if not any(str(getattr(c, "type", "")) in {"<-", "=", "<<-"} for c in children):
        return False
    return any(str(getattr(c, "type", "")) == "function_definition" for c in children)


# Node-level predicates for declarations the type name alone cannot identify.
_LANGUAGE_DECLARATION_PREDICATE: dict[str, Any] = {
    "r": _r_binds_function,
}

_KIND_HINTS = (
    ("method", "method"), ("function", "function"), ("constructor", "constructor"),
    ("class", "class"), ("struct", "struct"), ("enum", "enum"),
    ("trait", "trait"), ("interface", "interface"), ("protocol", "interface"),
    ("impl", "implementation"), ("namespace", "namespace"),
    ("module", "module"), ("type", "type"), ("macro", "macro"),
    ("procedure", "function"), ("subroutine", "function"),
)


@dataclass(frozen=True)
class StructuralSpan:
    """One exact source span backed by a concrete parser result."""

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


@dataclass(frozen=True)
class StructuralProfile:
    """Parser-derived control-shape metrics for one declaration."""

    name: str
    kind: str
    start_byte: int
    end_byte: int
    decision_points: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    max_control_nesting: int
    parameter_count: int
    return_points: int


_T = TypeVar("_T")


@dataclass(frozen=True)
class StructuralExtraction(Generic[_T]):
    """Bounded extraction result that makes traversal completeness explicit."""

    items: tuple[_T, ...]
    language: str
    complete: bool
    nodes_visited: int
    max_nodes: int
    backend: str


@dataclass
class _TraversalState:
    visited: int = 0
    complete: bool = True


def _true(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _false(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off"}


def _pack() -> Any | None:
    try:
        import tree_sitter_language_pack as pack
    except (ImportError, OSError):
        return None
    return pack


def _normalize_language(value: object) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    return text or None


def language_for_path(file_path: str) -> str | None:
    """Detect language through the live registry, with an offline fallback."""
    pack = _pack()
    if pack is not None:
        for function_name in ("detect_language_from_path", "detect_language"):
            detector = getattr(pack, function_name, None)
            if callable(detector):
                try:
                    language = _normalize_language(detector(str(file_path)))
                except Exception:
                    language = None
                if language:
                    return language
    return LANGUAGE_BY_SUFFIX.get(Path(file_path).suffix.lower())


def language_for_source(file_path: str, source: str) -> str | None:
    language = language_for_path(file_path)
    if language:
        return language
    pack = _pack()
    detector = getattr(pack, "detect_language_from_content", None) if pack else None
    if callable(detector):
        try:
            return _normalize_language(detector(source))
        except Exception:
            return None
    return None


def available_parser_languages() -> tuple[str, ...]:
    """Return registry languages without triggering parser downloads."""
    pack = _pack()
    available = getattr(pack, "available_languages", None) if pack else None
    if not callable(available):
        return tuple(sorted(set(LANGUAGE_BY_SUFFIX.values())))
    try:
        return tuple(sorted({_normalize_language(item) for item in available()} - {None}))
    except Exception:
        return tuple(sorted(set(LANGUAGE_BY_SUFFIX.values())))


def _downloads_allowed() -> bool:
    """Return whether parser acquisition is explicitly authorized for this process."""
    if _true(os.getenv("ENTROLY_AIR_GAP")):
        return False
    return _true(os.getenv("ENTROLY_TREE_SITTER_ALLOW_DOWNLOAD"))


def _get_local_parser(language: str) -> Any | None:
    pack = _pack()
    if pack is None:
        return None
    if not _downloads_allowed():
        normalized = _normalize_language(language)
        locally_loadable = False
        has_language = getattr(pack, "has_language", None)
        if callable(has_language):
            try:
                locally_loadable = bool(has_language(language))
            except Exception:
                return None
        else:
            local_languages: set[str | None] = set()
            available = getattr(pack, "available_languages", None)
            if callable(available):
                try:
                    local_languages.update(
                        _normalize_language(item) for item in available()
                    )
                except Exception:
                    pass
            downloaded = getattr(pack, "downloaded_languages", None)
            if callable(downloaded):
                try:
                    local_languages.update(
                        _normalize_language(item) for item in downloaded()
                    )
                except Exception:
                    pass
            locally_loadable = normalized in local_languages
        if not locally_loadable:
            return None
    try:
        return pack.get_parser(language)
    except Exception:
        return None


def _walk(root: Any, max_nodes: int, state: _TraversalState) -> Iterable[Any]:
    stack = [root]
    while stack:
        if state.visited >= max_nodes:
            state.complete = False
            return
        node = stack.pop()
        state.visited += 1
        yield node
        children = getattr(node, "children", ())
        stack.extend(reversed(children))


def _first_identifier(node: Any) -> Any | None:
    stack = [node]
    while stack:
        current = stack.pop()
        # Error-recovery subtrees hold fragments in the wrong place: a dense
        # one-line `class W{ fun run(){} }` mis-parses, and descending into the
        # ERROR node returns an inner function name as the class name. Skipping
        # them keeps resolution on the real, well-formed structure.
        if str(getattr(current, "type", "")) == "ERROR" or bool(
            getattr(current, "is_error", False)
        ):
            continue
        if getattr(current, "type", "") in {
            "identifier", "type_identifier", "field_identifier", "property_identifier",
            "constant", "operator_name", "name",
        }:
            return current
        stack.extend(reversed(getattr(current, "children", ())))
    return None


# Grammars that name a declaration with a bare positional child and expose no
# `name` field for it. The value is the child node type that holds the name, and
# only a DIRECT child is taken: Kotlin's `function_declaration` lists
# `simple_identifier` (the name), `function_value_parameters`, `user_type` (the
# return type) and `function_body` as siblings with no fields, so the generic
# `_first_identifier` descends into the return type and returns `Int` for
# `fun helper(): Int`, and finds nothing for a body-less nested `fun run()`.
# Taking the first direct `simple_identifier` returns the real name and cannot
# reach a parameter (nested in function_value_parameters) or a return type
# (nested in user_type). A class uses `type_identifier`, not `simple_identifier`,
# so this does not fire for it and the existing resolution still applies.
_LANGUAGE_NAME_CHILD_TYPE: dict[str, str] = {
    "kotlin": "simple_identifier",
    "zig": "IDENTIFIER",
    "proto": "message_name",
    "sql": "object_reference",
    # An R function binding names itself with the identifier left of the arrow.
    "r": "identifier",
}

# Grammars whose pack structure processing returns partial declarations (a named
# class but null-named functions, no method descent), where the name-resolving
# tree walk is more complete. Add a language only after verifying the walk does
# strictly better for it, since the walk's own per-grammar coverage varies.
_PACK_STRUCTURE_UNRELIABLE: frozenset[str] = frozenset(
    {"kotlin", "dart", "go", "solidity", "r"}
)


def _name_node(node: Any, language: str = "") -> Any | None:
    preferred = _LANGUAGE_NAME_CHILD_TYPE.get(language) if language else None
    if preferred:
        for child in getattr(node, "children", ()):
            if str(getattr(child, "type", "")) == preferred:
                return child
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
    lowered = node_type.lower()
    for hint, kind in _KIND_HINTS:
        if hint in lowered:
            return kind
    return "declaration"


def _language_scoped_node_ok(node: Any, node_type: str, language: str) -> bool:
    """A language-scoped declaration must carry a resolvable name.

    The scoped node names are broad on purpose, and some are overloaded within
    their own grammar: Haskell's `function` denotes both a definition
    (`helper x = x + 1`, which has a `name`) and a function TYPE (`Int -> Int`,
    which has none). A name is resolvable when the node has an explicit `name`
    field, or -- for the positional grammars that expose none -- a direct child
    of the type this language uses for names (Zig's `IDENTIFIER` under
    `FnProto`). The Haskell function type has neither, so it is still rejected
    and the walk does not emit `Int` as a function.
    """
    if node_type not in _LANGUAGE_DECLARATION_TYPES.get(language, ()):
        return True
    field = getattr(node, "child_by_field_name", None)
    if callable(field) and field("name") is not None:
        return True
    preferred = _LANGUAGE_NAME_CHILD_TYPE.get(language)
    if preferred:
        return any(
            str(getattr(child, "type", "")) == preferred
            for child in getattr(node, "children", ())
        )
    return False


def _is_declaration_type(node_type: str, language: str = "") -> bool:
    if language and node_type in _LANGUAGE_DECLARATION_TYPES.get(language, ()):
        return True
    lowered = node_type.lower()
    if lowered in _DECLARATION_TYPES:
        return True
    if not any(hint in lowered for hint in _DECLARATION_HINTS):
        return False
    return lowered.endswith((
        "_definition", "_declaration", "_item", "_specifier",
    )) or lowered in {"method", "module", "subroutine"}


_CALL_TYPES = frozenset({
    "call", "call_expression", "function_call", "function_call_expression",
    "invocation_expression", "method_invocation", "method_call_expression",
})


def _is_call_type(node_type: str) -> bool:
    lowered = node_type.lower()
    if lowered in _CALL_TYPES:
        return True
    if "argument" in lowered or "callable" in lowered:
        return False
    return lowered.endswith(("_call", "_invocation", "_call_expression"))


def _parse_source(
    source: str,
    file_path: str,
    max_bytes: int,
) -> tuple[bytes, Any, str] | None:
    language = language_for_source(file_path, source)
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
        return raw, parser.parse(raw), language
    except Exception:
        return None


def _structure_kind(value: object) -> str:
    raw = getattr(value, "name", None) or str(value)
    text = str(raw).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    text = text.replace("structurekind", "").replace("_", "-").strip(" -")
    if text.startswith("other("):
        return "declaration"
    return {
        "impl": "implementation",
        "type-alias": "type",
    }.get(text, text or "declaration")


def _pack_structure_spans(
    source: str,
    file_path: str,
    raw: bytes,
    language: str,
) -> list[StructuralSpan] | None:
    """Use the registry's cross-language normalized structure when available."""
    pack = _pack()
    process = getattr(pack, "process", None) if pack else None
    config_type = getattr(pack, "ProcessConfig", None) if pack else None
    if not callable(process) or config_type is None:
        return None
    try:
        config = config_type(
            language=language,
            structure=True,
            imports=False,
            exports=False,
            comments=False,
            docstrings=False,
            symbols=False,
            diagnostics=True,
        )
        result = process(source, config)
        roots = list(getattr(result, "structure", ()))
    except Exception:
        return None

    spans: list[StructuralSpan] = []
    stack = list(reversed(roots))
    while stack:
        item = stack.pop()
        children = list(getattr(item, "children", ()))
        stack.extend(reversed(children))
        name = str(getattr(item, "name", "") or "").strip()
        span = getattr(item, "span", None)
        if not name or span is None:
            continue
        start = int(getattr(span, "start_byte", -1))
        end = int(getattr(span, "end_byte", -1))
        if not (0 <= start < end <= len(raw)):
            continue
        source_slice = raw[start:end].decode("utf-8", errors="surrogateescape")
        signature = str(getattr(item, "signature", "") or "").strip()
        if not signature:
            signature = source_slice.splitlines()[0].strip() if source_slice else ""
        first_line = source_slice.splitlines()[0] if source_slice.splitlines() else ""
        spans.append(StructuralSpan(
            name=name,
            kind=_structure_kind(getattr(item, "kind", "declaration")),
            start_line=int(getattr(span, "start_line", 0)) + 1,
            end_line=int(getattr(span, "end_line", 0)) + 1,
            source=source_slice,
            signature=signature[:600],
            indent=len(first_line) - len(first_line.lstrip()),
            start_byte=start,
            end_byte=end,
        ))
    return sorted(spans, key=lambda item: (item.start_byte, -item.end_byte, item.name))


def _spans_from_tree(
    raw: bytes,
    tree: Any,
    language: str,
    max_nodes: int,
) -> StructuralExtraction[StructuralSpan]:
    spans: list[StructuralSpan] = []
    seen: set[tuple[int, int, str]] = set()
    state = _TraversalState()
    suppressed = _LANGUAGE_SUPPRESS_TYPES.get(language, frozenset())
    predicate = _LANGUAGE_DECLARATION_PREDICATE.get(language)
    for node in _walk(tree.root_node, max_nodes, state):
        node_type = str(getattr(node, "type", ""))
        if bool(getattr(node, "is_error", False)):
            continue
        is_declaration = (
            node_type not in suppressed and _is_declaration_type(node_type, language)
        ) or (predicate is not None and predicate(node))
        if not is_declaration:
            continue
        if not _language_scoped_node_ok(node, node_type, language):
            continue
        # A declaration with a direct ERROR child was recovered from a syntax
        # error, so its own structure is unreliable -- a dense `class W{ ... }`
        # can otherwise take an inner method name as the class name. Skipping it
        # keeps a malformed span out rather than emitting a mislabelled one; the
        # well-formed declarations in the same file are unaffected.
        if any(str(getattr(c, "type", "")) == "ERROR" for c in getattr(node, "children", ())):
            continue
        start = int(getattr(node, "start_byte", 0))
        end = int(getattr(node, "end_byte", 0))
        if end <= start or end > len(raw):
            continue
        name_node = _name_node(node, language)
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
        signature = raw[start:max(start, min(signature_end, end))].decode(
            "utf-8", errors="surrogateescape"
        ).strip()
        if not signature or len(signature) > 600:
            signature = raw[start:end].decode(
                "utf-8", errors="surrogateescape"
            ).splitlines()[0].strip()
        block_source = raw[start:end].decode("utf-8", errors="surrogateescape")
        first_line = block_source.splitlines()[0] if block_source.splitlines() else ""
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
    return StructuralExtraction(
        items=tuple(spans),
        language=language,
        complete=state.complete,
        nodes_visited=state.visited,
        max_nodes=max_nodes,
        backend="tree-sitter-raw",
    )


# Single-file-component languages whose script is embedded as opaque text, and
# the language to parse that text as. Svelte and Vue keep the script inside a
# `script_element` whose only child is a `raw_text` node -- the grammar does not
# descend into it, so the component's functions are invisible until that text is
# re-parsed as JavaScript and its spans are shifted back to file coordinates.
_EMBEDDED_SCRIPT_LANGUAGES: dict[str, str] = {
    "svelte": "javascript",
    "vue": "javascript",
}


def _embedded_script_spans(
    raw: bytes, tree: Any, language: str, max_nodes: int
) -> list[StructuralSpan]:
    """Extract declarations from the `<script>` block of a component file."""
    inner_language = _EMBEDDED_SCRIPT_LANGUAGES.get(language)
    if inner_language is None:
        return []
    parser = _get_local_parser(inner_language)
    if parser is None:
        return []
    spans: list[StructuralSpan] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        parent = getattr(node, "parent", None)
        if (
            str(getattr(node, "type", "")) == "raw_text"
            and parent is not None
            and str(getattr(parent, "type", "")) == "script_element"
        ):
            offset = int(getattr(node, "start_byte", 0))
            line_offset = int(node.start_point[0])
            inner_raw = raw[offset:int(getattr(node, "end_byte", 0))]
            inner_source = inner_raw.decode("utf-8", errors="surrogateescape")
            # Prefer the pack's structure, which is clean for JavaScript; the
            # raw tree walk is only the fallback, matching the main path so the
            # embedded script is analysed exactly as a standalone file would be.
            inner_items = _pack_structure_spans(
                inner_source, "embedded.js", inner_raw, inner_language
            )
            if not inner_items:
                inner_items = list(
                    _spans_from_tree(
                        inner_raw, parser.parse(inner_raw), inner_language, max_nodes
                    ).items
                )
            for item in inner_items:
                spans.append(StructuralSpan(
                    name=item.name,
                    kind=item.kind,
                    start_line=item.start_line + line_offset,
                    end_line=item.end_line + line_offset,
                    source=item.source,
                    signature=item.signature,
                    indent=item.indent,
                    start_byte=item.start_byte + offset,
                    end_byte=item.end_byte + offset,
                ))
        stack.extend(getattr(node, "children", ()))
    return spans


_CONTROL_TYPES = frozenset({
    "if_statement", "if_expression", "elif_clause", "unless_expression",
    "for_statement", "for_expression", "enhanced_for_statement",
    "while_statement", "while_expression", "do_statement",
    "catch_clause", "except_clause", "rescue", "rescue_clause",
    "case_statement", "case_clause", "switch_case", "match_arm",
    "when_entry", "conditional_expression", "ternary_expression",
})
_RETURN_TYPES = frozenset({
    "return_statement", "yield_expression", "yield_statement",
    "throw_statement", "raise_statement",
})
_PARAMETER_CONTAINER_TYPES = frozenset({
    "parameters", "formal_parameters", "parameter_list",
    "lambda_parameters", "closure_parameters",
})


def _profile_for_node(node: Any, raw: bytes) -> StructuralProfile | None:
    name_node = _name_node(node)
    if name_node is None:
        return None
    name = raw[int(name_node.start_byte):int(name_node.end_byte)].decode(
        "utf-8", errors="surrogateescape"
    ).strip()
    if not name:
        return None
    decisions = 0
    cognitive = 0
    max_nesting = 0
    returns = 0
    parameters = 0
    stack: list[tuple[Any, int]] = [(node, 0)]
    while stack:
        current, nesting = stack.pop()
        node_type = str(getattr(current, "type", ""))
        if current is not node and _is_declaration_type(node_type):
            continue
        is_control = current is not node and node_type in _CONTROL_TYPES
        child_nesting = nesting
        if is_control:
            decisions += 1
            cognitive += 1 + nesting
            child_nesting = nesting + 1
            max_nesting = max(max_nesting, child_nesting)
        if current is not node and node_type in _RETURN_TYPES:
            returns += 1
        if node_type in _PARAMETER_CONTAINER_TYPES:
            named_children = list(getattr(current, "named_children", ()))
            parameters = max(parameters, sum(
                1 for child in named_children
                if "comment" not in str(getattr(child, "type", ""))
            ))
        children = list(getattr(current, "named_children", ()))
        stack.extend((child, child_nesting) for child in reversed(children))
    return StructuralProfile(
        name=name,
        kind=_kind(str(getattr(node, "type", ""))),
        start_byte=int(getattr(node, "start_byte", 0)),
        end_byte=int(getattr(node, "end_byte", 0)),
        decision_points=decisions,
        cyclomatic_complexity=1 + decisions,
        cognitive_complexity=cognitive,
        max_control_nesting=max_nesting,
        parameter_count=parameters,
        return_points=returns,
    )


def extract_structural_profiles_report(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> StructuralExtraction[StructuralProfile] | None:
    parsed = _parse_source(source, file_path, max_bytes)
    if parsed is None:
        return None
    raw, tree, language = parsed
    profiles: list[StructuralProfile] = []
    state = _TraversalState()
    for node in _walk(tree.root_node, max_nodes, state):
        node_type = str(getattr(node, "type", ""))
        if not _is_declaration_type(node_type, language) or bool(getattr(node, "is_error", False)):
            continue
        if not _language_scoped_node_ok(node, node_type, language):
            continue
        if not any(hint in node_type.lower() for hint in (
            "function", "method", "constructor", "procedure", "subroutine",
        )):
            continue
        profile = _profile_for_node(node, raw)
        if profile is not None and 0 <= profile.start_byte < profile.end_byte <= len(raw):
            profiles.append(profile)
    return StructuralExtraction(
        items=tuple(sorted(profiles, key=lambda item: (item.start_byte, item.end_byte, item.name))),
        language=language,
        complete=state.complete,
        nodes_visited=state.visited,
        max_nodes=max_nodes,
        backend="tree-sitter-raw",
    )


def extract_structural_profiles(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> list[StructuralProfile] | None:
    report = extract_structural_profiles_report(
        source, file_path, max_bytes=max_bytes, max_nodes=max_nodes
    )
    if report is None or not report.complete:
        return None
    return list(report.items) or None


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


def extract_structural_calls_report(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> StructuralExtraction[StructuralCall] | None:
    parsed = _parse_source(source, file_path, max_bytes)
    if parsed is None:
        return None
    raw, tree, language = parsed
    calls: list[StructuralCall] = []
    seen: set[tuple[int, int, str]] = set()
    state = _TraversalState()
    for node in _walk(tree.root_node, max_nodes, state):
        node_type = str(getattr(node, "type", ""))
        if not _is_call_type(node_type) or bool(getattr(node, "is_error", False)):
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
    return StructuralExtraction(
        items=tuple(calls),
        language=language,
        complete=state.complete,
        nodes_visited=state.visited,
        max_nodes=max_nodes,
        backend="tree-sitter-raw",
    )


def extract_structural_calls(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> list[StructuralCall] | None:
    report = extract_structural_calls_report(
        source, file_path, max_bytes=max_bytes, max_nodes=max_nodes
    )
    if report is None or not report.complete:
        return None
    return list(report.items)


def extract_structural_spans_report(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> StructuralExtraction[StructuralSpan] | None:
    parsed = _parse_source(source, file_path, max_bytes)
    if parsed is None:
        return None
    raw, tree, language = parsed
    if language in _EMBEDDED_SCRIPT_LANGUAGES:
        # The component's own markup has no declarations; its script does, and
        # the grammar leaves that as opaque text. Re-parse the script block.
        embedded = _embedded_script_spans(raw, tree, language, max_nodes)
        metrics_state = _TraversalState()
        for _node in _walk(tree.root_node, max_nodes, metrics_state):
            pass
        return StructuralExtraction(
            items=tuple(embedded),
            language=language,
            complete=metrics_state.complete,
            nodes_visited=metrics_state.visited,
            max_nodes=max_nodes,
            backend="embedded-script",
        )
    # The pack's structure processing under-resolves some grammars: for Kotlin
    # it returns the top-level `fun` with a null name (dropped for lack of a
    # name) and omits a class's own methods entirely, yielding just the class.
    # For these the name-resolving tree walk below is strictly more complete, so
    # the pack path is skipped rather than trusted for its partial output. Kept
    # to an explicit set: forcing the fallback generally regressed JavaScript,
    # whose anonymous arrow is legitimately nameless and whose pack result is
    # already correct.
    normalized = (
        None
        if language in _PACK_STRUCTURE_UNRELIABLE
        else _pack_structure_spans(source, file_path, raw, language)
    )
    if normalized:
        # Registry processing owns its traversal and provides exact spans. Keep
        # the raw parser's node count only as a bounded completeness check.
        metrics_state = _TraversalState()
        for _node in _walk(tree.root_node, max_nodes, metrics_state):
            pass
        return StructuralExtraction(
            items=tuple(normalized),
            language=language,
            complete=metrics_state.complete,
            nodes_visited=metrics_state.visited,
            max_nodes=max_nodes,
            backend="language-pack-process",
        )
    return _spans_from_tree(raw, tree, language, max_nodes)


def extract_structural_spans(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> list[StructuralSpan] | None:
    report = extract_structural_spans_report(
        source, file_path, max_bytes=max_bytes, max_nodes=max_nodes
    )
    if report is None or not report.complete:
        return None
    return list(report.items) or None


def validate_structural_syntax(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
) -> bool | None:
    parsed = _parse_source(source, file_path, max_bytes)
    if parsed is None:
        return None
    _raw, tree, _language = parsed
    root = getattr(tree, "root_node", None)
    return not bool(getattr(root, "has_error", True))


__all__ = [
    "LANGUAGE_BY_SUFFIX",
    "StructuralCall",
    "StructuralExtraction",
    "StructuralProfile",
    "StructuralSpan",
    "available_parser_languages",
    "extract_structural_calls",
    "extract_structural_calls_report",
    "extract_structural_profiles",
    "extract_structural_profiles_report",
    "extract_structural_spans",
    "extract_structural_spans_report",
    "language_for_path",
    "language_for_source",
    "validate_structural_syntax",
]
