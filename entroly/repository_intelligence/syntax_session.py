"""One raw parser session for universal syntax facts.

Repository intelligence needs several raw-tree facts (syntax validity, calls,
control shape). Re-parsing the same source independently for each feature makes
language breadth expensive and creates inconsistent truncation behavior. This
module owns one parser invocation and one bounded traversal, then exposes
separately bounded fact families with explicit completeness flags.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..tree_sitter_support import (
    StructuralCall,
    _call_target,
    _get_local_parser,
    _is_call_type,
    language_for_source,
)

SYNTAX_SESSION_SCHEMA_VERSION = "entroly.syntax-session.v1"


@dataclass(frozen=True)
class SyntaxFlowShape:
    node_id: str
    kind: str
    grammar_type: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    evidence_sha256: str
    parent_id: str | None


@dataclass(frozen=True)
class SyntaxSession:
    file_path: str
    language: str
    raw: bytes
    tree: Any
    source_sha256: str
    syntax_valid: bool


@dataclass(frozen=True)
class SyntaxScan:
    language: str
    source_sha256: str
    calls: tuple[StructuralCall, ...]
    flow_shapes: tuple[SyntaxFlowShape, ...]
    traversal_complete: bool
    calls_complete: bool
    flow_complete: bool
    nodes_visited: int
    max_nodes: int
    max_calls: int
    max_flow_shapes: int


def _flow_kind(node_type: str) -> str | None:
    value = node_type.casefold().replace("-", "_")
    if not value or value in {"program", "source_file", "module"}:
        return None
    if any(token in value for token in (
        "if_statement", "if_expression", "conditional_expression",
        "switch_statement", "switch_expression", "match_expression",
        "match_statement", "case_statement", "case_clause", "match_arm",
        "when_entry",
    )):
        return "branch"
    if any(token in value for token in (
        "for_statement", "for_expression", "for_in_statement",
        "while_statement", "while_expression", "do_statement",
        "loop_expression", "repeat_statement",
    )):
        return "loop"
    if any(token in value for token in (
        "return_statement", "return_expression", "yield_statement",
        "yield_expression", "throw_statement", "throw_expression",
        "raise_statement",
    )):
        return "terminal"
    if any(token in value for token in (
        "break_statement", "break_expression", "continue_statement",
        "continue_expression", "goto_statement",
    )):
        return "jump"
    if "assignment" in value or value in {
        "augmented_assignment", "assignment_expression"
    }:
        return "assignment"
    if any(token in value for token in (
        "variable_declaration", "variable_declarator", "let_declaration",
        "const_declaration", "declaration_statement", "init_declarator",
    )):
        return "declaration"
    if _is_call_type(value):
        return "call"
    return None


def _shape_id(
    path: str,
    kind: str,
    grammar_type: str,
    start: int,
    end: int,
) -> str:
    material = f"{path}\0{kind}\0{grammar_type}\0{start}\0{end}".encode("utf-8")
    return "syntax-flow:" + hashlib.sha256(material).hexdigest()[:24]


def build_syntax_session(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
) -> SyntaxSession | None:
    """Parse one source artifact exactly once, or return ``None`` safely."""
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
        tree = parser.parse(raw)
    except Exception:
        return None
    root = getattr(tree, "root_node", None)
    if root is None:
        return None
    return SyntaxSession(
        file_path=file_path.replace("\\", "/"),
        language=language,
        raw=raw,
        tree=tree,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        syntax_valid=not bool(getattr(root, "has_error", True)),
    )


def scan_syntax_session(
    session: SyntaxSession,
    *,
    max_nodes: int = 100_000,
    max_calls: int = 50_000,
    max_flow_shapes: int = 20_000,
) -> SyntaxScan:
    """Extract calls and flow-relevant syntax in one bounded traversal.

    The traversal continues after a per-family budget is hit so another family
    can still be complete. Absence from a family with ``*_complete=False`` must
    never be interpreted as negative evidence.
    """
    traversal_limit = max(1, min(int(max_nodes), 1_000_000))
    call_limit = max(1, min(int(max_calls), 500_000))
    flow_limit = max(1, min(int(max_flow_shapes), 200_000))
    root = session.tree.root_node
    stack: list[tuple[Any, str | None]] = [(root, None)]
    visited = 0
    traversal_complete = True
    calls_complete = True
    flow_complete = True
    calls: list[StructuralCall] = []
    call_seen: set[tuple[int, int, str]] = set()
    flow_shapes: list[SyntaxFlowShape] = []
    flow_seen: set[str] = set()

    while stack:
        if visited >= traversal_limit:
            traversal_complete = False
            calls_complete = False
            flow_complete = False
            break
        node, flow_parent = stack.pop()
        visited += 1
        node_type = str(getattr(node, "type", ""))
        is_error = bool(getattr(node, "is_error", False))
        start = int(getattr(node, "start_byte", 0))
        end = int(getattr(node, "end_byte", 0))
        valid_span = 0 <= start < end <= len(session.raw)

        if not is_error and valid_span and _is_call_type(node_type):
            if len(calls) >= call_limit:
                calls_complete = False
            else:
                target = _call_target(node, session.raw)
                key = (start, end, target)
                if target and key not in call_seen:
                    call_seen.add(key)
                    calls.append(StructuralCall(
                        target=target,
                        start_line=int(getattr(node, "start_point", (0, 0))[0]) + 1,
                        start_byte=start,
                        end_byte=end,
                        evidence_sha256=hashlib.sha256(session.raw[start:end]).hexdigest(),
                    ))

        kind = _flow_kind(node_type) if not is_error and valid_span else None
        next_parent = flow_parent
        if kind is not None:
            node_id = _shape_id(session.file_path, kind, node_type, start, end)
            if node_id not in flow_seen:
                if len(flow_shapes) >= flow_limit:
                    flow_complete = False
                else:
                    flow_seen.add(node_id)
                    start_point = getattr(node, "start_point", (0, 0))
                    end_point = getattr(node, "end_point", start_point)
                    flow_shapes.append(SyntaxFlowShape(
                        node_id=node_id,
                        kind=kind,
                        grammar_type=node_type,
                        start_byte=start,
                        end_byte=end,
                        start_line=int(start_point[0]) + 1,
                        end_line=int(end_point[0]) + 1,
                        evidence_sha256=hashlib.sha256(session.raw[start:end]).hexdigest(),
                        parent_id=flow_parent,
                    ))
                    next_parent = node_id

        children = list(
            getattr(node, "named_children", ())
            or getattr(node, "children", ())
        )
        stack.extend((child, next_parent) for child in reversed(children))

    return SyntaxScan(
        language=session.language,
        source_sha256=session.source_sha256,
        calls=tuple(sorted(
            calls,
            key=lambda item: (item.start_byte, item.end_byte, item.target),
        )),
        flow_shapes=tuple(sorted(
            flow_shapes,
            key=lambda item: (
                item.start_byte, item.end_byte, item.kind, item.node_id
            ),
        )),
        traversal_complete=traversal_complete,
        calls_complete=calls_complete and traversal_complete,
        flow_complete=flow_complete and traversal_complete,
        nodes_visited=visited,
        max_nodes=traversal_limit,
        max_calls=call_limit,
        max_flow_shapes=flow_limit,
    )


__all__ = [
    "SYNTAX_SESSION_SCHEMA_VERSION",
    "SyntaxFlowShape",
    "SyntaxScan",
    "SyntaxSession",
    "build_syntax_session",
    "scan_syntax_session",
]
