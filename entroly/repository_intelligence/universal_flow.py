"""Language-neutral, parser-verified control-shape extraction.

This module deliberately stops short of claiming a control-flow graph or data-
flow proof. Across arbitrary Tree-sitter grammars we can verify that concrete
source spans are branches, loops, calls, assignments, jumps, or terminals by
syntax shape. We cannot generically prove path feasibility, aliasing, binding,
or reaching definitions. Those stronger facts belong to compiler/LSP/static-
analysis adapters and carry stronger epistemic classes elsewhere.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from ..tree_sitter_support import _get_local_parser, language_for_source

UNIVERSAL_FLOW_SCHEMA_VERSION = "entroly.universal-syntactic-flow.v1"


@dataclass(frozen=True)
class SyntacticFlowNode:
    node_id: str
    kind: str
    grammar_type: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    evidence_sha256: str
    parent_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "grammar_type": self.grammar_type,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "evidence_sha256": self.evidence_sha256,
            "parent_id": self.parent_id,
            "epistemic_class": "parser-verified",
        }


@dataclass(frozen=True)
class SyntacticFlowEdge:
    source_id: str
    target_id: str
    relation: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "epistemic_class": "parser-verified",
        }


@dataclass(frozen=True)
class SyntacticFlowGraph:
    language: str
    source_sha256: str
    nodes: tuple[SyntacticFlowNode, ...]
    edges: tuple[SyntacticFlowEdge, ...]
    complete: bool
    nodes_visited: int
    max_nodes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": UNIVERSAL_FLOW_SCHEMA_VERSION,
            "language": self.language,
            "source_sha256": self.source_sha256,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "complete": self.complete,
            "nodes_visited": self.nodes_visited,
            "max_nodes": self.max_nodes,
            "analysis_contract": {
                "guarantee": "parser-verified-control-shape-only",
                "not_claimed": [
                    "control-flow-feasibility",
                    "definition-use-binding",
                    "alias-or-heap-flow",
                    "dynamic-dispatch",
                    "exception-propagation",
                    "macro-or-generated-semantics",
                ],
            },
        }


def _kind(node_type: str) -> str | None:
    value = node_type.casefold().replace("-", "_")
    if not value or value in {"program", "source_file", "module"}:
        return None
    if any(token in value for token in ("if_statement", "if_expression", "conditional_expression")):
        return "branch"
    if any(token in value for token in (
        "switch_statement", "switch_expression", "match_expression", "match_statement",
        "case_statement", "case_clause", "match_arm", "when_entry",
    )):
        return "branch"
    if any(token in value for token in (
        "for_statement", "for_expression", "for_in_statement", "while_statement",
        "while_expression", "do_statement", "loop_expression", "repeat_statement",
    )):
        return "loop"
    if any(token in value for token in (
        "return_statement", "return_expression", "yield_statement", "yield_expression",
        "throw_statement", "throw_expression", "raise_statement",
    )):
        return "terminal"
    if any(token in value for token in (
        "break_statement", "break_expression", "continue_statement", "continue_expression",
        "goto_statement",
    )):
        return "jump"
    if "assignment" in value or value in {"augmented_assignment", "assignment_expression"}:
        return "assignment"
    if any(token in value for token in (
        "variable_declaration", "variable_declarator", "let_declaration",
        "const_declaration", "declaration_statement", "init_declarator",
    )):
        return "declaration"
    if any(token in value for token in (
        "call_expression", "call", "function_call", "invocation_expression",
        "method_invocation", "method_call_expression",
    )):
        return "call"
    return None


def _node_id(path: str, kind: str, grammar_type: str, start: int, end: int) -> str:
    material = f"{path}\0{kind}\0{grammar_type}\0{start}\0{end}".encode("utf-8")
    return "syntax-flow:" + hashlib.sha256(material).hexdigest()[:24]


def build_syntactic_flow_graph(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
    max_flow_nodes: int = 20_000,
) -> SyntacticFlowGraph | None:
    """Return exact flow-relevant syntax spans for any available grammar.

    A bounded traversal that reaches ``max_nodes`` is explicitly marked
    incomplete. Consumers must not treat the absence of a node/edge in an
    incomplete graph as negative evidence.
    """
    language = language_for_source(file_path, source)
    if not language or not source.strip():
        return None
    raw = source.encode("utf-8", errors="surrogateescape")
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

    traversal_limit = max(1, min(int(max_nodes), 1_000_000))
    flow_limit = max(1, min(int(max_flow_nodes), 200_000))
    stack: list[tuple[Any, str | None]] = [(root, None)]
    nodes: list[SyntacticFlowNode] = []
    edges: list[SyntacticFlowEdge] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()
    visited = 0
    traversal_exhausted = False
    flow_exhausted = False

    while stack:
        if visited >= traversal_limit:
            traversal_exhausted = True
            break
        current, semantic_parent = stack.pop()
        visited += 1
        grammar_type = str(getattr(current, "type", ""))
        kind = _kind(grammar_type)
        next_parent = semantic_parent
        if kind is not None and not bool(getattr(current, "is_error", False)):
            start = int(getattr(current, "start_byte", 0))
            end = int(getattr(current, "end_byte", 0))
            if 0 <= start < end <= len(raw):
                node_id = _node_id(file_path, kind, grammar_type, start, end)
                if node_id not in seen_nodes:
                    if len(nodes) >= flow_limit:
                        flow_exhausted = True
                        break
                    seen_nodes.add(node_id)
                    evidence = hashlib.sha256(raw[start:end]).hexdigest()
                    start_point = getattr(current, "start_point", (0, 0))
                    end_point = getattr(current, "end_point", start_point)
                    node = SyntacticFlowNode(
                        node_id=node_id,
                        kind=kind,
                        grammar_type=grammar_type,
                        start_byte=start,
                        end_byte=end,
                        start_line=int(start_point[0]) + 1,
                        end_line=int(end_point[0]) + 1,
                        evidence_sha256=evidence,
                        parent_id=semantic_parent,
                    )
                    nodes.append(node)
                    if semantic_parent is not None:
                        edge_key = (semantic_parent, node_id, "contains-flow-shape")
                        if edge_key not in seen_edges:
                            seen_edges.add(edge_key)
                            edges.append(SyntacticFlowEdge(*edge_key))
                next_parent = node_id

        children = list(getattr(current, "named_children", ()) or getattr(current, "children", ()))
        stack.extend((child, next_parent) for child in reversed(children))

    complete = not traversal_exhausted and not flow_exhausted and not stack
    nodes.sort(key=lambda item: (item.start_byte, item.end_byte, item.kind, item.node_id))
    edges.sort(key=lambda item: (item.source_id, item.target_id, item.relation))
    return SyntacticFlowGraph(
        language=language,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        complete=complete,
        nodes_visited=visited,
        max_nodes=traversal_limit,
    )


__all__ = [
    "UNIVERSAL_FLOW_SCHEMA_VERSION",
    "SyntacticFlowEdge",
    "SyntacticFlowGraph",
    "SyntacticFlowNode",
    "build_syntactic_flow_graph",
]
