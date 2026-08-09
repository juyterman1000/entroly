"""Language-neutral, parser-verified control-shape view.

The raw parser work is owned by :mod:`syntax_session`. This module converts the
shared bounded scan into a stable public graph contract without reparsing source.
It deliberately stops short of claiming a control-flow graph or data-flow proof.
"""
from __future__ import annotations

from dataclasses import dataclass

from .syntax_session import SyntaxScan, build_syntax_session, scan_syntax_session

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


def build_syntactic_flow_graph_from_scan(scan: SyntaxScan) -> SyntacticFlowGraph:
    """Project one already-computed syntax scan into the flow-shape contract."""
    nodes = tuple(
        SyntacticFlowNode(
            node_id=item.node_id,
            kind=item.kind,
            grammar_type=item.grammar_type,
            start_byte=item.start_byte,
            end_byte=item.end_byte,
            start_line=item.start_line,
            end_line=item.end_line,
            evidence_sha256=item.evidence_sha256,
            parent_id=item.parent_id,
        )
        for item in scan.flow_shapes
    )
    known = {node.node_id for node in nodes}
    edges = tuple(sorted(
        (
            SyntacticFlowEdge(
                source_id=node.parent_id,
                target_id=node.node_id,
                relation="contains-flow-shape",
            )
            for node in nodes
            if node.parent_id is not None and node.parent_id in known
        ),
        key=lambda item: (item.source_id, item.target_id, item.relation),
    ))
    return SyntacticFlowGraph(
        language=scan.language,
        source_sha256=scan.source_sha256,
        nodes=nodes,
        edges=edges,
        complete=scan.flow_complete,
        nodes_visited=scan.nodes_visited,
        max_nodes=scan.max_nodes,
    )


def build_syntactic_flow_graph(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
    max_flow_nodes: int = 20_000,
) -> SyntacticFlowGraph | None:
    """Standalone convenience wrapper using one parser invocation/traversal."""
    session = build_syntax_session(source, file_path, max_bytes=max_bytes)
    if session is None:
        return None
    scan = scan_syntax_session(
        session,
        max_nodes=max_nodes,
        max_flow_shapes=max_flow_nodes,
    )
    return build_syntactic_flow_graph_from_scan(scan)


__all__ = [
    "UNIVERSAL_FLOW_SCHEMA_VERSION",
    "SyntacticFlowEdge",
    "SyntacticFlowGraph",
    "SyntacticFlowNode",
    "build_syntactic_flow_graph",
    "build_syntactic_flow_graph_from_scan",
]
