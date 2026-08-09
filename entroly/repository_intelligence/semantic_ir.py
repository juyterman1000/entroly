"""Language-independent, evidence-carrying semantic representation for source code.

The IR deliberately hides grammar-specific Tree-sitter node names from the
rest of Entroly.  Parser/compiler/LSP frontends may strengthen the same graph
without changing the agent-facing schema.  Unknown languages still receive a
bounded exact-source structural skeleton rather than an "unsupported" error.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path

from ..tree_sitter_support import (
    StructuralCall,
    StructuralSpan,
    extract_structural_calls,
    extract_structural_spans,
    language_for_path,
    validate_structural_syntax,
)

SEMANTIC_IR_SCHEMA_VERSION = "entroly.semantic-ir.v1"
_MAX_FALLBACK_REGIONS = 2_000
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


class SemanticLevel(IntEnum):
    """Progressive semantic strength available for a source artifact."""

    SOURCE = 0
    SYNTAX = 1
    STRUCTURE = 2
    SEMANTICS = 3
    FLOW = 4
    TRANSFORMATION = 5


class EpistemicClass(StrEnum):
    """How strongly a node/edge is justified."""

    EXACT_SOURCE = "exact-source"
    PARSER_VERIFIED = "parser-verified"
    SOUND_STATIC = "sound-static"
    OBSERVED_RUNTIME = "observed-runtime"
    INFERRED = "inferred"
    LEARNED_PROPOSAL = "learned-proposal"
    HEURISTIC = "heuristic"


@dataclass(frozen=True)
class SourceEvidence:
    path: str
    start_byte: int
    end_byte: int
    sha256: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
            "sha256": self.sha256,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass(frozen=True)
class SemanticNode:
    node_id: str
    kind: str
    name: str
    language: str
    evidence: SourceEvidence
    epistemic_class: EpistemicClass
    parent_id: str | None = None
    signature: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "name": self.name,
            "language": self.language,
            "parent_id": self.parent_id,
            "signature": self.signature,
            "epistemic_class": self.epistemic_class.value,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class SemanticEdge:
    edge_id: str
    source_id: str
    relation: str
    target_id: str
    epistemic_class: EpistemicClass
    evidence: SourceEvidence | None = None
    target_name: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "relation": self.relation,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "epistemic_class": self.epistemic_class.value,
            "evidence": self.evidence.to_dict() if self.evidence else None,
        }


@dataclass(frozen=True)
class SemanticCapabilities:
    language: str
    level: SemanticLevel
    exact_source: bool = True
    parser_available: bool = False
    structure: bool = False
    semantic_binding: bool = False
    control_data_flow: bool = False
    verified_transformations: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "level": self.level.name.lower(),
            "level_number": int(self.level),
            "exact_source": self.exact_source,
            "parser_available": self.parser_available,
            "structure": self.structure,
            "semantic_binding": self.semantic_binding,
            "control_data_flow": self.control_data_flow,
            "verified_transformations": self.verified_transformations,
        }


@dataclass
class UniversalSemanticDocument:
    path: str
    language: str
    source_sha256: str
    byte_length: int
    capabilities: SemanticCapabilities
    nodes: list[SemanticNode] = field(default_factory=list)
    edges: list[SemanticEdge] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SEMANTIC_IR_SCHEMA_VERSION,
            "path": self.path,
            "language": self.language,
            "source_sha256": self.source_sha256,
            "byte_length": self.byte_length,
            "capabilities": self.capabilities.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "diagnostics": list(self.diagnostics),
        }


def _line_offsets(raw: bytes) -> list[int]:
    offsets = [0]
    cursor = 0
    while True:
        newline = raw.find(b"\n", cursor)
        if newline < 0:
            break
        cursor = newline + 1
        offsets.append(cursor)
    return offsets


def _line_for_offset(offsets: list[int], position: int) -> int:
    lo = 0
    hi = len(offsets)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= position:
            lo = mid
        else:
            hi = mid
    return lo + 1


def _evidence(path: str, raw: bytes, start: int, end: int, offsets: list[int]) -> SourceEvidence:
    start = max(0, min(start, len(raw)))
    end = max(start, min(end, len(raw)))
    return SourceEvidence(
        path=path,
        start_byte=start,
        end_byte=end,
        sha256=hashlib.sha256(raw[start:end]).hexdigest(),
        start_line=_line_for_offset(offsets, start),
        end_line=_line_for_offset(offsets, max(start, end - 1)),
    )


def _node_id(path: str, kind: str, name: str, start: int, end: int) -> str:
    material = f"{path}\0{kind}\0{name}\0{start}\0{end}".encode("utf-8")
    return "node:" + hashlib.sha256(material).hexdigest()[:24]


def _edge_id(source: str, relation: str, target: str, evidence: SourceEvidence | None) -> str:
    span = "" if evidence is None else f"{evidence.path}:{evidence.start_byte}:{evidence.end_byte}"
    material = f"{source}\0{relation}\0{target}\0{span}".encode("utf-8")
    return "edge:" + hashlib.sha256(material).hexdigest()[:24]


def _semantic_nodes(
    path: str,
    language: str,
    raw: bytes,
    spans: list[StructuralSpan],
    offsets: list[int],
) -> tuple[list[SemanticNode], list[SemanticEdge]]:
    nodes: list[SemanticNode] = []
    edges: list[SemanticEdge] = []
    active: list[SemanticNode] = []
    ordered = sorted(spans, key=lambda item: (item.start_byte, -item.end_byte, item.name))
    for span in ordered:
        while active and span.start_byte >= active[-1].evidence.end_byte:
            active.pop()
        parent = (
            active[-1]
            if active and span.end_byte <= active[-1].evidence.end_byte
            else None
        )
        evidence = _evidence(path, raw, span.start_byte, span.end_byte, offsets)
        node = SemanticNode(
            node_id=_node_id(path, span.kind, span.name, span.start_byte, span.end_byte),
            kind=span.kind,
            name=span.name,
            language=language,
            parent_id=parent.node_id if parent else None,
            signature=span.signature,
            epistemic_class=EpistemicClass.PARSER_VERIFIED,
            evidence=evidence,
        )
        nodes.append(node)
        if parent is not None:
            edges.append(SemanticEdge(
                edge_id=_edge_id(parent.node_id, "contains", node.node_id, evidence),
                source_id=parent.node_id,
                relation="contains",
                target_id=node.node_id,
                epistemic_class=EpistemicClass.PARSER_VERIFIED,
                evidence=evidence,
                target_name=node.name,
            ))
        active.append(node)
    return nodes, edges


def _owner_for_call(nodes: list[SemanticNode], call: StructuralCall) -> SemanticNode | None:
    candidates = [
        node
        for node in nodes
        if node.evidence.start_byte <= call.start_byte
        and call.end_byte <= node.evidence.end_byte
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item.evidence.end_byte - item.evidence.start_byte)


def _call_edges(
    path: str,
    raw: bytes,
    nodes: list[SemanticNode],
    calls: list[StructuralCall],
    offsets: list[int],
) -> list[SemanticEdge]:
    edges: list[SemanticEdge] = []
    root_id = _node_id(path, "file", Path(path).name, 0, len(raw))
    for call in calls:
        owner = _owner_for_call(nodes, call)
        source_id = owner.node_id if owner else root_id
        evidence = _evidence(path, raw, call.start_byte, call.end_byte, offsets)
        target_id = "unresolved-name:" + hashlib.sha256(
            call.target.encode("utf-8", errors="surrogateescape")
        ).hexdigest()[:24]
        edges.append(SemanticEdge(
            edge_id=_edge_id(source_id, "invokes-name", target_id, evidence),
            source_id=source_id,
            relation="invokes-name",
            target_id=target_id,
            target_name=call.target,
            # A parser proves the call expression and spelling, not binding.
            epistemic_class=EpistemicClass.PARSER_VERIFIED,
            evidence=evidence,
        ))
    return edges


def _fallback_regions(path: str, raw: bytes, offsets: list[int]) -> list[SemanticNode]:
    """Create an exact-source region skeleton without claiming language semantics."""
    nodes: list[SemanticNode] = []
    stack: list[tuple[int, int]] = []
    quote: int | None = None
    escaped = False
    for index, byte in enumerate(raw):
        if len(nodes) >= _MAX_FALLBACK_REGIONS:
            break
        if quote is not None:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == quote:
                quote = None
            continue
        if byte in (0x22, 0x27):  # rough string shielding; remains heuristic
            quote = byte
            continue
        if byte == 0x7B:  # {
            stack.append((index, len(stack)))
        elif byte == 0x7D and stack:
            start, depth = stack.pop()
            end = index + 1
            prefix_start = max(raw.rfind(b"\n", 0, start) + 1, start - 160)
            prefix = raw[prefix_start:start].decode("utf-8", errors="surrogateescape")
            identifiers = _IDENTIFIER.findall(prefix)
            name = identifiers[-1] if identifiers else f"region_{start}"
            evidence = _evidence(path, raw, start, end, offsets)
            nodes.append(SemanticNode(
                node_id=_node_id(path, "region", name, start, end),
                kind="region",
                name=name,
                language="unknown",
                parent_id=None,
                signature=prefix.strip()[-160:],
                epistemic_class=EpistemicClass.HEURISTIC,
                evidence=evidence,
            ))
    return sorted(nodes, key=lambda item: (item.evidence.start_byte, item.evidence.end_byte))


def build_universal_semantic_document(
    source: str,
    file_path: str,
    *,
    max_bytes: int = 2 * 1024 * 1024,
    max_nodes: int = 100_000,
) -> UniversalSemanticDocument:
    """Normalize any source artifact into the stable Entroly semantic IR."""
    raw = source.encode("utf-8", errors="surrogateescape")
    if len(raw) > max_bytes:
        raise ValueError(f"source exceeds max_bytes={max_bytes}")
    path = file_path.replace("\\", "/")
    language = language_for_path(path) or "unknown"
    offsets = _line_offsets(raw)
    source_sha256 = hashlib.sha256(raw).hexdigest()

    root_evidence = _evidence(path, raw, 0, len(raw), offsets)
    root = SemanticNode(
        node_id=_node_id(path, "file", Path(path).name, 0, len(raw)),
        kind="file",
        name=Path(path).name,
        language=language,
        parent_id=None,
        signature="",
        epistemic_class=EpistemicClass.EXACT_SOURCE,
        evidence=root_evidence,
    )

    syntax_valid = validate_structural_syntax(source, path, max_bytes=max_bytes)
    spans = extract_structural_spans(source, path, max_bytes=max_bytes, max_nodes=max_nodes)
    calls = extract_structural_calls(source, path, max_bytes=max_bytes, max_nodes=max_nodes)
    parser_available = syntax_valid is not None
    parser_structure = spans is not None

    nodes = [root]
    edges: list[SemanticEdge] = []
    diagnostics: list[str] = []
    if parser_structure:
        structure_nodes, structure_edges = _semantic_nodes(
            path, language, raw, spans or [], offsets
        )
        nodes.extend(structure_nodes)
        edges.extend(structure_edges)
        for node in structure_nodes:
            if node.parent_id is None:
                edges.append(SemanticEdge(
                    edge_id=_edge_id(root.node_id, "contains", node.node_id, node.evidence),
                    source_id=root.node_id,
                    relation="contains",
                    target_id=node.node_id,
                    target_name=node.name,
                    epistemic_class=EpistemicClass.PARSER_VERIFIED,
                    evidence=node.evidence,
                ))
        if calls is not None:
            edges.extend(_call_edges(path, raw, structure_nodes, calls, offsets))
    else:
        fallback = _fallback_regions(path, raw, offsets)
        nodes.extend(fallback)
        for node in fallback:
            edges.append(SemanticEdge(
                edge_id=_edge_id(root.node_id, "contains-region", node.node_id, node.evidence),
                source_id=root.node_id,
                relation="contains-region",
                target_id=node.node_id,
                target_name=node.name,
                epistemic_class=EpistemicClass.HEURISTIC,
                evidence=node.evidence,
            ))
        diagnostics.append(
            "parser structure unavailable; emitted exact-source heuristic regions"
        )

    if syntax_valid is False:
        diagnostics.append("parser reported syntax errors; structural evidence may be partial")
    if len(nodes) >= max_nodes:
        diagnostics.append("semantic node budget reached")

    level = SemanticLevel.STRUCTURE if parser_structure else (
        SemanticLevel.SYNTAX if parser_available else SemanticLevel.SOURCE
    )
    capabilities = SemanticCapabilities(
        language=language,
        level=level,
        exact_source=True,
        parser_available=parser_available,
        structure=parser_structure,
        semantic_binding=False,
        control_data_flow=False,
        verified_transformations=False,
    )
    return UniversalSemanticDocument(
        path=path,
        language=language,
        source_sha256=source_sha256,
        byte_length=len(raw),
        capabilities=capabilities,
        nodes=nodes,
        edges=edges,
        diagnostics=diagnostics,
    )


__all__ = [
    "SEMANTIC_IR_SCHEMA_VERSION",
    "EpistemicClass",
    "SemanticCapabilities",
    "SemanticEdge",
    "SemanticLevel",
    "SemanticNode",
    "SourceEvidence",
    "UniversalSemanticDocument",
    "build_universal_semantic_document",
]
