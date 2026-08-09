"""Language-independent, evidence-carrying semantic representation for source code.

The IR deliberately hides grammar-specific parser node names from the rest of
Entroly. Parser/compiler/LSP frontends may strengthen the same graph without
changing the agent-facing schema. Unknown languages still receive a bounded
exact-source structural skeleton rather than an "unsupported" error.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path

from ..tree_sitter_support import (
    StructuralCall,
    StructuralSpan,
    extract_structural_calls_report,
    extract_structural_spans_report,
    language_for_source,
    validate_structural_syntax,
)
from .registry_frontend import RegistryFacts, RegistrySpan, extract_registry_facts

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


class EpistemicClass(str, Enum):
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


def _evidence(
    path: str,
    raw: bytes,
    start: int,
    end: int,
    offsets: list[int],
) -> SourceEvidence:
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


def _registry_evidence(path: str, span: RegistrySpan) -> SourceEvidence:
    return SourceEvidence(
        path=path,
        start_byte=span.start_byte,
        end_byte=span.end_byte,
        sha256=span.evidence_sha256,
        start_line=span.start_line,
        end_line=span.end_line,
    )


def _node_id(path: str, kind: str, name: str, start: int, end: int) -> str:
    material = f"{path}\0{kind}\0{name}\0{start}\0{end}".encode("utf-8")
    return "node:" + hashlib.sha256(material).hexdigest()[:24]


def _edge_id(
    source: str,
    relation: str,
    target: str,
    evidence: SourceEvidence | None,
) -> str:
    span = "" if evidence is None else (
        f"{evidence.path}:{evidence.start_byte}:{evidence.end_byte}"
    )
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
    ordered = sorted(
        spans,
        key=lambda item: (item.start_byte, -item.end_byte, item.name),
    )
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


def _registry_graph(
    root: SemanticNode,
    language: str,
    facts: RegistryFacts,
) -> tuple[list[SemanticNode], list[SemanticEdge]]:
    """Materialize normalized parser facts without claiming semantic binding."""
    nodes: list[SemanticNode] = []
    edges: list[SemanticEdge] = []

    def add_fact(kind: str, name: str, evidence: SourceEvidence, signature: str = "") -> SemanticNode:
        node = SemanticNode(
            node_id=_node_id(
                evidence.path, kind, name, evidence.start_byte, evidence.end_byte
            ),
            kind=kind,
            name=name,
            language=language,
            parent_id=root.node_id,
            signature=signature,
            epistemic_class=EpistemicClass.PARSER_VERIFIED,
            evidence=evidence,
        )
        nodes.append(node)
        edges.append(SemanticEdge(
            edge_id=_edge_id(root.node_id, "contains", node.node_id, evidence),
            source_id=root.node_id,
            relation="contains",
            target_id=node.node_id,
            target_name=name,
            epistemic_class=EpistemicClass.PARSER_VERIFIED,
            evidence=evidence,
        ))
        return node

    for item in facts.imports:
        evidence = _registry_evidence(root.evidence.path, item.span)
        node = add_fact("import", item.source, evidence)
        target_id = "external-module:" + hashlib.sha256(
            item.source.encode("utf-8", errors="surrogateescape")
        ).hexdigest()[:24]
        edges.append(SemanticEdge(
            edge_id=_edge_id(node.node_id, "imports-source", target_id, evidence),
            source_id=node.node_id,
            relation="imports-source",
            target_id=target_id,
            target_name=item.source,
            epistemic_class=EpistemicClass.PARSER_VERIFIED,
            evidence=evidence,
        ))
        for imported_name in item.items:
            item_target = "external-symbol:" + hashlib.sha256(
                f"{item.source}\0{imported_name}".encode("utf-8")
            ).hexdigest()[:24]
            edges.append(SemanticEdge(
                edge_id=_edge_id(node.node_id, "imports-name", item_target, evidence),
                source_id=node.node_id,
                relation="imports-name",
                target_id=item_target,
                target_name=imported_name,
                epistemic_class=EpistemicClass.PARSER_VERIFIED,
                evidence=evidence,
            ))

    for item in facts.exports:
        evidence = _registry_evidence(root.evidence.path, item.span)
        node = add_fact("export", item.name, evidence, signature=item.kind)
        edges.append(SemanticEdge(
            edge_id=_edge_id(root.node_id, "exports-name", node.node_id, evidence),
            source_id=root.node_id,
            relation="exports-name",
            target_id=node.node_id,
            target_name=item.name,
            epistemic_class=EpistemicClass.PARSER_VERIFIED,
            evidence=evidence,
        ))

    for item in facts.symbols:
        evidence = _registry_evidence(root.evidence.path, item.span)
        add_fact(
            f"symbol:{item.kind}",
            item.name,
            evidence,
            signature=item.type_annotation,
        )
    return nodes, edges


def _owner_for_call(
    nodes: list[SemanticNode],
    call: StructuralCall,
) -> SemanticNode | None:
    candidates = [
        node
        for node in nodes
        if node.evidence.start_byte <= call.start_byte
        and call.end_byte <= node.evidence.end_byte
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: item.evidence.end_byte - item.evidence.start_byte,
    )


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
            # Parsing proves an invocation expression and its spelling, not the
            # semantic binding of that spelling to a declaration.
            epistemic_class=EpistemicClass.PARSER_VERIFIED,
            evidence=evidence,
        ))
    return edges


def _fallback_regions(
    path: str,
    language: str,
    raw: bytes,
    offsets: list[int],
) -> list[SemanticNode]:
    """Create exact-source regions without pretending they are declarations."""
    nodes: list[SemanticNode] = []
    stack: list[int] = []
    quote: int | None = None
    escaped = False
    for index, byte in enumerate(raw):
        if len(nodes) >= _MAX_FALLBACK_REGIONS:
            break
        if quote is not None:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == quote:
                quote = None
            continue
        if byte in (0x22, 0x27):
            quote = byte
            continue
        if byte == 0x7B:
            stack.append(index)
        elif byte == 0x7D and stack:
            start = stack.pop()
            end = index + 1
            prefix_start = max(raw.rfind(b"\n", 0, start) + 1, start - 160)
            prefix = raw[prefix_start:start].decode(
                "utf-8", errors="surrogateescape"
            )
            identifiers = _IDENTIFIER.findall(prefix)
            name = identifiers[-1] if identifiers else f"region_{start}"
            evidence = _evidence(path, raw, start, end, offsets)
            nodes.append(SemanticNode(
                node_id=_node_id(path, "region", name, start, end),
                kind="region",
                name=name,
                language=language,
                parent_id=None,
                signature=prefix.strip()[-160:],
                epistemic_class=EpistemicClass.HEURISTIC,
                evidence=evidence,
            ))
    return sorted(
        nodes,
        key=lambda item: (item.evidence.start_byte, item.evidence.end_byte),
    )


def _deduplicate_graph(
    nodes: list[SemanticNode],
    edges: list[SemanticEdge],
) -> tuple[list[SemanticNode], list[SemanticEdge]]:
    node_map = {node.node_id: node for node in nodes}
    edge_map = {edge.edge_id: edge for edge in edges}
    return (
        sorted(
            node_map.values(),
            key=lambda item: (
                item.evidence.start_byte,
                item.evidence.end_byte,
                item.kind,
                item.name,
                item.node_id,
            ),
        ),
        sorted(
            edge_map.values(),
            key=lambda item: (
                item.evidence.start_byte if item.evidence else -1,
                item.relation,
                item.source_id,
                item.target_id,
            ),
        ),
    )


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
    language = language_for_source(path, source) or "unknown"
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
    span_report = extract_structural_spans_report(
        source, path, max_bytes=max_bytes, max_nodes=max_nodes
    )
    call_report = extract_structural_calls_report(
        source, path, max_bytes=max_bytes, max_nodes=max_nodes
    )
    registry_facts = (
        extract_registry_facts(
            source, language, max_bytes=max_bytes, max_nodes=max_nodes
        )
        if language != "unknown"
        else None
    )
    parser_available = syntax_valid is not None
    parser_structure = span_report is not None and span_report.complete

    nodes = [root]
    edges: list[SemanticEdge] = []
    diagnostics: list[str] = []
    if parser_structure:
        spans = list(span_report.items) if span_report else []
        structure_nodes, structure_edges = _semantic_nodes(
            path, language, raw, spans, offsets
        )
        nodes.extend(structure_nodes)
        edges.extend(structure_edges)
        for node in structure_nodes:
            if node.parent_id is None:
                edges.append(SemanticEdge(
                    edge_id=_edge_id(
                        root.node_id, "contains", node.node_id, node.evidence
                    ),
                    source_id=root.node_id,
                    relation="contains",
                    target_id=node.node_id,
                    target_name=node.name,
                    epistemic_class=EpistemicClass.PARSER_VERIFIED,
                    evidence=node.evidence,
                ))
        if call_report is not None and call_report.complete:
            edges.extend(_call_edges(
                path,
                raw,
                structure_nodes,
                list(call_report.items),
                offsets,
            ))
        elif call_report is not None:
            diagnostics.append(
                "call traversal reached its node bound; parser calls were omitted"
            )
    else:
        fallback = _fallback_regions(path, language, raw, offsets)
        nodes.extend(fallback)
        for node in fallback:
            edges.append(SemanticEdge(
                edge_id=_edge_id(
                    root.node_id, "contains-region", node.node_id, node.evidence
                ),
                source_id=root.node_id,
                relation="contains-region",
                target_id=node.node_id,
                target_name=node.name,
                epistemic_class=EpistemicClass.HEURISTIC,
                evidence=node.evidence,
            ))
        if span_report is not None and not span_report.complete:
            diagnostics.append(
                "structural traversal reached its node bound; partial parser "
                "results were rejected and exact-source fallback was used"
            )
        else:
            diagnostics.append(
                "parser structure unavailable; emitted exact-source heuristic regions"
            )

    if registry_facts is not None:
        if registry_facts.complete:
            fact_nodes, fact_edges = _registry_graph(root, language, registry_facts)
            nodes.extend(fact_nodes)
            edges.extend(fact_edges)
        else:
            diagnostics.append(
                "registry fact extraction exceeded its node bound; facts were omitted"
            )
        for item in registry_facts.diagnostics[:100]:
            if item.message:
                diagnostics.append(
                    f"parser diagnostic ({item.severity}): {item.message}"
                )

    if syntax_valid is False:
        diagnostics.append("parser reported syntax errors")

    nodes, edges = _deduplicate_graph(nodes, edges)
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
        diagnostics=sorted(dict.fromkeys(diagnostics)),
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
