"""Entroly Memory Graph — receipt-backed knowledge graph over agent memories.

Structured entity-relationship graph that links memories via extracted entities.
Every edge is receipt-backed, pointing back to the exact memory entry that
established the relationship.

Architecture
------------
This is NOT a full knowledge graph engine. It is a lightweight, local-first
adjacency structure that:

1. Extracts entities from memory entries (using memory.py's _extract_entities)
2. Creates edges between entities that co-occur in the same memory
3. Supports multi-hop traversal for associative recall (à la HippoRAG)
4. Attaches a receipt to every edge linking to the source memory ID

The graph is stored as a JSON adjacency list under .entroly/memory_graph/.
No external database or embedding service is required.

Provenance and Auditability
---------------------------
Entroly's graph memory is receipt-backed: every edge can be traced to the exact
memory entry that created it, supporting audit, supersession tracking, and exact recovery.

References
----------
- HippoRAG (2024–2025): biologically-inspired associative graph memory
- GroundedKG-RAG (2026): every graph node points to original source sentences
- "Governed Shared Memory" (2026): temporal supersession in multi-agent graphs
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "memory-graph.v1"


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class GraphEdge:
    """A receipt-backed relationship between two entities.

    Each edge records which memory entry established the link, when it was
    created, and optional relationship metadata. Edges are directional but
    the graph supports bidirectional traversal via the adjacency index.
    """
    edge_id: str
    source_entity: str
    target_entity: str
    memory_id: str        # the memory entry that established this edge
    relationship: str     # auto-detected or user-provided label
    created_at: float
    weight: float = 1.0   # reinforced on co-occurrence
    superseded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphEdge:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class GraphNode:
    """An entity node in the memory graph."""
    entity: str
    first_seen: float
    last_seen: float
    memory_ids: list[str] = field(default_factory=list)  # all memories mentioning this entity
    mention_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GraphNode:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TraversalResult:
    """Result of a multi-hop graph traversal."""
    query_entities: list[str]
    reached_entities: list[str]
    memory_ids: list[str]   # unique memory IDs discovered
    edges_traversed: list[GraphEdge]
    hops: int
    receipt: dict[str, Any]  # audit trail of the traversal


# ── Graph Store ───────────────────────────────────────────────────────────

class MemoryGraph:
    """Local-first entity-relationship graph over agent memories.

    Usage::

        graph = MemoryGraph()

        # Index a memory entry's entities
        graph.index_memory("mem_000001", ["Alice", "BackendTeam"],
                           "Alice manages the BackendTeam")

        # Traverse from an entity
        result = graph.traverse(["Alice"], max_hops=2)
        # result.memory_ids → all memories reachable from Alice within 2 hops

        # Persist
        graph.save()
    """

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path(os.environ.get("ENTROLY_DIR", ".entroly"))
        self._root = Path(root) / "memory_graph"
        self._root.mkdir(parents=True, exist_ok=True)
        self._nodes_path = self._root / "nodes.json"
        self._edges_path = self._root / "edges.jsonl"

        # In-memory indices
        self._nodes: dict[str, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        # Adjacency: entity → list of (edge_index, neighbor_entity)
        self._adj: dict[str, list[tuple[int, str]]] = defaultdict(list)
        # Memory→edges index
        self._memory_edges: dict[str, list[int]] = defaultdict(list)

        self._load()

    # ── Indexing ──────────────────────────────────────────────────────

    def index_memory(
        self,
        memory_id: str,
        entities: list[str],
        content: str = "",
        relationship: str = "co_occurs",
    ) -> list[GraphEdge]:
        """Index a memory's entities into the graph.

        Creates/updates nodes for each entity and creates edges between all
        entity pairs that co-occur in the same memory. Returns new edges.

        This is the primary write path. Call it after MemoryOS.remember().
        """
        if len(entities) < 1:
            return []

        now = time.time()
        new_edges: list[GraphEdge] = []

        # Upsert nodes
        for entity in entities:
            key = entity.lower()
            if key in self._nodes:
                node = self._nodes[key]
                node.last_seen = now
                node.mention_count += 1
                if memory_id not in node.memory_ids:
                    node.memory_ids.append(memory_id)
            else:
                self._nodes[key] = GraphNode(
                    entity=entity,
                    first_seen=now,
                    last_seen=now,
                    memory_ids=[memory_id],
                    mention_count=1,
                )

        # Create edges between all pairs (undirected, stored as two directed edges)
        for i, src in enumerate(entities):
            for tgt in entities[i + 1:]:
                src_key, tgt_key = src.lower(), tgt.lower()

                # Check if this pair already has an edge from this memory
                existing = self._find_edge(src_key, tgt_key, memory_id)
                if existing is not None:
                    existing.weight += 0.5  # reinforce
                    continue

                # Detect relationship type from content
                rel = self._detect_relationship(src, tgt, content) if content else relationship

                edge = GraphEdge(
                    edge_id=self._make_edge_id(src_key, tgt_key, memory_id),
                    source_entity=src_key,
                    target_entity=tgt_key,
                    memory_id=memory_id,
                    relationship=rel,
                    created_at=now,
                )
                idx = len(self._edges)
                self._edges.append(edge)
                self._adj[src_key].append((idx, tgt_key))
                self._adj[tgt_key].append((idx, src_key))
                self._memory_edges[memory_id].append(idx)
                new_edges.append(edge)

        return new_edges

    def supersede_memory(self, old_memory_id: str, new_memory_id: str) -> int:
        """Mark all edges from old_memory_id as superseded.

        Returns the number of edges superseded. The edges remain in the graph
        for audit but are excluded from traversal by default.
        """
        count = 0
        for idx in self._memory_edges.get(old_memory_id, []):
            if not self._edges[idx].superseded:
                self._edges[idx].superseded = True
                count += 1
        return count

    # ── Traversal ─────────────────────────────────────────────────────────

    def traverse(
        self,
        seed_entities: list[str],
        *,
        max_hops: int = 2,
        max_results: int = 50,
        include_superseded: bool = False,
    ) -> TraversalResult:
        """Multi-hop traversal from seed entities.

        Returns all memory IDs reachable within max_hops, with full audit trail.
        This implements associative recall à la HippoRAG — given a query's
        entities, discover related memories via the entity graph.
        """
        seeds = [e.lower() for e in seed_entities if e.lower() in self._nodes]
        if not seeds:
            return TraversalResult(
                query_entities=seed_entities,
                reached_entities=[],
                memory_ids=[],
                edges_traversed=[],
                hops=0,
                receipt={"status": "no_matching_entities", "seeds": seed_entities},
            )

        visited: set[str] = set()
        frontier: set[str] = set(seeds)
        discovered_memories: set[str] = set()
        traversed_edges: list[GraphEdge] = []
        hops_taken = 0

        for hop in range(max_hops):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for entity in frontier:
                visited.add(entity)
                node = self._nodes.get(entity)
                if node:
                    discovered_memories.update(node.memory_ids)
                for edge_idx, neighbor in self._adj.get(entity, []):
                    edge = self._edges[edge_idx]
                    if edge.superseded and not include_superseded:
                        continue
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        if edge not in traversed_edges:
                            traversed_edges.append(edge)
            frontier = next_frontier - visited
            hops_taken = hop + 1

        # Collect remaining frontier node memories
        for entity in frontier:
            node = self._nodes.get(entity)
            if node:
                discovered_memories.update(node.memory_ids)
                visited.add(entity)

        memory_list = sorted(discovered_memories)[:max_results]
        entity_list = sorted(visited)

        receipt = {
            "traversal_type": "multi_hop_associative",
            "seeds": seed_entities,
            "matched_seeds": seeds,
            "hops": hops_taken,
            "max_hops": max_hops,
            "entities_visited": len(entity_list),
            "memories_discovered": len(memory_list),
            "edges_traversed": len(traversed_edges),
            "superseded_excluded": not include_superseded,
        }

        return TraversalResult(
            query_entities=seed_entities,
            reached_entities=entity_list,
            memory_ids=memory_list,
            edges_traversed=traversed_edges,
            hops=hops_taken,
            receipt=receipt,
        )

    def entity_memories(self, entity: str) -> list[str]:
        """Get all memory IDs associated with an entity."""
        node = self._nodes.get(entity.lower())
        return list(node.memory_ids) if node else []

    def neighbors(self, entity: str, *, include_superseded: bool = False) -> list[tuple[str, str, str]]:
        """Get (neighbor_entity, relationship, memory_id) tuples for an entity."""
        result = []
        for edge_idx, neighbor in self._adj.get(entity.lower(), []):
            edge = self._edges[edge_idx]
            if edge.superseded and not include_superseded:
                continue
            result.append((neighbor, edge.relationship, edge.memory_id))
        return result

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> None:
        """Persist graph to disk."""
        # Nodes
        nodes_data = {
            "version": _SCHEMA_VERSION,
            "saved_at": time.time(),
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
        }
        self._nodes_path.write_text(
            json.dumps(nodes_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Edges (append-only JSONL)
        with open(self._edges_path, "w", encoding="utf-8") as f:
            for edge in self._edges:
                f.write(json.dumps(edge.to_dict(), ensure_ascii=False) + "\n")

        logger.info(
            "MemoryGraph saved: %d nodes, %d edges",
            len(self._nodes), len(self._edges),
        )

    def _load(self) -> None:
        """Load graph from disk."""
        # Nodes
        if self._nodes_path.exists():
            try:
                data = json.loads(self._nodes_path.read_text(encoding="utf-8"))
                for key, node_dict in data.get("nodes", {}).items():
                    self._nodes[key] = GraphNode.from_dict(node_dict)
            except (json.JSONDecodeError, TypeError, KeyError) as exc:
                logger.warning("Failed to load memory graph nodes: %s", exc)

        # Edges
        if self._edges_path.exists():
            try:
                with open(self._edges_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        edge = GraphEdge.from_dict(json.loads(line))
                        idx = len(self._edges)
                        self._edges.append(edge)
                        self._adj[edge.source_entity].append((idx, edge.target_entity))
                        self._adj[edge.target_entity].append((idx, edge.source_entity))
                        self._memory_edges[edge.memory_id].append(idx)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to load memory graph edges: %s", exc)

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        active_edges = sum(1 for e in self._edges if not e.superseded)
        superseded_edges = len(self._edges) - active_edges
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "active_edges": active_edges,
            "superseded_edges": superseded_edges,
            "top_entities": sorted(
                [(k, v.mention_count) for k, v in self._nodes.items()],
                key=lambda x: -x[1],
            )[:20],
        }

    # ── Internal ─────────────────────────────────────────────────────

    def _find_edge(self, src: str, tgt: str, memory_id: str) -> GraphEdge | None:
        """Find an existing edge between src and tgt from a specific memory."""
        for edge_idx, neighbor in self._adj.get(src, []):
            edge = self._edges[edge_idx]
            if neighbor == tgt and edge.memory_id == memory_id:
                return edge
        return None

    @staticmethod
    def _make_edge_id(src: str, tgt: str, memory_id: str) -> str:
        raw = f"{src}|{tgt}|{memory_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _detect_relationship(src: str, tgt: str, content: str) -> str:
        """Heuristic relationship detection from content mentioning both entities."""
        import re
        text = content.lower()
        src_l, tgt_l = src.lower(), tgt.lower()

        # Check for common relationship verbs between the entities
        patterns = [
            (r"manages?", "manages"),
            (r"leads?", "leads"),
            (r"owns?", "owns"),
            (r"reports?\s+to", "reports_to"),
            (r"works?\s+(?:with|at|for|on)", "works_with"),
            (r"uses?", "uses"),
            (r"depends?\s+on", "depends_on"),
            (r"imports?", "imports"),
            (r"extends?", "extends"),
            (r"implements?", "implements"),
            (r"calls?", "calls"),
            (r"contains?", "contains"),
        ]

        for pattern, label in patterns:
            # Check src <verb> tgt
            if re.search(rf"\b{re.escape(src_l)}\b.*\b{pattern}\b.*\b{re.escape(tgt_l)}\b", text):
                return label
            # Check tgt <verb> src (reverse)
            if re.search(rf"\b{re.escape(tgt_l)}\b.*\b{pattern}\b.*\b{re.escape(src_l)}\b", text):
                return f"{label}_by"

        return "co_occurs"


__all__ = [
    "MemoryGraph",
    "GraphEdge",
    "GraphNode",
    "TraversalResult",
]
