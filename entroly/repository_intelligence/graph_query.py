"""Bounded, freshness-checked queries over the repository property graph."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .models import RepositoryIndex, Symbol, normalize_relative

GRAPH_QUERY_SCHEMA_VERSION = "entroly.verified-graph-query.v1"
_OPERATIONS = frozenset({"explain", "neighbors", "path", "related", "impact"})
_DIRECTIONS = frozenset({"incoming", "outgoing", "both"})


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    kind: str
    evidence: Mapping[str, object]

    @property
    def edge_id(self) -> str:
        canonical = json.dumps(
            [self.source, self.target, self.kind, dict(self.evidence)],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return "edge:" + hashlib.sha256(canonical).hexdigest()[:24]

    def to_dict(self, traversal: str) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "traversal": traversal,
            "evidence": dict(self.evidence),
        }


class _Freshness:
    def __init__(self, root: Path, index: RepositoryIndex) -> None:
        self.root = root
        self.index = index
        self.status: dict[str, str] = {}
        self.verified: dict[str, str] = {}

    def check(self, path: str) -> bool:
        cached = self.status.get(path)
        if cached is not None:
            return cached == "verified"
        record = self.index.files.get(path)
        if record is None:
            self.status[path] = "unknown-path"
            return False
        try:
            candidate = (self.root / path).resolve(strict=True)
            candidate.relative_to(self.root)
            raw = candidate.read_bytes()
        except (OSError, RuntimeError, ValueError):
            self.status[path] = "unsafe-or-unreadable-source"
            return False
        digest = hashlib.sha256(raw).hexdigest()
        if digest != record.sha256:
            self.status[path] = "stale-source"
            return False
        self.status[path] = "verified"
        self.verified[path] = digest
        return True

    def omissions(self) -> dict[str, int]:
        return dict(sorted(Counter(
            status for status in self.status.values() if status != "verified"
        ).items()))


def _file_node(path: str) -> str:
    return f"file:{path}"


def _symbol_node(symbol_id: str) -> str:
    return f"symbol:{symbol_id}"


def _node_path(node_id: str, index: RepositoryIndex) -> str | None:
    if node_id.startswith("file:"):
        path = node_id[5:]
        return path if path in index.files else None
    if node_id.startswith("symbol:"):
        symbol = index.symbols.get(node_id[7:])
        return symbol.path if symbol else None
    return None


def _resolve(
    index: RepositoryIndex,
    query: str,
) -> tuple[str, list[str]]:
    clean = query.strip()
    if clean.startswith("file:"):
        clean = clean[5:]
    normalized = normalize_relative(clean)
    if normalized in index.files:
        return "resolved", [_file_node(normalized)]
    symbol_query = clean[7:] if clean.startswith("symbol:") else clean
    lowered = symbol_query.lower()
    matches = sorted(
        (
            _symbol_node(symbol.symbol_id)
            for symbol in index.symbols.values()
            if lowered in {
                symbol.symbol_id.lower(),
                symbol.qualified_name.lower(),
                symbol.name.lower(),
            }
        )
    )
    if len(matches) == 1:
        return "resolved", matches
    return ("ambiguous" if matches else "not-found"), matches[:100]


def _graph(
    index: RepositoryIndex,
) -> tuple[dict[str, list[_Edge]], dict[str, list[_Edge]]]:
    outgoing: dict[str, list[_Edge]] = defaultdict(list)
    incoming: dict[str, list[_Edge]] = defaultdict(list)

    def add(edge: _Edge) -> None:
        outgoing[edge.source].append(edge)
        incoming[edge.target].append(edge)

    for source, targets in sorted(index.file_dependencies.items()):
        if source not in index.files:
            continue
        for target in sorted(targets):
            if target in index.files:
                add(_Edge(
                    _file_node(source),
                    _file_node(target),
                    "imports",
                    {
                        "source_sha256": index.files[source].sha256,
                        "target_sha256": index.files[target].sha256,
                        "trust": "parser-resolved-file-relationship",
                    },
                ))
    for symbol in sorted(index.symbols.values(), key=lambda item: item.symbol_id):
        add(_Edge(
            _file_node(symbol.path),
            _symbol_node(symbol.symbol_id),
            "contains",
            {
                "source_sha256": index.files[symbol.path].sha256,
                "start_byte": symbol.start_byte,
                "end_byte": symbol.end_byte,
                "evidence_sha256": symbol.evidence_sha256,
                "parse_backend": symbol.parse_backend,
                "trust": "verified-declaration-span",
            },
        ))
    for edge in index.call_edges:
        if edge.caller_id not in index.symbols or edge.callee_id not in index.symbols:
            continue
        add(_Edge(
            _symbol_node(edge.caller_id),
            _symbol_node(edge.callee_id),
            edge.kind,
            {
                "path": edge.path,
                "line": edge.line,
                "start_byte": edge.start_byte,
                "end_byte": edge.end_byte,
                "evidence_sha256": edge.evidence_sha256,
                "confidence": edge.confidence,
                "resolution": edge.resolution,
                "trust": "verified-call-site-span",
            },
        ))
    for graph in (outgoing, incoming):
        for edges in graph.values():
            edges.sort(key=lambda edge: (
                edge.kind, edge.source, edge.target, edge.edge_id
            ))
    return outgoing, incoming


def prepare_graph_query(
    index: RepositoryIndex,
) -> tuple[dict[str, list[_Edge]], dict[str, list[_Edge]]]:
    """Prepare immutable-snapshot adjacency once for repeated service queries."""
    return _graph(index)


def _steps(
    node: str,
    outgoing: Mapping[str, list[_Edge]],
    incoming: Mapping[str, list[_Edge]],
    direction: str,
) -> Iterable[tuple[str, _Edge, str]]:
    if direction in {"outgoing", "both"}:
        for edge in outgoing.get(node, ()):
            yield edge.target, edge, "forward"
    if direction in {"incoming", "both"}:
        for edge in incoming.get(node, ()):
            yield edge.source, edge, "reverse"


def _node_payload(node_id: str, index: RepositoryIndex) -> dict[str, object]:
    if node_id.startswith("file:"):
        path = node_id[5:]
        record = index.files[path]
        return {
            "node_id": node_id,
            "kind": "file",
            "path": path,
            "language": record.language,
            "source_sha256": record.sha256,
            "symbol_count": len(index.symbols_for_path(path)),
            "is_test": record.is_test,
        }
    symbol: Symbol = index.symbols[node_id[7:]]
    return {
        "node_id": node_id,
        "kind": "symbol",
        **symbol.to_dict(),
    }


def _witness_path(
    node: str,
    parent: Mapping[str, tuple[str, _Edge, str]],
) -> tuple[list[str], list[dict[str, object]]]:
    nodes = [node]
    edges: list[dict[str, object]] = []
    current = node
    while current in parent:
        previous, edge, traversal = parent[current]
        edges.append(edge.to_dict(traversal))
        nodes.append(previous)
        current = previous
    nodes.reverse()
    edges.reverse()
    return nodes, edges


def _commit(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["graph_query_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_verified_graph_query(
    root: Path,
    index: RepositoryIndex,
    query: str,
    *,
    index_digest: str,
    operation: str = "neighbors",
    target_query: str | None = None,
    direction: str = "both",
    max_depth: int = 4,
    limit: int = 100,
    max_visited: int = 10_000,
    prepared_graph: tuple[
        dict[str, list[_Edge]], dict[str, list[_Edge]]
    ] | None = None,
) -> dict[str, object]:
    """Execute a typed graph query without traversing stale source nodes."""
    root = root.expanduser().resolve(strict=True)
    clean_operation = str(operation).strip().lower()
    clean_direction = str(direction).strip().lower()
    if clean_operation not in _OPERATIONS:
        raise ValueError("operation must be explain, neighbors, path, related, or impact")
    if clean_direction not in _DIRECTIONS:
        raise ValueError("direction must be incoming, outgoing, or both")
    if clean_operation == "impact":
        clean_direction = "incoming"
    if clean_operation == "explain":
        max_depth = 1
    depth_limit = max(0, min(int(max_depth), 20))
    result_limit = max(1, min(int(limit), 5_000))
    visit_limit = max(result_limit, min(int(max_visited), 100_000))
    resolution, candidates = _resolve(index, query)
    target_resolution = "not-requested"
    target_candidates: list[str] = []
    if clean_operation == "path":
        if not isinstance(target_query, str) or not target_query.strip():
            raise ValueError("path operation requires target_query")
        target_resolution, target_candidates = _resolve(index, target_query)

    base: dict[str, object] = {
        "schema_version": GRAPH_QUERY_SCHEMA_VERSION,
        "index_digest": index_digest,
        "operation": clean_operation,
        "direction": clean_direction,
        "query": str(query).strip()[:1_000],
        "resolution": resolution,
        "candidates": candidates,
        "target_query": str(target_query).strip()[:1_000] if target_query else None,
        "target_resolution": target_resolution,
        "target_candidates": target_candidates,
        "root_node": candidates[0] if resolution == "resolved" else None,
        "target_node": (
            target_candidates[0] if target_resolution == "resolved" else None
        ),
        "nodes": [],
        "edges": [],
        "results": [],
        "truncated": False,
        "receipt": {
            "freshness": "not-applicable",
            "checked_sources": {},
            "omissions_by_reason": {},
            "remote_calls": 0,
            "dynamic_and_unindexed_relationships_may_remain": True,
            "commitment_scope": (
                "payload-excluding-generation-command-and-graph-query-sha256"
            ),
        },
    }
    if resolution != "resolved" or (
        clean_operation == "path" and target_resolution != "resolved"
    ):
        return _commit(base)

    start = candidates[0]
    target = target_candidates[0] if target_candidates else None
    freshness = _Freshness(root, index)
    start_path = _node_path(start, index)
    if start_path is None or not freshness.check(start_path):
        base["resolution"] = "stale-or-unreadable-source"
        base["receipt"]["omissions_by_reason"] = freshness.omissions()  # type: ignore[index]
        return _commit(base)
    if target is not None:
        target_path = _node_path(target, index)
        if target_path is None or not freshness.check(target_path):
            base["target_resolution"] = "stale-or-unreadable-source"
            base["receipt"]["omissions_by_reason"] = freshness.omissions()  # type: ignore[index]
            return _commit(base)

    outgoing, incoming = prepared_graph if prepared_graph is not None else _graph(index)
    queue = deque([(start, 0)])
    parent: dict[str, tuple[str, _Edge, str]] = {}
    depths = {start: 0}
    ordered = [start]
    while queue:
        node, depth = queue.popleft()
        if target is not None and node == target:
            break
        if depth >= depth_limit:
            continue
        for neighbor, edge, traversal in _steps(
            node, outgoing, incoming, clean_direction
        ):
            if neighbor in depths:
                continue
            path = _node_path(neighbor, index)
            if path is None or not freshness.check(path):
                continue
            if len(depths) >= visit_limit:
                base["truncated"] = True
                break
            depths[neighbor] = depth + 1
            parent[neighbor] = (node, edge, traversal)
            ordered.append(neighbor)
            queue.append((neighbor, depth + 1))
        if base["truncated"]:
            break

    selected: list[str]
    results: list[dict[str, object]] = []
    if clean_operation == "path":
        selected = [] if target not in depths else _witness_path(target, parent)[0]  # type: ignore[arg-type]
        if selected:
            nodes, edges = _witness_path(target, parent)  # type: ignore[arg-type]
            results.append({
                "kind": "shortest-path",
                "distance": len(edges),
                "nodes": nodes,
                "edges": edges,
            })
    else:
        ranked = sorted(
            (node for node in ordered if node != start),
            key=lambda node: (depths[node], node),
        )
        if clean_operation == "related":
            degree = {
                node: len(outgoing.get(node, ())) + len(incoming.get(node, ()))
                for node in ranked
            }
            ranked.sort(key=lambda node: (
                -(1.0 / (1 + depths[node])) * (1.0 + min(degree[node], 20) / 20),
                depths[node],
                node,
            ))
        selected = [start, *ranked[:result_limit]]
        for node in ranked[:result_limit]:
            nodes, edges = _witness_path(node, parent)
            result: dict[str, object] = {
                "node_id": node,
                "distance": depths[node],
                "witness_nodes": nodes,
                "witness_edges": edges,
            }
            if clean_operation == "related":
                degree = len(outgoing.get(node, ())) + len(incoming.get(node, ()))
                result["score"] = round(
                    (1.0 / (1 + depths[node])) * (1.0 + min(degree, 20) / 20),
                    8,
                )
                result["score_policy"] = "inverse-distance-times-bounded-degree"
            results.append(result)
        if len(ranked) > result_limit:
            base["truncated"] = True

    unique_selected = list(dict.fromkeys(selected))
    selected_set = set(unique_selected)
    tree_edges = {
        edge.edge_id: edge.to_dict(traversal)
        for node in unique_selected
        if node in parent
        for _previous, edge, traversal in [parent[node]]
    }
    base.update({
        "nodes": [_node_payload(node, index) for node in unique_selected],
        "edges": [tree_edges[key] for key in sorted(tree_edges)],
        "results": results,
        "receipt": {
            "freshness": "verified-on-traversal-against-indexed-source-sha256",
            "checked_sources": dict(sorted(freshness.verified.items())),
            "omissions_by_reason": freshness.omissions(),
            "visited_node_count": len(depths),
            "returned_node_count": len(selected_set),
            "returned_edge_count": len(tree_edges),
            "bounds": {
                "max_depth": depth_limit,
                "result_limit": result_limit,
                "max_visited": visit_limit,
            },
            "remote_calls": 0,
            "dynamic_and_unindexed_relationships_may_remain": True,
            "commitment_scope": (
                "payload-excluding-generation-command-and-graph-query-sha256"
            ),
        },
    })
    return _commit(base)


def verify_graph_query_commitment(payload: Mapping[str, object]) -> bool:
    """Verify a detached graph-query receipt without workspace access."""
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("generation", None)
        candidate.pop("command", None)
        if candidate.get("schema_version") != GRAPH_QUERY_SCHEMA_VERSION:
            return False
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("graph_query_sha256"))
        canonical = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "GRAPH_QUERY_SCHEMA_VERSION",
    "build_verified_graph_query",
    "prepare_graph_query",
    "verify_graph_query_commitment",
]
