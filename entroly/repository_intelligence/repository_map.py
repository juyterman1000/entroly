"""Deterministic, evidence-carrying repository maps.

The map ranks files and symbols together so dependency hubs, call targets, and
query-relevant definitions can compete in one bounded view.  Ranking never
turns an indexed location into evidence: every emitted signature is re-read,
checked against the indexed file digest, and committed in the receipt.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .models import RepositoryIndex, Symbol

REPOSITORY_MAP_SCHEMA_VERSION = "entroly.verified-repository-map.v1"
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "code", "for",
    "from", "in", "is", "it", "of", "on", "or", "repo", "repository",
    "the", "this", "to", "what", "where", "which", "with",
})


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return {
        item.lower()
        for item in _TOKEN_RE.findall(expanded.replace("-", "_"))
        if len(item) > 1 and item.lower() not in _STOPWORDS
    }


def _token_cost(value: str) -> int:
    return max(1, math.ceil(len(value.encode("utf-8", errors="surrogateescape")) / 3))


def _contains_identifier(text: str, identifier: str) -> bool:
    """Check lexical identity without compiling one regex per symbol."""
    if not identifier:
        return False
    start = 0
    while True:
        offset = text.find(identifier, start)
        if offset < 0:
            return False
        end = offset + len(identifier)
        left_ok = offset == 0 or not (text[offset - 1].isalnum() or text[offset - 1] in "_$")
        right_ok = end == len(text) or not (text[end].isalnum() or text[end] in "_$")
        if left_ok and right_ok:
            return True
        start = offset + 1


def _file_node(path: str) -> str:
    return f"file:{path}"


def _symbol_node(symbol_id: str) -> str:
    return f"symbol:{symbol_id}"


def _add_edge(
    graph: dict[str, dict[str, float]],
    source: str,
    target: str,
    weight: float,
) -> None:
    graph[source][target] = graph[source].get(target, 0.0) + weight


def _graph(index: RepositoryIndex) -> dict[str, dict[str, float]]:
    """Build a typed graph without inventing unresolved relationships."""
    graph: dict[str, dict[str, float]] = defaultdict(dict)
    for path in sorted(index.files):
        graph[_file_node(path)]
    for symbol_id, symbol in sorted(index.symbols.items()):
        node = _symbol_node(symbol_id)
        file_node = _file_node(symbol.path)
        graph[node]
        _add_edge(graph, file_node, node, 1.0)
        _add_edge(graph, node, file_node, 1.0)
        if symbol.parent_id in index.symbols:
            parent = _symbol_node(str(symbol.parent_id))
            _add_edge(graph, parent, node, 1.5)
            _add_edge(graph, node, parent, 1.5)
    for edge in index.call_edges:
        if edge.caller_id in index.symbols and edge.callee_id in index.symbols:
            _add_edge(
                graph,
                _symbol_node(edge.caller_id),
                _symbol_node(edge.callee_id),
                4.0,
            )
    for source, dependencies in sorted(index.file_dependencies.items()):
        if source not in index.files:
            continue
        for target in dependencies:
            if target in index.files:
                _add_edge(graph, _file_node(source), _file_node(target), 2.5)
    return {node: dict(sorted(edges.items())) for node, edges in sorted(graph.items())}


def _relevance(index: RepositoryIndex, query: str) -> dict[str, float]:
    query_tokens = _tokens(query)
    if not query_tokens:
        return {}
    scores: dict[str, float] = {}
    lowered = query.lower()
    for path, record in index.files.items():
        overlap = len(_tokens(path) & query_tokens)
        language = 1 if record.language.lower() in query_tokens else 0
        if overlap or language:
            scores[_file_node(path)] = float(5 * overlap + 2 * language)
    for symbol_id, symbol in index.symbols.items():
        exact = _contains_identifier(lowered, symbol.name.lower())
        score = (
            20 * int(exact)
            + 8 * len(_tokens(symbol.name) & query_tokens)
            + 5 * len(_tokens(symbol.qualified_name) & query_tokens)
            + 3 * len(_tokens(symbol.path) & query_tokens)
            + len(_tokens(symbol.signature) & query_tokens)
        )
        if score:
            scores[_symbol_node(symbol_id)] = float(score)
    return scores


def _personalization(
    nodes: list[str],
    relevance: dict[str, float],
) -> dict[str, float]:
    uniform = 1.0 / max(1, len(nodes))
    total = sum(relevance.get(node, 0.0) for node in nodes)
    if total <= 0:
        return {node: uniform for node in nodes}
    # Preserve a global-centrality floor so a narrow query cannot erase the
    # architectural backbone of the map.
    return {
        node: 0.15 * uniform + 0.85 * relevance.get(node, 0.0) / total
        for node in nodes
    }


def _page_rank(
    graph: dict[str, dict[str, float]],
    personalization: dict[str, float],
    *,
    damping: float = 0.85,
    tolerance: float = 1e-9,
    max_iterations: int = 50,
) -> tuple[dict[str, float], int, float]:
    nodes = sorted(graph)
    if not nodes:
        return {}, 0, 0.0
    node_indexes = {node: index for index, node in enumerate(nodes)}
    teleport = [personalization[node] for node in nodes]
    rank = list(teleport)
    outgoing: list[list[tuple[int, float]]] = []
    for node in nodes:
        total = sum(graph[node].values())
        outgoing.append([
            (node_indexes[target], weight / total)
            for target, weight in graph[node].items()
        ] if total > 0 else [])
    residual = 0.0
    for iteration in range(1, max_iterations + 1):
        dangling = sum(rank[index] for index, edges in enumerate(outgoing) if not edges)
        restart_mass = 1.0 - damping + damping * dangling
        updated = [restart_mass * value for value in teleport]
        for source_index, edges in enumerate(outgoing):
            contribution = damping * rank[source_index]
            for target_index, normalized_weight in edges:
                updated[target_index] += contribution * normalized_weight
        residual = sum(abs(updated[index] - rank[index]) for index in range(len(nodes)))
        rank = updated
        if residual <= tolerance:
            return dict(zip(nodes, rank)), iteration, residual
    return dict(zip(nodes, rank)), max_iterations, residual


def _components(graph: dict[str, dict[str, float]]) -> dict[str, str]:
    neighbors: dict[str, set[str]] = {node: set() for node in graph}
    for source, targets in graph.items():
        for target in targets:
            neighbors[source].add(target)
            neighbors[target].add(source)
    assigned: dict[str, str] = {}
    for seed in sorted(neighbors):
        if seed in assigned:
            continue
        pending = [seed]
        members: list[str] = []
        seen = {seed}
        while pending:
            node = pending.pop()
            members.append(node)
            for neighbor in sorted(neighbors[node], reverse=True):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        component = "component:" + hashlib.sha256(
            "\n".join(sorted(members)).encode("utf-8")
        ).hexdigest()[:12]
        for node in members:
            assigned[node] = component
    return assigned


def _verified_signature(
    root: Path,
    index: RepositoryIndex,
    symbol: Symbol,
    source_cache: dict[str, tuple[bytes | None, str]],
) -> tuple[dict[str, object] | None, str | None]:
    cached = source_cache.get(symbol.path)
    if cached is None:
        record = index.files.get(symbol.path)
        if record is None:
            cached = (None, "missing-file-record")
        else:
            try:
                candidate = (root / symbol.path).resolve(strict=True)
                candidate.relative_to(root)
                raw = candidate.read_bytes()
            except (OSError, RuntimeError, ValueError):
                cached = (None, "unsafe-or-unreadable")
            else:
                cached = (
                    (raw, "verified")
                    if hashlib.sha256(raw).hexdigest() == record.sha256
                    else (None, "stale-index")
                )
        source_cache[symbol.path] = cached
    raw, status = cached
    if raw is None:
        return None, status
    if not symbol.signature:
        return None, "missing-signature"
    signature_raw = symbol.signature.encode("utf-8", errors="surrogateescape")
    range_start, range_end = 0, len(raw)
    if 0 <= symbol.start_byte < symbol.end_byte <= len(raw):
        range_start, range_end = symbol.start_byte, symbol.end_byte
    start = raw.find(signature_raw, range_start, range_end)
    if start < 0:
        return None, "unverifiable-signature"
    if raw.find(signature_raw, start + 1, range_end) >= 0:
        return None, "ambiguous-signature"
    end = start + len(signature_raw)
    line = raw[:start].count(b"\n") + 1
    rendered = (
        f"{symbol.path}:{line} {symbol.kind} "
        f"{symbol.qualified_name} :: {symbol.signature}"
    )
    return {
        "symbol_id": symbol.symbol_id,
        "path": symbol.path,
        "language": symbol.language,
        "kind": symbol.kind,
        "qualified_name": symbol.qualified_name,
        "line": line,
        "start_byte": start,
        "end_byte": end,
        "signature": symbol.signature,
        "rendered": rendered,
        "estimated_tokens": _token_cost(rendered),
        "source_sha256": index.files[symbol.path].sha256,
        "evidence_sha256": hashlib.sha256(raw[start:end]).hexdigest(),
        "parse_backend": symbol.parse_backend,
        "trust": "untrusted-source-bytes",
    }, None


def _commit(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["repository_map_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_verified_repository_map(
    root: Path,
    index: RepositoryIndex,
    query: str = "",
    *,
    index_digest: str,
    token_budget: int = 2_000,
    max_entries: int = 100,
) -> dict[str, object]:
    """Rank a repository into a freshness-checked, budgeted structural map."""
    clean_query = query.strip()
    if len(clean_query) > 4_000:
        raise ValueError("query must be at most 4000 characters")
    budget = max(128, min(int(token_budget), 32_768))
    entry_limit = max(1, min(int(max_entries), 1_000))
    graph = _graph(index)
    nodes = sorted(graph)
    relevance = _relevance(index, clean_query)
    personalization = _personalization(nodes, relevance)
    ranks, iterations, residual = _page_rank(graph, personalization)
    components = _components(graph)

    candidates = sorted(
        index.symbols.values(),
        key=lambda symbol: (-ranks.get(_symbol_node(symbol.symbol_id), 0.0), symbol.symbol_id),
    )
    entries: list[dict[str, object]] = []
    omissions: dict[str, int] = defaultdict(int)
    source_cache: dict[str, tuple[bytes | None, str]] = {}
    remaining = budget
    max_relevance = max(relevance.values(), default=0.0)
    for symbol in candidates:
        if len(entries) >= entry_limit:
            omissions["entry-limit"] += 1
            continue
        estimated_rendered = (
            f"{symbol.path}:{symbol.line_start} {symbol.kind} "
            f"{symbol.qualified_name} :: {symbol.signature}"
        )
        if _token_cost(estimated_rendered) > remaining:
            omissions["budget"] += 1
            continue
        entry, reason = _verified_signature(root, index, symbol, source_cache)
        if entry is None:
            omissions[reason or "unknown"] += 1
            continue
        cost = int(entry["estimated_tokens"])
        if cost > remaining:
            omissions["budget"] += 1
            continue
        node = _symbol_node(symbol.symbol_id)
        entry["rank"] = len(entries) + 1
        entry["score"] = round(ranks.get(node, 0.0), 12)
        entry["query_relevance"] = round(
            relevance.get(node, 0.0) / max_relevance if max_relevance else 0.0,
            6,
        )
        entry["component"] = components[node]
        entries.append(entry)
        remaining -= cost

    edge_count = sum(len(targets) for targets in graph.values())
    payload: dict[str, object] = {
        "schema_version": REPOSITORY_MAP_SCHEMA_VERSION,
        "query": clean_query,
        "query_sha256": hashlib.sha256(clean_query.encode("utf-8")).hexdigest(),
        "index_digest": index_digest,
        "ranking": {
            "algorithm": "typed-personalized-pagerank",
            "damping": 0.85,
            "tolerance": 1e-9,
            "iterations": iterations,
            "residual_l1": round(residual, 15),
            "node_count": len(nodes),
            "directed_edge_count": edge_count,
            "edge_types": ["calls", "contains", "file-membership", "imports"],
            "personalization": "query-85-percent-global-15-percent"
            if relevance else "uniform-global",
        },
        "budget": {
            "token_budget": budget,
            "estimated_tokens": budget - remaining,
            "token_estimator": "ceil(rendered-utf8-bytes/3)",
            "max_entries": entry_limit,
        },
        "entries": entries,
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "selected_entry_count": len(entries),
            "omitted_candidate_count": sum(omissions.values()),
            "omissions_by_reason": dict(sorted(omissions.items())),
            "component_count": len(set(components.values())),
            "remote_calls": 0,
            "commitment_scope": (
                "payload-excluding-generation-command-and-repository-map-sha256"
            ),
        },
    }
    return _commit(payload)


def verify_repository_map_commitment(payload: dict[str, object]) -> bool:
    """Verify a repository-map receipt without workspace access."""
    try:
        candidate = copy.deepcopy(payload)
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("repository_map_sha256"))
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
    "REPOSITORY_MAP_SCHEMA_VERSION",
    "build_verified_repository_map",
    "verify_repository_map_commitment",
]
