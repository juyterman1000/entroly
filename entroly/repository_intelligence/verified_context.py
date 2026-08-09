"""Evidence-carrying, budgeted code context over a repository index.

The graph is a retrieval aid, never an authority. Every emitted source fragment
is re-read from the fixed workspace, checked against the indexed file hash, and
addressed by an exact fragment hash. Ambiguous calls remain explicit negative
evidence instead of being promoted into invented graph edges.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .models import CallEdge, RepositoryIndex, Symbol
from .git_history import collect_git_history

CONTEXT_SCHEMA_VERSION = "entroly.verified-code-context.v1"
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "code", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the",
    "this", "to", "what", "where", "which", "with",
})


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return {
        item.lower()
        for item in _TOKEN_RE.findall(expanded.replace("-", "_"))
        if len(item) > 1 and item.lower() not in _STOPWORDS
    }


def _token_cost(value: str) -> int:
    # Deliberately conservative and tokenizer-independent. The receipt names
    # the estimator so this number is never confused with provider billing.
    return max(1, math.ceil(len(value.encode("utf-8", errors="surrogateescape")) / 3))


def _symbol_score(symbol: Symbol, query: str, query_tokens: set[str]) -> float:
    name = symbol.name.lower()
    score = 0.0
    if name and re.search(rf"(?<![\w$]){re.escape(name)}(?![\w$])", query.lower()):
        score += 120.0
    score += 36.0 * len(_tokens(symbol.name) & query_tokens)
    score += 22.0 * len(_tokens(symbol.qualified_name) & query_tokens)
    score += 12.0 * len(_tokens(symbol.path) & query_tokens)
    score += 5.0 * len(_tokens(symbol.signature) & query_tokens)
    if "test" in query_tokens and symbol.kind == "test":
        score += 25.0
    if {"class", "type", "interface"} & query_tokens and symbol.kind in {
        "class", "struct", "interface", "trait", "type",
    }:
        score += 18.0
    if {"function", "method", "call", "caller", "callee"} & query_tokens and symbol.kind in {
        "function", "fn", "method", "constructor",
    }:
        score += 12.0
    return score


def _adjacency(
    index: RepositoryIndex,
) -> tuple[dict[str, list[tuple[str, str, CallEdge | None]]], dict[str, list[Symbol]]]:
    graph: dict[str, list[tuple[str, str, CallEdge | None]]] = defaultdict(list)
    children: dict[str, list[Symbol]] = defaultdict(list)
    by_path: dict[str, list[Symbol]] = defaultdict(list)
    for symbol in index.symbols.values():
        by_path[symbol.path].append(symbol)
        if symbol.parent_id and symbol.parent_id in index.symbols:
            graph[symbol.symbol_id].append((symbol.parent_id, "contained-by", None))
            graph[symbol.parent_id].append((symbol.symbol_id, "contains", None))
            children[symbol.parent_id].append(symbol)
    for edge in index.call_edges:
        if (
            edge.caller_id in index.symbols
            and edge.callee_id in index.symbols
            and edge.evidence_sha256
            and 0 <= edge.start_byte < edge.end_byte
        ):
            graph[edge.caller_id].append((edge.callee_id, "calls", edge))
            graph[edge.callee_id].append((edge.caller_id, "called-by", edge))
    for source_path, dependencies in index.file_dependencies.items():
        source_symbols = sorted(
            by_path.get(source_path, ()),
            key=lambda symbol: (symbol.parent_id is not None, symbol.line_start, symbol.symbol_id),
        )[:3]
        for target_path in dependencies:
            target_symbols = sorted(
                by_path.get(target_path, ()),
                key=lambda symbol: (
                    symbol.parent_id is not None,
                    symbol.line_start,
                    symbol.symbol_id,
                ),
            )[:3]
            for source_symbol in source_symbols:
                for target_symbol in target_symbols:
                    graph[source_symbol.symbol_id].append((
                        target_symbol.symbol_id, "imports", None
                    ))
                    graph[target_symbol.symbol_id].append((
                        source_symbol.symbol_id, "imported-by", None
                    ))
    return graph, children


def _rank_candidates(
    index: RepositoryIndex,
    query: str,
    *,
    max_hops: int,
    max_candidates: int,
) -> tuple[list[tuple[float, str, tuple[str, ...]]], set[str]]:
    query_tokens = _tokens(query)
    scored = [
        (_symbol_score(symbol, query, query_tokens), symbol.symbol_id)
        for symbol in index.symbols.values()
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    positive = [(score, symbol_id) for score, symbol_id in scored if score > 0]
    threshold = max(12.0, positive[0][0] * 0.45) if positive else 0.0
    seeds = [
        (score, symbol_id)
        for score, symbol_id in positive
        if score >= threshold
    ][:8]
    if not seeds:
        seeds = [(1.0, symbol_id) for _, symbol_id in scored[:3]]

    graph, _ = _adjacency(index)
    heap: list[tuple[float, int, str, tuple[str, ...]]] = []
    for score, symbol_id in seeds:
        heapq.heappush(heap, (-score, 0, symbol_id, ("query-match",)))
    best: dict[str, tuple[float, tuple[str, ...]]] = {}
    seed_ids = {symbol_id for _, symbol_id in seeds}
    while heap and len(best) < max_candidates:
        negative, depth, symbol_id, reasons = heapq.heappop(heap)
        score = -negative
        previous = best.get(symbol_id)
        if previous is not None and previous[0] >= score:
            continue
        best[symbol_id] = (score, reasons)
        if depth >= max_hops:
            continue
        for neighbor, relation, _evidence in graph.get(symbol_id, ()):
            factor = {
                "calls": 0.82,
                "called-by": 0.74,
                "contained-by": 0.88,
                "contains": 0.68,
                "imports": 0.62,
                "imported-by": 0.55,
            }[relation]
            next_score = score * factor
            if next_score < 1.0:
                continue
            heapq.heappush(
                heap,
                (-next_score, depth + 1, neighbor, (*reasons, relation)),
            )
    ranked = [
        (score, symbol_id, reasons)
        for symbol_id, (score, reasons) in best.items()
    ]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked, seed_ids


def _read_verified_fragment(
    root: Path,
    index: RepositoryIndex,
    symbol: Symbol,
    remaining_tokens: int,
) -> tuple[dict[str, object] | None, str | None]:
    record = index.files.get(symbol.path)
    if record is None:
        return None, "missing-file-record"
    try:
        candidate = (root / symbol.path).resolve(strict=True)
        candidate.relative_to(root)
        raw = candidate.read_bytes()
    except (OSError, RuntimeError, ValueError):
        return None, "unsafe-or-unreadable"
    source_sha256 = hashlib.sha256(raw).hexdigest()
    if source_sha256 != record.sha256:
        return None, "stale-index"

    start = symbol.start_byte
    end = symbol.end_byte
    resolution = "full"
    if not (0 <= start < end <= len(raw)):
        lines = raw.decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
        start = len("".join(lines[: max(0, symbol.line_start - 1)]).encode(
            "utf-8", errors="surrogateescape"
        ))
        end = len("".join(lines[: symbol.line_end]).encode(
            "utf-8", errors="surrogateescape"
        ))
    content_raw = raw[start:end]
    content = content_raw.decode("utf-8", errors="surrogateescape")
    cost = _token_cost(content)
    if cost > remaining_tokens and symbol.signature:
        signature_raw = symbol.signature.encode("utf-8", errors="surrogateescape")
        offset = raw.find(signature_raw, start, end)
        if offset < 0:
            return None, "unverifiable-signature"
        start = offset
        end = offset + len(signature_raw)
        content_raw = raw[start:end]
        content = content_raw.decode("utf-8", errors="surrogateescape")
        cost = _token_cost(content)
        resolution = "signature"
    if cost > remaining_tokens:
        return None, "budget"
    return {
        "symbol_id": symbol.symbol_id,
        "path": symbol.path,
        "language": symbol.language,
        "kind": symbol.kind,
        "qualified_name": symbol.qualified_name,
        "line_start": symbol.line_start,
        "line_end": symbol.line_end,
        "start_byte": start,
        "end_byte": end,
        "resolution": resolution,
        "content": content,
        "estimated_tokens": cost,
        "source_sha256": source_sha256,
        "fragment_sha256": hashlib.sha256(content_raw).hexdigest(),
        "parse_backend": symbol.parse_backend,
        "trust": "untrusted-source-bytes",
    }, None


def _verified_evidence_status(
    root: Path,
    index: RepositoryIndex,
    path: str,
    start_byte: int,
    end_byte: int,
    expected_sha256: str,
    source_cache: dict[str, tuple[bytes | None, str]],
) -> str:
    cached = source_cache.get(path)
    if cached is None:
        record = index.files.get(path)
        if record is None:
            cached = (None, "missing-file-record")
        else:
            try:
                candidate = (root / path).resolve(strict=True)
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
        source_cache[path] = cached
    raw, status = cached
    if raw is None:
        return status
    if not expected_sha256:
        return "missing-evidence-hash"
    if not (0 <= start_byte < end_byte <= len(raw)):
        return "invalid-evidence-range"
    actual = hashlib.sha256(raw[start_byte:end_byte]).hexdigest()
    return "verified" if actual == expected_sha256 else "evidence-hash-mismatch"


def _selected_relations(
    root: Path,
    index: RepositoryIndex,
    selected: set[str],
    source_cache: dict[str, tuple[bytes | None, str]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    relations: list[dict[str, object]] = []
    omissions: dict[str, int] = defaultdict(int)

    for symbol_id in sorted(selected):
        symbol = index.symbols[symbol_id]
        if symbol.parent_id in selected:
            relations.append({
                "source": symbol.parent_id,
                "target": symbol_id,
                "kind": "contains",
                "confidence": "parser-backed",
            })
    for edge in index.call_edges:
        if edge.caller_id in selected and edge.callee_id in selected:
            status = _verified_evidence_status(
                root,
                index,
                edge.path,
                edge.start_byte,
                edge.end_byte,
                edge.evidence_sha256,
                source_cache,
            )
            if status != "verified":
                omissions[status] += 1
                continue
            relations.append({
                "source": edge.caller_id,
                "target": edge.callee_id,
                "kind": edge.kind,
                "confidence": edge.confidence,
                "resolution": edge.resolution,
                "path": edge.path,
                "line": edge.line,
                "start_byte": edge.start_byte,
                "end_byte": edge.end_byte,
                "evidence_sha256": edge.evidence_sha256,
                "evidence_status": status,
            })
    selected_paths = {index.symbols[symbol_id].path for symbol_id in selected}
    for source_path, dependencies in sorted(index.file_dependencies.items()):
        if source_path not in selected_paths:
            continue
        for target_path in dependencies:
            if target_path in selected_paths:
                relations.append({
                    "source": source_path,
                    "target": target_path,
                    "kind": "imports",
                    "confidence": "resolved-file",
                })
    relations.sort(key=lambda item: (
        str(item["source"]), str(item["target"]), str(item["kind"])
    ))
    return relations, omissions


def build_verified_context(
    root: Path,
    index: RepositoryIndex,
    query: str,
    *,
    index_digest: str,
    token_budget: int = 2_000,
    max_hops: int = 2,
    max_fragments: int = 24,
    include_history: bool = False,
    max_history_commits: int = 20,
) -> dict[str, object]:
    """Build a deterministic partial code graph with a content receipt."""
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")
    if len(clean_query) > 4_000:
        raise ValueError("query must be at most 4000 characters")
    budget = max(128, min(int(token_budget), 32_768))
    hops = max(0, min(int(max_hops), 6))
    fragment_limit = max(1, min(int(max_fragments), 100))
    ranked, seed_ids = _rank_candidates(
        index,
        clean_query,
        max_hops=hops,
        max_candidates=max(fragment_limit * 8, 64),
    )

    fragments: list[dict[str, object]] = []
    omissions: dict[str, int] = defaultdict(int)
    remaining = budget
    selected: set[str] = set()
    for score, symbol_id, reasons in ranked:
        if len(fragments) >= fragment_limit:
            omissions["fragment-limit"] += 1
            continue
        fragment, omitted = _read_verified_fragment(
            root,
            index,
            index.symbols[symbol_id],
            remaining,
        )
        if fragment is None:
            omissions[omitted or "unknown"] += 1
            continue
        fragment["score"] = round(score, 6)
        fragment["selection_path"] = list(reasons)
        fragments.append(fragment)
        selected.add(symbol_id)
        remaining -= int(fragment["estimated_tokens"])

    source_cache: dict[str, tuple[bytes | None, str]] = {}
    relevant_unresolved = []
    for call in index.unresolved_calls:
        if call.caller_id not in selected:
            continue
        status = _verified_evidence_status(
            root,
            index,
            call.path,
            call.start_byte,
            call.end_byte,
            call.evidence_sha256,
            source_cache,
        )
        if status != "verified":
            omissions[f"unresolved-call-{status}"] += 1
            continue
        item = call.to_dict()
        item["evidence_status"] = status
        relevant_unresolved.append(item)
    relevant_unresolved = relevant_unresolved[:100]
    relations, relation_omissions = _selected_relations(
        root,
        index,
        selected,
        source_cache,
    )
    for reason, count in relation_omissions.items():
        omissions[f"relation-{reason}"] += count
    history = (
        collect_git_history(
            root,
            (str(fragment["path"]) for fragment in fragments),
            max_commits=max_history_commits,
        )
        if include_history
        else {
            "available": False,
            "commits": [],
            "diagnostic": "not-requested",
            "remote_calls": 0,
        }
    )
    selected_seeds = seed_ids & selected
    seed_coverage = len(selected_seeds) / max(1, len(seed_ids))
    payload: dict[str, object] = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "query": clean_query,
        "query_sha256": hashlib.sha256(clean_query.encode("utf-8")).hexdigest(),
        "index_digest": index_digest,
        "retrieval": {
            "policy": "selective-query-partial-graph",
            "max_hops": hops,
            "token_budget": budget,
            "estimated_tokens": budget - remaining,
            "token_estimator": "ceil(utf8_bytes/3)",
            "seed_count": len(seed_ids),
            "selected_seed_count": len(selected_seeds),
            "seed_coverage": round(seed_coverage, 6),
            "sufficient": bool(fragments) and seed_coverage >= 0.5,
        },
        "fragments": fragments,
        "relations": relations,
        "unresolved_calls": relevant_unresolved,
        "history": history,
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "selected_fragment_count": len(fragments),
            "selected_relation_count": len(relations),
            "omitted_candidate_count": sum(omissions.values()),
            "omissions_by_reason": dict(sorted(omissions.items())),
            "ambiguous_or_unresolved_calls": len(relevant_unresolved),
            "remote_calls": 0,
            "history_requested": bool(include_history),
            "commitment_scope": "payload-excluding-generation-command-and-context-sha256",
        },
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    payload["receipt"]["context_sha256"] = hashlib.sha256(canonical).hexdigest()  # type: ignore[index]
    return payload


def verify_context_commitment(payload: dict[str, object]) -> bool:
    """Verify the deterministic payload commitment without workspace access."""
    try:
        candidate = copy.deepcopy(payload)
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("context_sha256"))
        canonical = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


def build_symbol_graph(
    root: Path,
    index: RepositoryIndex,
    symbol_query: str,
    *,
    index_digest: str,
    direction: str = "both",
    max_depth: int = 3,
    limit: int = 200,
) -> dict[str, object]:
    """Return a freshness-checked call graph after unambiguous symbol lookup."""
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    query = symbol_query.strip()
    if not query or len(query) > 1_000:
        raise ValueError("symbol_query must contain 1 to 1000 characters")
    selected_direction = direction.strip().lower()
    if selected_direction not in {"callers", "callees", "both"}:
        raise ValueError("direction must be callers, callees, or both")
    depth_limit = max(0, min(int(max_depth), 12))
    node_limit = max(1, min(int(limit), 5_000))
    lowered = query.lower()
    matches = sorted(
        (
            symbol
            for symbol in index.symbols.values()
            if symbol.symbol_id.lower() == lowered
            or symbol.qualified_name.lower() == lowered
            or symbol.name.lower() == lowered
        ),
        key=lambda symbol: symbol.symbol_id,
    )
    resolution = "resolved" if len(matches) == 1 else "ambiguous" if matches else "not-found"
    file_cache: dict[str, tuple[bytes | None, str]] = {}

    def verified_file(path: str) -> tuple[bytes | None, str]:
        cached = file_cache.get(path)
        if cached is not None:
            return cached
        record = index.files.get(path)
        if record is None:
            result = (None, "missing-file-record")
        else:
            try:
                candidate = (root / path).resolve(strict=True)
                candidate.relative_to(root)
                raw = candidate.read_bytes()
            except (OSError, RuntimeError, ValueError):
                result = (None, "unsafe-or-unreadable")
            else:
                result = (
                    (raw, "verified")
                    if hashlib.sha256(raw).hexdigest() == record.sha256
                    else (None, "stale-index")
                )
        file_cache[path] = result
        return result

    def candidate_payload(symbol: Symbol) -> dict[str, object]:
        payload = symbol.to_dict()
        _raw, status = verified_file(symbol.path)
        payload["source_status"] = status
        return payload

    def edge_is_verified(edge: CallEdge) -> tuple[bool, str]:
        raw, status = verified_file(edge.path)
        if raw is None:
            return False, status
        if not (0 <= edge.start_byte < edge.end_byte <= len(raw)):
            return False, "invalid-evidence-range"
        actual = hashlib.sha256(raw[edge.start_byte:edge.end_byte]).hexdigest()
        if not edge.evidence_sha256 or actual != edge.evidence_sha256:
            return False, "evidence-hash-mismatch"
        return True, "verified"

    def finish(payload: dict[str, object]) -> dict[str, object]:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        receipt = payload["receipt"]
        assert isinstance(receipt, dict)
        receipt["graph_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload

    base: dict[str, object] = {
        "schema_version": "entroly.verified-symbol-graph.v1",
        "index_digest": index_digest,
        "query": query,
        "resolution": resolution,
        "candidates": [candidate_payload(symbol) for symbol in matches[:100]],
        "nodes": [],
        "edges": [],
        "truncated": len(matches) > 100,
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "selected_node_count": 0,
            "selected_edge_count": 0,
            "omissions_by_reason": {},
            "remote_calls": 0,
            "commitment_scope": "payload-excluding-generation-command-and-graph-sha256",
        },
    }
    if len(matches) != 1:
        return finish(base)

    root_symbol = matches[0]
    _root_raw, root_status = verified_file(root_symbol.path)
    if root_status != "verified":
        base["resolution"] = root_status
        base["receipt"]["omissions_by_reason"] = {root_status: 1}  # type: ignore[index]
        return finish(base)

    outgoing: dict[str, list[CallEdge]] = defaultdict(list)
    incoming: dict[str, list[CallEdge]] = defaultdict(list)
    for edge in index.call_edges:
        if edge.caller_id in index.symbols and edge.callee_id in index.symbols:
            outgoing[edge.caller_id].append(edge)
            incoming[edge.callee_id].append(edge)
    queue: list[tuple[str, int]] = [(root_symbol.symbol_id, 0)]
    seen = {root_symbol.symbol_id}
    selected_edges: set[CallEdge] = set()
    omissions: dict[str, int] = defaultdict(int)
    cursor = 0
    truncated = False
    while cursor < len(queue):
        symbol_id, depth = queue[cursor]
        cursor += 1
        if depth >= depth_limit:
            continue
        choices: list[tuple[str, CallEdge]] = []
        if selected_direction in {"callees", "both"}:
            choices.extend((edge.callee_id, edge) for edge in outgoing.get(symbol_id, ()))
        if selected_direction in {"callers", "both"}:
            choices.extend((edge.caller_id, edge) for edge in incoming.get(symbol_id, ()))
        choices.sort(key=lambda item: (item[0], item[1].path, item[1].line))
        for neighbor, edge in choices:
            verified, reason = edge_is_verified(edge)
            if not verified:
                omissions[reason] += 1
                continue
            neighbor_symbol = index.symbols[neighbor]
            _neighbor_raw, neighbor_status = verified_file(neighbor_symbol.path)
            if neighbor_status != "verified":
                omissions[neighbor_status] += 1
                continue
            if len(selected_edges) >= node_limit:
                truncated = True
                omissions["result-limit"] += 1
                continue
            selected_edges.add(edge)
            if neighbor in seen:
                continue
            if len(seen) >= node_limit:
                truncated = True
                continue
            seen.add(neighbor)
            queue.append((neighbor, depth + 1))
    base["nodes"] = [
        candidate_payload(index.symbols[symbol_id]) for symbol_id in sorted(seen)
    ]
    base["edges"] = [
        edge.to_dict()
        for edge in sorted(
            selected_edges,
            key=lambda edge: (
                edge.caller_id, edge.callee_id, edge.path, edge.line
            ),
        )
        if edge.caller_id in seen and edge.callee_id in seen
    ]
    base["truncated"] = truncated
    base["root_symbol_id"] = root_symbol.symbol_id
    base["direction"] = selected_direction
    base["max_depth"] = depth_limit
    base["receipt"] = {
        "freshness": "verified-against-indexed-source-sha256",
        "selected_node_count": len(base["nodes"]),
        "selected_edge_count": len(base["edges"]),
        "omissions_by_reason": dict(sorted(omissions.items())),
        "remote_calls": 0,
        "commitment_scope": "payload-excluding-generation-command-and-graph-sha256",
    }
    return finish(base)


def verify_symbol_graph_commitment(payload: dict[str, object]) -> bool:
    """Verify a symbol-graph receipt without workspace access."""
    try:
        candidate = copy.deepcopy(payload)
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("graph_sha256"))
        canonical = json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


def selected_symbol_ids(payload: dict[str, object]) -> Iterable[str]:
    """Yield selected IDs without exposing payload representation details."""
    fragments = payload.get("fragments", ())
    if not isinstance(fragments, list):
        return ()
    return (
        str(fragment["symbol_id"])
        for fragment in fragments
        if isinstance(fragment, dict) and "symbol_id" in fragment
    )
