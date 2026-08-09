"""Demand-driven, proof-carrying program graph across programming languages.

This layer composes Entroly's universal semantic IR with the existing verified
Python CFG/data-flow engines.  It intentionally does not pretend every language
has equal semantic depth: every selected file reports the strongest verified
semantic level currently available, while the agent-facing graph schema stays
stable as stronger compiler/LSP/flow adapters are added.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping

from .interprocedural_flow import build_verified_interprocedural_flow
from .models import RepositoryIndex, Symbol
from .program_graph import build_verified_program_graph
from .semantic_ir import build_universal_semantic_document
from .verified_context import build_verified_context

ADAPTIVE_PROGRAM_GRAPH_SCHEMA_VERSION = "entroly.adaptive-program-graph.v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _exact_matches(index: RepositoryIndex, query: str) -> list[Symbol]:
    normalized = query.casefold()
    return sorted(
        (
            symbol
            for symbol in index.symbols.values()
            if normalized
            in {
                symbol.symbol_id.casefold(),
                symbol.qualified_name.casefold(),
                symbol.name.casefold(),
            }
        ),
        key=lambda item: item.symbol_id,
    )


def _fresh_source(
    root: Path,
    index: RepositoryIndex,
    path: str,
) -> tuple[str | None, str]:
    record = index.files.get(path)
    if record is None:
        return None, "not-indexed"
    try:
        candidate = (root / path).resolve(strict=True)
        candidate.relative_to(root)
        raw = candidate.read_bytes()
    except (OSError, RuntimeError, ValueError):
        return None, "unsafe-or-unreadable"
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record.sha256:
        return None, "stale-index"
    try:
        return raw.decode("utf-8", errors="surrogateescape"), "verified"
    except UnicodeError:
        return None, "decode-failed"


def _context_paths(context: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    fragments = context.get("fragments")
    if not isinstance(fragments, list):
        return result
    for fragment in fragments:
        if not isinstance(fragment, Mapping):
            continue
        path = str(fragment.get("path", "")).replace("\\", "/").strip("/")
        if path and path not in result:
            result.append(path)
    return result


def _selected_symbols(
    index: RepositoryIndex,
    exact: list[Symbol],
    context: Mapping[str, object],
    *,
    limit: int,
) -> list[Symbol]:
    selected: list[Symbol] = []
    seen: set[str] = set()

    def add(symbol: Symbol) -> None:
        if symbol.symbol_id in seen or len(selected) >= limit:
            return
        seen.add(symbol.symbol_id)
        selected.append(symbol)

    for symbol in exact:
        add(symbol)
    fragments = context.get("fragments")
    if isinstance(fragments, list):
        for fragment in fragments:
            if not isinstance(fragment, Mapping):
                continue
            symbol = index.symbols.get(str(fragment.get("symbol_id", "")))
            if symbol is not None:
                add(symbol)
    return selected


def _deep_adapter_available(symbol: Symbol) -> bool:
    return symbol.language == "python" and symbol.kind in {
        "function", "method", "test"
    }


def _commit(payload: dict[str, object]) -> dict[str, object]:
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["adaptive_program_graph_sha256"] = hashlib.sha256(
        _canonical(payload)
    ).hexdigest()
    return payload


def build_adaptive_program_graph(
    root: Path,
    index: RepositoryIndex,
    query: str,
    *,
    index_digest: str,
    token_budget: int = 4_000,
    max_hops: int = 3,
    max_fragments: int = 32,
    max_files: int = 16,
    max_symbols: int = 8,
    max_semantic_nodes: int = 20_000,
    max_semantic_edges: int = 50_000,
    program_graph_limit: int = 2_000,
    interprocedural_depth: int = 3,
) -> dict[str, object]:
    """Materialize only the program facts needed for one task.

    The graph has a stable cross-language contract.  Universal parser facts are
    always evidence-classified; deeper CFG/data-flow facts are attached only
    when an adapter can verify them.  Missing depth is an explicit capability
    boundary, never a fabricated edge.
    """
    root = root.expanduser().resolve(strict=True)
    clean_query = query.strip()
    if not clean_query or len(clean_query) > 4_000:
        raise ValueError("query must contain 1 to 4000 characters")
    file_limit = max(1, min(int(max_files), 64))
    symbol_limit = max(1, min(int(max_symbols), 32))
    node_limit = max(1, min(int(max_semantic_nodes), 200_000))
    edge_limit = max(1, min(int(max_semantic_edges), 500_000))

    context = build_verified_context(
        root,
        index,
        clean_query,
        index_digest=index_digest,
        token_budget=max(128, min(int(token_budget), 32_768)),
        max_hops=max(0, min(int(max_hops), 6)),
        max_fragments=max(1, min(int(max_fragments), 100)),
    )
    exact = _exact_matches(index, clean_query)
    selected_symbols = _selected_symbols(
        index, exact, context, limit=symbol_limit
    )
    selected_paths: list[str] = []
    for symbol in selected_symbols:
        if symbol.path not in selected_paths:
            selected_paths.append(symbol.path)
    for path in _context_paths(context):
        if path not in selected_paths:
            selected_paths.append(path)
    selected_paths = selected_paths[:file_limit]

    semantic_files: list[dict[str, object]] = []
    diagnostics: list[dict[str, str]] = []
    total_nodes = 0
    total_edges = 0
    epistemic = Counter()
    capabilities = Counter()
    selected_file_digests: dict[str, str] = {}

    for path in selected_paths:
        if total_nodes >= node_limit or total_edges >= edge_limit:
            diagnostics.append({
                "path": path,
                "status": "global-semantic-budget-reached",
            })
            break
        source, freshness = _fresh_source(root, index, path)
        if source is None:
            diagnostics.append({"path": path, "status": freshness})
            continue
        record = index.files[path]
        selected_file_digests[path] = record.sha256
        remaining_nodes = max(1, node_limit - total_nodes)
        document = build_universal_semantic_document(
            source,
            path,
            max_bytes=max(record.byte_length, 1),
            max_nodes=remaining_nodes,
        )
        payload = document.to_dict()
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", [])
        if not isinstance(nodes, list) or not isinstance(edges, list):
            diagnostics.append({"path": path, "status": "invalid-semantic-document"})
            continue
        if len(edges) > edge_limit - total_edges:
            # Partial graph edges are never silently presented as complete.
            payload["edges"] = edges[: max(0, edge_limit - total_edges)]
            payload.setdefault("diagnostics", []).append(
                "adaptive graph edge budget reached; remaining semantic edges omitted"
            )
            edges = payload["edges"]
        total_nodes += len(nodes)
        total_edges += len(edges)
        capability = payload.get("capabilities", {})
        if isinstance(capability, Mapping):
            capabilities[str(capability.get("level", "unknown"))] += 1
        for node in nodes:
            if isinstance(node, Mapping):
                epistemic[str(node.get("epistemic_class", "unknown"))] += 1
        for edge in edges:
            if isinstance(edge, Mapping):
                epistemic[str(edge.get("epistemic_class", "unknown"))] += 1
        semantic_files.append(payload)

    deep_semantics: list[dict[str, object]] = []
    adapter_boundaries: list[dict[str, str]] = []
    for symbol in selected_symbols:
        if _deep_adapter_available(symbol):
            program = build_verified_program_graph(
                root,
                index,
                symbol.symbol_id,
                index_digest=index_digest,
                limit=max(16, min(int(program_graph_limit), 10_000)),
            )
            inter = build_verified_interprocedural_flow(
                root,
                index,
                symbol.symbol_id,
                index_digest=index_digest,
                direction="both",
                max_depth=max(0, min(int(interprocedural_depth), 12)),
                max_call_edges=2_000,
                max_flow_edges=20_000,
                max_nodes=20_000,
            )
            deep_semantics.append({
                "symbol_id": symbol.symbol_id,
                "language": symbol.language,
                "semantic_level": "flow",
                "program_graph": program,
                "interprocedural_flow": inter,
            })
        else:
            adapter_boundaries.append({
                "symbol_id": symbol.symbol_id,
                "language": symbol.language,
                "available": "structure",
                "missing": "verified-language-specific-flow-adapter",
            })

    context_receipt = context.get("receipt")
    context_sha = (
        str(context_receipt.get("context_sha256", ""))
        if isinstance(context_receipt, Mapping)
        else ""
    )
    payload: dict[str, object] = {
        "schema_version": ADAPTIVE_PROGRAM_GRAPH_SCHEMA_VERSION,
        "query": clean_query,
        "query_sha256": hashlib.sha256(clean_query.encode("utf-8")).hexdigest(),
        "index_digest": index_digest,
        "query_route": {
            "exact_candidates": [item.to_dict() for item in exact[:100]],
            "exact_candidate_count": len(exact),
            "selected_symbol_ids": [item.symbol_id for item in selected_symbols],
            "selected_paths": selected_paths,
        },
        "verified_context": context,
        "semantic_files": semantic_files,
        "deep_semantics": deep_semantics,
        "adapter_boundaries": adapter_boundaries,
        "coverage": {
            "semantic_files": len(semantic_files),
            "semantic_nodes": total_nodes,
            "semantic_edges": total_edges,
            "capability_levels": dict(sorted(capabilities.items())),
            "epistemic_facts": dict(sorted(epistemic.items())),
            "answer_sufficiency": "unproven",
            "interpretation": (
                "selected-evidence-coverage-not-whole-program-completeness-or-answer-quality"
            ),
        },
        "analysis_contract": {
            "materialization": "query-time-selected-files-only",
            "universal_layer": "exact-source-plus-parser-evidence",
            "deep_adapter_policy": "verified-adapters-only",
            "missing-adapter_behavior": "report-boundary-never-invent-flow",
            "learned_facts_allowed": False,
            "remote_calls": 0,
        },
        "diagnostics": diagnostics,
        "receipt": {
            "context_sha256": context_sha,
            "selected_file_sha256": dict(sorted(selected_file_digests.items())),
            "remote_calls": 0,
            "commitment_scope": (
                "payload-excluding-generation-command-and-adaptive-program-graph-sha256"
            ),
        },
    }
    return _commit(payload)


def verify_adaptive_program_graph_commitment(payload: Mapping[str, object]) -> bool:
    """Verify the detached top-level adaptive-graph commitment."""
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("generation", None)
        candidate.pop("command", None)
        if candidate.get("schema_version") != ADAPTIVE_PROGRAM_GRAPH_SCHEMA_VERSION:
            return False
        receipt = candidate.get("receipt")
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("adaptive_program_graph_sha256"))
        return hashlib.sha256(_canonical(candidate)).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "ADAPTIVE_PROGRAM_GRAPH_SCHEMA_VERSION",
    "build_adaptive_program_graph",
    "verify_adaptive_program_graph_commitment",
]
