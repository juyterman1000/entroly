"""Neuro-symbolic, proof-carrying partial program slices.

Caller-supplied learned scores may propose indexed symbols. They can influence
ranking, but they cannot create identities, edges, source spans, or confidence.
Every admitted fact is reconstructed by deterministic repository analyses and
bound into a detached receipt.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .interprocedural_flow import (
    build_verified_interprocedural_flow,
    verify_interprocedural_flow_commitment,
)
from .models import RepositoryIndex, Symbol
from .program_graph import (
    build_verified_program_graph,
    verify_program_graph_commitment,
)
from .verified_context import build_verified_context, verify_context_commitment

PROGRAM_SLICE_SCHEMA_VERSION = "entroly.verified-program-slice.v1"


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


def _entry_points(
    index: RepositoryIndex,
    exact: list[Symbol],
    context: Mapping[str, object],
    *,
    limit: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(symbol: Symbol, origin: str, score: float | None = None) -> None:
        if symbol.symbol_id in seen or len(selected) >= limit:
            return
        if symbol.language != "python" or symbol.kind not in {
            "function",
            "method",
            "test",
        }:
            return
        seen.add(symbol.symbol_id)
        item: dict[str, object] = {
            "symbol_id": symbol.symbol_id,
            "qualified_name": symbol.qualified_name,
            "path": symbol.path,
            "origin": origin,
        }
        if score is not None:
            item["retrieval_score"] = round(float(score), 8)
        selected.append(item)

    for symbol in exact:
        add(
            symbol,
            "exact-query" if len(exact) == 1 else "ambiguous-exact-candidate",
        )
    fragments = context.get("fragments")
    if isinstance(fragments, list):
        for fragment in fragments:
            if not isinstance(fragment, Mapping):
                continue
            symbol_id = fragment.get("symbol_id")
            symbol = index.symbols.get(str(symbol_id))
            if symbol is None:
                continue
            try:
                score = float(fragment.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            origin = (
                "verified-external-proposal"
                if "proposal_score" in fragment
                else "retrieved-query-candidate"
            )
            add(symbol, origin, score)
    return selected


def _commit(payload: dict[str, object]) -> dict[str, object]:
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["program_slice_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def build_verified_program_slice(
    root: Path,
    index: RepositoryIndex,
    query: str,
    *,
    index_digest: str,
    token_budget: int = 4_000,
    max_hops: int = 3,
    max_fragments: int = 32,
    max_entry_points: int = 3,
    flow_direction: str = "outgoing",
    flow_depth: int = 3,
    program_graph_limit: int = 1_000,
    max_call_edges: int = 1_000,
    max_flow_edges: int = 10_000,
    proposal_scores: Iterable[Mapping[str, object]] = (),
    proposal_provider: str = "caller-supplied",
) -> dict[str, object]:
    """Build a query-time slice from verified context and program semantics."""
    root = root.expanduser().resolve(strict=True)
    clean_query = query.strip()
    if not clean_query or len(clean_query) > 4_000:
        raise ValueError("query must contain 1 to 4000 characters")
    entry_limit = max(1, min(int(max_entry_points), 8))
    exact = _exact_matches(index, clean_query)
    if len(exact) == 1:
        route = "code-entity"
        identity = "unique-exact"
    elif exact:
        route = "code-entity"
        identity = "ambiguous-exact"
    else:
        route = "natural-language"
        identity = "not-applicable"

    context = build_verified_context(
        root,
        index,
        clean_query,
        index_digest=index_digest,
        token_budget=token_budget,
        max_hops=max_hops,
        max_fragments=max_fragments,
        proposal_scores=proposal_scores,
        proposal_provider=proposal_provider,
    )
    entries = _entry_points(index, exact, context, limit=entry_limit)
    intra: list[dict[str, object]] = []
    inter: list[dict[str, object]] = []
    for entry in entries:
        symbol_id = str(entry["symbol_id"])
        program = build_verified_program_graph(
            root,
            index,
            symbol_id,
            index_digest=index_digest,
            limit=max(16, min(int(program_graph_limit), 10_000)),
        )
        intra.append(program)
        flow = build_verified_interprocedural_flow(
            root,
            index,
            symbol_id,
            index_digest=index_digest,
            direction=flow_direction,
            max_depth=max(0, min(int(flow_depth), 12)),
            max_call_edges=max(1, min(int(max_call_edges), 100_000)),
            max_flow_edges=max(1, min(int(max_flow_edges), 100_000)),
            max_nodes=max(1, min(int(max_flow_edges) * 2, 100_000)),
        )
        inter.append(flow)

    context_receipt = context.get("receipt")
    context_sha = (
        str(context_receipt.get("context_sha256"))
        if isinstance(context_receipt, Mapping)
        else ""
    )
    program_shas = []
    for graph in intra:
        receipt = graph.get("receipt")
        if isinstance(receipt, Mapping):
            program_shas.append(str(receipt.get("program_graph_sha256", "")))
    flow_shas = []
    for flow in inter:
        receipt = flow.get("receipt")
        if isinstance(receipt, Mapping):
            flow_shas.append(str(receipt.get("interprocedural_flow_sha256", "")))
    known_calls = sum(len(flow.get("call_relations", [])) for flow in inter)
    known_flows = sum(len(flow.get("flow_edges", [])) for flow in inter)
    unresolved = sum(len(flow.get("unresolved_boundary", [])) for flow in inter)
    payload: dict[str, object] = {
        "schema_version": PROGRAM_SLICE_SCHEMA_VERSION,
        "query": clean_query,
        "query_sha256": hashlib.sha256(clean_query.encode("utf-8")).hexdigest(),
        "index_digest": index_digest,
        "query_route": {
            "kind": route,
            "identity_status": identity,
            "exact_candidates": [symbol.to_dict() for symbol in exact[:100]],
            "exact_candidates_omitted": max(0, len(exact) - 100),
        },
        "entry_points": entries,
        "verified_context": context,
        "intraprocedural_graphs": intra,
        "interprocedural_flows": inter,
        "coverage": {
            "answer_sufficiency": "unproven",
            "selected_entry_points": len(entries),
            "verified_call_relations": known_calls,
            "verified_value_flow_edges": known_flows,
            "unresolved_call_boundary": unresolved,
            "interpretation": (
                "structural-evidence-coverage-only-not-answer-quality-or-completeness"
            ),
        },
        "neuro_symbolic_contract": {
            "learned_role": "optional-entry-point-ranking-proposal",
            "symbolic_gate": (
                "indexed-identity-source-freshness-exact-span-and-static-resolution"
            ),
            "proposal_may_create_facts": False,
            "proposal_may_raise_confidence": False,
            "remote_calls": 0,
        },
        "receipt": {
            "context_sha256": context_sha,
            "program_graph_sha256": program_shas,
            "interprocedural_flow_sha256": flow_shas,
            "remote_calls": 0,
            "commitment_scope": (
                "payload-excluding-generation-command-and-program-slice-sha256"
            ),
        },
    }
    return _commit(payload)


def verify_program_slice_commitment(payload: Mapping[str, object]) -> bool:
    """Verify the slice and every nested evidence commitment."""
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("generation", None)
        candidate.pop("command", None)
        if candidate.get("schema_version") != PROGRAM_SLICE_SCHEMA_VERSION:
            return False
        context = candidate.get("verified_context")
        intra = candidate.get("intraprocedural_graphs")
        inter = candidate.get("interprocedural_flows")
        if not isinstance(context, dict) or not verify_context_commitment(context):
            return False
        if not isinstance(intra, list) or not all(
            isinstance(item, dict) and verify_program_graph_commitment(item)
            for item in intra
        ):
            return False
        if not isinstance(inter, list) or not all(
            isinstance(item, dict) and verify_interprocedural_flow_commitment(item)
            for item in inter
        ):
            return False
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("program_slice_sha256"))
        return hashlib.sha256(_canonical(candidate)).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "PROGRAM_SLICE_SCHEMA_VERSION",
    "build_verified_program_slice",
    "verify_program_slice_commitment",
]
