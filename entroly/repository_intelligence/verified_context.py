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
from itertools import islice
from pathlib import Path
from typing import Iterable, Mapping

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
    proposal_scores: Mapping[str, float] | None = None,
) -> tuple[list[tuple[float, str, tuple[str, ...]]], set[str]]:
    query_tokens = _tokens(query)
    proposals = proposal_scores or {}
    scored = [
        (
            _symbol_score(symbol, query, query_tokens)
            + 80.0 * proposals.get(symbol.symbol_id, 0.0),
            symbol.symbol_id,
        )
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
    proposed_seeds = sorted(
        (
            (_symbol_score(index.symbols[symbol_id], query, query_tokens) + 80.0 * score, symbol_id)
            for symbol_id, score in proposals.items()
            if symbol_id in index.symbols and score > 0.0
        ),
        key=lambda item: (-item[0], item[1]),
    )[:8]
    seed_by_id = {symbol_id: score for score, symbol_id in seeds}
    for score, symbol_id in proposed_seeds:
        seed_by_id[symbol_id] = max(score, seed_by_id.get(symbol_id, 0.0))
    seeds = sorted(
        ((score, symbol_id) for symbol_id, score in seed_by_id.items()),
        key=lambda item: (-item[0], item[1]),
    )[:8]
    if not seeds:
        seeds = [(1.0, symbol_id) for _, symbol_id in scored[:3]]

    graph, _ = _adjacency(index)
    heap: list[tuple[float, int, str, tuple[str, ...]]] = []
    for score, symbol_id in seeds:
        reasons = ["query-match"]
        if symbol_id in proposals:
            reasons.append("verified-external-proposal-identity")
        heapq.heappush(heap, (-score, 0, symbol_id, tuple(reasons)))
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


def _validated_proposals(
    index: RepositoryIndex,
    proposals: Iterable[Mapping[str, object]],
) -> tuple[dict[str, float], list[dict[str, object]], dict[str, int]]:
    scores: dict[str, float] = {}
    accepted: dict[str, dict[str, object]] = {}
    omissions: dict[str, int] = defaultdict(int)
    for position, raw in enumerate(islice(proposals, 1_001)):
        if position >= 1_000:
            omissions["proposal-limit"] += 1
            break
        if not isinstance(raw, Mapping):
            omissions["invalid-proposal"] += 1
            continue
        symbol_id = raw.get("symbol_id")
        raw_score = raw.get("score")
        if not isinstance(symbol_id, str) or symbol_id not in index.symbols:
            omissions["unknown-symbol"] += 1
            continue
        if isinstance(raw_score, bool):
            omissions["invalid-score"] += 1
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            omissions["invalid-score"] += 1
            continue
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            omissions["invalid-score"] += 1
            continue
        if score > scores.get(symbol_id, -1.0):
            scores[symbol_id] = score
            accepted[symbol_id] = {
                "symbol_id": symbol_id,
                "score": round(score, 8),
            }
    return (
        scores,
        [accepted[symbol_id] for symbol_id in sorted(accepted)],
        dict(sorted(omissions.items())),
    )


def _verified_source_span(
    root: Path,
    index: RepositoryIndex,
    symbol: Symbol,
) -> tuple[tuple[bytes, bytes, int, int, str] | None, str | None]:
    """Read one symbol span only when it still matches the indexed source."""

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
    if not (0 <= start < end <= len(raw)):
        lines = raw.decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
        start = len("".join(lines[: max(0, symbol.line_start - 1)]).encode(
            "utf-8", errors="surrogateescape"
        ))
        end = len("".join(lines[: symbol.line_end]).encode(
            "utf-8", errors="surrogateescape"
        ))
    content_raw = raw[start:end]
    return (raw, content_raw, start, end, source_sha256), None


def _fragment_ref(symbol_id: str, start: int, end: int, fragment_sha256: str) -> str:
    return f"{symbol_id}#bytes={start}:{end}@sha256:{fragment_sha256}"


def _recovery_descriptor(
    root: Path,
    index: RepositoryIndex,
    symbol: Symbol,
    *,
    omission_reason: str,
    score: float,
    selection_path: Iterable[str],
) -> tuple[dict[str, object] | None, str | None]:
    verified, error = _verified_source_span(root, index, symbol)
    if verified is None:
        return None, error
    _raw, content_raw, start, end, source_sha256 = verified
    fragment_sha256 = hashlib.sha256(content_raw).hexdigest()
    return {
        "context_ref": _fragment_ref(symbol.symbol_id, start, end, fragment_sha256),
        "symbol_id": symbol.symbol_id,
        "path": symbol.path,
        "language": symbol.language,
        "kind": symbol.kind,
        "qualified_name": symbol.qualified_name,
        "line_start": symbol.line_start,
        "line_end": symbol.line_end,
        "start_byte": start,
        "end_byte": end,
        "estimated_tokens": _token_cost(
            content_raw.decode("utf-8", errors="surrogateescape")
        ),
        "source_sha256": source_sha256,
        "fragment_sha256": fragment_sha256,
        "parse_backend": symbol.parse_backend,
        "omission_reason": omission_reason,
        "score": round(score, 6),
        "selection_path": list(selection_path),
    }, None


def _read_verified_fragment(
    root: Path,
    index: RepositoryIndex,
    symbol: Symbol,
    remaining_tokens: int,
) -> tuple[dict[str, object] | None, str | None]:
    verified, error = _verified_source_span(root, index, symbol)
    if verified is None:
        return None, error
    raw, content_raw, start, end, source_sha256 = verified
    resolution = "full"
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
    fragment_sha256 = hashlib.sha256(content_raw).hexdigest()
    return {
        "context_ref": _fragment_ref(symbol.symbol_id, start, end, fragment_sha256),
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
        "fragment_sha256": fragment_sha256,
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


def _context_sha256(payload: Mapping[str, object]) -> str:
    candidate = copy.deepcopy(dict(payload))
    candidate.pop("generation", None)
    candidate.pop("command", None)
    receipt = candidate.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("context receipt must be an object")
    receipt.pop("context_sha256", None)
    canonical = json.dumps(
        candidate,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _seal_context(payload: dict[str, object]) -> dict[str, object]:
    receipt = payload.get("receipt")
    if not isinstance(receipt, dict):
        raise ValueError("context receipt must be an object")
    receipt["context_sha256"] = _context_sha256(payload)
    return payload


def _current_fragment(
    root: Path,
    index: RepositoryIndex,
    raw_fragment: Mapping[str, object],
) -> dict[str, object]:
    """Validate one selected fragment against the current indexed source."""

    symbol_id = raw_fragment.get("symbol_id")
    if not isinstance(symbol_id, str) or symbol_id not in index.symbols:
        raise ValueError("selected fragment references an unknown symbol")
    symbol = index.symbols[symbol_id]
    verified, error = _verified_source_span(root, index, symbol)
    if verified is None:
        raise ValueError(f"selected fragment source is not current: {error}")
    raw, full_raw, full_start, full_end, source_sha256 = verified
    try:
        start = int(raw_fragment["start_byte"])
        end = int(raw_fragment["end_byte"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("selected fragment has an invalid byte range") from None
    if not (full_start <= start < end <= full_end):
        raise ValueError("selected fragment falls outside its indexed symbol")
    content_raw = raw[start:end]
    content = content_raw.decode("utf-8", errors="surrogateescape")
    fragment_sha256 = hashlib.sha256(content_raw).hexdigest()
    resolution = raw_fragment.get("resolution")
    if resolution == "full":
        if start != full_start or end != full_end or content_raw != full_raw:
            raise ValueError("full fragment does not match its indexed symbol span")
    elif resolution == "signature":
        if not symbol.signature or content != symbol.signature:
            raise ValueError("signature fragment does not match its indexed signature")
    else:
        raise ValueError("selected fragment has an invalid resolution")
    expected_ref = _fragment_ref(symbol_id, start, end, fragment_sha256)
    required = {
        "context_ref": expected_ref,
        "path": symbol.path,
        "source_sha256": source_sha256,
        "fragment_sha256": fragment_sha256,
        "content": content,
        "estimated_tokens": _token_cost(content),
    }
    if any(raw_fragment.get(key) != value for key, value in required.items()):
        raise ValueError("selected fragment metadata does not match current source")
    return dict(raw_fragment)


def _current_descriptor(
    root: Path,
    index: RepositoryIndex,
    raw_descriptor: Mapping[str, object],
) -> dict[str, object]:
    """Validate a content-free recovery descriptor against current source."""

    symbol_id = raw_descriptor.get("symbol_id")
    omission_reason = raw_descriptor.get("omission_reason")
    selection_path = raw_descriptor.get("selection_path")
    score = raw_descriptor.get("score")
    if (
        not isinstance(symbol_id, str)
        or symbol_id not in index.symbols
        or not isinstance(omission_reason, str)
        or not omission_reason
        or not isinstance(selection_path, list)
        or not all(isinstance(item, str) for item in selection_path)
        or isinstance(score, bool)
    ):
        raise ValueError("recovery descriptor is malformed")
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        raise ValueError("recovery descriptor score is invalid") from None
    if not math.isfinite(numeric_score):
        raise ValueError("recovery descriptor score is invalid")
    expected, error = _recovery_descriptor(
        root,
        index,
        index.symbols[symbol_id],
        omission_reason=omission_reason,
        score=numeric_score,
        selection_path=selection_path,
    )
    if expected is None:
        raise ValueError(f"recovery descriptor source is not current: {error}")
    if dict(raw_descriptor) != expected:
        raise ValueError("recovery descriptor does not match current source")
    return expected


def apply_context_fault(
    root: Path,
    index: RepositoryIndex,
    payload: Mapping[str, object],
    context_ref: str,
    *,
    index_digest: str,
    token_budget: int | None = None,
) -> dict[str, object]:
    """Recover one omitted fragment and commit a new bounded working set.

    The supplied context receipt is treated as an immutable parent. Recovery
    succeeds only when both that commitment and every carried source reference
    still match the current repository index and workspace bytes.
    """

    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not isinstance(context_ref, str) or not context_ref.strip():
        raise ValueError("context_ref must not be empty")
    if not isinstance(payload, Mapping):
        raise ValueError("context payload must be an object")
    parent = copy.deepcopy(dict(payload))
    if not verify_context_commitment(parent):
        raise ValueError("context receipt commitment is invalid")
    if parent.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported context schema version")
    if parent.get("index_digest") != index_digest:
        raise ValueError("context index digest does not match the current snapshot")

    raw_fragments = parent.get("fragments")
    raw_descriptors = parent.get("recoverable_fragments")
    retrieval = parent.get("retrieval")
    receipt = parent.get("receipt")
    if (
        not isinstance(raw_fragments, list)
        or not all(isinstance(item, Mapping) for item in raw_fragments)
        or not isinstance(raw_descriptors, list)
        or not all(isinstance(item, Mapping) for item in raw_descriptors)
        or not isinstance(retrieval, dict)
        or not isinstance(receipt, dict)
    ):
        raise ValueError("context payload shape is invalid")

    fragments = [
        _current_fragment(root, index, item)
        for item in raw_fragments
    ]
    for fragment in fragments:
        if fragment.get("context_ref") == context_ref:
            return parent

    descriptors = [
        _current_descriptor(root, index, item)
        for item in raw_descriptors
    ]
    matches = [item for item in descriptors if item["context_ref"] == context_ref]
    if len(matches) != 1:
        raise ValueError("context_ref is not a unique recoverable fragment")
    target_descriptor = matches[0]
    target_symbol = index.symbols[str(target_descriptor["symbol_id"])]
    verified, error = _verified_source_span(root, index, target_symbol)
    if verified is None:
        raise ValueError(f"recovery source is not current: {error}")
    _raw, content_raw, start, end, source_sha256 = verified
    content = content_raw.decode("utf-8", errors="surrogateescape")
    target = {
        key: value
        for key, value in target_descriptor.items()
        if key != "omission_reason"
    }
    target.update({
        "start_byte": start,
        "end_byte": end,
        "resolution": "full",
        "content": content,
        "estimated_tokens": _token_cost(content),
        "source_sha256": source_sha256,
        "fragment_sha256": hashlib.sha256(content_raw).hexdigest(),
        "selection_path": [*target_descriptor["selection_path"], "context-fault"],
        "trust": "untrusted-source-bytes",
    })

    configured_budget = retrieval.get("token_budget", 2_000)
    selected_budget = configured_budget if token_budget is None else token_budget
    if isinstance(selected_budget, bool):
        raise ValueError("token_budget must be an integer")
    try:
        budget = max(128, min(int(selected_budget), 32_768))
    except (TypeError, ValueError):
        raise ValueError("token_budget must be an integer") from None
    target_cost = int(target["estimated_tokens"])
    if target_cost > budget:
        raise ValueError("recovered fragment exceeds the active token budget")

    # A full recovery supersedes any signature-only selection for the symbol.
    target_symbol_id = target["symbol_id"]
    retained = [
        fragment for fragment in fragments
        if fragment.get("symbol_id") != target_symbol_id
    ]
    evicted: list[dict[str, object]] = []
    total = target_cost + sum(int(item["estimated_tokens"]) for item in retained)
    for candidate in sorted(
        retained,
        key=lambda item: (
            float(item.get("score", 0.0)),
            str(item.get("context_ref", "")),
        ),
    ):
        if total <= budget:
            break
        retained.remove(candidate)
        evicted.append(candidate)
        total -= int(candidate["estimated_tokens"])
    fragments = [*retained, target]
    fragments.sort(key=lambda item: (
        -float(item.get("score", 0.0)),
        str(item.get("symbol_id", "")),
        str(item.get("context_ref", "")),
    ))

    next_descriptors: list[dict[str, object]] = []
    next_refs: set[str] = set()
    for fragment in evicted:
        symbol = index.symbols[str(fragment["symbol_id"])]
        descriptor, descriptor_error = _recovery_descriptor(
            root,
            index,
            symbol,
            omission_reason="context-fault-eviction",
            score=float(fragment.get("score", 0.0)),
            selection_path=[*fragment.get("selection_path", []), "context-fault-eviction"],
        )
        if descriptor is None:
            raise ValueError(f"evicted fragment is not recoverable: {descriptor_error}")
        ref = str(descriptor["context_ref"])
        if ref not in next_refs:
            next_refs.add(ref)
            next_descriptors.append(descriptor)
    for descriptor in descriptors:
        ref = str(descriptor["context_ref"])
        if ref == context_ref or ref in next_refs:
            continue
        if len(next_descriptors) >= 256:
            break
        next_refs.add(ref)
        next_descriptors.append(descriptor)

    selected = {str(fragment["symbol_id"]) for fragment in fragments}
    source_cache: dict[str, tuple[bytes | None, str]] = {}
    relations, relation_omissions = _selected_relations(
        root, index, selected, source_cache
    )
    unresolved_calls: list[dict[str, object]] = []
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
        if status == "verified":
            item = call.to_dict()
            item["evidence_status"] = status
            unresolved_calls.append(item)
        if len(unresolved_calls) >= 100:
            break

    omissions = receipt.get("omissions_by_reason")
    if not isinstance(omissions, dict):
        raise ValueError("context omission receipt is invalid")
    merged_omissions = {
        str(reason): int(count)
        for reason, count in omissions.items()
        if isinstance(reason, str) and isinstance(count, int) and count >= 0
    }
    recovered_reason = str(target_descriptor["omission_reason"])
    if merged_omissions.get(recovered_reason, 0) > 0:
        merged_omissions[recovered_reason] -= 1
        if merged_omissions[recovered_reason] == 0:
            del merged_omissions[recovered_reason]
    if evicted:
        merged_omissions["context-fault-eviction"] = (
            merged_omissions.get("context-fault-eviction", 0) + len(evicted)
        )
    for reason in tuple(merged_omissions):
        if reason.startswith("relation-"):
            del merged_omissions[reason]
    for reason, count in relation_omissions.items():
        merged_omissions[f"relation-{reason}"] = (
            count
        )

    parent_sha256 = str(receipt["context_sha256"])
    parent["fragments"] = fragments
    parent["recoverable_fragments"] = next_descriptors
    parent["relations"] = relations
    parent["unresolved_calls"] = unresolved_calls
    retrieval["token_budget"] = budget
    retrieval["estimated_tokens"] = total
    receipt["selected_fragment_count"] = len(fragments)
    receipt["recoverable_fragment_count"] = len(next_descriptors)
    receipt["selected_relation_count"] = len(relations)
    receipt["ambiguous_or_unresolved_calls"] = len(unresolved_calls)
    receipt["omissions_by_reason"] = dict(sorted(merged_omissions.items()))
    receipt["omitted_candidate_count"] = sum(merged_omissions.values())
    receipt["context_fault_count"] = int(receipt.get("context_fault_count", 0)) + 1
    parent["context_fault"] = {
        "status": "exact-source-recovered",
        "parent_context_sha256": parent_sha256,
        "recovered_ref": context_ref,
        "evicted_refs": sorted(str(item["context_ref"]) for item in evicted),
    }
    return _seal_context(parent)


def validate_context_sources(
    root: Path,
    index: RepositoryIndex,
    payload: Mapping[str, object],
    *,
    index_digest: str,
) -> None:
    """Fail unless a committed context still matches the indexed workspace."""

    root = root.expanduser().resolve(strict=True)
    candidate = copy.deepcopy(dict(payload))
    if not verify_context_commitment(candidate):
        raise ValueError("context receipt commitment is invalid")
    if candidate.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported context schema version")
    if candidate.get("index_digest") != index_digest:
        raise ValueError("context index digest does not match the current snapshot")
    fragments = candidate.get("fragments")
    descriptors = candidate.get("recoverable_fragments")
    if (
        not isinstance(fragments, list)
        or not all(isinstance(item, Mapping) for item in fragments)
        or not isinstance(descriptors, list)
        or not all(isinstance(item, Mapping) for item in descriptors)
    ):
        raise ValueError("context payload shape is invalid")
    for fragment in fragments:
        _current_fragment(root, index, fragment)
    for descriptor in descriptors:
        _current_descriptor(root, index, descriptor)


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
    proposal_scores: Iterable[Mapping[str, object]] = (),
    proposal_provider: str = "caller-supplied",
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
    provider = proposal_provider.strip()
    if not provider or len(provider) > 200:
        raise ValueError("proposal provider must contain 1 to 200 characters")
    proposal_map, accepted_proposals, proposal_omissions = _validated_proposals(
        index, proposal_scores
    )
    ranked, seed_ids = _rank_candidates(
        index,
        clean_query,
        max_hops=hops,
        max_candidates=max(fragment_limit * 8, 64),
        proposal_scores=proposal_map,
    )

    fragments: list[dict[str, object]] = []
    recoverable_fragments: list[dict[str, object]] = []
    recoverable_refs: set[str] = set()
    omissions: dict[str, int] = defaultdict(int)
    remaining = budget
    selected: set[str] = set()
    for score, symbol_id, reasons in ranked:
        symbol = index.symbols[symbol_id]
        if len(fragments) >= fragment_limit:
            omissions["fragment-limit"] += 1
            descriptor, descriptor_error = _recovery_descriptor(
                root,
                index,
                symbol,
                omission_reason="fragment-limit",
                score=score,
                selection_path=reasons,
            )
            if descriptor is not None:
                ref = str(descriptor["context_ref"])
                if ref not in recoverable_refs and len(recoverable_fragments) < 256:
                    recoverable_refs.add(ref)
                    recoverable_fragments.append(descriptor)
            elif descriptor_error:
                omissions[f"recovery-{descriptor_error}"] += 1
            continue
        fragment, omitted = _read_verified_fragment(
            root,
            index,
            symbol,
            remaining,
        )
        if fragment is None:
            omissions[omitted or "unknown"] += 1
            if omitted == "budget":
                descriptor, descriptor_error = _recovery_descriptor(
                    root,
                    index,
                    symbol,
                    omission_reason="budget",
                    score=score,
                    selection_path=reasons,
                )
                if descriptor is not None:
                    ref = str(descriptor["context_ref"])
                    if ref not in recoverable_refs and len(recoverable_fragments) < 256:
                        recoverable_refs.add(ref)
                        recoverable_fragments.append(descriptor)
                elif descriptor_error:
                    omissions[f"recovery-{descriptor_error}"] += 1
            continue
        fragment["score"] = round(score, 6)
        fragment["selection_path"] = list(reasons)
        if symbol_id in proposal_map:
            fragment["proposal_score"] = round(proposal_map[symbol_id], 8)
        fragments.append(fragment)
        selected.add(symbol_id)
        remaining -= int(fragment["estimated_tokens"])
        if fragment["resolution"] == "signature":
            omissions["signature-only"] += 1
            descriptor, descriptor_error = _recovery_descriptor(
                root,
                index,
                symbol,
                omission_reason="signature-only",
                score=score,
                selection_path=reasons,
            )
            if descriptor is not None and descriptor["context_ref"] != fragment["context_ref"]:
                ref = str(descriptor["context_ref"])
                if ref not in recoverable_refs and len(recoverable_fragments) < 256:
                    recoverable_refs.add(ref)
                    recoverable_fragments.append(descriptor)
            elif descriptor_error:
                omissions[f"recovery-{descriptor_error}"] += 1

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
            "sufficiency_scope": (
                "query-seed-selection-coverage-only-not-answer-sufficiency"
            ),
        },
        "fragments": fragments,
        "recoverable_fragments": recoverable_fragments,
        "relations": relations,
        "unresolved_calls": relevant_unresolved,
        "history": history,
        "proposal_overlay": {
            "provider": provider,
            "trust": "untrusted-ranking-proposal-verified-against-symbol-index",
            "accepted": accepted_proposals,
            "omissions_by_reason": proposal_omissions,
            "may_affect_ranking_only": True,
            "may_create_symbols_or_edges": False,
        },
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "selected_fragment_count": len(fragments),
            "recoverable_fragment_count": len(recoverable_fragments),
            "selected_relation_count": len(relations),
            "omitted_candidate_count": sum(omissions.values()),
            "omissions_by_reason": dict(sorted(omissions.items())),
            "ambiguous_or_unresolved_calls": len(relevant_unresolved),
            "remote_calls": 0,
            "history_requested": bool(include_history),
            "accepted_proposal_count": len(accepted_proposals),
            "omitted_proposal_count": sum(proposal_omissions.values()),
            "commitment_scope": "payload-excluding-generation-command-and-context-sha256",
        },
    }
    return _seal_context(payload)


def verify_context_commitment(payload: dict[str, object]) -> bool:
    """Verify the deterministic payload commitment without workspace access."""
    try:
        receipt = payload["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = receipt.get("context_sha256")
        return isinstance(expected, str) and _context_sha256(payload) == expected
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
