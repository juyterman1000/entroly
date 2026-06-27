"""Hybrid retrieval primitives for Context Receipts."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Protocol

from .models import ContextIndex, DocumentChunk, RankedChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*", re.UNICODE)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "does",
    "have",
    "has",
    "from",
    "into",
    "which",
    "what",
    "where",
    "when",
    "shall",
    "will",
}


class SemanticScorer(Protocol):
    def score(self, query: str, chunks: Sequence[DocumentChunk]) -> Mapping[str, float]:
        """Return semantic/vector scores by chunk id."""


Reranker = Callable[[str, list[RankedChunk]], object]


def _finite_score(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return score if math.isfinite(score) else default


def _reason_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _rank_value(
    rank: RankedChunk | Mapping[str, object], name: str, default: object
) -> object:
    if isinstance(rank, Mapping):
        return rank.get(name, default)
    return getattr(rank, name, default)


def _sanitize_rank(rank: RankedChunk | Mapping[str, object]) -> RankedChunk:
    return RankedChunk(
        chunk_id=str(_rank_value(rank, "chunk_id", "")),
        lexical_score=round(
            _finite_score(_rank_value(rank, "lexical_score", 0.0)), 6
        ),
        semantic_score=round(
            _finite_score(_rank_value(rank, "semantic_score", 0.0)), 6
        ),
        rerank_score=round(_finite_score(_rank_value(rank, "rerank_score", 0.0)), 6),
        final_score=round(_finite_score(_rank_value(rank, "final_score", 0.0)), 6),
        reasons=_reason_list(_rank_value(rank, "reasons", [])),
    )


def _sanitize_reranked(
    reranked: Iterable[RankedChunk | Mapping[str, object]] | object,
    original: Sequence[RankedChunk],
    valid_chunk_ids: set[str],
) -> list[RankedChunk]:
    """Normalize custom reranker output without losing receipt audit coverage."""
    normalized: list[RankedChunk] = []
    seen: set[str] = set()
    if isinstance(reranked, Mapping):
        candidates = (reranked,) if "chunk_id" in reranked else reranked.values()
    else:
        candidates = reranked if isinstance(reranked, Iterable) else ()
    for rank in candidates:
        sanitized = _sanitize_rank(rank)
        if sanitized.chunk_id not in valid_chunk_ids or sanitized.chunk_id in seen:
            continue
        normalized.append(sanitized)
        seen.add(sanitized.chunk_id)

    for rank in original:
        if rank.chunk_id not in seen:
            normalized.append(_sanitize_rank(rank))
            seen.add(rank.chunk_id)
    return normalized


def tokenize(text: str) -> list[str]:
    text = text.replace("-", " ")
    return [
        t.lower()
        for t in TOKEN_RE.findall(text)
        if len(t) > 1 and t.lower() not in STOPWORDS
    ]


def _score_reason(
    chunk: DocumentChunk, query_terms: list[str], matched_terms: list[str]
) -> list[str]:
    reasons: list[str] = []
    if matched_terms:
        reasons.append("lexical match: " + ", ".join(matched_terms[:8]))
    heading = (chunk.section_heading or "").lower()
    heading_hits = sorted({t for t in query_terms if t in heading})
    if heading_hits:
        reasons.append("section heading match: " + ", ".join(heading_hits[:6]))
    source_lower = chunk.source_path.lower()
    source_hits = sorted({t for t in query_terms if t in source_lower})
    if source_hits:
        reasons.append("file name/path match: " + ", ".join(source_hits[:6]))
    lower = chunk.text.lower()
    if any(
        phrase in lower
        for phrase in ("as defined in", "subject to", "pursuant to", "see section")
    ):
        reasons.append("contains explicit dependency/reference language")
    if not reasons:
        reasons.append("low lexical overlap; retained as lower-ranked candidate")
    return reasons


def rank_chunks(
    index: ContextIndex,
    query: str,
    *,
    semantic_scorer: SemanticScorer | None = None,
    reranker: Reranker | None = None,
) -> list[RankedChunk]:
    """Rank chunks with BM25-style lexical scoring plus optional semantic/rerank hooks."""

    query_terms = tokenize(query)
    if not query_terms:
        query_terms = [t.lower() for t in TOKEN_RE.findall(query)]

    docs_tokens = {chunk.chunk_id: tokenize(chunk.text) for chunk in index.chunks}
    df: dict[str, int] = defaultdict(int)
    for terms in docs_tokens.values():
        for term in set(terms):
            df[term] += 1
    doc_count = max(1, len(index.chunks))
    avg_len = (
        sum(len(t) for t in docs_tokens.values()) / doc_count if docs_tokens else 1.0
    )

    raw_semantic_scores = (
        semantic_scorer.score(query, index.chunks) if semantic_scorer else {}
    )
    semantic_scores = (
        raw_semantic_scores if isinstance(raw_semantic_scores, Mapping) else {}
    )
    ranked: list[RankedChunk] = []
    for chunk in index.chunks:
        terms = docs_tokens[chunk.chunk_id]
        tf = Counter(terms)
        doc_len = max(1, len(terms))
        lexical = 0.0
        matched: list[str] = []
        for term in query_terms:
            freq = tf.get(term, 0)
            in_heading = term in (chunk.section_heading or "").lower()
            in_path = term in chunk.source_path.lower()
            if freq or in_heading or in_path:
                matched.append(term)
            idf = math.log(
                (doc_count - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0
            )
            if freq:
                lexical += idf * (
                    (freq * 2.2)
                    / (freq + 1.2 * (1 - 0.75 + 0.75 * doc_len / max(1.0, avg_len)))
                )
            if in_heading:
                lexical += idf * 2.5
            if in_path:
                lexical += idf * 1.5
        coverage = len(set(matched)) / max(1, len(set(query_terms)))
        lexical *= 1.0 + coverage
        semantic = _finite_score(semantic_scores.get(chunk.chunk_id, 0.0))
        final = lexical + 0.25 * semantic
        ranked.append(
            RankedChunk(
                chunk_id=chunk.chunk_id,
                lexical_score=round(lexical, 6),
                semantic_score=round(semantic, 6),
                rerank_score=0.0,
                final_score=round(final, 6),
                reasons=_score_reason(chunk, query_terms, sorted(set(matched))),
            )
        )

    ranked.sort(key=lambda item: (-item.final_score, item.chunk_id))
    if reranker is not None:
        ranked = _sanitize_reranked(
            reranker(query, ranked),
            ranked,
            set(docs_tokens),
        )
    else:
        ranked = [_sanitize_rank(rank) for rank in ranked]
    return ranked
