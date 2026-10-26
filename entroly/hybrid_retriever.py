"""Entroly Hybrid Retriever — multi-signal memory recall.

Fuses BM25 keyword search, entity graph traversal, and optional neural embedding
scores via Reciprocal Rank Fusion (RRF).

Multi-signal retrieval fans queries across vector + BM25 + entity matching
simultaneously, adding receipt-backed audit trails for the fusion process.

Architecture
------------
    query → ┌─ BM25 keyword ranking
             ├─ Entity graph traversal (from memory_graph.py)
             └─ Neural embedding ranking (optional, from neural_evidence_selector.py)
                          │
                          ▼
              Reciprocal Rank Fusion
                          │
                          ▼
              Budget-aware selection + receipt

References
----------
- Cormack et al. (2009): Reciprocal Rank Fusion outperforms individual
  retrieval methods and learned fusion with minimal tuning
- HippoRAG 2 (2025): graph-augmented associative retrieval
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, Sequence

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


# ── Protocols ─────────────────────────────────────────────────────────────

class MemoryLike(Protocol):
    """Minimal interface for the MemoryOS entries we need."""
    @property
    def _entries(self) -> dict: ...
    @property
    def _lock(self) -> Any: ...


class GraphLike(Protocol):
    """Minimal interface for MemoryGraph."""
    def traverse(
        self,
        seed_entities: list[str],
        *,
        max_hops: int = 2,
        max_results: int = 50,
    ) -> Any: ...


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class ScoredMemory:
    """A memory entry scored by the hybrid retriever."""
    memory_id: str
    content: str
    combined_score: float
    bm25_rank: int | None = None
    entity_rank: int | None = None
    neural_rank: int | None = None
    signals_present: int = 0
    fusion_detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HybridRetrievalReceipt:
    """Audit trail for hybrid retrieval."""
    query: str
    bm25_candidates: int
    entity_candidates: int
    neural_candidates: int
    fused_results: int
    rrf_k: int
    signals_used: list[str]
    top_results: list[dict[str, Any]]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion across multiple ranking lists.

    For each item, the RRF score is:
        score(d) = Σ 1 / (k + rank_i(d))

    where rank_i is the 1-indexed rank in the i-th ranking list.
    Items not present in a list are assigned rank = infinity (contribute 0).

    k=60 is the standard from the original paper.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_0, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank_0 + 1)
    return scores


# ── Hybrid Retriever ──────────────────────────────────────────────────────

class HybridRetriever:
    """Multi-signal memory retrieval via Reciprocal Rank Fusion.

    Usage::

        from entroly.memory import MemoryOS
        from entroly.memory_graph import MemoryGraph
        from entroly.hybrid_retriever import HybridRetriever

        mem = MemoryOS()
        graph = MemoryGraph()
        retriever = HybridRetriever(mem, graph=graph)

        results = retriever.recall("What does Alice work on currently?", top_k=10)
        for scored in results:
            print(f"{scored.memory_id}: {scored.combined_score:.4f}")
    """

    def __init__(
        self,
        memory_os: Any,  # MemoryOS — Any to avoid circular import
        *,
        graph: Any | None = None,  # MemoryGraph
        neural_encoder: Any | None = None,  # Optional embedding model
        rrf_k: int = 60,
    ):
        self._mem = memory_os
        self._graph = graph
        self._encoder = neural_encoder
        self.rrf_k = rrf_k

    def recall(
        self,
        query: str,
        *,
        agent_id: str = "default",
        top_k: int = 20,
        include_shared: bool = True,
        graph_hops: int = 2,
    ) -> list[ScoredMemory]:
        """Multi-signal recall with RRF fusion.

        Returns top_k memories ranked by fused BM25 + entity + neural score.
        """
        # Collect candidate pool
        candidates = self._get_candidate_pool(agent_id, include_shared)
        if not candidates:
            return []

        rankings: list[list[str]] = []
        signal_names: list[str] = []

        # Signal 1: BM25 keyword ranking
        bm25_ranking = self._bm25_rank(query, candidates)
        rankings.append(bm25_ranking)
        signal_names.append("bm25")

        # Signal 2: Entity graph traversal
        entity_ranking: list[str] = []
        if self._graph is not None:
            entity_ranking = self._entity_rank(query, candidates, graph_hops)
            if entity_ranking:
                rankings.append(entity_ranking)
                signal_names.append("entity_graph")

        # Signal 3: Neural embedding ranking (optional)
        neural_ranking: list[str] = []
        if self._encoder is not None:
            neural_ranking = self._neural_rank(query, candidates)
            if neural_ranking:
                rankings.append(neural_ranking)
                signal_names.append("neural")

        # Fuse via RRF
        fused_scores = _reciprocal_rank_fusion(rankings, k=self.rrf_k)

        # Build results
        bm25_rank_map = {mid: i for i, mid in enumerate(bm25_ranking)}
        entity_rank_map = {mid: i for i, mid in enumerate(entity_ranking)}
        neural_rank_map = {mid: i for i, mid in enumerate(neural_ranking)}

        results: list[ScoredMemory] = []
        for mem_id, score in sorted(fused_scores.items(), key=lambda x: -x[1]):
            if mem_id not in candidates:
                continue
            entry = candidates[mem_id]
            signals = sum([
                mem_id in bm25_rank_map,
                mem_id in entity_rank_map,
                mem_id in neural_rank_map,
            ])
            results.append(ScoredMemory(
                memory_id=mem_id,
                content=entry.content,
                combined_score=round(score, 6),
                bm25_rank=bm25_rank_map.get(mem_id),
                entity_rank=entity_rank_map.get(mem_id),
                neural_rank=neural_rank_map.get(mem_id),
                signals_present=signals,
                fusion_detail={
                    "bm25_contribution": round(1.0 / (self.rrf_k + bm25_rank_map[mem_id] + 1), 6) if mem_id in bm25_rank_map else 0.0,
                    "entity_contribution": round(1.0 / (self.rrf_k + entity_rank_map[mem_id] + 1), 6) if mem_id in entity_rank_map else 0.0,
                    "neural_contribution": round(1.0 / (self.rrf_k + neural_rank_map[mem_id] + 1), 6) if mem_id in neural_rank_map else 0.0,
                },
            ))
            if len(results) >= top_k:
                break

        return results

    def recall_with_receipt(
        self,
        query: str,
        *,
        agent_id: str = "default",
        top_k: int = 20,
        include_shared: bool = True,
        graph_hops: int = 2,
    ) -> tuple[list[ScoredMemory], HybridRetrievalReceipt]:
        """Recall with a full audit receipt."""
        results = self.recall(
            query,
            agent_id=agent_id,
            top_k=top_k,
            include_shared=include_shared,
            graph_hops=graph_hops,
        )

        signals = ["bm25"]
        bm25_count = len(self._get_candidate_pool(agent_id, include_shared))
        entity_count = 0
        neural_count = 0

        if self._graph is not None:
            signals.append("entity_graph")
            entity_count = sum(1 for r in results if r.entity_rank is not None)
        if self._encoder is not None:
            signals.append("neural")
            neural_count = sum(1 for r in results if r.neural_rank is not None)

        receipt = HybridRetrievalReceipt(
            query=query,
            bm25_candidates=bm25_count,
            entity_candidates=entity_count,
            neural_candidates=neural_count,
            fused_results=len(results),
            rrf_k=self.rrf_k,
            signals_used=signals,
            top_results=[r.to_dict() for r in results[:5]],
        )

        return results, receipt

    # ── Signal implementations ────────────────────────────────────────

    def _get_candidate_pool(self, agent_id: str, include_shared: bool) -> dict:
        """Get all eligible memory entries."""
        candidates = {}
        try:
            with self._mem._lock:
                for mem_id, entry in self._mem._entries.items():
                    if entry.agent_id == agent_id or (include_shared and entry.agent_id == "shared"):
                        candidates[mem_id] = entry
        except Exception:
            pass
        return candidates

    def _bm25_rank(self, query: str, candidates: dict) -> list[str]:
        """BM25 ranking of candidates."""
        query_terms = _tokens(query)
        if not query_terms or not candidates:
            return list(candidates.keys())

        k1, b = 1.2, 0.75
        N = len(candidates)
        entries = list(candidates.values())
        avg_dl = sum(len(e.content.split()) for e in entries) / max(1, N)

        # Document frequency
        df: dict[str, int] = {}
        for entry in entries:
            doc_terms = _tokens(entry.content)
            for t in query_terms & doc_terms:
                df[t] = df.get(t, 0) + 1

        # Score each candidate
        scored: list[tuple[float, str]] = []
        for mem_id, entry in candidates.items():
            doc_tokens = entry.content.lower().split()
            dl = len(doc_tokens)
            tf: dict[str, int] = {}
            for t in doc_tokens:
                tl = t.lower()
                if tl in query_terms:
                    tf[tl] = tf.get(tl, 0) + 1

            score = 0.0
            for term, freq in tf.items():
                n = df.get(term, 0)
                idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
                tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / max(1, avg_dl)))
                score += idf * tf_norm
            scored.append((score, mem_id))

        scored.sort(key=lambda x: -x[0])
        return [mem_id for _, mem_id in scored]

    def _entity_rank(self, query: str, candidates: dict, max_hops: int) -> list[str]:
        """Rank candidates by entity graph proximity."""
        if self._graph is None:
            return []

        # Extract entities from query
        from .memory import _extract_entities
        query_entities = _extract_entities(query)
        if not query_entities:
            # Fall back to capitalized words in query
            query_entities = [w for w in query.split() if w and w[0].isupper() and len(w) > 2]

        if not query_entities:
            return []

        result = self._graph.traverse(query_entities, max_hops=max_hops)
        # Filter to candidates and preserve discovery order
        ranked = [mid for mid in result.memory_ids if mid in candidates]
        return ranked

    def _neural_rank(self, query: str, candidates: dict) -> list[str]:
        """Rank candidates by neural embedding similarity."""
        if self._encoder is None:
            return []

        try:
            texts = [(mid, entry.content) for mid, entry in candidates.items()]
            if not texts:
                return []

            query_emb = self._encoder.encode([query])[0]
            doc_embs = self._encoder.encode([t[1] for t in texts])

            import numpy as np
            similarities = np.dot(doc_embs, query_emb) / (
                np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
            )
            ranked_indices = np.argsort(-similarities)
            return [texts[i][0] for i in ranked_indices]
        except Exception as exc:
            logger.warning("Neural ranking failed: %s", exc)
            return []


__all__ = [
    "HybridRetriever",
    "ScoredMemory",
    "HybridRetrievalReceipt",
]
