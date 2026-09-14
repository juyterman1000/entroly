"""Tests for EmbeddingScorer adapter and prefix/substring graph resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from entroly.context_receipts.embedding_scorer import EmbeddingScorer
from entroly.context_receipts.ingest import ingest_documents
from entroly.context_receipts.retrieval import rank_chunks
from entroly.repository_intelligence import build_repository_index
from entroly.repository_intelligence.graph_query import build_verified_graph_query


# ---------------------------------------------------------------------------
# Mock encoder for testing without sentence-transformers
# ---------------------------------------------------------------------------

@dataclass
class _MockEncoder:
    _dim: int = 4

    @property
    def model_id(self) -> str:
        return "mock-encoder"

    @property
    def fingerprint(self) -> str:
        return "a" * 64

    def encode(self, texts):
        import math

        vecs = []
        for text in texts:
            raw = [float(ord(c) % 7) for c in (text[:self._dim] or "a")]
            raw.extend([0.0] * (self._dim - len(raw)))
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vecs.append([x / norm for x in raw])
        return vecs


# ---------------------------------------------------------------------------
# EmbeddingScorer tests
# ---------------------------------------------------------------------------

def test_embedding_scorer_returns_scores_for_all_chunks(tmp_path: Path) -> None:
    encoder = _MockEncoder()
    scorer = EmbeddingScorer(encoder, tmp_path / "cache.sqlite3")

    index = ingest_documents(
        [("doc.md", "alpha bravo charlie delta echo foxtrot golf hotel")],
        chunk_tokens=8,
        overlap_tokens=0,
    )
    scores = scorer.score("alpha bravo", index.chunks)

    assert isinstance(scores, dict)
    assert set(scores.keys()) == {chunk.chunk_id for chunk in index.chunks}
    assert all(isinstance(v, float) for v in scores.values())


def test_embedding_scorer_caches_embeddings_in_sqlite(tmp_path: Path) -> None:
    encoder = _MockEncoder()
    db_path = tmp_path / "cache.sqlite3"
    scorer = EmbeddingScorer(encoder, db_path)

    index = ingest_documents(
        [("doc.md", "alpha bravo charlie delta echo foxtrot golf hotel")],
        chunk_tokens=8,
        overlap_tokens=0,
    )
    first = scorer.score("alpha", index.chunks)
    second = scorer.score("alpha", index.chunks)

    assert first == second
    assert db_path.exists()


def test_embedding_scorer_integrates_with_rank_chunks(tmp_path: Path) -> None:
    encoder = _MockEncoder()
    scorer = EmbeddingScorer(encoder, tmp_path / "cache.sqlite3")

    index = ingest_documents(
        [("doc.md", "alpha bravo charlie delta echo foxtrot golf hotel")],
        chunk_tokens=8,
        overlap_tokens=0,
    )
    ranked_without = rank_chunks(index, "alpha")
    ranked_with = rank_chunks(index, "alpha", semantic_scorer=scorer)

    assert len(ranked_without) == len(ranked_with)
    has_nonzero = any(r.semantic_score != 0.0 for r in ranked_with)
    assert has_nonzero


def test_embedding_scorer_returns_empty_on_no_chunks(tmp_path: Path) -> None:
    encoder = _MockEncoder()
    scorer = EmbeddingScorer(encoder, tmp_path / "cache.sqlite3")
    assert scorer.score("query", []) == {}


# ---------------------------------------------------------------------------
# graph_query._resolve prefix/substring tests
# ---------------------------------------------------------------------------

def _write(root: Path, path: str, text: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_resolve_prefix_match_finds_unique_symbol(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "service.py",
        "def execute_query(value):\n    return value + 1\n",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_graph_query(
        tmp_path,
        index,
        "execute_q",
        index_digest="sha256:test",
        operation="neighbors",
    )
    assert payload["resolution"] == "resolved"
    assert any("execute_query" in str(n) for n in payload["candidates"])


def test_resolve_substring_match_finds_unique_symbol(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "service.py",
        "def handle_request(value):\n    return value + 1\n",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_graph_query(
        tmp_path,
        index,
        "request",
        index_digest="sha256:test",
        operation="neighbors",
    )
    assert payload["resolution"] in ("resolved", "ambiguous")
    assert any("handle_request" in str(c) for c in payload["candidates"])


def test_resolve_exact_match_takes_priority_over_prefix(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "funcs.py",
        "def run():\n    return 1\n\n"
        "def run_fast():\n    return 2\n",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_graph_query(
        tmp_path,
        index,
        "run",
        index_digest="sha256:test",
        operation="neighbors",
    )
    assert payload["resolution"] == "resolved"
    assert any(c.endswith("::run::function") for c in payload["candidates"])
    assert not any("run_fast" in c for c in payload["candidates"])


def test_resolve_prefix_returns_ambiguous_for_multiple_matches(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "workers.py",
        "def process_alpha():\n    return 1\n\n"
        "def process_beta():\n    return 2\n",
    )
    index = build_repository_index(tmp_path)
    payload = build_verified_graph_query(
        tmp_path,
        index,
        "process_",
        index_digest="sha256:test",
        operation="neighbors",
    )
    assert payload["resolution"] == "ambiguous"
    assert len(payload["candidates"]) == 2
