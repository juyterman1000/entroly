"""Thin adapter: SemanticEncoder -> SQLite-cached SemanticScorer.

Bridges the ``SemanticEncoder`` contract (neural_evidence_selector) to the
``SemanticScorer`` protocol consumed by ``rank_chunks``. Embeddings are
persisted in a per-model SQLite cache so repeat queries skip the encoder.

Returns ``None`` from ``default_scorer()`` when the ``[neural]`` extra is
absent or no local model path is configured, preserving the existing
lexical-only behavior.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import struct
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..neural_evidence_selector import SemanticEncoder
    from .models import DocumentChunk


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pack_embedding(embedding: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _unpack_embedding(data: bytes) -> list[float]:
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


class EmbeddingScorer:
    """SQLite-cached vector scorer bridging SemanticEncoder to SemanticScorer."""

    def __init__(
        self,
        encoder: SemanticEncoder,
        cache_path: str | Path,
    ) -> None:
        self._encoder = encoder
        self._model_prefix = encoder.fingerprint[:16]
        self._lock = threading.RLock()
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(path), timeout=30.0, check_same_thread=False
        )
        with self._lock:
            if str(path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS embeddings (
                    model_prefix TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    PRIMARY KEY (model_prefix, text_hash)
                )"""
            )

    def _get_cached(self, text_hash: str) -> list[float] | None:
        row = self._conn.execute(
            "SELECT embedding FROM embeddings"
            " WHERE model_prefix=? AND text_hash=?",
            (self._model_prefix, text_hash),
        ).fetchone()
        return _unpack_embedding(row[0]) if row else None

    def _put_cached(self, text_hash: str, embedding: Sequence[float]) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO embeddings"
            " (model_prefix, text_hash, embedding) VALUES (?, ?, ?)",
            (self._model_prefix, text_hash, _pack_embedding(embedding)),
        )
        self._conn.commit()

    def _encode_with_cache(self, texts: Sequence[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        to_encode: list[tuple[int, str]] = []

        with self._lock:
            for i, text in enumerate(texts):
                cached = self._get_cached(_text_hash(text))
                if cached is not None:
                    results[i] = cached
                else:
                    to_encode.append((i, text))

        if to_encode:
            fresh = self._encoder.encode([text for _, text in to_encode])
            with self._lock:
                for (i, text), embedding in zip(to_encode, fresh):
                    packed = _pack_embedding(embedding)
                    vec = _unpack_embedding(packed)
                    results[i] = vec
                    self._put_cached(_text_hash(text), vec)

        return [r for r in results if r is not None]  # type: ignore[misc]

    def score(
        self, query: str, chunks: Sequence[DocumentChunk]
    ) -> Mapping[str, float]:
        if not chunks:
            return {}
        try:
            texts = [query] + [chunk.text for chunk in chunks]
            embeddings = self._encode_with_cache(texts)
            if len(embeddings) != len(texts):
                return {}
            query_vec = embeddings[0]
            scores: dict[str, float] = {}
            for chunk, chunk_vec in zip(chunks, embeddings[1:]):
                scores[chunk.chunk_id] = _dot(query_vec, chunk_vec)
            return scores
        except Exception:
            return {}


_scorer_lock = threading.Lock()
_scorer_instance: EmbeddingScorer | None = None
_scorer_attempted = False


def default_scorer() -> EmbeddingScorer | None:
    """Return a cached EmbeddingScorer if a local model is configured, else None.

    Checks ``ENTROLY_SEMANTIC_MODEL_PATH`` for a local SentenceTransformer
    directory. Returns ``None`` instantly when the variable is unset or the
    ``[neural]`` extra is not installed.
    """
    global _scorer_instance, _scorer_attempted

    if _scorer_attempted:
        return _scorer_instance

    with _scorer_lock:
        if _scorer_attempted:
            return _scorer_instance
        _scorer_attempted = True

        model_path = os.environ.get("ENTROLY_SEMANTIC_MODEL_PATH")
        if not model_path:
            return None

        path = Path(model_path).expanduser().resolve()
        if not path.is_dir():
            return None

        try:
            from ..neural_evidence_selector import LocalTransformerEncoder
        except Exception:
            return None

        try:
            encoder = LocalTransformerEncoder(path)
        except (ValueError, RuntimeError):
            return None

        entroly_dir = Path(
            os.environ.get(
                "ENTROLY_DIR", os.path.join(os.getcwd(), ".entroly")
            )
        )
        cache_path = entroly_dir / "embedding_cache.sqlite3"

        try:
            _scorer_instance = EmbeddingScorer(encoder, cache_path)
        except Exception:
            return None

        return _scorer_instance
