"""
Entroly CCR — Compressed Context Retrieval
============================================

Lossless compression with full reversibility. When Entroly compresses a
fragment to a skeleton or reference, the original content is stored in
the CCR store. The LLM can call `entroly_retrieve` to get the full
original back when it needs more detail.

This eliminates the "silent truncation" problem architecturally:
nothing is ever permanently lost.

Thread-safe. Memory-efficient (LRU eviction at configurable capacity).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any


class CompressedContextStore:
    """Lossless compressed context store with LRU eviction.

    When fragments are compressed (skeleton/reference), their originals
    are stored here. The LLM can retrieve them on demand via the
    `/retrieve` endpoint or `entroly_retrieve` MCP tool.

    Architecture:
        compress_and_store(source, original, compressed)
            → stores original, returns compressed
        retrieve(source)
            → returns original content (full resolution)
        list_available()
            → returns all stored source keys with metadata

    Memory bound: max_entries controls LRU eviction (default 500).
    At ~10KB avg per fragment, 500 entries ≈ 5MB — negligible.
    """

    def __init__(self, max_entries: int = 500):
        self._max_entries = max_entries
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._total_stored = 0
        self._total_retrieved = 0

    def store(
        self,
        source: str,
        original_content: str,
        compressed_content: str,
        resolution: str = "skeleton",
        original_tokens: int = 0,
        compressed_tokens: int = 0,
    ) -> None:
        """Store the original content for a compressed fragment.

        Args:
            source: Fragment source identifier (e.g., "file:src/auth.py")
            original_content: Full, uncompressed content
            compressed_content: What was actually sent to the LLM
            resolution: Compression level applied ("skeleton", "reference", "belief")
            original_tokens: Token count of original
            compressed_tokens: Token count of compressed version
        """
        with self._lock:
            # Move to end if exists (LRU refresh)
            if source in self._store:
                self._store.move_to_end(source)
            self._store[source] = {
                "original": original_content,
                "compressed": compressed_content,
                "resolution": resolution,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
            }
            self._total_stored += 1

            # LRU eviction
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def retrieve(self, source: str) -> dict[str, Any] | None:
        """Retrieve the original content for a compressed fragment.

        Returns dict with 'original', 'compressed', 'resolution', etc.
        Returns None if source not found in store.
        """
        with self._lock:
            entry = self._store.get(source)
            if entry is not None:
                self._store.move_to_end(source)  # LRU refresh
                self._total_retrieved += 1
                return dict(entry)
            return None

    def list_available(self) -> list[dict[str, Any]]:
        """List all stored fragments with metadata (no content).

        Returns list of dicts with source, resolution, token counts.
        Used by the LLM to decide WHICH fragments to retrieve.
        """
        with self._lock:
            return [
                {
                    "source": source,
                    "resolution": entry["resolution"],
                    "original_tokens": entry["original_tokens"],
                    "compressed_tokens": entry["compressed_tokens"],
                    "tokens_recoverable": entry["original_tokens"] - entry["compressed_tokens"],
                }
                for source, entry in self._store.items()
            ]

    def clear(self) -> None:
        """Clear all stored originals."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        with self._lock:
            total_original = sum(e["original_tokens"] for e in self._store.values())
            total_compressed = sum(e["compressed_tokens"] for e in self._store.values())
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "total_stored": self._total_stored,
                "total_retrieved": self._total_retrieved,
                "tokens_in_store": total_original,
                "tokens_compressed": total_compressed,
                "tokens_recoverable": total_original - total_compressed,
            }


# Module-level singleton (shared across proxy and MCP)
_global_store: CompressedContextStore | None = None
_store_lock = threading.Lock()


def get_ccr_store(max_entries: int = 500) -> CompressedContextStore:
    """Get or create the global CCR store singleton."""
    global _global_store
    with _store_lock:
        if _global_store is None:
            _global_store = CompressedContextStore(max_entries=max_entries)
        return _global_store
