"""
Entroly Cache Aligner — Provider KV Cache Optimization
========================================================

Stabilizes message prefixes so LLM provider KV caches actually work.

Problem: Anthropic offers a 90% read discount on cached prefixes, and
OpenAI caches repeated prefixes automatically. But if Entroly injects
different context on every request, the prefix changes every time and
the cache never hits.

Solution: CacheAligner hashes the injected context and stabilizes it
when the content hasn't materially changed. If the new context is
>90% similar to the previous injection for the same client, we reuse
the previous context verbatim — preserving the provider's cache prefix.

This is free money: same quality, 90% cheaper on cache-hit turns.

Thread-safe. Per-client tracking with LRU eviction.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any


class CacheAligner:
    """Stabilize context prefixes for LLM provider KV cache optimization.

    Tracks per-client context injections. When a new injection is
    sufficiently similar to the previous one (>similarity_threshold),
    the previous injection is reused verbatim to preserve cache hits.

    Similarity is measured by token-level Jaccard coefficient:
        |A ∩ B| / |A ∪ B|

    This is cheaper than computing embeddings and perfectly adequate
    for detecting near-identical context blocks.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        max_clients: int = 100,
    ):
        self._threshold = similarity_threshold
        self._max_clients = max_clients
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def align(self, client_key: str, context: str) -> tuple[str, bool]:
        """Align context for cache stability.

        Args:
            client_key: Client identifier (hashed API key)
            context: The new context block to inject

        Returns:
            (aligned_context, cache_hit): The context to use and whether
            the previous cached version was reused.
        """
        context_tokens = set(context.split())

        with self._lock:
            prev = self._cache.get(client_key)

            if prev is not None:
                prev_tokens = prev["tokens"]

                # Jaccard similarity
                intersection = len(context_tokens & prev_tokens)
                union = len(context_tokens | prev_tokens)
                similarity = intersection / max(union, 1)

                if similarity >= self._threshold:
                    # Cache hit — reuse previous context verbatim
                    self._cache.move_to_end(client_key)
                    self._hits += 1
                    return prev["context"], True

            # Cache miss — store new context
            self._cache[client_key] = {
                "context": context,
                "tokens": context_tokens,
            }
            self._cache.move_to_end(client_key)
            self._misses += 1

            # LRU eviction
            while len(self._cache) > self._max_clients:
                self._cache.popitem(last=False)

            return context, False

    def invalidate(self, client_key: str) -> None:
        """Force-invalidate a client's cached context."""
        with self._lock:
            self._cache.pop(client_key, None)

    def stats(self) -> dict[str, Any]:
        """Return cache alignment statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1), 4),
                "active_clients": len(self._cache),
                "estimated_savings_pct": round(self._hits * 90 / max(total, 1), 1),
            }
