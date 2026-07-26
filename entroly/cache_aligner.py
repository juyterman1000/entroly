"""
Entroly Cache Aligner — Provider KV Cache Optimization
========================================================

Stabilizes message prefixes so LLM provider KV caches actually work.

Provider prompt caches match exact prefixes.  Reusing an older context merely
because its token set is similar can silently hide changed evidence.  This
aligner therefore reuses a block only when its canonical bytes are identical.
Provider-reported usage remains the source of truth for cache savings.

Thread-safe. Per-client tracking with LRU eviction.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

# Live proxy integration point for Evidence-Locked Compression.
#
# proxy.py imports proxy_transform first and CacheAligner immediately after it.
# That makes this the smallest safe hook point: proxy_transform is already fully
# initialized, while the giant proxy.py request handler does not need to be
# rewritten. The installer is feature-flagged and returns immediately unless
# ENTROLY_COMPRESSION_PROXY_MODE=elc, so normal imports remain side-effect free.
try:  # pragma: no cover - behavior covered by integration tests
    from .compression_proxy_live import install_live_compression_proxy as _install_elc_proxy

    _install_elc_proxy()
except Exception:
    pass


class CacheAligner:
    """Stabilize context prefixes for LLM provider KV cache optimization.

    Tracks per-client context injections and returns the previous object only
    for an exact SHA-256 match.  ``similarity_threshold`` remains accepted for
    API compatibility, but approximate reuse is intentionally disabled:
    correctness and fresh evidence take precedence over a speculative cache hit.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        max_clients: int = 100,
    ):
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between zero and one")
        if max_clients < 1:
            raise ValueError("max_clients must be at least one")
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
        if not client_key:
            raise ValueError("client_key is required")
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()

        with self._lock:
            prev = self._cache.get(client_key)

            if prev is not None and prev["digest"] == digest:
                # Exact hit — reuse the existing string object byte-for-byte.
                self._cache.move_to_end(client_key)
                self._hits += 1
                return prev["context"], True

            # Cache miss — store new context
            self._cache[client_key] = {
                "context": context,
                "digest": digest,
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
                "match_policy": "exact_sha256",
            }
