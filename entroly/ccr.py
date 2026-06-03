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
import re
import threading
from collections import OrderedDict
from typing import Any


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _retrieval_handle(source: str, original_content: str) -> str:
    digest = hashlib.sha256(
        f"{source}\0{original_content}".encode("utf-8")
    ).hexdigest()
    return f"ccr:{digest[:24]}"


_RECOVERY_TERM_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_RECOVERY_GAP = "\n\n[... exact excerpt gap; retrieve full source by handle ...]\n\n"


def slice_recovery_content(
    content: str,
    query: str,
    token_budget: int,
) -> tuple[str, bool]:
    """Return a bounded provenance-preserving slice of an oversized source.

    Half the character budget preserves the source lead, a strong prior for prose
    and declarations. The other half selects exact query-local windows, a
    strong prior for code, logs, and long documents. No generated summary is
    introduced: emitted excerpts are exact spans and omitted text is marked.
    """
    if not content or token_budget < 32:
        return "", False
    max_chars = token_budget * 4
    if len(content) <= max_chars:
        return content, False

    payload_chars = max_chars - len(_RECOVERY_GAP)
    if payload_chars <= 0:
        return "", False
    lead_chars = payload_chars // 2
    local_chars = payload_chars - lead_chars
    lead = content[:lead_chars]

    query_terms = {
        term.lower() for term in _RECOVERY_TERM_RE.findall(query)
    }
    lowered = content.lower()
    latest_start = max(lead_chars, len(content) - local_chars)
    starts = {lead_chars}
    for term in query_terms:
        for match in re.finditer(re.escape(term), lowered[lead_chars:]):
            centered = lead_chars + match.start() - local_chars // 2
            starts.add(min(latest_start, max(lead_chars, centered)))

    def rank(start: int) -> tuple[int, int]:
        window = lowered[start:start + local_chars]
        return (
            sum(window.count(term) for term in query_terms),
            -start,
        )

    local_start = max(starts, key=rank)
    local = content[local_start:local_start + local_chars]
    return f"{lead}{_RECOVERY_GAP}{local}", True


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
        self._latest_by_source: dict[str, str] = {}
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
        fragment_id: str = "",
        relevance: float = 0.0,
        entropy_score: float = 0.0,
    ) -> str:
        """Store the original content for a compressed fragment.

        Args:
            source: Fragment source identifier (e.g., "file:src/auth.py")
            original_content: Full, uncompressed content
            compressed_content: What was actually sent to the LLM
            resolution: Compression level applied ("skeleton", "reference", "belief")
            original_tokens: Token count of original
            compressed_tokens: Token count of compressed version

        Returns:
            A content-addressed retrieval handle. The source key always points
            to the newest stored version, while older handles remain valid
            until LRU eviction.
        """
        handle = _retrieval_handle(source, original_content)
        with self._lock:
            # Move to end if exists (LRU refresh)
            if handle in self._store:
                self._store.move_to_end(handle)
            self._store[handle] = {
                "source": source,
                "retrieval_handle": handle,
                "original": original_content,
                "compressed": compressed_content,
                "resolution": resolution,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "content_sha256": _content_hash(original_content),
                "compressed_sha256": _content_hash(compressed_content),
                "fragment_id": fragment_id,
                "relevance": relevance,
                "entropy_score": entropy_score,
            }
            self._latest_by_source[source] = handle
            self._total_stored += 1

            # LRU eviction
            while len(self._store) > self._max_entries:
                evicted_handle, evicted = self._store.popitem(last=False)
                source_key = evicted["source"]
                if self._latest_by_source.get(source_key) == evicted_handle:
                    del self._latest_by_source[source_key]
        return handle

    def retrieve(self, source_or_handle: str) -> dict[str, Any] | None:
        """Retrieve the original content for a compressed fragment.

        Returns dict with 'original', 'compressed', 'resolution', etc.
        Accepts either a source key or an exact content-addressed handle.
        Returns None if no matching entry is stored.
        """
        with self._lock:
            handle = self._latest_by_source.get(source_or_handle, source_or_handle)
            entry = self._store.get(handle)
            if entry is not None:
                self._store.move_to_end(handle)  # LRU refresh
                self._total_retrieved += 1
                return dict(entry)
            return None

    def retrieve_or_materialize(
        self,
        source_or_handle: str,
        fragment_lookup: Any | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve an entry, lazily materializing an ingested source if needed.

        Exact handles never fall back to source lookup: a missing handle means
        that exact historical version was evicted. Source keys may be resolved
        from the engine's live fragment store, which keeps hierarchical
        overview reads recoverable without eagerly copying the whole corpus.
        """
        entry = self.retrieve(source_or_handle)
        if entry is not None or fragment_lookup is None:
            return entry
        if source_or_handle.startswith("ccr:"):
            return None

        fragment = fragment_lookup(source_or_handle)
        if not fragment:
            return None
        original = fragment.get("content", "")
        if not original:
            return None
        handle = self.store(
            source=fragment.get("source", source_or_handle),
            original_content=original,
            compressed_content=f"[ref] {fragment.get('source', source_or_handle)}",
            resolution="reference",
            original_tokens=int(fragment.get("token_count", 0) or 0),
            compressed_tokens=max(3, len(source_or_handle) // 4),
            fragment_id=fragment.get("id", fragment.get("fragment_id", "")),
            entropy_score=float(fragment.get("entropy_score", 0.0) or 0.0),
        )
        return self.retrieve(handle)

    def list_available(self) -> list[dict[str, Any]]:
        """List all stored fragments with metadata (no content).

        Returns list of dicts with source, resolution, token counts.
        Used by the LLM to decide WHICH fragments to retrieve.
        """
        with self._lock:
            return [
                {
                    "source": entry["source"],
                    "retrieval_handle": entry["retrieval_handle"],
                    "resolution": entry["resolution"],
                    "original_tokens": entry["original_tokens"],
                    "compressed_tokens": entry["compressed_tokens"],
                    "tokens_recoverable": entry["original_tokens"] - entry["compressed_tokens"],
                    "content_sha256": entry["content_sha256"],
                    "fragment_id": entry["fragment_id"],
                }
                for entry in self._store.values()
            ]

    def clear(self) -> None:
        """Clear all stored originals."""
        with self._lock:
            self._store.clear()
            self._latest_by_source.clear()

    def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        with self._lock:
            total_original = sum(e["original_tokens"] for e in self._store.values())
            total_compressed = sum(e["compressed_tokens"] for e in self._store.values())
            return {
                "entries": len(self._store),
                "sources": len(self._latest_by_source),
                "max_entries": self._max_entries,
                "total_stored": self._total_stored,
                "total_retrieved": self._total_retrieved,
                "tokens_in_store": total_original,
                "tokens_compressed": total_compressed,
                "tokens_recoverable": total_original - total_compressed,
            }


def capture_recoverable_fragments(
    fragments: list[dict[str, Any]],
    fragment_lookup: Any,
    store: CompressedContextStore | None = None,
) -> list[str]:
    """Store originals for compressed selections and attach retrieval handles.

    The optimizer deliberately returns the compressed variant to downstream
    callers. This bridge resolves the engine's original fragment by id/source,
    stores it in CCR, and annotates the selected fragment with the exact handle
    that can recover the omitted detail.
    """
    ccr_store = store or get_ccr_store()
    handles: list[str] = []
    for fragment in fragments:
        resolution = fragment.get("variant", "full")
        if resolution == "full":
            continue
        fragment_id = fragment.get("id", fragment.get("fragment_id", ""))
        source = fragment.get("source", "")
        original = fragment_lookup(fragment_id) if fragment_id else None
        if original is None and source:
            original = fragment_lookup(source)
        if not original:
            continue
        original_content = original.get("content", "")
        if not original_content:
            continue
        compressed_content = fragment.get("content", fragment.get("preview", ""))
        handle = ccr_store.store(
            source=source or original.get("source", fragment_id),
            original_content=original_content,
            compressed_content=compressed_content,
            resolution=resolution,
            original_tokens=int(original.get("token_count", 0) or 0),
            compressed_tokens=int(fragment.get("token_count", 0) or 0),
            fragment_id=fragment_id,
            relevance=float(fragment.get("relevance", 0.0) or 0.0),
            entropy_score=float(fragment.get("entropy_score", 0.0) or 0.0),
        )
        fragment["retrieval_handle"] = handle
        fragment["recoverable"] = True
        handles.append(handle)
    return handles


def hierarchical_context_fragments(
    hcc_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return the fragments represented by a rendered hierarchical context.

    Level 3 fragments are full resolution. Level 2 fragments are rendered
    skeletons and therefore need eager CCR handles. Level 1 remains lazily
    recoverable by the visible source paths in its overview map.
    """
    return [
        *hcc_result.get("level3_fragments", []),
        *hcc_result.get("level2_fragments", []),
    ]


def capture_ranked_recovery_candidates(
    excluded: list[dict[str, Any]],
    fragment_lookup: Any,
    *,
    selected_sources: set[str] | None = None,
    selected_keys: set[str] | None = None,
    max_candidates: int = 8,
    store: CompressedContextStore | None = None,
) -> list[dict[str, Any]]:
    """Register a bounded set of query-ranked omissions for exact replay.

    These references are not rendered into the first prompt. They are standby
    candidates for verification-triggered recovery when the first selection
    missed evidence entirely rather than compressing it too aggressively.
    """
    if max_candidates <= 0:
        return []
    selected_sources = selected_sources or set()
    selected_keys = selected_keys or set()

    def rank(candidate: dict[str, Any]) -> tuple[float, float]:
        scores = candidate.get("scores", {})
        return (
            float(scores.get("semantic", 0.0) or 0.0),
            float(scores.get("composite", 0.0) or 0.0),
        )

    candidates: list[dict[str, Any]] = []
    emitted_keys: set[str] = set()
    for omitted in sorted(excluded, key=rank, reverse=True):
        if len(candidates) >= max_candidates:
            break
        fragment_id = omitted.get("id", omitted.get("fragment_id", ""))
        source = omitted.get("source", "")
        candidate_key = fragment_id or source
        if (
            not source
            or source in selected_sources
            or candidate_key in selected_keys
            or candidate_key in emitted_keys
        ):
            continue
        original = fragment_lookup(fragment_id) if fragment_id else None
        if original is None:
            original = fragment_lookup(source)
        if not original or not original.get("content"):
            continue
        semantic, composite = rank(omitted)
        if semantic <= 0.0 and composite <= 0.0:
            continue
        emitted_keys.add(candidate_key)
        candidates.append({
            "id": fragment_id or original.get("id", ""),
            "source": source,
            "content": f"[omitted exact replay available] {source}",
            "token_count": max(3, len(source) // 4),
            "variant": "reference",
            "relevance": semantic or composite,
            "entropy_score": float(
                original.get("entropy_score", 0.0) or 0.0
            ),
            "recovery_candidate": True,
        })

    capture_recoverable_fragments(candidates, fragment_lookup, store=store)
    return candidates


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
