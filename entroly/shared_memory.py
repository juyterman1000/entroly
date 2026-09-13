"""
Cross-agent shared memory — a content-addressed store that multiple AI agents
(Claude, Codex, Cursor, Gemini, etc.) can read and write simultaneously with
automatic deduplication and agent provenance tracking.

Storage lives under `.entroly/shared_memory/` alongside the vault, using the
same local-first philosophy. Each entry is a JSON record with content hash,
agent identity, SimHash for near-duplicate detection, and BM25-searchable text.

Concurrency: file-locking via platform-native advisory locks ensures safe
multi-process writes. Reads are lock-free (append-only log + periodic compaction).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SharedEntry:
    entry_id: str
    content_hash: str
    content: str
    agent_id: str
    session_id: str
    tags: list[str] = field(default_factory=list)
    simhash: int = 0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SharedEntry:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# File locking (cross-platform)
# ---------------------------------------------------------------------------

class _FileLock:
    """Advisory file lock. Uses msvcrt on Windows, fcntl on POSIX."""

    def __init__(self, path: Path):
        self._path = path
        self._fh = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w")
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self._fh:
            if os.name == "nt":
                import msvcrt
                try:
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Core store
# ---------------------------------------------------------------------------

def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _compute_simhash(content: str) -> int:
    """Compute SimHash, using Rust engine if available, else Python fallback."""
    try:
        from entroly_core import py_simhash
        return py_simhash(content)
    except (ImportError, AttributeError):
        pass
    # Python fallback: 64-bit SimHash via word-level hashing
    v = [0] * 64
    tokens = content.lower().split()
    for token in tokens:
        h = int(hashlib.md5(token.encode(), usedforsecurity=False).hexdigest(), 16) & ((1 << 64) - 1)
        for i in range(64):
            if h & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1
    result = 0
    for i in range(64):
        if v[i] > 0:
            result |= (1 << i)
    return result


def _hamming_distance(a: int, b: int) -> int:
    """Hamming distance between two 64-bit SimHashes."""
    try:
        from entroly_core import py_hamming_distance
        return py_hamming_distance(a, b)
    except (ImportError, AttributeError):
        pass
    x = a ^ b
    count = 0
    while x:
        count += 1
        x &= x - 1
    return count


class SharedMemoryStore:
    """
    Content-addressed shared memory for cross-agent context.

    Multiple agent processes (Claude Code, Codex, Cursor, etc.) can write to
    and read from the same store concurrently. Automatic deduplication via
    SimHash prevents redundant entries across agents.
    """

    DEDUP_THRESHOLD = 6  # hamming distance <= 6 → near-duplicate

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path(os.environ.get("ENTROLY_DIR", ".entroly"))
        self._root = Path(root) / "shared_memory"
        self._log_path = self._root / "entries.jsonl"
        self._lock_path = self._root / ".lock"
        self._root.mkdir(parents=True, exist_ok=True)

    # -- Write ---------------------------------------------------------------

    def write(
        self,
        content: str,
        agent_id: str = "unknown",
        session_id: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        dedup: bool = True,
    ) -> SharedEntry | None:
        """
        Write a shared memory entry. Returns the entry if written, or None if
        deduplicated away (near-duplicate already exists from another agent).
        """
        content_hash = _content_hash(content)
        simhash = _compute_simhash(content)

        if dedup:
            existing = self._find_duplicate(content_hash, simhash)
            if existing is not None:
                logger.debug(
                    "Dedup: content from %s matches existing entry %s by %s",
                    agent_id, existing.entry_id, existing.agent_id,
                )
                return None

        entry = SharedEntry(
            entry_id=uuid.uuid4().hex[:12],
            content_hash=content_hash,
            content=content,
            agent_id=agent_id,
            session_id=session_id or uuid.uuid4().hex[:8],
            tags=tags or [],
            simhash=simhash,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        with _FileLock(self._lock_path):
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

        logger.info(
            "SharedMemory: wrote %s from agent=%s tags=%s",
            entry.entry_id, agent_id, tags,
        )
        return entry

    # -- Read ----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        agent_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[SharedEntry]:
        """BM25-ranked search across shared memory entries."""
        entries = self._load_entries()
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        if tags:
            tag_set = set(tags)
            entries = [e for e in entries if tag_set & set(e.tags)]

        scored = self._bm25_rank(query, entries)
        return [e for _, e in sorted(scored, key=lambda x: -x[0])[:top_k]]

    def list_entries(
        self,
        agent_id: str | None = None,
        limit: int = 50,
    ) -> list[SharedEntry]:
        """List entries, optionally filtered by agent, newest first."""
        entries = self._load_entries()
        if agent_id:
            entries = [e for e in entries if e.agent_id == agent_id]
        entries.sort(key=lambda e: -e.timestamp)
        return entries[:limit]

    def get(self, entry_id: str) -> SharedEntry | None:
        """Retrieve a specific entry by ID."""
        for e in self._load_entries():
            if e.entry_id == entry_id:
                return e
        return None

    def forget(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found and removed."""
        entries = self._load_entries()
        remaining = [e for e in entries if e.entry_id != entry_id]
        if len(remaining) == len(entries):
            return False
        self._write_entries(remaining)
        return True

    def stats(self) -> dict:
        """Summary statistics about the shared memory store."""
        entries = self._load_entries()
        agents = {}
        tag_counts = {}
        for e in entries:
            agents[e.agent_id] = agents.get(e.agent_id, 0) + 1
            for t in e.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

        total_tokens = sum(len(e.content.split()) for e in entries)
        return {
            "total_entries": len(entries),
            "total_tokens": total_tokens,
            "agents": agents,
            "top_tags": dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10]),
            "oldest": min((e.timestamp for e in entries), default=0),
            "newest": max((e.timestamp for e in entries), default=0),
        }

    # -- Internal ------------------------------------------------------------

    def _load_entries(self) -> list[SharedEntry]:
        if not self._log_path.exists():
            return []
        entries = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(SharedEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
        return entries

    def _write_entries(self, entries: list[SharedEntry]) -> None:
        with _FileLock(self._lock_path):
            with open(self._log_path, "w", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")

    def _find_duplicate(
        self, content_hash: str, simhash: int,
    ) -> SharedEntry | None:
        """Check for exact hash match or SimHash near-duplicate."""
        for entry in self._load_entries():
            if entry.content_hash == content_hash:
                return entry
            if _hamming_distance(entry.simhash, simhash) <= self.DEDUP_THRESHOLD:
                return entry
        return None

    def _bm25_rank(
        self, query: str, entries: list[SharedEntry],
    ) -> list[tuple[float, SharedEntry]]:
        """Rank entries by BM25 relevance to query."""
        try:
            from entroly_core import py_shared_memory_search
            texts = [e.content for e in entries]
            results = py_shared_memory_search(texts, query, len(entries))
            scored = [(score, entries[idx]) for idx, score in results]
            return scored
        except (ImportError, AttributeError):
            pass

        # Python BM25 fallback
        import math
        query_terms = set(query.lower().split())
        k1, b = 1.2, 0.75
        N = len(entries)
        if N == 0:
            return []

        avg_dl = sum(len(e.content.split()) for e in entries) / N
        df = {}
        for e in entries:
            doc_terms = set(e.content.lower().split())
            for t in query_terms & doc_terms:
                df[t] = df.get(t, 0) + 1

        scored = []
        for entry in entries:
            doc_terms = entry.content.lower().split()
            dl = len(doc_terms)
            tf = {}
            for t in doc_terms:
                if t in query_terms:
                    tf[t] = tf.get(t, 0) + 1

            score = 0.0
            for term, freq in tf.items():
                n = df.get(term, 0)
                idf = math.log((N - n + 0.5) / (n + 0.5) + 1.0)
                tf_norm = (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avg_dl))
                score += idf * tf_norm
            scored.append((score, entry))
        return scored
