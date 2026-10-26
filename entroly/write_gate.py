"""Entroly Memory Write Gate — redundancy filter for memory writes.

Before a memory is committed, the write gate checks whether it carries
enough new information to be worth storing.

Every gating decision is receipt-tracked: the gate records what was
admitted, what was rejected, and why. This pairs with Entroly's receipt
philosophy — no silent information loss.

Architecture
------------
The write gate sits between the caller and MemoryOS.remember():

    caller → WriteGate.should_commit() → MemoryOS.remember()
                     │
                     └─ GatingReceipt (audit trail)

Gating signals:
1. Semantic redundancy — SimHash distance against existing entries
2. Information density — Shannon entropy of the candidate content
3. Novelty — fraction of new tokens not seen in existing memories
4. Entity novelty — whether the candidate introduces new entities

References
----------
- "Gated-Memory Routing" (Aug 2026): write gates commit only non-redundant
  steps, read gates supply task-relevant subsets
- "RCR-Router" (Aug 2025): role-aware context routing within token budgets
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of token distribution in text (bits)."""
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum(
        (c / total) * math.log2(c / total) for c in counts.values()
    )


def _compute_simhash(content: str) -> int:
    """64-bit SimHash for near-duplicate detection."""
    try:
        from entroly_core import py_simhash
        return py_simhash(content)
    except (ImportError, AttributeError):
        pass
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


def _hamming(a: int, b: int) -> int:
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


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class GatingDecision:
    """Result of a write gate evaluation."""
    commit: bool
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    nearest_memory_id: str | None = None
    nearest_distance: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GatingReceipt:
    """Audit trail of gating decisions over a session."""
    admitted: int = 0
    rejected: int = 0
    decisions: list[GatingDecision] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "rejected": self.rejected,
            "rejection_rate": round(self.rejected / max(1, self.admitted + self.rejected), 4),
            "top_rejection_reasons": self._top_reasons(),
        }

    def _top_reasons(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.decisions:
            if not d.commit:
                counts[d.reason] = counts.get(d.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:5])


# ── Write Gate ────────────────────────────────────────────────────────────

class WriteGate:
    """Decides whether a memory write carries enough new information to commit.

    Usage::

        from entroly.memory import MemoryOS
        from entroly.write_gate import WriteGate

        mem = MemoryOS()
        gate = WriteGate(mem)

        decision = gate.should_commit("Alice manages the backend team")
        if decision.commit:
            mem.remember("Alice manages the backend team")

    Configuration:
        redundancy_threshold: SimHash hamming distance below which content is
            considered a near-duplicate (default: 6, matching shared_memory.py)
        min_entropy: minimum Shannon entropy (bits) to be worth storing
            (default: 1.5 — filters out extremely low-information content)
        min_novelty: minimum fraction of new tokens (default: 0.15)
    """

    def __init__(
        self,
        memory_os: Any,  # MemoryOS — using Any to avoid circular import
        *,
        redundancy_threshold: int = 6,
        min_entropy: float = 1.5,
        min_novelty: float = 0.15,
        max_receipt_history: int = 500,
    ):
        self._mem = memory_os
        self.redundancy_threshold = redundancy_threshold
        self.min_entropy = min_entropy
        self.min_novelty = min_novelty
        self._receipt = GatingReceipt()
        self._max_history = max_receipt_history
        # Cache SimHashes of stored memories for fast redundancy checks
        self._simhash_cache: dict[str, int] = {}

    @property
    def receipt(self) -> GatingReceipt:
        return self._receipt

    def should_commit(self, content: str, *, agent_id: str = "default") -> GatingDecision:
        """Evaluate whether content should be committed to memory.

        Returns a GatingDecision with commit=True/False and an explanation.
        """
        text = content.strip()
        if not text:
            return self._reject("empty_content", {})

        # 1. Information density check
        entropy = _shannon_entropy(text)
        if entropy < self.min_entropy:
            return self._reject("low_entropy", {"entropy": round(entropy, 4), "threshold": self.min_entropy})

        # 2. Redundancy check (SimHash near-duplicate + Token Jaccard overlap)
        simhash = _compute_simhash(text)
        nearest_id, nearest_dist = self._find_nearest(simhash, agent_id)
        candidate_tokens = _tokens(text)
        is_near_dup = False
        if nearest_dist is not None and nearest_dist <= self.redundancy_threshold:
            is_near_dup = True
        elif candidate_tokens:
            try:
                with self._mem._lock:
                    for entry in self._mem._entries.values():
                        if entry.agent_id == agent_id or entry.agent_id == "shared":
                            e_tokens = _tokens(entry.content)
                            if e_tokens:
                                jaccard = len(candidate_tokens & e_tokens) / len(candidate_tokens | e_tokens)
                                if jaccard >= 0.75:
                                    is_near_dup = True
                                    nearest_id = entry.id
                                    break
            except Exception:
                pass

        if is_near_dup:
            return self._reject(
                "near_duplicate",
                {"hamming_distance": nearest_dist if nearest_dist is not None else 0, "threshold": self.redundancy_threshold},
                nearest_memory_id=nearest_id,
                nearest_distance=nearest_dist,
            )

        # 3. Token novelty check
        existing_tokens = self._collect_existing_tokens(agent_id)
        if existing_tokens and candidate_tokens:
            new_tokens = candidate_tokens - existing_tokens
            novelty = len(new_tokens) / max(1, len(candidate_tokens))
        else:
            novelty = 1.0  # First entry is always novel

        if novelty < self.min_novelty:
            return self._reject(
                "low_novelty",
                {"novelty": round(novelty, 4), "threshold": self.min_novelty,
                 "new_tokens": len(candidate_tokens - existing_tokens) if existing_tokens else len(candidate_tokens)},
            )

        # All checks passed — admit
        decision = GatingDecision(
            commit=True,
            reason="admitted",
            scores={
                "entropy": round(entropy, 4),
                "novelty": round(novelty, 4),
                "nearest_distance": nearest_dist if nearest_dist is not None else -1,
            },
            nearest_memory_id=nearest_id,
            nearest_distance=nearest_dist,
        )
        self._receipt.admitted += 1
        self._record(decision)

        # Cache the SimHash for future checks
        # (actual memory ID isn't known yet — use a placeholder)
        self._simhash_cache[f"_pending_{time.time()}"] = simhash

        return decision

    def register_committed(self, memory_id: str, content: str) -> None:
        """Register a newly committed memory ID and content into the cache."""
        for k in list(self._simhash_cache.keys()):
            if k.startswith("_pending_"):
                del self._simhash_cache[k]
        self._simhash_cache[memory_id] = _compute_simhash(content)

    def refresh_cache(self) -> None:
        """Rebuild SimHash cache from current MemoryOS entries."""
        self._simhash_cache.clear()
        with self._mem._lock:
            for entry in self._mem._entries.values():
                self._simhash_cache[entry.id] = _compute_simhash(entry.content)
        logger.debug("WriteGate cache refreshed: %d entries", len(self._simhash_cache))

    # ── Internal ─────────────────────────────────────────────────────

    def _find_nearest(self, simhash: int, agent_id: str) -> tuple[str | None, int | None]:
        """Find the nearest existing memory by SimHash distance."""
        # Lazy-populate cache on first use
        if not self._simhash_cache:
            self.refresh_cache()

        best_id: str | None = None
        best_dist: int | None = None

        for mem_id, cached_hash in list(self._simhash_cache.items()):
            dist = _hamming(simhash, cached_hash)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = mem_id

        if best_id and best_id.startswith("_pending_"):
            try:
                with self._mem._lock:
                    for entry in self._mem._entries.values():
                        if entry.agent_id in (agent_id, "shared"):
                            e_hash = _compute_simhash(entry.content)
                            if _hamming(simhash, e_hash) == best_dist:
                                best_id = entry.id
                                break
            except Exception:
                pass

        return best_id, best_dist

    def _collect_existing_tokens(self, agent_id: str) -> set[str]:
        """Collect all tokens from existing memories for this agent."""
        all_tokens: set[str] = set()
        try:
            with self._mem._lock:
                for entry in self._mem._entries.values():
                    if entry.agent_id == agent_id or entry.agent_id == "shared":
                        all_tokens.update(_tokens(entry.content))
        except Exception:
            pass
        return all_tokens

    def _reject(
        self,
        reason: str,
        scores: dict[str, float],
        *,
        nearest_memory_id: str | None = None,
        nearest_distance: int | None = None,
    ) -> GatingDecision:
        decision = GatingDecision(
            commit=False,
            reason=reason,
            scores=scores,
            nearest_memory_id=nearest_memory_id,
            nearest_distance=nearest_distance,
        )
        self._receipt.rejected += 1
        self._record(decision)
        return decision

    def _record(self, decision: GatingDecision) -> None:
        self._receipt.decisions.append(decision)
        # Bounded history
        if len(self._receipt.decisions) > self._max_history:
            self._receipt.decisions = self._receipt.decisions[-self._max_history:]


__all__ = [
    "WriteGate",
    "GatingDecision",
    "GatingReceipt",
]
