"""
Entroly × Hippocampus — Long-Term Memory Integration
=====================================================

Bridges entroly's session-level context optimization with hippocampus-sharp-memory's
brain-inspired long-term retention engine. Together they solve a problem neither
solves alone:

  entroly alone:   Great at optimizing context for THIS call, but forgets everything
                   between sessions. Every new session starts cold.

  hippocampus alone: Great at remembering ACROSS sessions, but has no token budget
                     awareness. Can't tell the LLM "here are the 3 most relevant
                     memories that fit in your context window."

  entroly + hippocampus:
    1. hippocampus remembers high-value fragments across sessions (salience-based decay)
    2. entroly recalls relevant long-term memories at optimize_context() time
    3. entroly injects them as "pinned" fragments → knapsack gives them priority
    4. Result: LLM gets both fresh context AND relevant historical knowledge

Integration is zero-friction:
  - If hippocampus-sharp-memory is installed: long-term memory is active
  - If not installed: entroly works exactly as before (graceful degradation)
  - No configuration needed. No user interaction.

Salience mapping (automatic):
  - Pinned/critical fragments → salience=100 (survives ~460 ticks)
  - High-entropy fragments → salience=50 (survives ~230 ticks)
  - Normal fragments → salience=20 (survives ~92 ticks)
  - Explored/low-value fragments → salience=5 (survives ~23 ticks)

A "tick" = one optimize_context() call. So salience=100 means the memory
survives ~460 optimization rounds — effectively permanent for most sessions.

Usage (automatic — no user action required):
    # hippocampus is initialized automatically when EntrolyEngine starts
    # High-value fragments from optimize_context() are auto-remembered
    # Long-term memories are auto-injected into subsequent optimize_context() calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("entroly.long_term_memory")

# ── Availability check ─────────────────────────────────────────────────

_HIPPOCAMPUS_AVAILABLE = False
_HippocampusEngine = None
_create_memory = None

try:
    from hippocampus_sharp_memory import (
        HippocampusEngine as _HippocampusEngine,
    )
    from hippocampus_sharp_memory import (
        create_memory as _create_memory,
    )
    _HIPPOCAMPUS_AVAILABLE = True
    logger.info("Long-term memory: hippocampus-sharp-memory detected +")
except ImportError:
    logger.debug(
        "hippocampus-sharp-memory not installed — long-term memory disabled. "
        "Install with: pip install hippocampus-sharp-memory"
    )


def is_available() -> bool:
    """Check if hippocampus-sharp-memory is installed and available."""
    return _HIPPOCAMPUS_AVAILABLE


# ── Salience mapping ────────────────────────────────────────────────────

@dataclass
class SalienceProfile:
    """Maps entroly fragment properties to hippocampus salience values."""
    pinned: float = 100.0        # Critical/safety files → ~460 ticks
    high_entropy: float = 50.0   # High information density → ~230 ticks
    selected: float = 30.0       # Selected by knapsack → ~138 ticks
    normal: float = 15.0         # Normal fragments → ~69 ticks
    low_value: float = 3.0       # Low relevance → ~14 ticks

    def compute(
        self,
        is_pinned: bool = False,
        entropy_score: float = 0.0,
        was_selected: bool = False,
        relevance: float = 0.0,
    ) -> float:
        """Compute salience for a fragment based on its properties."""
        if is_pinned:
            return self.pinned
        if entropy_score > 0.7:
            return self.high_entropy
        if was_selected and relevance > 0.5:
            return self.selected
        if was_selected:
            return self.normal
        return self.low_value


# ── Long-Term Memory Adapter ───────────────────────────────────────────

class LongTermMemory:
    """
    Adapter between entroly's session-level context engine and hippocampus'
    cross-session memory.

    Lifecycle:
      1. Created once when EntrolyEngine starts (in server.py)
      2. On each optimize_context(): recall relevant long-term memories → inject as fragments
      3. On each successful optimization: remember high-value selected fragments
      4. On each advance_turn(): tick the hippocampus clock (triggers decay)

    Thread-safe: hippocampus engine is Rust-backed with internal locking.
    Memory-efficient: default capacity is 10K episodes (~10MB) for MCP use.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        consolidation_interval: int = 50,
        recall_reinforcement: float = 1.3,
    ):
        self._engine = None
        self._salience = SalienceProfile()
        self._total_remembered = 0
        self._total_recalled = 0
        self._active = False

        if not _HIPPOCAMPUS_AVAILABLE:
            return

        try:
            self._engine = _create_memory(
                capacity=capacity,
                consolidation_interval=consolidation_interval,
                recall_reinforcement=recall_reinforcement,
            )
            self._active = True
            logger.info(
                f"Long-term memory initialized: capacity={capacity}, "
                f"consolidation_interval={consolidation_interval}"
            )
        except Exception as e:
            logger.warning(f"Long-term memory init failed (non-fatal): {e}")

    @property
    def active(self) -> bool:
        return self._active and self._engine is not None

    def tick(self) -> None:
        """Advance the hippocampus clock by 1 tick (called on each advance_turn)."""
        if self.active:
            try:
                self._engine.tick()
            except Exception:
                pass

    def remember_fragments(
        self,
        fragments: list[dict],
        selected_ids: set[str] | None = None,
    ) -> int:
        """
        Remember high-value fragments from an optimization result.

        Called automatically after optimize_context(). Only remembers
        fragments that were selected by the knapsack optimizer (high
        value fragments that the LLM actually saw).

        Args:
            fragments: List of fragment dicts with 'content', 'source',
                      'entropy_score', 'is_pinned', 'relevance'.
            selected_ids: Set of fragment IDs that were selected.

        Returns:
            Number of fragments remembered.
        """
        if not self.active:
            return 0

        selected_ids = selected_ids or set()
        remembered = 0

        for frag in fragments:
            fid = frag.get("id", frag.get("fragment_id", ""))
            content = frag.get("content", "")
            source = frag.get("source", "")
            was_selected = fid in selected_ids

            if not content or not was_selected:
                continue

            salience = self._salience.compute(
                is_pinned=frag.get("is_pinned", False),
                entropy_score=frag.get("entropy_score", 0.0),
                was_selected=was_selected,
                relevance=frag.get("relevance", 0.0),
            )

            # Emotional tag: critical/safety files get amplified (3x retention)
            emotional_tag = 0
            criticality = frag.get("criticality", "")
            if "Safety" in criticality or "Critical" in criticality:
                emotional_tag = 3  # 3x salience boost
            elif "Important" in criticality:
                emotional_tag = 2  # 1.5x boost

            try:
                self._engine.remember(
                    content=f"[{source}] {content[:500]}",  # Cap at 500 chars to save memory
                    salience=salience,
                    source=source,
                    emotional_tag=emotional_tag,
                )
                remembered += 1
            except Exception:
                pass

        self._total_remembered += remembered
        return remembered

    def recall_relevant(
        self,
        query: str,
        top_k: int = 5,
        min_retention: float = 0.2,
    ) -> list[dict]:
        """
        Recall relevant long-term memories for a query.

        Called automatically at the start of optimize_context()
        to inject historical context into the current optimization.

        Args:
            query: The optimization query (passed to hippocampus recall).
            top_k: Maximum number of memories to return.
            min_retention: Minimum retention threshold (0.0-1.0).
                          Below this, memories are too faded to be useful.

        Returns:
            List of dicts with 'content', 'source', 'salience',
            'retention', 'age_ticks', 'from_long_term_memory'.
        """
        if not self.active or not query:
            return []

        try:
            results = self._engine.recall(query, top_k=top_k)
        except Exception:
            return []

        memories = []
        for r in results:
            # Skip faded memories
            if r.retention < min_retention:
                continue

            memories.append({
                "content": r.content,
                "source": getattr(r, "source", "long_term_memory"),
                "salience": r.salience,
                "retention": round(r.retention, 4),
                "age_ticks": getattr(r, "age_ticks", 0),
                "recall_count": getattr(r, "recall_count", 0),
                "consolidated": getattr(r, "consolidated", False),
                "from_long_term_memory": True,
            })

        self._total_recalled += len(memories)
        return memories

    def stats(self) -> dict:
        """Get long-term memory statistics for the dashboard."""
        if not self.active:
            return {
                "active": False,
                "reason": "hippocampus-sharp-memory not installed",
            }

        try:
            engine_stats = self._engine.stats()
            return {
                "active": True,
                "total_remembered": self._total_remembered,
                "total_recalled": self._total_recalled,
                "episode_count": getattr(engine_stats, "episode_count", 0),
                "consolidated_count": getattr(engine_stats, "consolidated_count", 0),
                "avg_retention": round(getattr(engine_stats, "avg_retention", 0.0), 4),
                "insight": (
                    f"Long-term memory active: {self._total_remembered} fragments remembered, "
                    f"{self._total_recalled} recalled across sessions. "
                    f"Powered by hippocampus-sharp-memory (Kanerva SDM + LSH)."
                ),
            }
        except Exception as e:
            return {
                "active": True,
                "total_remembered": self._total_remembered,
                "total_recalled": self._total_recalled,
                "error": str(e),
            }

    def consolidate(self) -> str:
        """Force a consolidation cycle (sleep-replay)."""
        if not self.active:
            return "Long-term memory not active"
        try:
            result = self._engine.consolidate_now()
            return str(result)
        except Exception as e:
            return f"Consolidation failed: {e}"
