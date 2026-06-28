"""Entroly Memory OS — budget-aware runtime memory for AI agents.

This module gives Python users a clean public product surface while the deeper
Rust memory stack continues to mature behind it.

It intentionally stays dependency-free:

- no embeddings API,
- no vector database,
- no network calls,
- no required native extension.

The goal is not to replace the Rust memory engine. It is to expose the same
product semantics in a stable, easy-to-demo API:

    remember -> recall under budget -> omit with reasons -> decay -> consolidate

When native bindings for the Rust MemoryManager/SCHIPC/ComplianceGate are
promoted to the public PyO3 module, this facade can delegate to them without
changing the user-facing API.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal

MemoryTier = Literal["working", "episodic", "semantic"]

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _estimate_tokens(text: str) -> int:
    # Code tends to be slightly denser than prose. 3.5 chars/token matches the
    # Rust memory estimate and is good enough for budget gating.
    return max(1, math.ceil(len(text or "") / 3.5))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class MemoryEntry:
    """One memory trace stored by MemoryOS."""

    id: str
    agent_id: str
    content: str
    importance: float = 0.5
    tier: MemoryTier = "working"
    source: str = "manual"
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    last_recalled_at: float = 0.0
    recall_count: int = 0
    salience: float = 25.0
    token_cost: int = 1
    consolidated: bool = False

    def retention(self, now_tick: float) -> float:
        """Ebbinghaus-style retention R(t)=exp(-age/salience)."""
        if self.tier == "semantic":
            return 1.0
        age = max(0.0, now_tick - self.created_at)
        if self.salience <= 0.0:
            return 0.0
        return math.exp(-age / self.salience)

    def reinforce(self, now_tick: float, factor: float = 1.3) -> None:
        self.recall_count += 1
        self.last_recalled_at = now_tick
        self.salience = min(10_000.0, self.salience * factor)


@dataclass(slots=True)
class SelectedMemory:
    id: str
    content: str
    source: str
    tier: MemoryTier
    token_cost: int
    score: float
    retention: float
    reason: str


@dataclass(slots=True)
class OmittedMemory:
    id: str
    source: str
    tier: MemoryTier
    token_cost: int
    score: float
    retention: float
    reason: str


@dataclass(slots=True)
class MemoryContext:
    """Recall result returned by MemoryOS.recall()."""

    selected: list[SelectedMemory]
    omitted: list[OmittedMemory]
    budget: int
    used_tokens: int
    risk: dict[str, object]

    def as_text(self) -> str:
        """Render selected memories as prompt-ready evidence."""
        if not self.selected:
            return ""
        blocks = []
        for mem in self.selected:
            blocks.append(
                f"[memory:{mem.id} tier={mem.tier} source={mem.source} "
                f"retention={mem.retention:.2f}]\n{mem.content}"
            )
        return "\n\n".join(blocks)

    def receipt(self) -> dict[str, object]:
        """Machine-readable memory receipt."""
        return {
            "budget": self.budget,
            "used_tokens": self.used_tokens,
            "selected": [asdict(m) for m in self.selected],
            "omitted": [asdict(m) for m in self.omitted],
            "risk": self.risk,
        }


class MemoryOS:
    """Budget-aware working/episodic/semantic memory for agents.

    This is the public, stable facade. It is intentionally conservative and
    local-first. It stores entries in process memory by default; callers that
    need persistence can serialize ``snapshot()`` and load it with
    ``from_snapshot()``.
    """

    def __init__(
        self,
        *,
        default_budget: int = 4096,
        death_threshold: float = 0.05,
        consolidation_retention: float = 0.50,
        consolidation_min_recalls: int = 2,
    ) -> None:
        self.default_budget = max(1, int(default_budget))
        self.death_threshold = _clamp01(death_threshold)
        self.consolidation_retention = _clamp01(consolidation_retention)
        self.consolidation_min_recalls = max(1, int(consolidation_min_recalls))
        self._tick = 0.0
        self._next_id = 1
        self._entries: dict[str, MemoryEntry] = {}

    @property
    def current_tick(self) -> float:
        return self._tick

    def remember(
        self,
        content: str,
        *,
        agent_id: str = "default",
        importance: float = 0.5,
        tier: MemoryTier = "working",
        source: str = "manual",
        tags: Iterable[str] | None = None,
    ) -> str:
        """Store a memory and return its ID.

        Identical content for the same agent/source is deduplicated and
        reinforced instead of duplicated.
        """
        safe_content = str(content or "").strip()
        if not safe_content:
            raise ValueError("memory content cannot be empty")
        safe_agent = str(agent_id or "default")
        safe_source = str(source or "manual")
        safe_tier: MemoryTier = tier if tier in ("working", "episodic", "semantic") else "working"  # type: ignore[assignment]
        importance = _clamp01(importance)
        tag_list = [str(t) for t in (tags or [])]

        # Deduplicate exact repeats for the same agent/source/content.
        for entry in self._entries.values():
            if (
                entry.agent_id == safe_agent
                and entry.source == safe_source
                and entry.content == safe_content
            ):
                entry.importance = max(entry.importance, importance)
                entry.reinforce(self._tick)
                return entry.id

        mem_id = f"mem_{self._next_id:06d}"
        self._next_id += 1

        emotional_multiplier = 1.0
        lowered_tags = {t.lower() for t in tag_list}
        if importance >= 0.9 or {"critical", "safety", "security"} & lowered_tags:
            emotional_multiplier = 3.0
        elif importance >= 0.7:
            emotional_multiplier = 1.5
        elif importance >= 0.5:
            emotional_multiplier = 1.2

        # importance=1.0 -> salience 50 ticks before emotional multiplier.
        salience = max(1.0, importance * 50.0 * emotional_multiplier)
        if safe_tier == "semantic":
            salience = 10_000.0

        self._entries[mem_id] = MemoryEntry(
            id=mem_id,
            agent_id=safe_agent,
            content=safe_content,
            importance=importance,
            tier=safe_tier,
            source=safe_source,
            tags=tag_list,
            created_at=self._tick,
            last_recalled_at=self._tick,
            salience=salience,
            token_cost=_estimate_tokens(safe_content),
        )
        return mem_id

    def recall(
        self,
        query: str = "",
        *,
        agent_id: str = "default",
        budget: int | None = None,
        tier: MemoryTier | None = None,
        include_shared: bool = True,
    ) -> MemoryContext:
        """Recall memories for an agent under a token budget."""
        safe_budget = max(1, int(budget or self.default_budget))
        safe_agent = str(agent_id or "default")
        query_terms = _tokens(query)

        candidates: list[tuple[MemoryEntry, float, float, str]] = []
        for entry in self._entries.values():
            if entry.agent_id != safe_agent and not (include_shared and entry.agent_id == "shared"):
                continue
            if tier is not None and entry.tier != tier:
                continue

            retention = entry.retention(self._tick)
            if entry.tier != "semantic" and retention < self.death_threshold:
                reason = "below_retention_threshold"
                candidates.append((entry, 0.0, retention, reason))
                continue

            content_terms = _tokens(entry.content)
            if query_terms:
                overlap = len(query_terms & content_terms) / max(1, len(query_terms))
                lexical = 0.25 + 0.75 * overlap
                reason = "query_overlap" if overlap > 0 else "salience_fallback"
            else:
                lexical = 1.0
                reason = "no_query_salience_recall"

            frequency = 1.0 + min(2.0, math.log1p(entry.recall_count))
            tier_bonus = {"working": 1.0, "episodic": 1.1, "semantic": 1.25}[entry.tier]
            score = retention * frequency * lexical * tier_bonus * (0.5 + entry.importance)
            candidates.append((entry, score, retention, reason))

        candidates.sort(key=lambda x: (x[1] / max(1, x[0].token_cost), x[1]), reverse=True)

        selected: list[SelectedMemory] = []
        omitted: list[OmittedMemory] = []
        used = 0

        for entry, score, retention, reason in candidates:
            if score <= 0.0:
                omitted.append(
                    OmittedMemory(entry.id, entry.source, entry.tier, entry.token_cost, score, retention, reason)
                )
                continue
            if used + entry.token_cost <= safe_budget:
                selected.append(
                    SelectedMemory(
                        entry.id,
                        entry.content,
                        entry.source,
                        entry.tier,
                        entry.token_cost,
                        round(score, 6),
                        round(retention, 6),
                        reason,
                    )
                )
                used += entry.token_cost
                entry.reinforce(self._tick)
            else:
                omitted.append(
                    OmittedMemory(
                        entry.id,
                        entry.source,
                        entry.tier,
                        entry.token_cost,
                        round(score, 6),
                        round(retention, 6),
                        "over_budget",
                    )
                )

        risk = {
            "selected_count": len(selected),
            "omitted_count": len(omitted),
            "budget": safe_budget,
            "used_tokens": used,
            "budget_utilization": round(used / safe_budget, 4),
            "stale_omitted": sum(1 for m in omitted if m.reason == "below_retention_threshold"),
            "over_budget_omitted": sum(1 for m in omitted if m.reason == "over_budget"),
            "query_terms": sorted(query_terms)[:12],
        }
        return MemoryContext(selected=selected, omitted=omitted, budget=safe_budget, used_tokens=used, risk=risk)

    def tick(self, count: int = 1) -> None:
        """Advance memory time and run opportunistic consolidation."""
        steps = max(1, int(count))
        for _ in range(steps):
            self._tick += 1.0
            if int(self._tick) % 100 == 0:
                self.consolidate()

    def forget(self, threshold: float | None = None) -> int:
        """Forget weak non-semantic memories below the retention threshold."""
        thresh = self.death_threshold if threshold is None else _clamp01(threshold)
        before = len(self._entries)
        self._entries = {
            mem_id: entry
            for mem_id, entry in self._entries.items()
            if entry.tier == "semantic" or entry.retention(self._tick) >= thresh
        }
        return before - len(self._entries)

    def consolidate(self) -> int:
        """Promote frequently recalled memories toward semantic memory."""
        promoted = 0
        for entry in self._entries.values():
            if entry.tier == "semantic":
                continue
            if (
                entry.retention(self._tick) >= self.consolidation_retention
                and entry.recall_count >= self.consolidation_min_recalls
            ):
                entry.tier = "episodic" if entry.tier == "working" else "semantic"
                entry.consolidated = True
                entry.salience = max(entry.salience, 100.0 if entry.tier == "episodic" else 10_000.0)
                promoted += 1
        self.forget()
        return promoted

    def stats(self) -> dict[str, object]:
        tiers = {"working": 0, "episodic": 0, "semantic": 0}
        tokens = {"working": 0, "episodic": 0, "semantic": 0}
        for entry in self._entries.values():
            tiers[entry.tier] += 1
            tokens[entry.tier] += entry.token_cost
        return {
            "total_entries": len(self._entries),
            "current_tick": self._tick,
            "tiers": tiers,
            "tokens": tokens,
            "default_budget": self.default_budget,
        }

    def snapshot(self) -> dict[str, object]:
        """Serialize MemoryOS state into plain Python data."""
        return {
            "version": "memory-os.v1",
            "created_at": time.time(),
            "current_tick": self._tick,
            "next_id": self._next_id,
            "entries": [asdict(entry) for entry in self._entries.values()],
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> "MemoryOS":
        """Restore MemoryOS from ``snapshot()`` output."""
        obj = cls()
        obj._tick = float(snapshot.get("current_tick", 0.0))
        obj._next_id = int(snapshot.get("next_id", 1))
        entries = snapshot.get("entries", [])
        if isinstance(entries, list):
            for raw in entries:
                if not isinstance(raw, dict):
                    continue
                entry = MemoryEntry(**raw)
                obj._entries[entry.id] = entry
        return obj


__all__ = [
    "MemoryOS",
    "MemoryEntry",
    "MemoryContext",
    "SelectedMemory",
    "OmittedMemory",
]
