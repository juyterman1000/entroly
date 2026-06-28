"""Entroly Memory OS — budget-aware runtime memory for AI agents.

This module gives Python users a stable public product surface while the deeper
Rust memory stack continues to mature behind it.

Production goals:

- local-first and dependency-free by default,
- bounded memory growth,
- safety checks before storing sensitive memory,
- deterministic budget-aware recall,
- explicit selected/omitted receipts,
- atomic save/load for durable local memory.

It does not require an embeddings API, vector database, network call, or native
extension. When native bindings for the Rust MemoryManager/SCHIPC/ComplianceGate
are promoted to public PyO3, this facade can delegate to them without changing
the user-facing API.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Iterable, Literal

MemoryTier = Literal["working", "episodic", "semantic"]
SafetyPolicy = Literal["block", "redact", "allow"]

_SCHEMA_VERSION = "memory-os.v2"
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"), "critical"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "critical"),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"), "critical"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "critical"),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "critical"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "critical"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "medium"),
    ("phone", re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}(?!\d)"), "medium"),
]

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("ignore_instructions", re.compile(r"ignore (?:all )?(?:previous|prior|above) instructions", re.I), "critical"),
    ("system_prompt_exfil", re.compile(r"(?:reveal|print|dump|show).{0,40}(?:system prompt|developer message|hidden instructions)", re.I), "critical"),
    ("tool_abuse", re.compile(r"(?:run|execute).{0,40}(?:rm -rf|curl .+\|\s*sh|powershell .+iex)", re.I), "critical"),
]


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def _estimate_tokens(text: str) -> int:
    # Code tends to be slightly denser than prose. 3.5 chars/token matches the
    # Rust memory estimate and is good enough for budget gating.
    return max(1, math.ceil(len(text or "") / 3.5))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_tier(tier: str) -> MemoryTier:
    if tier in ("working", "episodic", "semantic"):
        return tier  # type: ignore[return-value]
    return "working"


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = -1
    tmp_name = ""
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.chmod(tmp_name, 0o600)
        except OSError:
            pass
        os.replace(tmp_name, path)
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


@dataclass(slots=True)
class MemorySafetyFinding:
    """Sensitive or unsafe content detected before memory storage."""

    kind: str
    severity: str
    start: int
    end: int


@dataclass(slots=True)
class MemorySafetyResult:
    """Safety scan result for a candidate memory."""

    allowed: bool
    action: str
    findings: list[MemorySafetyFinding] = field(default_factory=list)
    redacted_content: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "findings": [asdict(f) for f in self.findings],
            "redacted_content": self.redacted_content,
        }


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
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class OmittedMemory:
    id: str
    source: str
    tier: MemoryTier
    token_cost: int
    score: float
    retention: float
    reason: str
    tags: list[str] = field(default_factory=list)


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
            tags = f" tags={','.join(mem.tags)}" if mem.tags else ""
            blocks.append(
                f"[memory:{mem.id} tier={mem.tier} source={mem.source}{tags} "
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

    The facade is intentionally conservative: local-only, bounded, safe by
    default, and deterministic. It stores entries in process memory; use
    ``save(path)`` and ``load(path)`` for durable local memory.
    """

    def __init__(
        self,
        *,
        default_budget: int = 4096,
        death_threshold: float = 0.05,
        consolidation_retention: float = 0.50,
        consolidation_min_recalls: int = 2,
        max_entries: int = 50_000,
        max_tokens: int = 500_000,
        safety_policy: SafetyPolicy = "block",
    ) -> None:
        self.default_budget = max(1, int(default_budget))
        self.death_threshold = _clamp01(death_threshold)
        self.consolidation_retention = _clamp01(consolidation_retention)
        self.consolidation_min_recalls = max(1, int(consolidation_min_recalls))
        self.max_entries = max(1, int(max_entries))
        self.max_tokens = max(1, int(max_tokens))
        self.safety_policy: SafetyPolicy = safety_policy if safety_policy in ("block", "redact", "allow") else "block"  # type: ignore[assignment]
        self._tick = 0.0
        self._next_id = 1
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = RLock()

    @property
    def current_tick(self) -> float:
        return self._tick

    def scan_safety(self, content: str, *, policy: SafetyPolicy | None = None) -> MemorySafetyResult:
        """Scan memory content for secrets, PII, and prompt-injection payloads."""
        effective_policy = policy or self.safety_policy
        findings: list[MemorySafetyFinding] = []
        text = str(content or "")
        for kind, pattern, severity in _SECRET_PATTERNS + _INJECTION_PATTERNS:
            for match in pattern.finditer(text):
                findings.append(MemorySafetyFinding(kind, severity, match.start(), match.end()))

        if not findings or effective_policy == "allow":
            return MemorySafetyResult(True, "allow", findings, None)

        redacted = text
        # Apply from the end to keep offsets stable.
        for f in sorted(findings, key=lambda x: x.start, reverse=True):
            redacted = redacted[: f.start] + f"[{f.kind.upper()}_REDACTED]" + redacted[f.end :]

        if effective_policy == "redact":
            return MemorySafetyResult(True, "redact", findings, redacted)

        return MemorySafetyResult(False, "block", findings, redacted)

    def remember(
        self,
        content: str,
        *,
        agent_id: str = "default",
        importance: float = 0.5,
        tier: MemoryTier = "working",
        source: str = "manual",
        tags: Iterable[str] | None = None,
        safety_policy: SafetyPolicy | None = None,
    ) -> str:
        """Store a memory and return its ID.

        Identical content for the same agent/source is deduplicated and
        reinforced instead of duplicated. By default, unsafe memory is blocked.
        Pass ``safety_policy='redact'`` to store a redacted copy, or ``'allow'``
        only for trusted offline test fixtures.
        """
        safe_content = str(content or "").strip()
        if not safe_content:
            raise ValueError("memory content cannot be empty")

        safety = self.scan_safety(safe_content, policy=safety_policy)
        if not safety.allowed:
            kinds = ", ".join(sorted({f.kind for f in safety.findings}))
            raise ValueError(f"memory content failed safety policy: {kinds}")
        if safety.action == "redact" and safety.redacted_content is not None:
            safe_content = safety.redacted_content

        safe_agent = str(agent_id or "default")
        safe_source = str(source or "manual")
        safe_tier = _safe_tier(str(tier))
        importance = _clamp01(importance)
        tag_list = [str(t) for t in (tags or [])]

        with self._lock:
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
            self._prune_to_capacity_locked()
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
        tier_filter = _safe_tier(str(tier)) if tier is not None else None

        with self._lock:
            candidates: list[tuple[MemoryEntry, float, float, str]] = []
            for entry in self._entries.values():
                if entry.agent_id != safe_agent and not (include_shared and entry.agent_id == "shared"):
                    continue
                if tier_filter is not None and entry.tier != tier_filter:
                    continue

                retention = entry.retention(self._tick)
                if entry.tier != "semantic" and retention < self.death_threshold:
                    candidates.append((entry, 0.0, retention, "below_retention_threshold"))
                    continue

                content_terms = _tokens(entry.content)
                source_terms = _tokens(entry.source.replace("/", " ").replace("\\", " "))
                tag_terms = {t.lower() for t in entry.tags}
                searchable = content_terms | source_terms | tag_terms
                if query_terms:
                    overlap = len(query_terms & searchable) / max(1, len(query_terms))
                    lexical = 0.20 + 0.80 * overlap
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
                        OmittedMemory(entry.id, entry.source, entry.tier, entry.token_cost, round(score, 6), round(retention, 6), reason, list(entry.tags))
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
                            list(entry.tags),
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
                            list(entry.tags),
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
        with self._lock:
            for _ in range(steps):
                self._tick += 1.0
                if int(self._tick) % 100 == 0:
                    self._consolidate_locked()

    def forget(self, threshold: float | None = None) -> int:
        """Forget weak non-semantic memories below the retention threshold."""
        thresh = self.death_threshold if threshold is None else _clamp01(threshold)
        with self._lock:
            before = len(self._entries)
            self._entries = {
                mem_id: entry
                for mem_id, entry in self._entries.items()
                if entry.tier == "semantic" or entry.retention(self._tick) >= thresh
            }
            return before - len(self._entries)

    def consolidate(self) -> int:
        """Promote frequently recalled memories toward semantic memory."""
        with self._lock:
            return self._consolidate_locked()

    def stats(self) -> dict[str, object]:
        with self._lock:
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
                "total_tokens": sum(tokens.values()),
                "default_budget": self.default_budget,
                "max_entries": self.max_entries,
                "max_tokens": self.max_tokens,
                "safety_policy": self.safety_policy,
            }

    def snapshot(self) -> dict[str, object]:
        """Serialize MemoryOS state into plain Python data."""
        with self._lock:
            return {
                "version": _SCHEMA_VERSION,
                "created_at": time.time(),
                "current_tick": self._tick,
                "next_id": self._next_id,
                "config": {
                    "default_budget": self.default_budget,
                    "death_threshold": self.death_threshold,
                    "consolidation_retention": self.consolidation_retention,
                    "consolidation_min_recalls": self.consolidation_min_recalls,
                    "max_entries": self.max_entries,
                    "max_tokens": self.max_tokens,
                    "safety_policy": self.safety_policy,
                },
                "entries": [asdict(entry) for entry in self._entries.values()],
            }

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Atomically save memory state to a local JSON file with owner-only permissions."""
        out = Path(path).expanduser()
        _atomic_write_json(out, self.snapshot())
        return out

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "MemoryOS":
        """Load MemoryOS from a local JSON snapshot."""
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("memory snapshot must be a JSON object")
        return cls.from_snapshot(data)

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, object]) -> "MemoryOS":
        """Restore MemoryOS from ``snapshot()`` output."""
        if not isinstance(snapshot, dict):
            raise ValueError("memory snapshot must be a dict")
        config = snapshot.get("config", {})
        if not isinstance(config, dict):
            config = {}
        obj = cls(
            default_budget=int(config.get("default_budget", 4096)),
            death_threshold=float(config.get("death_threshold", 0.05)),
            consolidation_retention=float(config.get("consolidation_retention", 0.50)),
            consolidation_min_recalls=int(config.get("consolidation_min_recalls", 2)),
            max_entries=int(config.get("max_entries", 50_000)),
            max_tokens=int(config.get("max_tokens", 500_000)),
            safety_policy=str(config.get("safety_policy", "block")),  # type: ignore[arg-type]
        )
        obj._tick = float(snapshot.get("current_tick", 0.0))
        obj._next_id = int(snapshot.get("next_id", 1))
        entries = snapshot.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("memory snapshot entries must be a list")
        for raw in entries:
            if not isinstance(raw, dict):
                continue
            entry = obj._entry_from_dict(raw)
            obj._entries[entry.id] = entry
        obj._prune_to_capacity_locked()
        return obj

    def _entry_from_dict(self, raw: dict[str, object]) -> MemoryEntry:
        content = str(raw.get("content", "")).strip()
        if not content:
            raise ValueError("memory entry content cannot be empty")
        token_cost = int(raw.get("token_cost", _estimate_tokens(content)))
        return MemoryEntry(
            id=str(raw.get("id", f"mem_{self._next_id:06d}")),
            agent_id=str(raw.get("agent_id", "default")),
            content=content,
            importance=_clamp01(float(raw.get("importance", 0.5))),
            tier=_safe_tier(str(raw.get("tier", "working"))),
            source=str(raw.get("source", "manual")),
            tags=[str(t) for t in raw.get("tags", [])] if isinstance(raw.get("tags", []), list) else [],
            created_at=float(raw.get("created_at", 0.0)),
            last_recalled_at=float(raw.get("last_recalled_at", 0.0)),
            recall_count=max(0, int(raw.get("recall_count", 0))),
            salience=max(0.001, float(raw.get("salience", 25.0))),
            token_cost=max(1, token_cost),
            consolidated=bool(raw.get("consolidated", False)),
        )

    def _consolidate_locked(self) -> int:
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
        self._forget_locked(self.death_threshold)
        self._prune_to_capacity_locked()
        return promoted

    def _forget_locked(self, threshold: float) -> int:
        before = len(self._entries)
        self._entries = {
            mem_id: entry
            for mem_id, entry in self._entries.items()
            if entry.tier == "semantic" or entry.retention(self._tick) >= threshold
        }
        return before - len(self._entries)

    def _prune_to_capacity_locked(self) -> int:
        """Evict weakest non-semantic memories until count/token budgets fit."""
        removed = 0
        while len(self._entries) > self.max_entries or self._total_tokens_locked() > self.max_tokens:
            evictable = [e for e in self._entries.values() if e.tier != "semantic"]
            if not evictable:
                break
            weakest = min(
                evictable,
                key=lambda e: (
                    e.retention(self._tick) * (0.5 + e.importance) * (1.0 + math.log1p(e.recall_count)),
                    e.last_recalled_at,
                ),
            )
            self._entries.pop(weakest.id, None)
            removed += 1
        return removed

    def _total_tokens_locked(self) -> int:
        return sum(entry.token_cost for entry in self._entries.values())


__all__ = [
    "MemoryOS",
    "MemoryEntry",
    "MemoryContext",
    "MemorySafetyFinding",
    "MemorySafetyResult",
    "SelectedMemory",
    "OmittedMemory",
]
