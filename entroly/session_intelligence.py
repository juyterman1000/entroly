"""Session intelligence primitives for realized savings and continuity.

This module separates exact savings from estimates, nets omitted-span retrievals
against gross compression savings, scores checkpoints by query relevance, extracts
compact decision state before compaction, forecasts cache-retention value without
issuing network calls, and detects repeated waste loops in a bounded window.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

_WORD_RE = re.compile(r"[A-Za-z0-9_./:-]{3,}")
_DECISION_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:decision|decided|choose|chosen|selected|use|ship|fix|root cause|remaining|next|todo|failure|failed|blocked)\b[:\-]?\s*(.+)$"
)
_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")


class SavingsConfidence(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    OPPORTUNITY = "opportunity"


@dataclass(frozen=True, slots=True)
class RealizedSavingsRecord:
    receipt_id: str
    original_tokens: int
    compressed_tokens: int
    retrieved_tokens: int = 0
    repeated_expansion_tokens: int = 0
    confidence: SavingsConfidence = SavingsConfidence.MEASURED

    @property
    def gross_saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.compressed_tokens)

    @property
    def net_realized_saved_tokens(self) -> int:
        return max(
            0,
            self.gross_saved_tokens
            - max(0, self.retrieved_tokens)
            - max(0, self.repeated_expansion_tokens),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "gross_saved_tokens": self.gross_saved_tokens,
            "retrieved_tokens": self.retrieved_tokens,
            "repeated_expansion_tokens": self.repeated_expansion_tokens,
            "net_realized_saved_tokens": self.net_realized_saved_tokens,
            "confidence": self.confidence.value,
        }


@dataclass(slots=True)
class SavingsTierLedger:
    _records: list[RealizedSavingsRecord] = field(default_factory=list)

    def add(self, record: RealizedSavingsRecord) -> None:
        self._records.append(record)

    def summary(self) -> dict[str, Any]:
        by_tier = {
            tier.value: {
                "records": 0,
                "gross_saved_tokens": 0,
                "retrieved_tokens": 0,
                "repeated_expansion_tokens": 0,
                "net_realized_saved_tokens": 0,
            }
            for tier in SavingsConfidence
        }
        for record in self._records:
            bucket = by_tier[record.confidence.value]
            bucket["records"] += 1
            bucket["gross_saved_tokens"] += record.gross_saved_tokens
            bucket["retrieved_tokens"] += max(0, record.retrieved_tokens)
            bucket["repeated_expansion_tokens"] += max(0, record.repeated_expansion_tokens)
            bucket["net_realized_saved_tokens"] += record.net_realized_saved_tokens
        return {"records": len(self._records), "by_confidence": by_tier}


@dataclass(frozen=True, slots=True)
class DecisionDigest:
    decisions: tuple[str, ...] = ()
    modified_paths: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    remaining_tasks: tuple[str, ...] = ()

    def as_metadata(self) -> dict[str, list[str]]:
        return {
            "decisions": list(self.decisions),
            "modified_paths": list(self.modified_paths),
            "failures": list(self.failures),
            "remaining_tasks": list(self.remaining_tasks),
        }


def extract_decision_digest(text: str, *, max_items: int = 12) -> DecisionDigest:
    decisions: list[str] = []
    failures: list[str] = []
    remaining: list[str] = []
    for match in _DECISION_RE.finditer(text or ""):
        raw = _clean_line(match.group(0))
        lower = raw.lower()
        if any(word in lower for word in ("failure", "failed", "blocked", "root cause")):
            failures.append(raw)
        elif any(word in lower for word in ("remaining", "next", "todo")):
            remaining.append(raw)
        else:
            decisions.append(raw)
    paths = sorted(set(_PATH_RE.findall(text or "")))
    return DecisionDigest(
        decisions=tuple(_dedupe(decisions)[:max_items]),
        modified_paths=tuple(paths[:max_items]),
        failures=tuple(_dedupe(failures)[:max_items]),
        remaining_tasks=tuple(_dedupe(remaining)[:max_items]),
    )


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    checkpoint_id: str
    score: float
    matched_terms: tuple[str, ...]
    age_seconds: float
    decision_hits: int
    path_hits: int
    is_trusted: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "score": round(self.score, 6),
            "matched_terms": list(self.matched_terms),
            "age_seconds": round(self.age_seconds, 3),
            "decision_hits": self.decision_hits,
            "path_hits": self.path_hits,
            "is_trusted": self.is_trusted,
        }


class CheckpointRelevanceScorer:
    def __init__(self, *, half_life_seconds: float = 86400.0) -> None:
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive")
        self.half_life_seconds = half_life_seconds

    def score(self, checkpoint: Any, query: str, *, now: float | None = None) -> CheckpointScore:
        timestamp = time.time() if now is None else float(now)
        created = float(getattr(checkpoint, "timestamp", timestamp) or timestamp)
        age = max(0.0, timestamp - created)
        terms = _terms(query)
        metadata = getattr(checkpoint, "metadata", {}) or {}
        stats = getattr(checkpoint, "stats", {}) or {}
        fragments = getattr(checkpoint, "fragments", []) or []
        text_parts: list[str] = []
        for key in ("task", "query", "summary", "current_step"):
            value = metadata.get(key)
            if value:
                text_parts.append(str(value))
        continuity = metadata.get("continuity")
        if isinstance(continuity, Mapping):
            for value in continuity.values():
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    text_parts.extend(str(item) for item in value)
                else:
                    text_parts.append(str(value))
        for fragment in fragments[:50]:
            if isinstance(fragment, Mapping):
                text_parts.append(str(fragment.get("source", "")))
                text_parts.append(str(fragment.get("content", ""))[:500])
        haystack = "\n".join(text_parts).lower()
        matched = tuple(sorted(term for term in terms if term in haystack))
        decision_hits = _count_hits(continuity, "decisions", terms)
        path_hits = _count_hits(continuity, "modified_paths", terms)
        freshness = 0.5 ** (age / self.half_life_seconds)
        fragment_strength = min(1.0, math.log1p(len(fragments)) / math.log(101))
        try:
            stats_strength = min(1.0, float(stats.get("coverage_pct", 0.0)) / 100.0) if isinstance(stats, Mapping) else 0.0
        except (TypeError, ValueError):
            stats_strength = 0.0
        is_trusted = not bool(metadata.get("untrusted", False))
        trust = 1.0 if is_trusted else 0.35
        term_score = len(matched) / max(1, len(terms))
        continuity_score = min(1.0, 0.2 * decision_hits + 0.25 * path_hits)
        score = trust * (0.55 * term_score + 0.20 * continuity_score + 0.15 * freshness + 0.05 * fragment_strength + 0.05 * stats_strength)
        return CheckpointScore(str(getattr(checkpoint, "checkpoint_id", "")), score, matched, age, decision_hits, path_hits, is_trusted)

    def rank(self, checkpoints: Iterable[Any], query: str, *, now: float | None = None, limit: int = 5) -> list[CheckpointScore]:
        scored = [self.score(checkpoint, query, now=now) for checkpoint in checkpoints]
        scored.sort(key=lambda item: (item.score, -item.age_seconds, item.checkpoint_id), reverse=True)
        return scored[: max(0, limit)]


@dataclass(frozen=True, slots=True)
class CacheRetentionOption:
    name: str
    ttl_seconds: float
    write_multiplier: float
    read_multiplier: float


@dataclass(frozen=True, slots=True)
class CacheRetentionDecision:
    selected: str
    expected_cost_usd: Mapping[str, float]
    expected_savings_usd: Mapping[str, float]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "expected_cost_usd": {k: round(v, 9) for k, v in self.expected_cost_usd.items()},
            "expected_savings_usd": {k: round(v, 9) for k, v in self.expected_savings_usd.items()},
            "reason": self.reason,
        }


class CacheRetentionForecaster:
    def __init__(self, options: Iterable[CacheRetentionOption] | None = None) -> None:
        self.options = tuple(options or (
            CacheRetentionOption("none", 0.0, 1.0, 1.0),
            CacheRetentionOption("short", 300.0, 1.0, 0.1),
            CacheRetentionOption("long", 3600.0, 1.25, 0.1),
        ))
        if not self.options:
            raise ValueError("at least one cache option is required")

    def decide(self, *, prefix_tokens: int, input_price_per_million: float, pause_samples_seconds: Sequence[float], expected_future_turns: int = 1, min_savings_usd: float = 0.0) -> CacheRetentionDecision:
        if prefix_tokens < 0 or input_price_per_million < 0:
            raise ValueError("prefix tokens and price must be non-negative")
        turns = max(1, int(expected_future_turns))
        pauses = [max(0.0, float(pause)) for pause in pause_samples_seconds] or [float("inf")]
        base_write = prefix_tokens * input_price_per_million / 1_000_000.0
        expected_cost: dict[str, float] = {}
        expected_savings: dict[str, float] = {}
        for option in self.options:
            hit_probability = sum(1 for pause in pauses if pause <= option.ttl_seconds) / len(pauses)
            write_cost = base_write * option.write_multiplier
            read_cost = base_write * option.read_multiplier * hit_probability * turns
            miss_cost = base_write * (1.0 - hit_probability) * turns
            total = write_cost + read_cost + miss_cost
            no_cache_total = base_write * (1 + turns)
            expected_cost[option.name] = total
            expected_savings[option.name] = max(0.0, no_cache_total - total)
        selected = min(expected_cost, key=lambda name: (expected_cost[name], name))
        if expected_savings.get(selected, 0.0) < min_savings_usd:
            selected = "none" if "none" in expected_cost else selected
            reason = "forecast:no_material_savings"
        else:
            reason = "forecast:lowest_expected_cost"
        return CacheRetentionDecision(selected, expected_cost, expected_savings, reason)


@dataclass(frozen=True, slots=True)
class BehaviorEvent:
    kind: str
    key: str
    timestamp: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BehaviorWasteReport:
    repeated_errors: int
    repeated_tool_calls: int
    retry_loops: int
    model_switch_churn: int
    total_events: int

    @property
    def waste_score(self) -> float:
        weighted = 1.5 * self.repeated_errors + self.repeated_tool_calls + 1.25 * self.retry_loops + 1.25 * self.model_switch_churn
        return round(min(1.0, weighted / max(1, self.total_events)), 6)

    def as_dict(self) -> dict[str, Any]:
        return {
            "repeated_errors": self.repeated_errors,
            "repeated_tool_calls": self.repeated_tool_calls,
            "retry_loops": self.retry_loops,
            "model_switch_churn": self.model_switch_churn,
            "total_events": self.total_events,
            "waste_score": self.waste_score,
        }


class BehavioralWasteDetector:
    def __init__(self, *, window_seconds: float = 300.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.window_seconds = window_seconds
        self._events: list[BehaviorEvent] = []

    def record(self, kind: str, key: str, *, timestamp: float | None = None, metadata: Mapping[str, Any] | None = None) -> BehaviorWasteReport:
        ts = time.time() if timestamp is None else float(timestamp)
        self._events.append(BehaviorEvent(kind, key, ts, dict(metadata or {})))
        self._evict(ts)
        return self.report(now=ts)

    def report(self, *, now: float | None = None) -> BehaviorWasteReport:
        ts = time.time() if now is None else float(now)
        self._evict(ts)
        repeated_errors = _repeat_count(self._events, "error")
        repeated_tool_calls = _repeat_count(self._events, "tool_call")
        retry_loops = _repeat_count(self._events, "retry")
        model_switch_churn = _churn_count([event for event in self._events if event.kind == "model"])
        return BehaviorWasteReport(repeated_errors, repeated_tool_calls, retry_loops, model_switch_churn, len(self._events))

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._events = [event for event in self._events if event.timestamp >= cutoff]


def _terms(text: str) -> set[str]:
    return {match.group(0).lower() for match in _WORD_RE.finditer(text or "")}


def _clean_line(text: str) -> str:
    return " ".join((text or "").strip().split())[:300]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _count_hits(continuity: Any, key: str, terms: set[str]) -> int:
    if not isinstance(continuity, Mapping):
        return 0
    values = continuity.get(key, ())
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence):
        return 0
    count = 0
    for value in values:
        lowered = str(value).lower()
        if any(term in lowered for term in terms):
            count += 1
    return count


def _repeat_count(events: Sequence[BehaviorEvent], kind: str) -> int:
    counts: dict[str, int] = {}
    for event in events:
        if event.kind == kind:
            counts[event.key] = counts.get(event.key, 0) + 1
    return sum(max(0, count - 1) for count in counts.values())


def _churn_count(events: Sequence[BehaviorEvent]) -> int:
    if len(events) < 2:
        return 0
    switches = 0
    previous = events[0].key
    for event in events[1:]:
        if event.key != previous:
            switches += 1
            previous = event.key
    return switches


__all__ = [
    "BehaviorEvent",
    "BehaviorWasteReport",
    "BehavioralWasteDetector",
    "CacheRetentionDecision",
    "CacheRetentionForecaster",
    "CacheRetentionOption",
    "CheckpointRelevanceScorer",
    "CheckpointScore",
    "DecisionDigest",
    "RealizedSavingsRecord",
    "SavingsConfidence",
    "SavingsTierLedger",
    "extract_decision_digest",
]
