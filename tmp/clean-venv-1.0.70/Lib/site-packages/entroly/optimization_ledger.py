"""Durable accounting for optimization savings and opportunities.

Provider usage belongs in :mod:`entroly.usage_ledger`.  This ledger stores the
different fact pattern produced by optimizers: gross savings, later retrieval
cost, and estimates that must never be presented as realized invoice savings.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class SavingsTier(str, Enum):
    """Confidence class for a savings claim."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    OPPORTUNITY = "opportunity"


@dataclass(frozen=True, slots=True)
class OptimizationEvent:
    event_id: str
    feature: str
    tier: SavingsTier
    gross_tokens_saved: int
    gross_micro_usd: int = 0
    cost_micro_usd: int = 0
    session_id: str = ""
    conversation_id: str = ""
    provider: str = ""
    model: str = ""
    occurred_at: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.feature:
            raise ValueError("event_id and feature are required")
        if self.gross_tokens_saved < 0:
            raise ValueError("gross_tokens_saved must be non-negative")
        if self.gross_micro_usd < 0 or self.cost_micro_usd < 0:
            raise ValueError("money fields must be non-negative")


@dataclass(frozen=True, slots=True)
class OptimizationAdjustment:
    adjustment_id: str
    event_id: str
    tokens_reexpanded: int
    cost_micro_usd: int = 0
    occurred_at: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.adjustment_id or not self.event_id:
            raise ValueError("adjustment_id and event_id are required")
        if self.tokens_reexpanded < 0 or self.cost_micro_usd < 0:
            raise ValueError("adjustment values must be non-negative")


@dataclass(frozen=True, slots=True)
class SavingsSummary:
    measured_gross_tokens: int = 0
    measured_reexpanded_tokens: int = 0
    measured_net_tokens: int = 0
    measured_net_micro_usd: int = 0
    estimated_tokens: int = 0
    estimated_micro_usd: int = 0
    opportunity_tokens: int = 0
    opportunity_micro_usd: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class OptimizationLedger:
    """SQLite-backed, idempotent optimization accounting.

    Events and retrieval adjustments have independent idempotency keys.  A
    measured event can therefore be recorded before a later retrieval, while a
    replayed MCP request cannot double-charge the same adjustment when its
    request id is supplied.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS optimization_events (
                    event_id TEXT PRIMARY KEY,
                    feature TEXT NOT NULL,
                    tier TEXT NOT NULL CHECK(tier IN ('measured','estimated','opportunity')),
                    gross_tokens_saved INTEGER NOT NULL CHECK(gross_tokens_saved >= 0),
                    gross_micro_usd INTEGER NOT NULL CHECK(gross_micro_usd >= 0),
                    cost_micro_usd INTEGER NOT NULL CHECK(cost_micro_usd >= 0),
                    session_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    occurred_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS optimization_adjustments (
                    adjustment_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL REFERENCES optimization_events(event_id),
                    tokens_reexpanded INTEGER NOT NULL CHECK(tokens_reexpanded >= 0),
                    cost_micro_usd INTEGER NOT NULL CHECK(cost_micro_usd >= 0),
                    occurred_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_optimization_events_time
                    ON optimization_events(occurred_at);
                CREATE INDEX IF NOT EXISTS idx_optimization_adjustments_event
                    ON optimization_adjustments(event_id);
                """
            )

    def record(self, event: OptimizationEvent) -> bool:
        """Insert an event, returning False for an idempotent replay."""
        payload = json.dumps(dict(event.metadata), sort_keys=True, default=str)
        proposed = (
            event.feature,
            event.tier.value,
            event.gross_tokens_saved,
            event.gross_micro_usd,
            event.cost_micro_usd,
            event.session_id,
            event.conversation_id,
            event.provider,
            event.model,
            payload,
        )
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """INSERT OR IGNORE INTO optimization_events VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.feature,
                    event.tier.value,
                    event.gross_tokens_saved,
                    event.gross_micro_usd,
                    event.cost_micro_usd,
                    event.session_id,
                    event.conversation_id,
                    event.provider,
                    event.model,
                    event.occurred_at,
                    payload,
                ),
            )
            if cursor.rowcount == 1:
                return True
            existing = db.execute(
                "SELECT * FROM optimization_events WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            if existing is None:
                raise RuntimeError("idempotent optimization insert lost its row")
            identity = tuple(
                existing[key]
                for key in (
                    "feature", "tier", "gross_tokens_saved", "gross_micro_usd",
                    "cost_micro_usd", "session_id", "conversation_id", "provider",
                    "model", "metadata_json",
                )
            )
            if identity != proposed:
                raise ValueError("event_id already has different optimization data")
            return False

    def adjust(self, adjustment: OptimizationAdjustment) -> bool:
        """Record retrieval/re-expansion cost, idempotently."""
        payload = json.dumps(dict(adjustment.metadata), sort_keys=True, default=str)
        with self._lock, self._connect() as db:
            cursor = db.execute(
                """INSERT OR IGNORE INTO optimization_adjustments VALUES
                (?, ?, ?, ?, ?, ?)""",
                (
                    adjustment.adjustment_id,
                    adjustment.event_id,
                    adjustment.tokens_reexpanded,
                    adjustment.cost_micro_usd,
                    adjustment.occurred_at,
                    payload,
                ),
            )
            if cursor.rowcount == 1:
                return True
            existing = db.execute(
                "SELECT * FROM optimization_adjustments WHERE adjustment_id = ?",
                (adjustment.adjustment_id,),
            ).fetchone()
            if existing is None:
                raise RuntimeError("idempotent adjustment insert lost its row")
            identity = (
                existing["event_id"],
                existing["tokens_reexpanded"],
                existing["cost_micro_usd"],
                existing["metadata_json"],
            )
            proposed = (
                adjustment.event_id,
                adjustment.tokens_reexpanded,
                adjustment.cost_micro_usd,
                payload,
            )
            if identity != proposed:
                raise ValueError("adjustment_id already has different retrieval data")
            return False

    def event(self, event_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute(
                """SELECT e.*,
                    COALESCE(SUM(a.tokens_reexpanded), 0) AS tokens_reexpanded,
                    COALESCE(SUM(a.cost_micro_usd), 0) AS adjustment_micro_usd
                FROM optimization_events e
                LEFT JOIN optimization_adjustments a ON a.event_id = e.event_id
                WHERE e.event_id = ? GROUP BY e.event_id""",
                (event_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["net_tokens_saved"] = (
            result["gross_tokens_saved"] - result["tokens_reexpanded"]
        )
        result["net_micro_usd"] = (
            result["gross_micro_usd"]
            - result["cost_micro_usd"]
            - result["adjustment_micro_usd"]
        )
        return result

    def summary(self, *, since: float | None = None) -> SavingsSummary:
        where = "WHERE e.occurred_at >= ?" if since is not None else ""
        params: tuple[float, ...] = (float(since),) if since is not None else ()
        query = f"""
            SELECT e.tier,
                COALESCE(SUM(e.gross_tokens_saved), 0) gross_tokens,
                COALESCE(SUM(e.gross_micro_usd - e.cost_micro_usd), 0) base_micro_usd,
                COALESCE(SUM(a.tokens_reexpanded), 0) reexpanded_tokens,
                COALESCE(SUM(a.adjustment_micro_usd), 0) adjustment_micro_usd
            FROM optimization_events e
            LEFT JOIN (
                SELECT event_id,
                    SUM(tokens_reexpanded) tokens_reexpanded,
                    SUM(cost_micro_usd) adjustment_micro_usd
                FROM optimization_adjustments GROUP BY event_id
            ) a ON a.event_id = e.event_id
            {where}
            GROUP BY e.tier
        """
        by_tier: dict[str, sqlite3.Row] = {}
        with self._lock, self._connect() as db:
            for row in db.execute(query, params).fetchall():
                by_tier[str(row["tier"])] = row

        def value(tier: SavingsTier, key: str) -> int:
            row = by_tier.get(tier.value)
            return int(row[key]) if row is not None else 0

        measured_gross = value(SavingsTier.MEASURED, "gross_tokens")
        measured_reexpanded = value(SavingsTier.MEASURED, "reexpanded_tokens")
        return SavingsSummary(
            measured_gross_tokens=measured_gross,
            measured_reexpanded_tokens=measured_reexpanded,
            measured_net_tokens=measured_gross - measured_reexpanded,
            measured_net_micro_usd=(
                value(SavingsTier.MEASURED, "base_micro_usd")
                - value(SavingsTier.MEASURED, "adjustment_micro_usd")
            ),
            estimated_tokens=value(SavingsTier.ESTIMATED, "gross_tokens"),
            estimated_micro_usd=value(SavingsTier.ESTIMATED, "base_micro_usd"),
            opportunity_tokens=value(SavingsTier.OPPORTUNITY, "gross_tokens"),
            opportunity_micro_usd=value(SavingsTier.OPPORTUNITY, "base_micro_usd"),
        )


__all__ = [
    "OptimizationAdjustment",
    "OptimizationEvent",
    "OptimizationLedger",
    "SavingsSummary",
    "SavingsTier",
]
