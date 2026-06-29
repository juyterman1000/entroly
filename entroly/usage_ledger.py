"""Durable provider-usage and spend ledger.

The ledger stores provider-reported token categories, not inferred savings. Costs
are represented as integer microdollars to avoid floating-point accounting
drift. Every request id is idempotent, and SQLite WAL provides transactional
cross-process durability for gateway and dashboard readers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class TokenUsage:
    uncached_input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.uncached_input_tokens,
                self.cache_read_tokens,
                self.cache_write_tokens,
                self.output_tokens,
            )
        ):
            raise ValueError("token usage cannot be negative")

    @property
    def total_input_tokens(self) -> int:
        return (
            self.uncached_input_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class UsagePricing:
    """USD per one million tokens, supplied from an auditable catalog."""

    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal
    cache_write_per_million: Decimal | None = None
    source: str = "explicit"

    @classmethod
    def from_values(
        cls,
        *,
        input_per_million: str | int | float,
        output_per_million: str | int | float,
        cache_read_per_million: str | int | float,
        cache_write_per_million: str | int | float | None = None,
        source: str = "explicit",
    ) -> "UsagePricing":
        return cls(
            input_per_million=Decimal(str(input_per_million)),
            output_per_million=Decimal(str(output_per_million)),
            cache_read_per_million=Decimal(str(cache_read_per_million)),
            cache_write_per_million=(
                None
                if cache_write_per_million is None
                else Decimal(str(cache_write_per_million))
            ),
            source=source,
        )

    def __post_init__(self) -> None:
        rates = (
            self.input_per_million,
            self.output_per_million,
            self.cache_read_per_million,
        )
        if any(rate < 0 for rate in rates):
            raise ValueError("pricing cannot be negative")
        if self.cache_write_per_million is not None and self.cache_write_per_million < 0:
            raise ValueError("cache-write pricing cannot be negative")

    @property
    def cache_write_rate(self) -> Decimal:
        return self.cache_write_per_million or self.input_per_million


@dataclass(frozen=True, slots=True)
class UsageEvent:
    request_id: str
    provider: str
    model: str
    usage: TokenUsage
    cost_micro_usd: int
    cache_savings_micro_usd: int
    occurred_at: float = field(default_factory=time.time)
    team: str = ""
    tool: str = ""
    project: str = ""
    conversation_id: str = ""
    pricing_source: str = "explicit"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def parse_provider_usage(provider: str, payload: Mapping[str, Any]) -> TokenUsage:
    """Normalize OpenAI, Anthropic, and Gemini usage payloads."""
    provider_key = provider.lower().strip()
    raw = payload.get("usage")
    if not isinstance(raw, Mapping):
        raw = payload.get("usage_metadata")
    if not isinstance(raw, Mapping):
        raw = payload

    if provider_key in {"openai", "azure_openai", "azure"}:
        input_total = _int(raw.get("input_tokens", raw.get("prompt_tokens")))
        output = _int(raw.get("output_tokens", raw.get("completion_tokens")))
        details = raw.get("input_tokens_details", raw.get("prompt_tokens_details"))
        cached = _int(details.get("cached_tokens")) if isinstance(details, Mapping) else 0
        return TokenUsage(
            uncached_input_tokens=max(0, input_total - cached),
            cache_read_tokens=min(cached, input_total),
            output_tokens=output,
        )

    if provider_key in {"anthropic", "claude"}:
        return TokenUsage(
            uncached_input_tokens=_int(raw.get("input_tokens")),
            cache_read_tokens=_int(raw.get("cache_read_input_tokens")),
            cache_write_tokens=_int(raw.get("cache_creation_input_tokens")),
            output_tokens=_int(raw.get("output_tokens")),
        )

    if provider_key in {"google", "gemini", "vertex"}:
        input_total = _int(
            raw.get("promptTokenCount", raw.get("prompt_token_count"))
        )
        cached = _int(
            raw.get("cachedContentTokenCount", raw.get("cached_content_token_count"))
        )
        output = _int(
            raw.get("candidatesTokenCount", raw.get("candidates_token_count"))
        )
        return TokenUsage(
            uncached_input_tokens=max(0, input_total - cached),
            cache_read_tokens=min(cached, input_total),
            output_tokens=output,
        )

    raise ValueError(f"unsupported provider usage format: {provider!r}")


def _micro_cost(tokens: int, rate_per_million: Decimal) -> int:
    # tokens * USD/1M * 1M microUSD/USD == tokens * rate.
    return int(
        (Decimal(tokens) * rate_per_million).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def price_usage(usage: TokenUsage, pricing: UsagePricing) -> tuple[int, int]:
    cost = (
        _micro_cost(usage.uncached_input_tokens, pricing.input_per_million)
        + _micro_cost(usage.cache_read_tokens, pricing.cache_read_per_million)
        + _micro_cost(usage.cache_write_tokens, pricing.cache_write_rate)
        + _micro_cost(usage.output_tokens, pricing.output_per_million)
    )
    no_cache_baseline = (
        _micro_cost(usage.total_input_tokens, pricing.input_per_million)
        + _micro_cost(usage.output_tokens, pricing.output_per_million)
    )
    return cost, max(0, no_cache_baseline - cost)


class UsageLedger:
    """Transactional, idempotent usage ledger backed by SQLite."""

    _FILTER_COLUMNS = frozenset(
        {"provider", "model", "team", "tool", "project", "conversation_id"}
    )

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS usage_events (
                request_id TEXT PRIMARY KEY,
                occurred_at REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                team TEXT NOT NULL,
                tool TEXT NOT NULL,
                project TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                uncached_input_tokens INTEGER NOT NULL,
                cache_read_tokens INTEGER NOT NULL,
                cache_write_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_micro_usd INTEGER NOT NULL,
                cache_savings_micro_usd INTEGER NOT NULL,
                pricing_source TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_usage_time
                ON usage_events(occurred_at);
            CREATE INDEX IF NOT EXISTS idx_usage_conversation
                ON usage_events(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_usage_dimensions
                ON usage_events(team, tool, project, provider, model);
            """
        )
        self._conn.commit()

    def record(self, event: UsageEvent) -> bool:
        """Insert once; return False when request_id was already recorded."""
        if not event.request_id:
            raise ValueError("request_id is required")
        row = (
            event.request_id,
            float(event.occurred_at),
            event.provider,
            event.model,
            event.team,
            event.tool,
            event.project,
            event.conversation_id,
            event.usage.uncached_input_tokens,
            event.usage.cache_read_tokens,
            event.usage.cache_write_tokens,
            event.usage.output_tokens,
            event.cost_micro_usd,
            event.cache_savings_micro_usd,
            event.pricing_source,
            json.dumps(
                dict(event.metadata),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO usage_events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                row,
            )
            return cursor.rowcount == 1

    def record_provider_payload(
        self,
        *,
        request_id: str,
        provider: str,
        model: str,
        payload: Mapping[str, Any],
        pricing: UsagePricing,
        occurred_at: float | None = None,
        team: str = "",
        tool: str = "",
        project: str = "",
        conversation_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> UsageEvent:
        usage = parse_provider_usage(provider, payload)
        cost, savings = price_usage(usage, pricing)
        event = UsageEvent(
            request_id=request_id,
            provider=provider,
            model=model,
            usage=usage,
            cost_micro_usd=cost,
            cache_savings_micro_usd=savings,
            occurred_at=time.time() if occurred_at is None else occurred_at,
            team=team,
            tool=tool,
            project=project,
            conversation_id=conversation_id,
            pricing_source=pricing.source,
            metadata=dict(metadata or {}),
        )
        self.record(event)
        return event

    def summary(self, **filters: str) -> dict[str, int | float]:
        unknown = set(filters) - self._FILTER_COLUMNS
        if unknown:
            raise ValueError(f"unsupported filters: {sorted(unknown)}")

        clauses: list[str] = []
        values: list[str] = []
        for column, value in sorted(filters.items()):
            clauses.append(f"{column} = ?")
            values.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._lock:
            row = self._conn.execute(
                f"""
                SELECT
                    COUNT(*) AS requests,
                    COALESCE(SUM(uncached_input_tokens), 0) AS uncached_input_tokens,
                    COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                    COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                    COALESCE(SUM(output_tokens), 0) AS output_tokens,
                    COALESCE(SUM(cost_micro_usd), 0) AS cost_micro_usd,
                    COALESCE(SUM(cache_savings_micro_usd), 0)
                        AS cache_savings_micro_usd
                FROM usage_events
                {where}
                """,
                values,
            ).fetchone()

        result = {key: int(row[key]) for key in row.keys()}
        cached = result["cache_read_tokens"]
        uncached = result["uncached_input_tokens"]
        result["cache_hit_token_ratio"] = cached / max(1, cached + uncached)
        return result

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "UsageLedger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = [
    "TokenUsage",
    "UsageEvent",
    "UsageLedger",
    "UsagePricing",
    "parse_provider_usage",
    "price_usage",
]
