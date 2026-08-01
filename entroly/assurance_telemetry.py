"""Local observability for assurance-gated compression decisions.

The ledger is deliberately local-only and stores decision metadata, hashes,
counts, and timings. It never stores raw context or model output. Callers opt in
by constructing :class:`AssuranceLedger`; importing Entroly creates no files.
"""
from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assurance_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL,
    decision TEXT NOT NULL,
    certificate_scope TEXT NOT NULL,
    certificate_verdict TEXT NOT NULL,
    requested_budget INTEGER NOT NULL,
    original_tokens INTEGER NOT NULL,
    delivered_tokens INTEGER NOT NULL,
    exact_identity INTEGER NOT NULL,
    budget_compliant INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    input_sha256 TEXT NOT NULL,
    output_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assurance_created_at
    ON assurance_events(created_at);
CREATE INDEX IF NOT EXISTS idx_assurance_decision
    ON assurance_events(decision);
"""


@dataclass(frozen=True)
class AssuranceSummary:
    events: int
    decision_counts: dict[str, int]
    scope_counts: dict[str, int]
    verdict_counts: dict[str, int]
    accepted_events: int
    bypass_events: int
    identity_events: int
    budget_compliant_events: int
    accepted_coverage: float
    bypass_rate: float
    mean_token_savings: float
    p50_latency_ms: float
    p95_latency_ms: float
    latest_created_at: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "decision_counts": dict(self.decision_counts),
            "scope_counts": dict(self.scope_counts),
            "verdict_counts": dict(self.verdict_counts),
            "accepted_events": self.accepted_events,
            "bypass_events": self.bypass_events,
            "identity_events": self.identity_events,
            "budget_compliant_events": self.budget_compliant_events,
            "accepted_coverage": self.accepted_coverage,
            "bypass_rate": self.bypass_rate,
            "mean_token_savings": self.mean_token_savings,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "latest_created_at": self.latest_created_at,
        }


def _bounded_metadata(metadata: Mapping[str, Any] | None) -> str:
    if not metadata:
        return "{}"
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        name = str(key)[:80]
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[name] = value if not isinstance(value, str) else value[:500]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            safe[name] = [str(item)[:120] for item in list(value)[:20]]
        else:
            safe[name] = str(value)[:500]
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class AssuranceLedger:
    """Thread-safe SQLite ledger containing no raw user context."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def record(
        self,
        *,
        kind: str,
        decision: str,
        requested_budget: int,
        original_tokens: int,
        delivered_tokens: int,
        exact_identity: bool,
        budget_compliant: bool,
        latency_ms: float,
        input_sha256: str,
        output_sha256: str,
        certificate_scope: str = "",
        certificate_verdict: str = "",
        metadata: Mapping[str, Any] | None = None,
        created_at: float | None = None,
    ) -> int:
        if requested_budget < 0 or original_tokens < 0 or delivered_tokens < 0:
            raise ValueError("token and budget fields must be non-negative")
        if latency_ms < 0 or not math.isfinite(latency_ms):
            raise ValueError("latency_ms must be finite and non-negative")
        timestamp = time.time() if created_at is None else float(created_at)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO assurance_events (
                    created_at, kind, decision, certificate_scope,
                    certificate_verdict, requested_budget, original_tokens,
                    delivered_tokens, exact_identity, budget_compliant,
                    latency_ms, input_sha256, output_sha256, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    str(kind)[:80],
                    str(decision)[:120],
                    str(certificate_scope)[:80],
                    str(certificate_verdict)[:80],
                    int(requested_budget),
                    int(original_tokens),
                    int(delivered_tokens),
                    int(bool(exact_identity)),
                    int(bool(budget_compliant)),
                    float(latency_ms),
                    str(input_sha256)[:128],
                    str(output_sha256)[:128],
                    _bounded_metadata(metadata),
                ),
            )
            return int(cursor.lastrowid)

    def record_selection(
        self,
        receipt: Mapping[str, Any],
        *,
        latency_ms: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        attempts = receipt.get("attempts") or []
        final_attempt = attempts[-1] if attempts and isinstance(attempts[-1], Mapping) else {}
        return self.record(
            kind="selection",
            decision=str(receipt.get("decision") or ""),
            certificate_scope=str(
                final_attempt.get("certificate_scope")
                or receipt.get("required_scope")
                or ""
            ),
            certificate_verdict=str(final_attempt.get("certificate_verdict") or ""),
            requested_budget=int(receipt.get("requested_budget") or 0),
            original_tokens=int(receipt.get("raw_tokens") or 0),
            delivered_tokens=int(receipt.get("delivered_tokens") or 0),
            exact_identity=bool(receipt.get("exact_identity")),
            budget_compliant=bool(receipt.get("budget_compliant")),
            latency_ms=latency_ms,
            input_sha256=str(receipt.get("input_sha256") or ""),
            output_sha256=str(receipt.get("output_sha256") or ""),
            metadata=metadata,
        )

    def record_domain(
        self,
        receipt: Mapping[str, Any],
        *,
        latency_ms: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        validation = receipt.get("validation")
        validation = validation if isinstance(validation, Mapping) else {}
        combined = dict(metadata or {})
        combined.setdefault("content_type", receipt.get("content_type"))
        combined.setdefault("validation_reasons", validation.get("reasons") or [])
        return self.record(
            kind="domain",
            decision=str(receipt.get("decision") or ""),
            certificate_scope="domain_validation",
            certificate_verdict="sufficient" if validation.get("valid") else "degraded",
            requested_budget=int(receipt.get("requested_budget") or 0),
            original_tokens=int(receipt.get("original_tokens") or 0),
            delivered_tokens=int(receipt.get("emitted_tokens") or 0),
            exact_identity=bool(receipt.get("exact_identity")),
            budget_compliant=bool(receipt.get("budget_compliant")),
            latency_ms=latency_ms,
            input_sha256=str(receipt.get("input_sha256") or ""),
            output_sha256=str(receipt.get("output_sha256") or ""),
            metadata=combined,
        )

    def summary(self, *, since: float | None = None, limit: int = 100_000) -> AssuranceSummary:
        if limit <= 0:
            raise ValueError("limit must be positive")
        query = (
            "SELECT created_at, decision, certificate_scope, certificate_verdict, "
            "original_tokens, delivered_tokens, exact_identity, budget_compliant, latency_ms "
            "FROM assurance_events"
        )
        params: list[Any] = []
        if since is not None:
            query += " WHERE created_at >= ?"
            params.append(float(since))
        query += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))
        with self._lock, self._connect() as connection:
            rows = list(connection.execute(query, params))

        decision_counts: dict[str, int] = {}
        scope_counts: dict[str, int] = {}
        verdict_counts: dict[str, int] = {}
        latencies: list[float] = []
        savings: list[float] = []
        accepted = bypassed = identities = compliant = 0
        latest: float | None = None
        for created_at, decision, scope, verdict, original, delivered, identity, budget_ok, latency in rows:
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            latencies.append(float(latency))
            identities += int(bool(identity))
            compliant += int(bool(budget_ok))
            upper_decision = str(decision).upper()
            if "CERTIFIED" in upper_decision or "VALIDATED" in upper_decision:
                accepted += 1
            if "BYPASS" in upper_decision:
                bypassed += 1
            if original:
                savings.append(1.0 - float(delivered) / float(original))
            latest = max(latest or float(created_at), float(created_at))

        count = len(rows)
        return AssuranceSummary(
            events=count,
            decision_counts=decision_counts,
            scope_counts=scope_counts,
            verdict_counts=verdict_counts,
            accepted_events=accepted,
            bypass_events=bypassed,
            identity_events=identities,
            budget_compliant_events=compliant,
            accepted_coverage=accepted / count if count else 0.0,
            bypass_rate=bypassed / count if count else 0.0,
            mean_token_savings=sum(savings) / len(savings) if savings else 0.0,
            p50_latency_ms=_percentile(latencies, 0.50),
            p95_latency_ms=_percentile(latencies, 0.95),
            latest_created_at=latest,
        )

    def prune(self, *, before: float) -> int:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM assurance_events WHERE created_at < ?", (float(before),)
            )
            return int(cursor.rowcount)


__all__ = ["AssuranceLedger", "AssuranceSummary"]
