"""Durable, idempotent delivery for Entroly operational event gateways.

The queue is intentionally local and dependency-free. It persists only bounded,
redacted payloads plus a SHA-256 destination fingerprint; webhook URLs, bot
tokens, credentials, and provider response bodies are never written to disk.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DELIVERY_SCHEMA = "entroly.event-delivery.v1"
RECEIPT_SCHEMA = "entroly.event-delivery-receipt.v1"
_MAX_TEXT_BYTES = 32_768
_MAX_ERROR_CHARS = 500

_SECRET_PATTERNS = (
    re.compile(r"https://hooks\.slack\.com/services/[^\s]+", re.IGNORECASE),
    re.compile(r"https://(?:canary\.|ptb\.)?discord(?:app)?\.com/api/webhooks/[^\s]+", re.IGNORECASE),
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+"),
)


class IdempotencyConflict(ValueError):
    """Raised when one idempotency key is reused with a different payload."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def redact_text(value: object, *, max_bytes: int = _MAX_TEXT_BYTES) -> str:
    """Return bounded single-string content with common credentials removed."""
    text = str(value or "").replace("\x00", "").replace("\r", "")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "…"


def _sanitize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the persisted event contract deliberately small and predictable."""
    text = redact_text(payload.get("text", ""))
    metadata_in = payload.get("metadata")
    metadata: dict[str, str] = {}
    if isinstance(metadata_in, Mapping):
        for key, value in list(metadata_in.items())[:32]:
            safe_key = re.sub(r"[^A-Za-z0-9_.:-]", "_", str(key))[:80]
            if safe_key:
                metadata[safe_key] = redact_text(value, max_bytes=1024)
    return {"text": text, "metadata": metadata}


@dataclass(frozen=True)
class DeliveryEvent:
    event_id: str
    channel: str
    destination_hash: str
    payload: dict[str, Any]
    payload_digest: str
    created_at: float
    not_before: float
    attempts: int
    max_attempts: int
    state: str
    lease_until: float | None
    last_error: str
    receipt_json: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DeliveryEvent":
        return cls(
            event_id=str(row["event_id"]),
            channel=str(row["channel"]),
            destination_hash=str(row["destination_hash"]),
            payload=json.loads(str(row["payload_json"])),
            payload_digest=str(row["payload_digest"]),
            created_at=float(row["created_at"]),
            not_before=float(row["not_before"]),
            attempts=int(row["attempts"]),
            max_attempts=int(row["max_attempts"]),
            state=str(row["state"]),
            lease_until=(
                float(row["lease_until"]) if row["lease_until"] is not None else None
            ),
            last_error=str(row["last_error"] or ""),
            receipt_json=str(row["receipt_json"] or ""),
        )


@dataclass(frozen=True)
class DeliveryReceipt:
    schema: str
    event_id: str
    channel: str
    destination_hash: str
    payload_digest: str
    attempts: int
    delivered_at: float
    provider_status: str
    receipt_digest: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "event_id": self.event_id,
            "channel": self.channel,
            "destination_hash": self.destination_hash,
            "payload_digest": self.payload_digest,
            "attempts": self.attempts,
            "delivered_at": self.delivered_at,
            "provider_status": self.provider_status,
        }

    def verify(self) -> bool:
        return _sha256_text(_canonical_json(self.payload())) == self.receipt_digest

    def to_json(self) -> str:
        value = self.payload()
        value["receipt_digest"] = self.receipt_digest
        return _canonical_json(value)

    @classmethod
    def from_json(cls, value: str) -> "DeliveryReceipt":
        raw = json.loads(value)
        return cls(**raw)


@dataclass(frozen=True)
class DeliveryOutcome:
    event_id: str
    state: str
    attempts: int
    inserted: bool = False
    receipt_digest: str = ""
    next_attempt_at: float | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.state == "delivered"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "queued": self.state == "pending",
            "dead": self.state == "dead",
            "state": self.state,
            "event_id": self.event_id,
            "attempts": self.attempts,
            "inserted": self.inserted,
            "receipt": self.receipt_digest,
            "next_attempt_at": self.next_attempt_at,
            "error": self.error,
        }


class EventDeliveryStore:
    """SQLite-backed event queue with leases, retries, and delivery receipts."""

    def __init__(
        self,
        path: str | Path = ".entroly/event-delivery.sqlite3",
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self._clock = clock
        self._schema_lock = threading.Lock()
        self._ready = False
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _ensure_schema(self) -> None:
        if self._ready:
            return
        with self._schema_lock:
            if self._ready:
                return
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS delivery_events (
                        event_id TEXT PRIMARY KEY,
                        channel TEXT NOT NULL,
                        destination_hash TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        payload_digest TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        not_before REAL NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        max_attempts INTEGER NOT NULL,
                        state TEXT NOT NULL CHECK(state IN ('pending','delivered','dead')),
                        lease_until REAL,
                        delivered_at REAL,
                        dead_at REAL,
                        last_error TEXT NOT NULL DEFAULT '',
                        receipt_json TEXT NOT NULL DEFAULT ''
                    );
                    CREATE INDEX IF NOT EXISTS idx_delivery_due
                    ON delivery_events(state, not_before, lease_until, created_at);
                    """
                )
            self._ready = True

    def enqueue(
        self,
        *,
        channel: str,
        destination_hash: str,
        payload: Mapping[str, Any],
        idempotency_key: str | None = None,
        max_attempts: int = 8,
        not_before: float | None = None,
    ) -> tuple[DeliveryEvent, bool]:
        if max_attempts < 1 or max_attempts > 100:
            raise ValueError("max_attempts must be between 1 and 100")
        clean_payload = _sanitize_payload(payload)
        payload_json = _canonical_json(clean_payload)
        payload_digest = _sha256_text(payload_json)
        key = idempotency_key or "auto:" + secrets.token_hex(16)
        event_id = _sha256_text(
            "\0".join((DELIVERY_SCHEMA, channel, destination_hash, key))
        )
        now = self._clock()
        due = now if not_before is None else max(now, float(not_before))

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO delivery_events (
                    event_id, channel, destination_hash, payload_json,
                    payload_digest, created_at, not_before, max_attempts, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    event_id,
                    channel,
                    destination_hash,
                    payload_json,
                    payload_digest,
                    now,
                    due,
                    max_attempts,
                ),
            )
            inserted = cursor.rowcount == 1
            row = connection.execute(
                "SELECT * FROM delivery_events WHERE event_id = ?", (event_id,)
            ).fetchone()
            connection.execute("COMMIT")

        if row is None:
            raise RuntimeError("delivery event disappeared after enqueue")
        event = DeliveryEvent.from_row(row)
        if event.payload_digest != payload_digest:
            raise IdempotencyConflict(
                "idempotency key already belongs to a different event payload"
            )
        return event, inserted

    def get(self, event_id: str) -> DeliveryEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM delivery_events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return DeliveryEvent.from_row(row) if row is not None else None

    def claim_due(
        self,
        *,
        channel: str,
        destination_hash: str,
        limit: int = 32,
        lease_seconds: float = 30.0,
    ) -> list[DeliveryEvent]:
        limit = max(1, min(int(limit), 1000))
        now = self._clock()
        lease_until = now + max(1.0, float(lease_seconds))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT event_id FROM delivery_events
                WHERE channel = ?
                  AND destination_hash = ?
                  AND state = 'pending'
                  AND not_before <= ?
                  AND (lease_until IS NULL OR lease_until <= ?)
                ORDER BY created_at, event_id
                LIMIT ?
                """,
                (channel, destination_hash, now, now, limit),
            ).fetchall()
            event_ids = [str(row["event_id"]) for row in rows]
            if event_ids:
                connection.executemany(
                    "UPDATE delivery_events SET lease_until = ? WHERE event_id = ?",
                    ((lease_until, event_id) for event_id in event_ids),
                )
                placeholders = ",".join("?" for _ in event_ids)
                claimed = connection.execute(
                    f"SELECT * FROM delivery_events WHERE event_id IN ({placeholders}) "
                    "ORDER BY created_at, event_id",
                    event_ids,
                ).fetchall()
            else:
                claimed = []
            connection.execute("COMMIT")
        return [DeliveryEvent.from_row(row) for row in claimed]

    def mark_delivered(
        self,
        event: DeliveryEvent,
        *,
        provider_status: object = "ok",
    ) -> DeliveryOutcome:
        now = self._clock()
        attempts = event.attempts + 1
        status = redact_text(provider_status, max_bytes=256)
        receipt_payload = {
            "schema": RECEIPT_SCHEMA,
            "event_id": event.event_id,
            "channel": event.channel,
            "destination_hash": event.destination_hash,
            "payload_digest": event.payload_digest,
            "attempts": attempts,
            "delivered_at": now,
            "provider_status": status,
        }
        digest = _sha256_text(_canonical_json(receipt_payload))
        receipt = DeliveryReceipt(receipt_digest=digest, **receipt_payload)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_events
                SET state = 'delivered', attempts = ?, delivered_at = ?,
                    lease_until = NULL, last_error = '', receipt_json = ?
                WHERE event_id = ? AND state = 'pending'
                """,
                (attempts, now, receipt.to_json(), event.event_id),
            )
        return DeliveryOutcome(
            event_id=event.event_id,
            state="delivered",
            attempts=attempts,
            receipt_digest=digest,
        )

    @staticmethod
    def _retry_delay(
        event_id: str,
        attempt: int,
        *,
        base_delay_s: float,
        max_delay_s: float,
    ) -> float:
        exponent = min(max(attempt - 1, 0), 20)
        raw = min(max_delay_s, base_delay_s * (2**exponent))
        seed = int(_sha256_text(f"{event_id}:{attempt}")[:8], 16) / 0xFFFFFFFF
        jitter = 0.85 + (0.30 * seed)
        return max(0.0, raw * jitter)

    def mark_failed(
        self,
        event: DeliveryEvent,
        *,
        error: object,
        base_delay_s: float,
        max_delay_s: float,
        retry_after_s: float | None = None,
    ) -> DeliveryOutcome:
        now = self._clock()
        attempts = event.attempts + 1
        error_text = redact_text(error, max_bytes=_MAX_ERROR_CHARS)
        if attempts >= event.max_attempts:
            state = "dead"
            next_attempt = None
        else:
            state = "pending"
            delay = self._retry_delay(
                event.event_id,
                attempts,
                base_delay_s=max(0.0, base_delay_s),
                max_delay_s=max(max_delay_s, base_delay_s),
            )
            if retry_after_s is not None:
                delay = max(delay, min(max_delay_s, max(0.0, retry_after_s)))
            next_attempt = now + delay

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE delivery_events
                SET state = ?, attempts = ?, not_before = ?, lease_until = NULL,
                    dead_at = ?, last_error = ?
                WHERE event_id = ? AND state = 'pending'
                """,
                (
                    state,
                    attempts,
                    next_attempt if next_attempt is not None else now,
                    now if state == "dead" else None,
                    error_text,
                    event.event_id,
                ),
            )
        return DeliveryOutcome(
            event_id=event.event_id,
            state=state,
            attempts=attempts,
            next_attempt_at=next_attempt,
            error=error_text,
        )

    def retry_dead(self, event_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE delivery_events
                SET state = 'pending', attempts = 0, not_before = ?,
                    lease_until = NULL, dead_at = NULL, last_error = ''
                WHERE event_id = ? AND state = 'dead'
                """,
                (self._clock(), event_id),
            )
        return cursor.rowcount == 1

    def stats(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM delivery_events GROUP BY state"
            ).fetchall()
        result = {"pending": 0, "delivered": 0, "dead": 0}
        for row in rows:
            result[str(row["state"])] = int(row["count"])
        result["total"] = sum(result.values())
        return result

    def prune_delivered(self, *, older_than_s: float = 30 * 86400) -> int:
        cutoff = self._clock() - max(0.0, older_than_s)
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM delivery_events WHERE state = 'delivered' AND delivered_at < ?",
                (cutoff,),
            )
        return int(cursor.rowcount)


Sender = Callable[[DeliveryEvent], Mapping[str, Any]]


class ReliableEventDispatcher:
    """Channel-neutral publish/flush facade used by Slack, Discord, and Telegram."""

    def __init__(
        self,
        *,
        channel: str,
        destination_identity: str,
        sender: Sender,
        db_path: str | Path = ".entroly/event-delivery.sqlite3",
        max_attempts: int = 8,
        base_delay_s: float = 2.0,
        max_delay_s: float = 900.0,
        lease_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.channel = channel
        self.destination_hash = _sha256_text(destination_identity)
        self._sender = sender
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._lease_seconds = lease_seconds
        self.store = EventDeliveryStore(db_path, clock=clock)
        self._flush_lock = threading.Lock()

    def publish(
        self,
        text: str,
        *,
        idempotency_key: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        immediate: bool = True,
    ) -> dict[str, Any]:
        event, inserted = self.store.enqueue(
            channel=self.channel,
            destination_hash=self.destination_hash,
            payload={"text": text, "metadata": dict(metadata or {})},
            idempotency_key=idempotency_key,
            max_attempts=self._max_attempts,
        )
        if immediate and event.state == "pending":
            self.flush()
        current = self.store.get(event.event_id) or event
        receipt_digest = ""
        if current.receipt_json:
            try:
                receipt_digest = DeliveryReceipt.from_json(
                    current.receipt_json
                ).receipt_digest
            except Exception:
                receipt_digest = ""
        return DeliveryOutcome(
            event_id=current.event_id,
            state=current.state,
            attempts=current.attempts,
            inserted=inserted,
            receipt_digest=receipt_digest,
            next_attempt_at=(current.not_before if current.state == "pending" else None),
            error=current.last_error,
        ).to_dict()

    def flush(self, *, limit: int = 64) -> list[DeliveryOutcome]:
        if not self._flush_lock.acquire(blocking=False):
            return []
        try:
            outcomes: list[DeliveryOutcome] = []
            for event in self.store.claim_due(
                channel=self.channel,
                destination_hash=self.destination_hash,
                limit=limit,
                lease_seconds=self._lease_seconds,
            ):
                try:
                    result = dict(self._sender(event))
                except Exception as exc:
                    result = {"ok": False, "error": str(exc)}
                if bool(result.get("ok")):
                    status = result.get("status", "ok")
                    outcomes.append(
                        self.store.mark_delivered(event, provider_status=status)
                    )
                else:
                    retry_after = result.get("retry_after_s")
                    try:
                        parsed_retry_after = (
                            float(retry_after) if retry_after is not None else None
                        )
                    except (TypeError, ValueError):
                        parsed_retry_after = None
                    outcomes.append(
                        self.store.mark_failed(
                            event,
                            error=result.get("error") or result.get("status") or "delivery_failed",
                            base_delay_s=self._base_delay_s,
                            max_delay_s=self._max_delay_s,
                            retry_after_s=parsed_retry_after,
                        )
                    )
            return outcomes
        finally:
            self._flush_lock.release()

    def stats(self) -> dict[str, int]:
        return self.store.stats()
