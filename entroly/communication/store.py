"""Private, durable storage for communication evidence and action state."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .models import (
    AssuranceDecision,
    CommunicationActionProposal,
    CommunicationEvent,
    CommunicationStateConflict,
    CommunicationStateError,
    MAX_METADATA_BYTES,
    SCHEMA,
    bounded_metadata,
    bounded_string,
    canonical_json,
)
from .preferences import CommunicationTaste

DEFAULT_RETENTION_DAYS = 90
MAX_RETENTION_DAYS = 3650


def default_store_path() -> Path:
    """Use user state, never the project/workspace, for private communication."""
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        root = Path(os.environ["LOCALAPPDATA"]) / "Entroly"
    else:
        root = Path(
            os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
        ) / "entroly"
    return root.expanduser().absolute() / "communication" / "communication.sqlite3"


def resolve_store_path(path: str | os.PathLike[str] | None) -> Path:
    if path is None or not str(path).strip():
        return default_store_path()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise CommunicationStateError(
            "communication store override must be an absolute path"
        )
    return candidate.absolute()


def normalize_retention_days(value: Any) -> int:
    if value is None:
        return DEFAULT_RETENTION_DAYS
    if isinstance(value, bool):
        raise CommunicationStateError("retention_days must be an integer")
    try:
        days = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CommunicationStateError("retention_days must be an integer") from exc
    if days < 0:
        raise CommunicationStateError("retention_days cannot be negative")
    return min(days, MAX_RETENTION_DAYS)


class CommunicationStore:
    """Transactional local store with exact-scope retrieval and idempotency."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        retention_days: int | None = None,
    ) -> None:
        self.path = resolve_store_path(path)
        self.retention_days = normalize_retention_days(retention_days)
        self._lock = threading.RLock()
        self._secure_parent()
        self._conn = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA secure_delete=ON")
            self._create_schema()
        self._secure_db()

    def _secure_parent(self) -> None:
        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            if parent.is_symlink() or not parent.is_dir():
                raise CommunicationStateError(
                    f"unsafe communication state directory: {parent}"
                )
            if self.path.exists() and self.path.is_symlink():
                raise CommunicationStateError(
                    f"refusing symlink communication database: {self.path}"
                )
            if os.name == "posix":
                parent.chmod(0o700)
        except OSError as exc:
            raise CommunicationStateError(
                f"cannot secure communication state directory: {exc}"
            ) from exc

    def _secure_db(self) -> None:
        if os.name != "posix":
            return
        try:
            self.path.chmod(0o600)
        except OSError as exc:
            raise CommunicationStateError(
                f"cannot secure communication database: {exc}"
            ) from exc

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS communication_events (
                event_id TEXT PRIMARY KEY,
                schema TEXT NOT NULL,
                direction TEXT NOT NULL,
                channel TEXT NOT NULL,
                account_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                conversation_kind TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                reply_to_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_timestamp REAL,
                observed_at REAL NOT NULL,
                content TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                delivery_state TEXT NOT NULL,
                identity_strength TEXT NOT NULL,
                provider_update_id TEXT NOT NULL,
                provider_update_kind TEXT NOT NULL,
                source TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                commitment_sha256 TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_communication_scope_time
              ON communication_events(
                channel, account_id, conversation_id, observed_at
              );
            CREATE INDEX IF NOT EXISTS idx_communication_sender_time
              ON communication_events(channel, account_id, sender_id, observed_at);
            CREATE INDEX IF NOT EXISTS idx_communication_message
              ON communication_events(
                channel, account_id, conversation_id, message_id
              );

            CREATE TABLE IF NOT EXISTS communication_actions (
                action_id TEXT PRIMARY KEY,
                action_type TEXT NOT NULL,
                channel TEXT NOT NULL,
                account_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                source_event_ids_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                category TEXT NOT NULL,
                risk_class TEXT NOT NULL,
                creates_commitment INTEGER NOT NULL,
                conversation_kind TEXT NOT NULL,
                policy_decision TEXT NOT NULL,
                policy_reasons_json TEXT NOT NULL,
                execution_state TEXT NOT NULL,
                outbound_message_id TEXT NOT NULL,
                error TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_communication_action_scope
              ON communication_actions(
                channel, account_id, conversation_id, updated_at
              );

            CREATE TABLE IF NOT EXISTS communication_preferences (
                scope_type TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                profile_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(scope_type, scope_id)
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "CommunicationStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def record_event(
        self,
        event: CommunicationEvent,
        *,
        observed_at: float | None = None,
    ) -> bool:
        now = time.time() if observed_at is None else float(observed_at)
        if not math.isfinite(now) or now < 0:
            raise CommunicationStateError(
                "observed_at must be finite and non-negative"
            )
        metadata_json = canonical_json(bounded_metadata(event.metadata))
        if len(metadata_json.encode("utf-8")) > MAX_METADATA_BYTES:
            raise CommunicationStateError("communication metadata is too large")
        row = (
            event.event_id,
            event.schema,
            event.direction,
            event.channel,
            event.account_id,
            event.conversation_id,
            event.conversation_kind,
            event.sender_id,
            event.recipient_id,
            event.message_id,
            event.reply_to_id,
            event.session_key,
            event.run_id,
            event.event_type,
            event.timestamp,
            now,
            event.content,
            event.content_sha256,
            event.delivery_state,
            event.identity_strength,
            event.provider_update_id,
            event.provider_update_kind,
            event.source,
            metadata_json,
            event.commitment_sha256,
        )
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO communication_events VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                row,
            )
            inserted = cursor.rowcount == 1
            if not inserted:
                existing = self._conn.execute(
                    """
                    SELECT commitment_sha256
                    FROM communication_events WHERE event_id = ?
                    """,
                    (event.event_id,),
                ).fetchone()
                if existing is None:
                    raise CommunicationStateError(
                        "idempotent insert lost stored event"
                    )
                if existing["commitment_sha256"] != event.commitment_sha256:
                    raise CommunicationStateConflict(
                        "stable communication event identity was reused "
                        "with different evidence"
                    )
        self.prune(now=now)
        return inserted

    def get_event(self, event_id: str) -> CommunicationEvent | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM communication_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return None if row is None else self._row_to_event(row)

    def events_for_scope(
        self,
        *,
        channel: str,
        conversation_id: str,
        account_id: str = "",
        limit: int = 200,
    ) -> list[CommunicationEvent]:
        """Retrieve only an exact authorized conversation scope."""
        if not channel or not conversation_id:
            raise CommunicationStateError(
                "channel and conversation_id are required for scoped retrieval"
            )
        count = max(1, min(int(limit), 10_000))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM communication_events
                WHERE channel = ? AND account_id = ? AND conversation_id = ?
                ORDER BY observed_at DESC, event_id DESC
                LIMIT ?
                """,
                (channel, account_id, conversation_id, count),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def recent_events(
        self,
        *,
        channel: str = "",
        account_id: str | None = None,
        limit: int = 1000,
    ) -> list[CommunicationEvent]:
        """Owner-review retrieval across scopes; callers must enforce authorization."""
        count = max(1, min(int(limit), 10_000))
        clauses: list[str] = []
        values: list[Any] = []
        if channel:
            clauses.append("channel = ?")
            values.append(channel)
        if account_id is not None:
            clauses.append("account_id = ?")
            values.append(account_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT * FROM communication_events
                {where}
                ORDER BY observed_at DESC, event_id DESC
                LIMIT ?
                """,
                (*values, count),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def _row_to_event(self, row: sqlite3.Row) -> CommunicationEvent:
        return CommunicationEvent(
            event_id=row["event_id"],
            schema=row["schema"],
            direction=row["direction"],
            channel=row["channel"],
            account_id=row["account_id"],
            conversation_id=row["conversation_id"],
            conversation_kind=row["conversation_kind"],
            sender_id=row["sender_id"],
            recipient_id=row["recipient_id"],
            message_id=row["message_id"],
            reply_to_id=row["reply_to_id"],
            session_key=row["session_key"],
            run_id=row["run_id"],
            event_type=row["event_type"],
            timestamp=row["event_timestamp"],
            content=row["content"],
            delivery_state=row["delivery_state"],
            identity_strength=row["identity_strength"],
            provider_update_id=row["provider_update_id"],
            provider_update_kind=row["provider_update_kind"],
            source=row["source"],
            metadata=json.loads(row["metadata_json"]),
        )

    def prune(self, *, now: float | None = None) -> int:
        """Apply the explicit raw-evidence retention window."""
        if self.retention_days <= 0:
            return 0
        timestamp = time.time() if now is None else float(now)
        cutoff = timestamp - (self.retention_days * 24 * 60 * 60)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM communication_events WHERE observed_at < ?",
                (cutoff,),
            )
            return max(0, cursor.rowcount)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            totals = self._conn.execute(
                """
                SELECT
                  COUNT(*) AS total,
                  SUM(CASE WHEN direction = 'inbound' THEN 1 ELSE 0 END)
                    AS inbound,
                  SUM(CASE WHEN direction = 'outbound' THEN 1 ELSE 0 END)
                    AS outbound,
                  COUNT(DISTINCT (
                    channel || char(31) || account_id ||
                    char(31) || conversation_id
                  )) AS conversations,
                  MIN(observed_at) AS oldest,
                  MAX(observed_at) AS newest
                FROM communication_events
                """
            ).fetchone()
        return {
            "schema": SCHEMA,
            "retention_days": self.retention_days,
            "events": int(totals["total"] or 0),
            "inbound": int(totals["inbound"] or 0),
            "outbound": int(totals["outbound"] or 0),
            "conversations": int(totals["conversations"] or 0),
            "oldest_observed_at": totals["oldest"],
            "newest_observed_at": totals["newest"],
        }

    def set_explicit_taste(
        self,
        taste: CommunicationTaste,
        *,
        now: float | None = None,
    ) -> None:
        """Replace current explicit preference state for one exact scope."""
        if taste.source != "explicit":
            raise CommunicationStateError(
                "only explicit taste belongs in authoritative preference state"
            )
        timestamp = time.time() if now is None else float(now)
        payload = canonical_json(taste.to_dict())
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO communication_preferences(
                    scope_type, scope_id, profile_json, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(scope_type, scope_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = excluded.updated_at
                """,
                (taste.scope_type, taste.scope_id, payload, timestamp),
            )

    def get_explicit_taste(
        self,
        *,
        scope_type: str,
        scope_id: str,
    ) -> CommunicationTaste | None:
        """Read current explicit preference state for one exact scope."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT profile_json
                FROM communication_preferences
                WHERE scope_type = ? AND scope_id = ?
                """,
                (str(scope_type), str(scope_id)),
            ).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(str(row["profile_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CommunicationStateError(
                "stored communication preference is corrupted"
            ) from exc
        if not isinstance(payload, dict):
            raise CommunicationStateError(
                "stored communication preference is corrupted"
            )
        try:
            return CommunicationTaste.build(
                scope_type=str(payload.get("scope_type") or ""),  # type: ignore[arg-type]
                scope_id=str(payload.get("scope_id") or ""),
                source="explicit",
                confidence=float(payload.get("confidence", 1.0)),
                evidence_event_ids=tuple(
                    str(item)
                    for item in payload.get("evidence_event_ids", [])
                    if str(item)
                )
                if isinstance(payload.get("evidence_event_ids"), list)
                else (),
                preferred_language=str(
                    payload.get("preferred_language") or "adaptive"
                ),
                formality=str(payload.get("formality") or "adaptive"),
                response_length=str(
                    payload.get("response_length") or "adaptive"
                ),
                emoji_level=str(payload.get("emoji_level") or "adaptive"),
                routine_action=str(payload.get("routine_action") or "none"),
                preferred_reaction=str(
                    payload.get("preferred_reaction") or ""
                ),
                greeting_style=str(payload.get("greeting_style") or ""),
                signoff_style=str(payload.get("signoff_style") or ""),
                notes=payload.get("notes")
                if isinstance(payload.get("notes"), dict)
                else {},
            )
        except (CommunicationStateError, TypeError, ValueError) as exc:
            raise CommunicationStateError(
                "stored communication preference is invalid"
            ) from exc

    def explicit_tastes_for_scopes(
        self,
        scopes: Sequence[tuple[str, str]],
    ) -> list[CommunicationTaste]:
        tastes: list[CommunicationTaste] = []
        for scope_type, scope_id in scopes:
            taste = self.get_explicit_taste(
                scope_type=scope_type,
                scope_id=scope_id,
            )
            if taste is not None:
                tastes.append(taste)
        return tastes

    def record_action(
        self,
        proposal: CommunicationActionProposal,
        *,
        decision: AssuranceDecision,
        reasons: Sequence[str],
        execution_state: str = "proposed",
        now: float | None = None,
    ) -> bool:
        """Persist a proposal only when all source evidence is in one scope."""
        if proposal.source_event_ids:
            placeholders = ",".join("?" for _ in proposal.source_event_ids)
            with self._lock:
                rows = self._conn.execute(
                    f"""
                    SELECT event_id, channel, account_id, conversation_id
                    FROM communication_events
                    WHERE event_id IN ({placeholders})
                    """,
                    proposal.source_event_ids,
                ).fetchall()
            if {row["event_id"] for row in rows} != set(
                proposal.source_event_ids
            ):
                raise CommunicationStateError(
                    "action source_event_ids must all exist in the evidence store"
                )
            if any(
                row["channel"] != proposal.channel
                or row["account_id"] != proposal.account_id
                or row["conversation_id"] != proposal.conversation_id
                for row in rows
            ):
                raise CommunicationStateError(
                    "action source evidence crosses conversation scope"
                )

        timestamp = time.time() if now is None else float(now)
        row = (
            proposal.action_id,
            proposal.action_type,
            proposal.channel,
            proposal.account_id,
            proposal.conversation_id,
            canonical_json(list(proposal.source_event_ids)),
            proposal.payload_sha256,
            proposal.category,
            proposal.risk_class,
            1 if proposal.creates_commitment else 0,
            proposal.conversation_kind,
            decision,
            canonical_json(sorted(set(str(reason) for reason in reasons))),
            execution_state,
            "",
            "",
            timestamp,
            timestamp,
        )
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO communication_actions VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                row,
            )
            if cursor.rowcount == 1:
                return True
            existing = self._conn.execute(
                """
                SELECT action_type, channel, account_id, conversation_id,
                       source_event_ids_json, payload_sha256, category, risk_class,
                       creates_commitment, conversation_kind,
                       execution_state
                FROM communication_actions WHERE action_id = ?
                """,
                (proposal.action_id,),
            ).fetchone()
            if existing is None:
                raise CommunicationStateError(
                    "idempotent action insert lost stored row"
                )
            if tuple(existing[:10]) != row[1:11]:
                raise CommunicationStateConflict(
                    "stable action identity was reused with different content"
                )
            # A successfully observed send is terminal for duplicate prevention.
            # Other states may be re-evaluated after an explicit policy change.
            if existing["execution_state"] != "sent":
                self._conn.execute(
                    """
                    UPDATE communication_actions
                    SET policy_decision = ?, policy_reasons_json = ?,
                        execution_state = ?, updated_at = ?
                    WHERE action_id = ? AND execution_state != 'sent'
                    """,
                    (
                        decision,
                        row[12],
                        execution_state,
                        timestamp,
                        proposal.action_id,
                    ),
                )
            return False

    def action_is_handled(self, action_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT execution_state
                FROM communication_actions WHERE action_id = ?
                """,
                (action_id,),
            ).fetchone()
        return row is not None and row["execution_state"] == "sent"

    def correlate_outbound_event(
        self,
        event: CommunicationEvent,
        *,
        now: float | None = None,
        window_seconds: float = 300.0,
        error: str = "",
    ) -> str | None:
        """Bind one observed outbound delivery to exactly one allowed proposal.

        Ambiguous matches are intentionally left unresolved rather than risking
        a false handled state.
        """
        if event.direction != "outbound" or not event.content_sha256:
            return None
        timestamp = time.time() if now is None else float(now)
        cutoff = timestamp - max(1.0, float(window_seconds))
        with self._lock, self._conn:
            rows = self._conn.execute(
                """
                SELECT action_id
                FROM communication_actions
                WHERE channel = ?
                  AND account_id = ?
                  AND conversation_id = ?
                  AND payload_sha256 = ?
                  AND policy_decision = 'allow'
                  AND execution_state = 'dispatching'
                  AND updated_at >= ?
                ORDER BY updated_at DESC, action_id DESC
                LIMIT 2
                """,
                (
                    event.channel,
                    event.account_id,
                    event.conversation_id,
                    event.content_sha256,
                    cutoff,
                ),
            ).fetchall()
            if len(rows) != 1:
                return None
            action_id = str(rows[0]["action_id"])
            self._conn.execute(
                """
                UPDATE communication_actions
                SET execution_state = ?, outbound_message_id = ?,
                    error = ?, updated_at = ?
                WHERE action_id = ?
                """,
                (
                    "sent" if event.delivery_state == "sent" else "failed",
                    bounded_string(event.message_id, 1024),
                    bounded_string(error, 4096),
                    timestamp,
                    action_id,
                ),
            )
            return action_id

    def begin_action_execution(
        self,
        action_id: str,
        *,
        now: float | None = None,
    ) -> bool:
        """Atomically claim one assured action for dispatch.

        False means the action is not in the exact dispatchable state. This
        provides duplicate suppression across retries/restarts; callers must
        never send when False is returned.
        """
        timestamp = time.time() if now is None else float(now)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE communication_actions
                SET execution_state = 'dispatching', updated_at = ?
                WHERE action_id = ?
                  AND policy_decision = 'allow'
                  AND execution_state = 'assured'
                """,
                (timestamp, action_id),
            )
            return cursor.rowcount == 1

    def fail_dispatch(
        self,
        action_id: str,
        *,
        error: str,
        now: float | None = None,
    ) -> bool:
        """Record an observed host dispatch exception without reopening send."""
        timestamp = time.time() if now is None else float(now)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE communication_actions
                SET execution_state = 'failed', error = ?, updated_at = ?
                WHERE action_id = ? AND execution_state = 'dispatching'
                """,
                (bounded_string(error, 4096), timestamp, action_id),
            )
            return cursor.rowcount == 1

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM communication_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["source_event_ids"] = json.loads(result.pop("source_event_ids_json"))
        result["policy_reasons"] = json.loads(result.pop("policy_reasons_json"))
        result["creates_commitment"] = bool(result["creates_commitment"])
        return result

    def record_action_outcome(
        self,
        action_id: str,
        *,
        success: bool,
        outbound_message_id: str = "",
        error: str = "",
        now: float | None = None,
    ) -> None:
        timestamp = time.time() if now is None else float(now)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE communication_actions
                SET execution_state = ?, outbound_message_id = ?,
                    error = ?, updated_at = ?
                WHERE action_id = ?
                """,
                (
                    "sent" if success else "failed",
                    bounded_string(outbound_message_id, 1024),
                    bounded_string(error, 4096),
                    timestamp,
                    action_id,
                ),
            )
            if cursor.rowcount != 1:
                raise CommunicationStateError(f"unknown action_id: {action_id}")
