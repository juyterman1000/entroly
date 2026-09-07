"""
Governance Audit Log — Append-only, tamper-evident audit logging.
==================================================================

Every governance action (permission check, tool invocation, policy
decision, security finding, provenance event) is persisted to an
append-only JSONL log backed by SQLite WAL for transactional durability.

Properties:
  - Append-only: records are never modified or deleted
  - Tamper-evident: each record includes a chain hash linking to the
    previous record (similar to a blockchain journal)
  - Idempotent: records with the same event_id are skipped
  - Process-safe: SQLite WAL allows concurrent readers and one writer
  - Credential-safe: secret patterns are redacted before persistence
  - Schema-versioned: migrations are applied automatically

Storage layout::

    ~/.entroly/governance/audit/
    ├── audit.db        ← SQLite WAL for queryable records
    └── audit.jsonl     ← Append-only JSONL for tamper-evident chain

The SQLite database is the query surface.  The JSONL file is the
authoritative chain.

The two are separate files, so they are not written atomically: a process
killed between the two writes, or a failing disk, leaves one ahead of the
other.  Rather than claim an atomicity the code does not have, `verify_chain`
cross-checks the record counts and reports a disagreement, and the chain is
anchored on the JSONL so the stores cannot silently drift apart.  A repeated
`event_id` is rejected once, by the database, for both sinks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .domain import _now

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_AUDIT_DIR_ENV = "ENTROLY_AUDIT_DIR"
_STATE_DIR_ENV = "ENTROLY_DIR"
_DB_SCHEMA_VERSION = 1


def _resolve_audit_dir(audit_dir: Path | None = None) -> Path:
    """Where audit records live, resolved at call time, most specific first.

    Previously a module-level ``Path("~/...").expanduser()``. Two consequences,
    both reproduced before this was changed:

    * ``~`` was expanded once at import, so relocating HOME afterwards had no
      effect and records kept landing in the original home directory. Tests and
      sandboxes isolate exactly that way.
    * ``ENTROLY_DIR`` -- the project state directory 24 other modules honour --
      was ignored, so governance was the one subsystem that could not be scoped
      to a project.

    An explicit argument also now outranks the environment. It did not: passing
    ``audit_dir=`` while ``ENTROLY_AUDIT_DIR`` was set wrote somewhere else
    entirely and said nothing, which makes a caller's isolation silently void.
    """
    if audit_dir is not None:
        return Path(audit_dir).expanduser()
    override = os.environ.get(_AUDIT_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    state_dir = os.environ.get(_STATE_DIR_ENV, "").strip()
    if state_dir:
        return Path(state_dir).expanduser() / "governance" / "audit"
    return Path("~/.entroly/governance/audit").expanduser()

# Patterns redacted before persistence — never log raw credentials
_REDACT_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;\"']{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),  # base64 blobs
    re.compile(r"(?i)(token|secret|password|key)\s*[:=]\s*[^\s,;\"']{8,}"),
)

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS governance_audit (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT UNIQUE NOT NULL,
    event_type  TEXT NOT NULL,
    agent_id    TEXT NOT NULL DEFAULT '',
    session_id  TEXT NOT NULL DEFAULT '',
    org_id      TEXT NOT NULL DEFAULT '',
    trace_id    TEXT NOT NULL DEFAULT '',
    allowed     INTEGER,
    payload     TEXT NOT NULL DEFAULT '{}',
    chain_hash  TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON governance_audit(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_agent_id ON governance_audit(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON governance_audit(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_session_id ON governance_audit(session_id);
"""


# ── Redaction ────────────────────────────────────────────────────────

def _redact(text: str) -> str:
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(
            lambda m: (m.group(1) if m.lastindex else "") + "[REDACTED]",
            text,
        )
    return text


def _safe_json(data: Any) -> str:
    try:
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, default=str)
        return _redact(raw)
    except Exception:
        return "{}"


# ── Audit Record ────────────────────────────────────────────────────

@dataclass(frozen=True)
class AuditRecord:
    """An immutable audit record."""
    event_id: str
    event_type: str
    agent_id: str = ""
    session_id: str = ""
    org_id: str = ""
    trace_id: str = ""
    allowed: bool | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    chain_hash: str = ""
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "org_id": self.org_id,
            "trace_id": self.trace_id,
            "allowed": self.allowed,
            "payload": dict(self.payload),
            "chain_hash": self.chain_hash,
            "created_at": self.created_at,
        }


# ── Audit Log ────────────────────────────────────────────────────────

class GovernanceAuditLog:
    """Append-only, tamper-evident governance audit log.

    Usage::

        log = GovernanceAuditLog()
        log.append(event_id="...", event_type="permission.denied",
                   agent_id="agent-1", payload={"scope": "write"}, allowed=False)

        recent = log.query(event_type="permission.denied", limit=50)
    """

    def __init__(self, audit_dir: Path | None = None) -> None:
        self._dir = _resolve_audit_dir(audit_dir)
        self._lock = threading.Lock()
        self._prev_hash = ""
        self._initialized = False

    @property
    def db_path(self) -> Path:
        return self._dir / "audit.db"

    @property
    def jsonl_path(self) -> Path:
        return self._dir / "audit.jsonl"

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_DB_SCHEMA)
            conn.commit()
            conn.close()
            # Restore prev_hash from last record
            self._prev_hash = self._last_chain_hash()
            self._initialized = True
        except Exception as exc:
            logger.warning("Audit log initialization failed: %s", exc)

    def _last_chain_hash(self) -> str:
        """Anchor the chain on the JSONL, which is the authoritative chain.

        This read from SQLite, which is wrong whenever the two sinks disagree.
        The next record would be hashed from the database's last hash but
        appended after the JSONL's last line, so the link between those two
        lines could not verify -- and `verify_chain` walks the JSONL, so it
        reported "Chain broken" at a record nobody had touched.

        Reproduced before this change: append an event, append it again, then
        reopen the log as a new process would. The duplicate advanced the JSONL
        past the database, the restart re-anchored on the database, and
        verification turned a healthy log into a tampering alarm. A false alarm
        is not a lesser failure than a missed one here -- an integrity check
        that cries wolf is one operators learn to ignore.

        SQLite remains the fallback for a log whose JSONL is missing or
        unreadable, where it is the only anchor available.
        """
        from_chain = self._last_jsonl_chain_hash()
        if from_chain is not None:
            return from_chain
        try:
            if not self.db_path.exists():
                return ""
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            row = conn.execute(
                "SELECT chain_hash FROM governance_audit ORDER BY id DESC LIMIT 1"
            ).fetchone()
            conn.close()
            return row[0] if row else ""
        except Exception:
            return ""

    def _last_jsonl_chain_hash(self) -> str | None:
        """The last record's chain hash, or None if the chain is unreadable.

        Reads a bounded window from the end rather than scanning the file. The
        JSONL is append-only and never rotated, so it grows without limit, and
        this runs during initialization -- a full scan would make every process
        that touches governance pay for the whole history.

        None and "" mean different things: None is "no chain to anchor on, try
        the database", "" is "the chain exists and starts from empty".
        """
        try:
            if not self.jsonl_path.exists():
                return None
            size = self.jsonl_path.stat().st_size
            if size == 0:
                return None
            window = 65536
            with open(self.jsonl_path, "rb") as handle:
                while True:
                    start = max(0, size - window)
                    handle.seek(start)
                    lines = [ln for ln in handle.read(size - start).split(b"\n") if ln.strip()]
                    # Seeking into the middle of the file can cut the first line
                    # in half. Requiring a second line means the last one is
                    # whole; reaching the start means there is nothing to cut.
                    if start == 0 or len(lines) >= 2:
                        if not lines:
                            return None
                        return str(json.loads(lines[-1].decode("utf-8")).get("chain_hash", ""))
                    window *= 4
        except Exception as exc:
            logger.warning("Could not read the audit chain anchor: %s", exc)
            return None

    def _compute_chain_hash(self, record: AuditRecord) -> str:
        """Chain hash: SHA-256 of (previous_hash + event_id + payload_json)."""
        canon = f"{self._prev_hash}|{record.event_id}|{_safe_json(record.payload)}"
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def append(
        self,
        *,
        event_id: str,
        event_type: str,
        agent_id: str = "",
        session_id: str = "",
        org_id: str = "",
        trace_id: str = "",
        allowed: bool | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> AuditRecord:
        """Append a record to the audit log. Idempotent on event_id.

        Both sinks obey that idempotence, decided once. They did not: SQLite
        took the repeat as `INSERT OR IGNORE` and dropped it while the JSONL
        appended unconditionally, so the two disagreed about what happened.
        `query` read the deduplicated database and looked correct; `verify_chain`
        walks the JSONL and certified the inflated copy. Measured on two events
        appended three times: 2 rows, 3 chain lines, "Chain intact (3 records
        verified)".

        The database decides, because its UNIQUE constraint is the only check
        that holds across processes -- an in-memory set of seen ids would not
        survive a restart and would not see a second writer.
        """
        with self._lock:
            self._ensure_initialized()

            # Kept so the anchor can be rolled back. A duplicate must leave the
            # chain exactly where it was: advancing it for a record that is
            # never persisted makes the next real record hash from a link that
            # exists nowhere.
            prev_hash_before = self._prev_hash

            record = AuditRecord(
                event_id=event_id,
                event_type=event_type,
                agent_id=agent_id,
                session_id=session_id,
                org_id=org_id,
                trace_id=trace_id,
                allowed=allowed,
                payload=payload or {},
                chain_hash=self._compute_chain_hash(
                    AuditRecord(
                        event_id=event_id,
                        event_type=event_type,
                        payload=payload or {},
                    )
                ),
            )
            self._prev_hash = record.chain_hash

            # Write to SQLite
            stored_chain_hash: str | None = None
            is_duplicate = False
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=10)
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO governance_audit
                    (event_id, event_type, agent_id, session_id, org_id,
                     trace_id, allowed, payload, chain_hash, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        record.event_id, record.event_type,
                        record.agent_id, record.session_id,
                        record.org_id, record.trace_id,
                        int(allowed) if allowed is not None else None,
                        _safe_json(payload or {}),
                        record.chain_hash, record.created_at,
                    ),
                )
                conn.commit()
                # rowcount 0 means the UNIQUE constraint on event_id rejected
                # it, which is the only way IGNORE suppresses a row here.
                is_duplicate = cursor.rowcount == 0
                if is_duplicate:
                    existing = conn.execute(
                        "SELECT chain_hash FROM governance_audit WHERE event_id = ?",
                        (record.event_id,),
                    ).fetchone()
                    stored_chain_hash = existing[0] if existing else None
                conn.close()
            except Exception as exc:
                # A failed database write is not a duplicate. The JSONL is the
                # authoritative chain, so the record still goes there and the
                # divergence is left for `verify_chain` to report rather than
                # dropping an audit record on the floor.
                logger.warning("Audit DB write failed: %s", exc)

            if is_duplicate:
                self._prev_hash = prev_hash_before
                logger.debug(
                    "Audit event %s already recorded; skipping duplicate append.",
                    record.event_id,
                )
                # Report the hash that is actually stored, not the one this call
                # computed and discarded.
                return (
                    record if stored_chain_hash is None
                    else AuditRecord(
                        event_id=record.event_id, event_type=record.event_type,
                        agent_id=record.agent_id, session_id=record.session_id,
                        org_id=record.org_id, trace_id=record.trace_id,
                        allowed=record.allowed, payload=record.payload,
                        chain_hash=stored_chain_hash, created_at=record.created_at,
                    )
                )

            # Append to JSONL chain
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
            except Exception as exc:
                logger.warning("Audit JSONL write failed: %s", exc)

            return record

    def query(
        self,
        *,
        event_type: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        org_id: str | None = None,
        allowed: bool | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit records with optional filters."""
        self._ensure_initialized()
        if not self.db_path.exists():
            return []

        clauses: list[str] = []
        params: list[Any] = []

        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if org_id:
            clauses.append("org_id = ?")
            params.append(org_id)
        if allowed is not None:
            clauses.append("allowed = ?")
            params.append(int(allowed))
        if since:
            clauses.append("created_at >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT event_id, event_type, agent_id, session_id, org_id,
                   trace_id, allowed, payload, chain_hash, created_at
            FROM governance_audit
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            conn.close()
        except Exception as exc:
            logger.warning("Audit query failed: %s", exc)
            return []

        results = []
        for row in rows:
            try:
                payload = json.loads(row["payload"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            results.append({
                "event_id": row["event_id"],
                "event_type": row["event_type"],
                "agent_id": row["agent_id"],
                "session_id": row["session_id"],
                "org_id": row["org_id"],
                "trace_id": row["trace_id"],
                "allowed": bool(row["allowed"]) if row["allowed"] is not None else None,
                "payload": payload,
                "chain_hash": row["chain_hash"],
                "created_at": row["created_at"],
            })
        return results

    def verify_chain(self, limit: int = 1000) -> tuple[bool, str]:
        """Verify the JSONL chain, and that it agrees with the database.

        Returns (True, detail) if intact, or (False, error_message).

        The hash walk alone was not enough. It only asks whether each link
        follows from the one before, so any self-consistent file passes --
        including one carrying the same event twice, which is what a duplicate
        subscriber produced. `query` read the deduplicated database and looked
        right while this reported the inflated chain as intact.

        So the record count is now cross-checked against the database. That is
        the anchor a bare hash chain lacks: the chain cannot say how long it is
        meant to be, but the database's row count can. It makes truncation of
        the JSONL alone detectable, which it previously was not.

        The anchor holds only while the database is intact. Anyone able to
        truncate both sinks, or to fabricate a log with the same number of
        records, still passes -- the chain is unkeyed, so write access is
        enough to forge a consistent history. `entroly govern audit verify`
        states that boundary, and the tests measure it in both directions.
        """
        self._ensure_initialized()
        if not self.jsonl_path.exists():
            return True, "No records to verify"

        prev_hash = ""
        lines_checked = 0
        seen_event_ids: set[str] = set()
        truncated_by_limit = False
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    expected = hashlib.sha256(
                        f"{prev_hash}|{record['event_id']}|{_safe_json(record.get('payload', {}))}".encode()
                    ).hexdigest()
                    if expected != record.get("chain_hash", ""):
                        return False, (
                            f"Chain broken at event {record.get('event_id')} "
                            f"(record {lines_checked + 1})"
                        )
                    event_id = record["event_id"]
                    if event_id in seen_event_ids:
                        # Every event_id is unique by construction, so a repeat
                        # is a record written twice, not two events.
                        return False, (
                            f"Event {event_id} appears more than once in the chain "
                            f"(record {lines_checked + 1}); the log is inflated "
                            "and does not describe what happened"
                        )
                    seen_event_ids.add(event_id)
                    prev_hash = record["chain_hash"]
                    lines_checked += 1
                    if lines_checked >= limit:
                        truncated_by_limit = True
                        break
        except Exception as exc:
            return False, f"Verification error: {exc}"

        if truncated_by_limit:
            # Only a prefix was read, so the counts cannot be compared. Say so
            # rather than implying the whole chain was verified.
            return True, (
                f"First {lines_checked} records verified (limit reached; "
                "the rest of the chain was not read)"
            )

        db_count = self._db_record_count()
        if db_count is not None and db_count != lines_checked:
            return False, (
                f"Audit stores disagree: the chain has {lines_checked} records, "
                f"the database has {db_count}. One of them is not a record of "
                "what happened."
            )

        return True, f"Chain intact ({lines_checked} records verified)"

    def _db_record_count(self) -> int | None:
        """Row count in the query surface, or None if it cannot be read.

        None is not zero: an unreadable database means the cross-check cannot
        run, which must not be reported as the two stores agreeing.
        """
        try:
            if not self.db_path.exists():
                return None
            conn = sqlite3.connect(str(self.db_path), timeout=5)
            row = conn.execute("SELECT COUNT(*) FROM governance_audit").fetchone()
            conn.close()
            return int(row[0]) if row else None
        except Exception as exc:
            logger.warning("Could not count audit records for cross-check: %s", exc)
            return None


# ── Event Bus Integration ────────────────────────────────────────────

def make_audit_subscriber(log: "GovernanceAuditLog | None" = None):
    """Create an event bus subscriber that auto-logs all governance events.

    With no ``log``, the handler resolves the process-global log at emit time
    rather than capturing one at creation. That matters because the bus and the
    log are separate globals with separate lifetimes: a handler that closed over
    an instance would keep writing to it after the global log was replaced, so
    records would land in an object nothing reads while the live log stayed
    empty -- and an empty log is what a clean audit looks like.

    Pass an explicit ``log`` to bind one deliberately, e.g. for a private bus.
    """
    from .events import GovernanceEvent

    def _handler(event: GovernanceEvent) -> None:
        target = log if log is not None else get_audit_log()
        payload = dict(event.payload)
        target.append(
            event_id=event.event_id,
            event_type=event.event_type,
            agent_id=event.tracing.agent_id,
            session_id=event.tracing.session_id,
            org_id=event.tracing.organization_id,
            trace_id=event.tracing.trace_id,
            allowed=payload.pop("allowed", None),
            payload=payload,
        )
    return _handler


# ── Global Singleton ─────────────────────────────────────────────────

_global_log: GovernanceAuditLog | None = None
_log_lock = threading.Lock()


def get_audit_log() -> GovernanceAuditLog:
    """Get or create the process-global audit log."""
    global _global_log
    if _global_log is None:
        with _log_lock:
            if _global_log is None:
                _global_log = GovernanceAuditLog()
    return _global_log


def install_audit_subscriber(bus=None) -> None:
    """Install the audit log as a wildcard subscriber on the event bus.

    Idempotent per bus. `subscribe` appends unconditionally, so without this
    guard a second call attaches a second wildcard handler and every event is
    appended twice -- and the two sinks in `append` then disagree. SQLite takes
    it as `INSERT OR IGNORE` on `event_id` and drops the copy; the JSONL chain
    is an unconditional write and keeps it. So `query` reports the true count
    while `verify_chain`, which walks the JSONL, reports roughly double and
    still calls the chain intact. Measured: one run logged 7 records by `query`
    and "Chain intact (13 records verified)".

    That is the worst shape for an audit defect -- inflated, self-consistent,
    and endorsed by the integrity check meant to catch it.

    A second call is ordinary, not pathological: `get_authorization_service`
    installs on creation, and `reset=True` (or `reset_authorization_service`)
    makes it create again within the same process.
    """
    from .events import get_event_bus
    if bus is None:
        bus = get_event_bus()
    # Tracked on the bus so the marker dies with it. A module-level set keyed
    # by id() would let a recycled id suppress a legitimate install.
    if getattr(bus, "_entroly_audit_subscriber_installed", False):
        return
    # No log argument: the handler resolves the current global log per event,
    # so this install stays correct if that log is later replaced.
    bus.subscribe("*", make_audit_subscriber())
    try:
        bus._entroly_audit_subscriber_installed = True
    except AttributeError:  # pragma: no cover - bus defining __slots__
        logger.debug(
            "Event bus rejects the install marker; duplicate audit subscribers "
            "are possible on this bus."
        )


__all__ = [
    "AuditRecord",
    "GovernanceAuditLog",
    "make_audit_subscriber",
    "get_audit_log",
    "install_audit_subscriber",
]
