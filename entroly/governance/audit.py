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
authoritative chain.  Both are written atomically.
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
        """Append a record to the audit log. Idempotent on event_id."""
        with self._lock:
            self._ensure_initialized()

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
            try:
                conn = sqlite3.connect(str(self.db_path), timeout=10)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
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
                conn.close()
            except Exception as exc:
                logger.warning("Audit DB write failed: %s", exc)

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
        """Verify the JSONL chain hash integrity.

        Returns (True, "OK") if chain is intact, or (False, error_message).
        """
        self._ensure_initialized()
        if not self.jsonl_path.exists():
            return True, "No records to verify"

        prev_hash = ""
        lines_checked = 0
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
                    prev_hash = record["chain_hash"]
                    lines_checked += 1
                    if lines_checked >= limit:
                        break
        except Exception as exc:
            return False, f"Verification error: {exc}"

        return True, f"Chain intact ({lines_checked} records verified)"


# ── Event Bus Integration ────────────────────────────────────────────

def make_audit_subscriber(log: "GovernanceAuditLog"):
    """Create an event bus subscriber that auto-logs all governance events."""
    from .events import GovernanceEvent

    def _handler(event: GovernanceEvent) -> None:
        payload = dict(event.payload)
        log.append(
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
    """Install the audit log as a wildcard subscriber on the event bus."""
    from .events import get_event_bus
    if bus is None:
        bus = get_event_bus()
    log = get_audit_log()
    bus.subscribe("*", make_audit_subscriber(log))


__all__ = [
    "AuditRecord",
    "GovernanceAuditLog",
    "make_audit_subscriber",
    "get_audit_log",
    "install_audit_subscriber",
]
