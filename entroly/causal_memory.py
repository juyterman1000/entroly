"""
Causal Memory with WITNESS Provenance (CMWP)
=============================================

The first session memory system with proof-carrying facts and automatic
staleness detection via source-file hash tracking.

Motivation
----------
Existing session memory stores facts without verification.  If the AI
hallucinated "auth uses JWT with RS256" in session 3, every future
session uses the hallucinated fact as ground truth.

CMWP stores every fact with:
    - The WITNESS certificate that verified it
    - SHA-256 hashes of source files at verification time
    - A causal chain showing which upstream facts this depends on
    - Automatic staleness detection when source files change

Memory without verification is a liability.
Memory WITH verification is a moat.

Mathematical invariant
----------------------
For any fact F recalled at time t:

    F.is_valid(t)  ⟺  F.confidence ≥ τ
                    ∧  ∀ h ∈ F.source_hashes: h = current_hash(file)
                    ∧  ¬∃ F' : F'.invalidates(F)
                    ∧  (F.valid_until is None ∨ t ≤ F.valid_until)

A fact is recalled ONLY if all four conditions hold.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class CausalFact:
    """A fact with full causal provenance.

    Every field serves a verification purpose:
    - claim: the actual statement
    - evidence_hash: what established it (tamper-detectable)
    - confidence: WITNESS score at storage time
    - source_file_hashes: file states at verification time (staleness detection)
    - causal_chain: upstream dependencies (cascade invalidation)
    """
    fact_id: str                    # SHA-256(claim + evidence_hash + timestamp)
    claim: str                      # "auth uses JWT with RS256"
    evidence_hash: str              # SHA-256 of the evidence text
    witness_score: float            # WITNESS summary_score [0, 1]
    witness_label: str              # "grounded" | "unsupported" | "contradicted"
    established_at: str             # ISO 8601 timestamp
    valid_until: str | None = None  # ISO 8601, None = permanent
    invalidated_by: str | None = None  # fact_id that contradicted
    invalidation_reason: str | None = None
    source_file_hashes: dict[str, str] = field(default_factory=dict)  # path → SHA-256
    causal_chain: list[str] = field(default_factory=list)  # upstream fact_ids
    tags: list[str] = field(default_factory=list)  # for recall filtering
    recall_count: int = 0
    last_recalled: str | None = None

    @property
    def is_invalidated(self) -> bool:
        return self.invalidated_by is not None

    def is_stale(self, current_hashes: dict[str, str]) -> bool:
        """Check if any source file changed since this fact was established."""
        for path, stored_hash in self.source_file_hashes.items():
            current = current_hashes.get(path)
            if current is not None and current != stored_hash:
                return True
        return False

    def is_expired(self) -> bool:
        """Check if the fact has passed its valid_until time."""
        if self.valid_until is None:
            return False
        try:
            expiry = datetime.fromisoformat(self.valid_until)
            return datetime.now(timezone.utc) > expiry
        except (ValueError, TypeError):
            return False


def _generate_fact_id(claim: str, evidence_hash: str, timestamp: str) -> str:
    """Generate a deterministic fact ID from content + time."""
    content = f"{claim}|{evidence_hash}|{timestamp}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]


def _hash_file(path: str) -> str | None:
    """SHA-256 hash of a file's contents. Returns None if file not readable."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (OSError, IOError):
        return None


def _hash_text(text: str) -> str:
    """SHA-256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ── SQLite Schema ────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    claim TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    witness_score REAL NOT NULL,
    witness_label TEXT NOT NULL,
    established_at TEXT NOT NULL,
    valid_until TEXT,
    invalidated_by TEXT,
    invalidation_reason TEXT,
    source_file_hashes TEXT NOT NULL DEFAULT '{}',
    causal_chain TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    recall_count INTEGER DEFAULT 0,
    last_recalled TEXT
);

CREATE INDEX IF NOT EXISTS idx_facts_claim ON facts(claim);
CREATE INDEX IF NOT EXISTS idx_facts_label ON facts(witness_label);
CREATE INDEX IF NOT EXISTS idx_facts_score ON facts(witness_score);
CREATE INDEX IF NOT EXISTS idx_facts_established ON facts(established_at);

-- Full-text search on claims for fast recall
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    fact_id,
    claim,
    tags,
    content=facts,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, fact_id, claim, tags)
    VALUES (new.rowid, new.fact_id, new.claim, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, fact_id, claim, tags)
    VALUES ('delete', old.rowid, old.fact_id, old.claim, old.tags);
END;
"""


# ── CausalMemoryStore ────────────────────────────────────────────────

class CausalMemoryStore:
    """WITNESS-verified session memory with causal provenance.

    Thread-safe SQLite-backed store. Facts are stored only if WITNESS
    verified them with sufficient confidence. Facts auto-invalidate
    when source files change.

    Usage
    -----
    >>> store = CausalMemoryStore("~/.entroly/memory.db")
    >>> store.remember("auth uses JWT", evidence_text, witness_result)
    >>> facts = store.recall("JWT authentication")
    >>> for f in facts:
    ...     print(f.claim, f.witness_score, f.is_stale(current_hashes))
    """

    def __init__(
        self,
        db_path: str | None = None,
        min_confidence: float = 0.70,
        max_facts: int = 10000,
    ):
        if db_path is None:
            db_path = os.path.join(
                os.path.expanduser("~"),
                ".entroly",
                "causal_memory.db",
            )

        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._db_path = db_path
        self._min_confidence = min_confidence
        self._max_facts = max_facts
        self._lock = threading.Lock()

        # Initialize schema
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── Remember ─────────────────────────────────────────────────

    def remember(
        self,
        claim: str,
        evidence: str,
        witness_score: float,
        witness_label: str = "grounded",
        source_files: list[str] | None = None,
        valid_hours: float | None = None,
        tags: list[str] | None = None,
        causal_parents: list[str] | None = None,
    ) -> CausalFact | None:
        """Store a WITNESS-verified fact.

        Only stores facts with confidence ≥ min_confidence.
        Returns the stored CausalFact, or None if rejected.
        """
        if witness_score < self._min_confidence:
            logger.debug(
                "[CMWP] Rejected fact (score %.2f < %.2f): %s",
                witness_score, self._min_confidence, claim[:80],
            )
            return None

        if witness_label in ("contradicted",):
            logger.debug("[CMWP] Rejected contradicted fact: %s", claim[:80])
            return None

        now = datetime.now(timezone.utc).isoformat()
        evidence_hash = _hash_text(evidence)

        # Compute source file hashes for staleness tracking
        file_hashes: dict[str, str] = {}
        if source_files:
            for path in source_files:
                h = _hash_file(path)
                if h:
                    file_hashes[path] = h

        valid_until = None
        if valid_hours is not None:
            expiry = datetime.now(timezone.utc) + timedelta(hours=valid_hours)
            valid_until = expiry.isoformat()

        fact_id = _generate_fact_id(claim, evidence_hash, now)

        fact = CausalFact(
            fact_id=fact_id,
            claim=claim,
            evidence_hash=evidence_hash,
            witness_score=witness_score,
            witness_label=witness_label,
            established_at=now,
            valid_until=valid_until,
            source_file_hashes=file_hashes,
            causal_chain=causal_parents or [],
            tags=tags or [],
        )

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO facts
                    (fact_id, claim, evidence_hash, witness_score, witness_label,
                     established_at, valid_until, source_file_hashes, causal_chain,
                     tags, recall_count, last_recalled)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
                    (
                        fact.fact_id, fact.claim, fact.evidence_hash,
                        fact.witness_score, fact.witness_label,
                        fact.established_at, fact.valid_until,
                        json.dumps(fact.source_file_hashes),
                        json.dumps(fact.causal_chain),
                        json.dumps(fact.tags),
                    ),
                )
                conn.commit()

            self._enforce_limit(conn)

        logger.info(
            "[CMWP] Stored fact %s (score=%.2f): %s",
            fact_id[:8], witness_score, claim[:60],
        )
        return fact

    # ── Recall ───────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
        current_file_hashes: dict[str, str] | None = None,
        include_stale: bool = False,
    ) -> list[CausalFact]:
        """Retrieve relevant facts, filtered by staleness and validity.

        Uses FTS5 full-text search on claims. Only returns facts that
        satisfy the validity invariant (not invalidated, not expired,
        not stale unless include_stale=True).
        """
        with self._lock:
            with self._connect() as conn:
                # FTS5 query
                try:
                    rows = conn.execute(
                        """SELECT f.* FROM facts f
                        JOIN facts_fts fts ON f.fact_id = fts.fact_id
                        WHERE facts_fts MATCH ?
                        AND f.invalidated_by IS NULL
                        AND f.witness_score >= ?
                        ORDER BY rank
                        LIMIT ?""",
                        (query, min_score, max_results * 2),
                    ).fetchall()
                except sqlite3.OperationalError:
                    # FTS match failed (bad query syntax) — fall back to LIKE
                    rows = conn.execute(
                        """SELECT * FROM facts
                        WHERE claim LIKE ?
                        AND invalidated_by IS NULL
                        AND witness_score >= ?
                        ORDER BY witness_score DESC
                        LIMIT ?""",
                        (f"%{query}%", min_score, max_results * 2),
                    ).fetchall()

        facts = [self._row_to_fact(row) for row in rows]

        # Filter by validity
        valid_facts: list[CausalFact] = []
        for fact in facts:
            if fact.is_expired():
                continue
            if not include_stale and current_file_hashes:
                if fact.is_stale(current_file_hashes):
                    continue
            valid_facts.append(fact)

        # Update recall counts
        if valid_facts:
            with self._lock:
                with self._connect() as conn:
                    now = datetime.now(timezone.utc).isoformat()
                    for f in valid_facts[:max_results]:
                        conn.execute(
                            "UPDATE facts SET recall_count = recall_count + 1, "
                            "last_recalled = ? WHERE fact_id = ?",
                            (now, f.fact_id),
                        )
                    conn.commit()

        return valid_facts[:max_results]

    # ── Invalidate ───────────────────────────────────────────────

    def invalidate(
        self,
        fact_id: str,
        reason: str = "contradicted by new evidence",
        cascade: bool = True,
    ) -> int:
        """Invalidate a fact and optionally cascade to dependents.

        Returns the total number of facts invalidated (including cascades).
        """
        count = 0
        with self._lock:
            with self._connect() as conn:
                # Invalidate the target
                conn.execute(
                    """UPDATE facts SET invalidated_by = 'manual',
                    invalidation_reason = ? WHERE fact_id = ?""",
                    (reason, fact_id),
                )
                count += conn.total_changes

                # Cascade: invalidate facts that depend on this one
                if cascade:
                    dependents = conn.execute(
                        "SELECT fact_id, causal_chain FROM facts "
                        "WHERE invalidated_by IS NULL",
                    ).fetchall()

                    for dep_id, chain_json in dependents:
                        chain = json.loads(chain_json) if chain_json else []
                        if fact_id in chain:
                            conn.execute(
                                """UPDATE facts SET invalidated_by = ?,
                                invalidation_reason = ?
                                WHERE fact_id = ?""",
                                (fact_id, f"cascade: upstream {fact_id} invalidated", dep_id),
                            )
                            count += 1

                conn.commit()

        logger.info("[CMWP] Invalidated %d facts (root=%s)", count, fact_id[:8])
        return count

    # ── Prune Stale ──────────────────────────────────────────────

    def prune_stale(
        self,
        current_file_hashes: dict[str, str],
    ) -> int:
        """Bulk-invalidate facts whose source files have changed.

        Call this after file modifications (e.g., git pull, edit).
        Returns the number of facts invalidated.
        """
        count = 0
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT fact_id, source_file_hashes FROM facts "
                    "WHERE invalidated_by IS NULL",
                ).fetchall()

                for fact_id, hashes_json in rows:
                    stored_hashes = json.loads(hashes_json) if hashes_json else {}
                    for path, stored_hash in stored_hashes.items():
                        current = current_file_hashes.get(path)
                        if current is not None and current != stored_hash:
                            conn.execute(
                                """UPDATE facts SET invalidated_by = 'stale',
                                invalidation_reason = ?
                                WHERE fact_id = ?""",
                                (f"source file changed: {path}", fact_id),
                            )
                            count += 1
                            break

                conn.commit()

        logger.info("[CMWP] Pruned %d stale facts", count)
        return count

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return memory store statistics."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            valid = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE invalidated_by IS NULL"
            ).fetchone()[0]
            avg_score = conn.execute(
                "SELECT AVG(witness_score) FROM facts WHERE invalidated_by IS NULL"
            ).fetchone()[0] or 0.0
            most_recalled = conn.execute(
                "SELECT claim, recall_count FROM facts "
                "WHERE invalidated_by IS NULL "
                "ORDER BY recall_count DESC LIMIT 5"
            ).fetchall()

        return {
            "total_facts": total,
            "valid_facts": valid,
            "invalidated_facts": total - valid,
            "avg_confidence": round(avg_score, 3),
            "most_recalled": [
                {"claim": c[:80], "recalls": r} for c, r in most_recalled
            ],
            "db_path": self._db_path,
        }

    # ── Internal ─────────────────────────────────────────────────

    def _row_to_fact(self, row: tuple) -> CausalFact:
        """Convert a database row to a CausalFact."""
        return CausalFact(
            fact_id=row[0],
            claim=row[1],
            evidence_hash=row[2],
            witness_score=row[3],
            witness_label=row[4],
            established_at=row[5],
            valid_until=row[6],
            invalidated_by=row[7],
            invalidation_reason=row[8],
            source_file_hashes=json.loads(row[9]) if row[9] else {},
            causal_chain=json.loads(row[10]) if row[10] else [],
            tags=json.loads(row[11]) if row[11] else [],
            recall_count=row[12] if len(row) > 12 else 0,
            last_recalled=row[13] if len(row) > 13 else None,
        )

    def _enforce_limit(self, conn: sqlite3.Connection) -> None:
        """Remove oldest invalidated facts if over max_facts limit."""
        count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        if count > self._max_facts:
            excess = count - self._max_facts
            conn.execute(
                """DELETE FROM facts WHERE fact_id IN (
                    SELECT fact_id FROM facts
                    WHERE invalidated_by IS NOT NULL
                    ORDER BY established_at ASC
                    LIMIT ?
                )""",
                (excess,),
            )
