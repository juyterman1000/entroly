"""Bitemporal belief ledger — time travel for the knowledge vault.

`VaultManager.write_belief` keeps one file per entity, so each write destroys
the previous version. This module preserves history without changing that
layout: an append-only, hash-chained JSONL ledger under ``vault/ledger/``
records every belief version, with bodies stored once in a content-addressed
object store (``vault/ledger/objects/<sha256>.md``).

Two time axes are recorded, and callers must not conflate them:

- **transaction time** (``tx_time``): when the vault learned it. "Answer using
  only what I knew last Tuesday" queries this axis (the default).
- **valid time** (``valid_time``): the artifact's ``last_checked`` — when the
  knowledge was last verified against reality.

Trust properties:

- append-only; every record carries ``prev_sha256`` forming a tamper-evident
  chain (same posture as SessionReceiptChain), verifiable offline;
- bodies are content-addressed, so identical bodies across versions are
  stored once and any body substitution is detectable via ``body_sha256``;
- unparseable ledger lines fail closed: queries raise rather than silently
  returning a partial past;
- backfill from pre-ledger belief files is explicit and flagged
  (``backfilled: true``), never silent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .path_safety import resolve_output_within

LEDGER_SCHEMA = "entroly.belief-ledger.v1"
_RECORD_HASH_FIELD = "record_sha256"


class LedgerIntegrityError(RuntimeError):
    """The ledger is unreadable or its hash chain is broken."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_when(when: str | datetime) -> str:
    """Normalize a query instant to a sortable UTC ISO string."""
    if isinstance(when, datetime):
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return when.astimezone(timezone.utc).isoformat()
    dt = datetime.fromisoformat(when)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _record_hash(record: dict[str, Any]) -> str:
    payload = {k: v for k, v in record.items() if k != _RECORD_HASH_FIELD}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class BeliefVersion:
    """One immutable version of a belief, as recorded in the ledger."""
    seq: int
    tx_time: str
    valid_time: str
    claim_id: str
    entity: str
    status: str
    confidence: float
    sources: tuple[str, ...]
    title: str
    body_sha256: str
    backfilled: bool = False

    @classmethod
    def from_record(cls, rec: dict[str, Any]) -> "BeliefVersion":
        return cls(
            seq=int(rec["seq"]),
            tx_time=str(rec["tx_time"]),
            valid_time=str(rec.get("valid_time", "")),
            claim_id=str(rec.get("claim_id", "")),
            entity=str(rec.get("entity", "")),
            status=str(rec.get("status", "unknown")),
            confidence=float(rec.get("confidence", 0.0)),
            sources=tuple(rec.get("sources", ())),
            title=str(rec.get("title", "")),
            body_sha256=str(rec.get("body_sha256", "")),
            backfilled=bool(rec.get("backfilled", False)),
        )


class BeliefLedger:
    """Append-only bitemporal history of vault beliefs."""

    def __init__(self, vault_base: str | Path):
        self._base = Path(vault_base)
        self._dir = self._base / "ledger"
        self._log = self._dir / "beliefs.jsonl"
        self._objects = self._dir / "objects"

    # ── Writing ──────────────────────────────────────────────────────

    def _last_record(self) -> dict[str, Any] | None:
        if not self._log.exists():
            return None
        last_line = ""
        for line in self._log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                last_line = line
        if not last_line:
            return None
        try:
            return json.loads(last_line)
        except json.JSONDecodeError as exc:
            raise LedgerIntegrityError(
                f"unreadable final ledger record: {exc}"
            ) from exc

    def record(self, artifact: Any, *, backfilled: bool = False,
               tx_time: str | None = None) -> dict[str, Any]:
        """Append one belief version. Called by VaultManager.write_belief.

        ``tx_time`` is injectable for backfill/tests only; live writes stamp
        the current UTC instant.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        self._objects.mkdir(parents=True, exist_ok=True)

        body = str(getattr(artifact, "body", "") or "")
        body_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        obj_path = self._objects / f"{body_sha}.md"
        safe_obj = resolve_output_within(self._dir, obj_path)
        if safe_obj is None:
            raise LedgerIntegrityError(f"object path escapes ledger: {obj_path}")
        if not safe_obj.exists():
            safe_obj.write_text(body, encoding="utf-8")

        last = self._last_record()
        record = {
            "schema": LEDGER_SCHEMA,
            "seq": (int(last["seq"]) + 1) if last else 1,
            "tx_time": tx_time or _utc_now_iso(),
            "valid_time": str(getattr(artifact, "last_checked", "") or ""),
            "claim_id": str(getattr(artifact, "claim_id", "") or ""),
            "entity": str(getattr(artifact, "entity", "") or ""),
            "status": str(getattr(artifact, "status", "") or "unknown"),
            "confidence": float(getattr(artifact, "confidence", 0.0) or 0.0),
            "sources": list(getattr(artifact, "sources", []) or []),
            "title": str(getattr(artifact, "title", "") or ""),
            "body_sha256": body_sha,
            "backfilled": bool(backfilled),
            "prev_sha256": last[_RECORD_HASH_FIELD] if last else "",
        }
        record[_RECORD_HASH_FIELD] = _record_hash(record)
        with self._log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        return {
            "status": "recorded",
            "seq": record["seq"],
            "record_sha256": record[_RECORD_HASH_FIELD],
        }

    def seed_from_current(self, beliefs_dir: str | Path) -> dict[str, Any]:
        """Backfill one version per existing belief file (explicit, flagged).

        tx_time is the file's mtime — the best available approximation of
        when the vault learned it. No-op for entities already in the ledger.
        """
        from .vault import BeliefArtifact, _extract_body, _parse_frontmatter

        known = {v.entity for v in self._iter_versions()} if self._log.exists() else set()
        seeded = 0
        for md in sorted(Path(beliefs_dir).rglob("*.md")):
            content = md.read_text(encoding="utf-8", errors="replace")
            fm = _parse_frontmatter(content) or {}
            entity = fm.get("entity", md.stem)
            if entity in known:
                continue
            artifact = BeliefArtifact(
                claim_id=fm.get("claim_id", ""),
                entity=entity,
                status=fm.get("status", "unknown"),
                confidence=float(fm.get("confidence", 0.0) or 0.0),
                sources=[],
                last_checked=fm.get("last_checked", ""),
                title=fm.get("title", md.stem),
                body=_extract_body(content),
            )
            mtime = datetime.fromtimestamp(md.stat().st_mtime, tz=timezone.utc)
            self.record(artifact, backfilled=True, tx_time=mtime.isoformat())
            seeded += 1
        return {"status": "seeded", "entities": seeded}

    # ── Reading ──────────────────────────────────────────────────────

    def _iter_versions(self) -> Iterator[BeliefVersion]:
        if not self._log.exists():
            return
        for line_no, line in enumerate(
            self._log.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                yield BeliefVersion.from_record(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise LedgerIntegrityError(
                    f"unreadable ledger record at line {line_no}: {exc}"
                ) from exc

    def body_of(self, version: BeliefVersion) -> str:
        obj = self._objects / f"{version.body_sha256}.md"
        if not obj.exists():
            raise LedgerIntegrityError(
                f"missing body object {version.body_sha256} for '{version.entity}'"
            )
        body = obj.read_text(encoding="utf-8")
        actual = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if actual != version.body_sha256:
            raise LedgerIntegrityError(
                f"body object tampered for '{version.entity}': "
                f"expected {version.body_sha256}, got {actual}"
            )
        return body

    def as_of(self, when: str | datetime, *,
              time_axis: str = "transaction") -> dict[str, BeliefVersion]:
        """Snapshot: the belief version per entity visible at ``when``.

        ``time_axis="transaction"`` answers "what did I know at T" (default).
        ``time_axis="valid"`` answers "what had been verified as of T".
        """
        if time_axis not in ("transaction", "valid"):
            raise ValueError(f"unknown time_axis: {time_axis!r}")
        cutoff = _parse_when(when)
        snapshot: dict[str, BeliefVersion] = {}
        for v in self._iter_versions():
            instant = v.tx_time if time_axis == "transaction" else v.valid_time
            if instant and instant <= cutoff:
                prev = snapshot.get(v.entity)
                if prev is None or v.seq > prev.seq:
                    snapshot[v.entity] = v
        return snapshot

    def diff(self, t1: str | datetime, t2: str | datetime, *,
             time_axis: str = "transaction") -> dict[str, Any]:
        """What changed between two instants (t1 < t2)."""
        a = self.as_of(t1, time_axis=time_axis)
        b = self.as_of(t2, time_axis=time_axis)
        added = sorted(set(b) - set(a))
        changed = []
        for entity in sorted(set(a) & set(b)):
            va, vb = a[entity], b[entity]
            if va.seq == vb.seq:
                continue
            changed.append({
                "entity": entity,
                "from_seq": va.seq,
                "to_seq": vb.seq,
                "status": [va.status, vb.status],
                "confidence": [va.confidence, vb.confidence],
                "body_changed": va.body_sha256 != vb.body_sha256,
            })
        return {
            "from": _parse_when(t1),
            "to": _parse_when(t2),
            "time_axis": time_axis,
            "added": added,
            "changed": changed,
        }

    def timeline(self, entity: str) -> list[BeliefVersion]:
        """Every recorded version of one entity, oldest first."""
        return sorted(
            (v for v in self._iter_versions() if v.entity == entity),
            key=lambda v: v.seq,
        )

    def verify_chain(self) -> dict[str, Any]:
        """Recompute the hash chain. Fail-closed: reports the first break."""
        if not self._log.exists():
            return {"status": "empty", "records": 0}
        prev_hash = ""
        count = 0
        for line_no, line in enumerate(
            self._log.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return {"status": "broken", "line": line_no,
                        "reason": "unparseable record"}
            if rec.get("prev_sha256", "") != prev_hash:
                return {"status": "broken", "line": line_no,
                        "reason": "prev_sha256 mismatch"}
            if _record_hash(rec) != rec.get(_RECORD_HASH_FIELD):
                return {"status": "broken", "line": line_no,
                        "reason": "record_sha256 mismatch"}
            prev_hash = rec[_RECORD_HASH_FIELD]
            count += 1
        return {"status": "intact", "records": count}
