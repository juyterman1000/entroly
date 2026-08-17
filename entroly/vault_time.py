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
  (``backfilled: true``), never silent;
- **redaction without repudiation**: ``redact`` deletes body objects and
  appends a chained tombstone, so sensitive *content* is provably gone while
  the hash chain still verifies. Structural metadata (hashes, timestamps,
  entity labels) necessarily remains — a chain cannot un-say its metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .path_safety import resolve_output_within

LEDGER_SCHEMA = "entroly.belief-ledger.v1"
_RECORD_HASH_FIELD = "record_sha256"
_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_STALE_SECONDS = 120.0
_LOCK_SETTLE_SECONDS = 1.0
_LOCK_REGION = 1024

logger = logging.getLogger(__name__)


def _lock_token() -> str:
    """Identifies one lock holder, uniquely across machines."""

    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


def _try_acquire(lock_path: Path, token: str) -> bool:
    """Claim the lock by creating its file exclusively.

    `O_CREAT | O_EXCL` is the one mutual-exclusion primitive that holds on a
    network filesystem: NFSv3 and later implement exclusive create atomically
    server-side, and SMB does the same. `fcntl.flock` does not -- without a
    working `lockd` it degrades to a local-only lock, so two machines sharing
    a vault over NFS would each believe they held it and interleave their
    read-modify-write appends exactly as unlocked processes did.

    The advisory lock is still taken underneath where the platform offers it,
    because it releases automatically when a process dies. The lock *file* is
    what makes the guarantee portable; the advisory lock is what makes the
    common local case recover instantly from a crash.
    """

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    except OSError:
        return False

    try:
        os.write(fd, f"{token}\n{time.time():.3f}\n".encode())
    finally:
        os.close(fd)
    return True


def _lock_is_stale(lock_path: Path) -> bool:
    """Whether a lock file was abandoned rather than held.

    A lock file outlives the process that made it, so a crash mid-append would
    block every later writer forever. Age alone is not proof of abandonment,
    though: a slow holder is not a dead one. The file's mtime is read twice
    across a settle interval and the lock is only broken when it has not moved,
    which distinguishes "nobody is there" from "someone is still working".

    mtime comes from the file server, so this comparison is unaffected by
    clock skew between clients -- both readings come from the same clock.
    """

    try:
        first = lock_path.stat().st_mtime
    except OSError:
        return False
    if time.time() - first < _LOCK_STALE_SECONDS:
        return False
    time.sleep(_LOCK_SETTLE_SECONDS)
    try:
        return lock_path.stat().st_mtime == first
    except OSError:
        return False


def _release(lock_path: Path, token: str) -> None:
    """Remove the lock, but only if it is still ours.

    A lock broken as stale can be re-taken by someone else while the original
    holder is still running. Deleting unconditionally would then release a
    lock the caller does not hold, letting two writers proceed at once -- so
    the token written at acquisition is checked first.
    """

    try:
        held = lock_path.read_text(encoding="utf-8").split("\n", 1)[0]
    except OSError:
        return
    if held != token:
        logger.warning(
            "vault ledger lock was taken over while held; not releasing another "
            "holder's lock"
        )
        return
    try:
        lock_path.unlink()
    except OSError:  # pragma: no cover - best effort release
        pass


def _advisory_lock(handle: Any) -> bool:
    """Best-effort OS lock, layered under the lock file. True when taken."""

    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except ImportError:
        pass
    except OSError:
        return False
    try:
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_REGION)
        return True
    except (ImportError, OSError):
        return False
    return False


def _advisory_unlock(handle: Any) -> None:
    try:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    except ImportError:
        pass
    except OSError:
        return
    try:
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_REGION)
    except (ImportError, OSError):  # pragma: no cover - best effort release
        pass


class LedgerIntegrityError(RuntimeError):
    """The ledger is unreadable or its hash chain is broken."""


class BeliefRedactedError(RuntimeError):
    """The requested body was deliberately redacted — not an integrity failure."""


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
    redacted: bool = False

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
        """The final record, read from the tail rather than the whole file.

        Every append needs the previous record's seq and hash, and this used
        to `read_text()` the entire ledger to get them -- 69% of the cost of
        writing a belief, on a file that only grows. Seeking from the end
        reads one line's worth regardless of history, which is what makes
        appending independent of how much history exists.

        The tail is authoritative, unlike `head.json`: a crash between the
        append and the head write would leave the head one record behind, and
        chaining onto that would silently fork the chain.
        """

        if not self._log.exists():
            return None
        try:
            with self._log.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if size == 0:
                    return None
                block = 4096
                buffer = b""
                while size > 0:
                    step = min(block, size)
                    size -= step
                    handle.seek(size)
                    buffer = handle.read(step) + buffer
                    lines = [ln for ln in buffer.split(b"\n") if ln.strip()]
                    # Need a complete line: unless the chunk reaches the file
                    # start, the first fragment may be cut mid-record.
                    if len(lines) >= 2 or size == 0:
                        break
                lines = [ln for ln in buffer.split(b"\n") if ln.strip()]
                if not lines:
                    return None
                last_line = lines[-1].decode("utf-8")
        except OSError:
            return None

        try:
            return json.loads(last_line)
        except json.JSONDecodeError as exc:
            raise LedgerIntegrityError(
                f"unreadable final ledger record: {exc}"
            ) from exc

    @property
    def _head(self) -> Path:
        return self._dir / "head.json"

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        """Serialize the read-tail -> append -> write-head sequence.

        Appending is a read-modify-write: a record's `seq` and `prev_sha256`
        come from whatever is currently last. Two processes that read the same
        tail both chain onto it, and the loser's record is overwritten by the
        winner's -- measured at three processes writing 90 beliefs, the ledger
        held 65-71 records and `verify_chain` reported a `prev_sha256`
        mismatch every time. `entroly serve` holding a vault open while
        `entroly compile` writes to it is exactly that shape.

        Blocking, unlike the best-effort non-blocking helpers in
        `checkpoint.py`: a lock that gives up when contended would leave the
        very case it exists for unprotected. If the platform offers no locking
        at all the append still proceeds -- a ledger that refuses to record is
        worse than one that can be raced -- and the chain check remains the
        backstop that makes such a race visible.
        """

        self._dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._dir / ".lock"
        token = _lock_token()
        acquired = False
        handle = None

        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        delay = 0.001
        while True:
            if _try_acquire(lock_path, token):
                acquired = True
                break
            if _lock_is_stale(lock_path):
                # Breaking a stale lock races with anyone else breaking it, so
                # the winner is decided by the exclusive create on the next
                # pass, not by the unlink.
                logger.warning("breaking an abandoned vault ledger lock")
                try:
                    lock_path.unlink()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                logger.warning(
                    "vault ledger lock timed out after %.0fs; appending "
                    "unserialized", _LOCK_TIMEOUT_SECONDS
                )
                break
            time.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, 0.05)

        if acquired:
            try:
                handle = open(lock_path, "r+")  # noqa: SIM115 - closed in finally
            except OSError:
                handle = None
            if handle is not None and not _advisory_lock(handle):
                handle.close()
                handle = None

        try:
            yield
        finally:
            if handle is not None:
                _advisory_unlock(handle)
                handle.close()
            if acquired:
                _release(lock_path, token)

    def _write_head(self, seq: int, record_hash: str) -> None:
        """Record where the chain currently ends.

        A hash chain proves that the records present are internally
        consistent. It cannot notice records that are *absent from the end*:
        deleting the last N lines leaves a shorter chain that still verifies,
        so the most recent history -- the part most worth erasing -- was the
        part the chain did not protect. Storing the expected head means a
        truncated log no longer matches what the ledger last committed to.

        This raises the bar rather than sealing it: an actor who can rewrite
        the log can also rewrite this file. Detecting that needs an attestation
        anchored outside the vault, which `receipt_attestation` provides for
        receipts and this deliberately does not reimplement.
        """

        # Reuses the vault's atomic write rather than repeating `os.replace`
        # here. A second copy was a second place to omit the jittered retry
        # that Windows needs when a reader holds the target open, and it
        # promptly failed under the concurrency test the first copy passes.
        # One implementation of "replace this file safely" is the point.
        from .vault import _atomic_write_text

        payload = json.dumps(
            {"schema": LEDGER_SCHEMA, "seq": int(seq), "record_sha256": record_hash},
            sort_keys=True,
        )
        _atomic_write_text(self._head, payload)

    def _append(self, record: dict[str, Any]) -> None:
        """Append one record and advance the head together.

        The single append path. Redaction appends a tombstone rather than
        rewriting history, and when only `record()` advanced the head that
        tombstone made the log look truncated -- a legitimate operation
        reported as tampering. Anything that appends must move the head, so
        there is one place that does both.
        """

        with self._log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self._write_head(record["seq"], record[_RECORD_HASH_FIELD])

    def _read_head(self) -> dict[str, Any] | None:
        if not self._head.exists():
            return None
        try:
            return json.loads(self._head.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

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

        # Held across read-tail, build and append: a record's seq and
        # prev_sha256 come from whatever is currently last, so reading outside
        # the lock lets two processes chain onto the same tail and lose one of
        # the two records.
        with self._exclusive():
            return self._record_locked(artifact, backfilled, tx_time, body_sha)

    def _record_locked(
        self, artifact: Any, backfilled: bool, tx_time: str | None, body_sha: str
    ) -> dict[str, Any]:
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
        self._append(record)
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

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        if not self._log.exists():
            return
        for line_no, line in enumerate(
            self._log.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerIntegrityError(
                    f"unreadable ledger record at line {line_no}: {exc}"
                ) from exc

    def _iter_versions(self) -> Iterator[BeliefVersion]:
        for rec in self._iter_records():
            if rec.get("kind") == "redaction":
                continue
            try:
                yield BeliefVersion.from_record(rec)
            except (KeyError, ValueError) as exc:
                raise LedgerIntegrityError(
                    f"unreadable ledger record seq={rec.get('seq')}: {exc}"
                ) from exc

    def _redaction_index(self) -> dict[int, str]:
        """seq -> redaction reason, from tombstone records."""
        index: dict[int, str] = {}
        for rec in self._iter_records():
            if rec.get("kind") == "redaction":
                for seq in rec.get("redacts_seqs", ()):
                    index[int(seq)] = str(rec.get("reason", ""))
        return index

    def _flag_redacted(self, version: BeliefVersion,
                       index: dict[int, str]) -> BeliefVersion:
        from dataclasses import replace
        if version.seq in index:
            return replace(version, redacted=True)
        return version

    def redact(self, *, claim_id: str = "", entity: str = "",
               reason: str = "user_requested_erasure") -> dict[str, Any]:
        """Erase belief content while keeping the hash chain verifiable.

        Deletes the content-addressed body objects of every belief version
        matching the selector and appends a chained tombstone record. The
        chain records themselves are never modified, so ``verify_chain``
        still passes — the *content* is provably gone, the *structure*
        (hashes, timestamps, entity labels, confidences) remains. If entity
        labels themselves are sensitive, that is a naming-policy decision at
        write time; a hash chain cannot un-say its metadata.

        A body object shared with a non-redacted version is NOT deleted
        (deleting it would corrupt the other belief); the redacted versions
        still refuse ``body_of`` by policy.
        """
        if bool(claim_id) == bool(entity):
            raise ValueError("redact requires exactly one of claim_id or entity")

        matched: list[dict[str, Any]] = []
        kept_shas: set[str] = set()
        already_redacted = self._redaction_index()
        for rec in self._iter_records():
            if rec.get("kind") == "redaction":
                continue
            is_match = (
                (claim_id and rec.get("claim_id") == claim_id)
                or (entity and rec.get("entity") == entity)
            )
            if is_match:
                matched.append(rec)
            elif int(rec["seq"]) not in already_redacted:
                kept_shas.add(str(rec.get("body_sha256", "")))
        if not matched:
            return {"status": "no_match", "claim_id": claim_id, "entity": entity}

        deleted_objects: list[str] = []
        retained_shared: list[str] = []
        for rec in matched:
            sha = str(rec.get("body_sha256", ""))
            obj = self._objects / f"{sha}.md"
            if sha in kept_shas:
                retained_shared.append(sha)
            elif obj.exists():
                obj.unlink()
                deleted_objects.append(sha)

        with self._exclusive():
            return self._tombstone_locked(matched, claim_id, entity, reason,
                                          deleted_objects, retained_shared)

    def _tombstone_locked(
        self, matched: list[dict[str, Any]], claim_id: str, entity: str,
        reason: str, deleted_objects: list[str], retained_shared: list[str],
    ) -> dict[str, Any]:
        last = self._last_record()
        tombstone = {
            "schema": LEDGER_SCHEMA,
            "kind": "redaction",
            "seq": (int(last["seq"]) + 1) if last else 1,
            "tx_time": _utc_now_iso(),
            "redacts_seqs": sorted(int(r["seq"]) for r in matched),
            "claim_id": claim_id,
            "entity": entity,
            "reason": reason,
            "deleted_objects": sorted(deleted_objects),
            "retained_shared_objects": sorted(set(retained_shared)),
            "prev_sha256": last[_RECORD_HASH_FIELD] if last else "",
        }
        tombstone[_RECORD_HASH_FIELD] = _record_hash(tombstone)
        # Through the same append path as a belief version: a redaction adds
        # to history rather than rewriting it, so it must advance the head too
        # or a lawful redaction reads as a truncated log.
        self._append(tombstone)
        return {
            "status": "redacted",
            "versions": len(matched),
            "objects_deleted": len(deleted_objects),
            "objects_retained_shared": len(set(retained_shared)),
            "tombstone_seq": tombstone["seq"],
        }

    def body_of(self, version: BeliefVersion) -> str:
        reason = self._redaction_index().get(version.seq)
        if reason is not None:
            raise BeliefRedactedError(
                f"body of '{version.entity}' seq={version.seq} was redacted: {reason}"
            )
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
        index = self._redaction_index()
        return {e: self._flag_redacted(v, index) for e, v in snapshot.items()}

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
        index = self._redaction_index()
        return sorted(
            (self._flag_redacted(v, index)
             for v in self._iter_versions() if v.entity == entity),
            key=lambda v: v.seq,
        )

    def verify_chain(self) -> dict[str, Any]:
        """Recompute the hash chain. Fail-closed: reports the first break.

        Held under the append lock. The log and the head are two files, and
        reading them at different instants let a concurrent append land in
        between: the log was counted before the append and the head read after
        it, so a healthy ledger reported "truncated: head expects 299, found
        298". A soak of three writers against a maintainer produced six such
        alarms in a minute, on a chain that was intact at the end.

        A tamper alarm that fires on healthy data is worse than none, so the
        verification takes a consistent snapshot rather than tolerating a
        window -- tolerating one would have blunted real truncation detection
        by exactly the amount of the tolerance.
        """

        if not self._log.exists():
            return {"status": "empty", "records": 0}
        with self._exclusive():
            return self._verify_locked()

    def _verify_locked(self) -> dict[str, Any]:
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

        # A chain proves the records present are consistent; it cannot notice
        # records missing from the *end*. Dropping the last N lines leaves a
        # shorter chain that still verifies, so the most recent history was
        # exactly the part the chain did not protect. Comparing against the
        # head the ledger last committed to closes that.
        head = self._read_head()
        if head is not None:
            expected = int(head.get("seq", 0))
            # Only a head expecting *more* than exists means records were
            # removed. A head expecting fewer is the opposite situation: the
            # log was appended and the process died before the head advanced,
            # leaving it one behind on a chain that is entirely intact.
            # Reporting that as tampering would raise a permanent false alarm
            # on a healthy ledger, and an alarm that cries wolf is worse than
            # none -- it is the reason the real one gets ignored.
            if expected > count:
                return {
                    "status": "broken",
                    "records": count,
                    "reason": f"ledger truncated: head expects {expected} "
                              f"record(s), found {count}",
                }
            if expected == count and head.get("record_sha256") != prev_hash:
                return {
                    "status": "broken",
                    "records": count,
                    "reason": "head does not match the final record",
                }
            if expected < count:
                return {
                    "status": "intact",
                    "records": count,
                    "note": f"head lagged at {expected}; the next append will "
                            "advance it",
                }

        return {"status": "intact", "records": count}
