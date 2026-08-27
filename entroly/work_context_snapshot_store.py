"""Hardened host-byte store for Work Graph context snapshots.

Semantic truth stays elsewhere:
- ContextReceipt/RecoveryHandle meaning is Rust-owned.
- Work Graph state stores only bounded receipt/handle references.
- This module stores the exact canonical host context JSON needed to service a
  later page fault without placing those source bytes inside the graph or MCP
  token.

A snapshot token is derived from the verified repository-context
``receipt.context_sha256``. The canonical Rust ContextReceipt already stores the
same value as ``source_commitment``; therefore another agent can reconstruct the
short snapshot locator from durable receipt evidence instead of depending on a
previous MCP response surviving verbatim.

The verified-context commitment intentionally excludes volatile host metadata
(``generation`` and ``command``). Those fields are stripped before persistence
as well, so one commitment maps to one stable byte representation rather than
multiple snapshots that merely differ in process-local metadata.

The store deliberately reuses ``WorkGraphStore``'s repository directory and
cross-process lock so context snapshots follow the same local isolation and
persistence boundary without inventing another coordination protocol.
"""
from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .work_graph_store import WorkGraphStore, WorkGraphStateError, _fsync_dir, _private_dir

CONTEXT_SNAPSHOT_TOKEN_PREFIX = "wctx1."
DEFAULT_MAX_CONTEXT_BYTES = 512 * 1024
DEFAULT_MAX_SNAPSHOTS = 8_192
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_VOLATILE_CONTEXT_FIELDS = frozenset({"generation", "command"})


class WorkContextSnapshotError(WorkGraphStateError):
    """A context snapshot is malformed, unsafe, missing, or exceeds bounds."""


def _verify_snapshot_bytes(raw: bytes, expected_digest: str) -> None:
    """Delegate v1 snapshot commitment semantics to the Rust kernel."""
    try:
        from entroly_core import verified_context_snapshot_verify_bytes
    except (ImportError, AttributeError) as exc:
        raise WorkContextSnapshotError(
            "native context snapshot verifier is unavailable"
        ) from exc
    try:
        actual = verified_context_snapshot_verify_bytes(raw, expected_digest)
    except (TypeError, ValueError) as exc:
        raise WorkContextSnapshotError(str(exc)) from exc
    if actual != expected_digest:
        raise WorkContextSnapshotError(
            "native context snapshot verifier returned a different commitment"
        )


class WorkContextSnapshotStore:
    """Repository-scoped exact context bytes keyed by verified context identity."""

    def __init__(
        self,
        graph_store: WorkGraphStore,
        *,
        max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    ) -> None:
        if not isinstance(graph_store, WorkGraphStore):
            raise TypeError("graph_store must be a WorkGraphStore")
        if isinstance(max_context_bytes, bool) or int(max_context_bytes) < 1_024:
            raise ValueError("max_context_bytes must be an integer >= 1024")
        if isinstance(max_snapshots, bool) or not 1 <= int(max_snapshots) <= 100_000:
            raise ValueError("max_snapshots must be an integer between 1 and 100000")
        if isinstance(max_total_bytes, bool) or int(max_total_bytes) < int(max_context_bytes):
            raise ValueError("max_total_bytes must be at least max_context_bytes")
        self.graph_store = graph_store
        self.max_context_bytes = int(max_context_bytes)
        self.max_snapshots = int(max_snapshots)
        self.max_total_bytes = int(max_total_bytes)
        self.context_dir = graph_store.repo_dir / "context-snapshots"
        _private_dir(self.context_dir)

    @staticmethod
    def _valid_digest(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(ch in _HEX for ch in value)
        )

    @classmethod
    def token_for_commitment(cls, commitment: str) -> str:
        """Derive the durable short locator carried by MCP/continuation state."""
        if not cls._valid_digest(commitment):
            raise WorkContextSnapshotError("context commitment is not a sha256 digest")
        return CONTEXT_SNAPSHOT_TOKEN_PREFIX + commitment

    @classmethod
    def _digest_from_token(cls, token: str) -> str:
        if not isinstance(token, str) or not token.startswith(CONTEXT_SNAPSHOT_TOKEN_PREFIX):
            raise WorkContextSnapshotError("unsupported context snapshot token")
        digest = token[len(CONTEXT_SNAPSHOT_TOKEN_PREFIX):]
        if not cls._valid_digest(digest):
            raise WorkContextSnapshotError("context snapshot token has an invalid digest")
        return digest

    @classmethod
    def _context_commitment(cls, payload: dict[str, Any]) -> str:
        receipt = payload.get("receipt")
        digest = receipt.get("context_sha256") if isinstance(receipt, dict) else None
        if not cls._valid_digest(digest):
            raise WorkContextSnapshotError(
                "context snapshot is missing a valid verified context commitment"
            )
        return str(digest)

    @staticmethod
    def _stable_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Return the exact semantic payload covered by ``context_sha256``.

        The verified-context commitment excludes only process-local
        ``generation``/``command`` fields. Persisting them under that commitment
        would violate content-address uniqueness, so storage follows the same
        scope exactly.
        """
        stable = copy.deepcopy(payload)
        for field in _VOLATILE_CONTEXT_FIELDS:
            stable.pop(field, None)
        return stable

    def _canonical_bytes(self, payload: dict[str, Any]) -> bytes:
        try:
            raw = json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("context snapshot must be canonical JSON") from exc
        if len(raw) > self.max_context_bytes:
            raise WorkContextSnapshotError(
                f"context snapshot exceeds {self.max_context_bytes} bytes"
            )
        return raw

    def _path(self, digest: str) -> Path:
        return self.context_dir / f"{digest}.json"

    def _read_path_unlocked(self, path: Path, expected_digest: str) -> dict[str, Any]:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise WorkContextSnapshotError("context snapshot is unavailable") from exc
        except OSError as exc:
            raise WorkContextSnapshotError(f"cannot inspect context snapshot: {exc}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise WorkContextSnapshotError("unsafe context snapshot path")
        if info.st_size > self.max_context_bytes:
            raise WorkContextSnapshotError("context snapshot exceeds its byte bound")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise WorkContextSnapshotError(f"cannot open context snapshot safely: {exc}") from exc
        try:
            current = os.fstat(fd)
            if not stat.S_ISREG(current.st_mode) or current.st_size > self.max_context_bytes:
                raise WorkContextSnapshotError("unsafe or oversized context snapshot")
            raw = bytearray()
            while len(raw) <= self.max_context_bytes:
                chunk = os.read(
                    fd,
                    min(1024 * 1024, self.max_context_bytes + 1 - len(raw)),
                )
                if not chunk:
                    break
                raw.extend(chunk)
        finally:
            os.close(fd)
        encoded = bytes(raw)
        if len(encoded) > self.max_context_bytes:
            raise WorkContextSnapshotError("context snapshot exceeds its byte bound")
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkContextSnapshotError("context snapshot is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise WorkContextSnapshotError("context snapshot root must be an object")
        if any(field in payload for field in _VOLATILE_CONTEXT_FIELDS):
            raise WorkContextSnapshotError("context snapshot contains volatile host metadata")
        # The store writes one canonical byte representation. A same-semantics
        # whitespace/key-order rewrite is still a storage mutation and is
        # rejected rather than silently normalized on read.
        if self._canonical_bytes(payload) != encoded:
            raise WorkContextSnapshotError("context snapshot bytes are not canonical")
        _verify_snapshot_bytes(encoded, expected_digest)
        return payload

    def _snapshot_usage_unlocked(self) -> tuple[int, int]:
        count = 0
        total_bytes = 0
        try:
            with os.scandir(self.context_dir) as entries:
                for entry in entries:
                    if entry.name.startswith(".context-") and entry.name.endswith(".tmp"):
                        continue
                    if not entry.name.endswith(".json"):
                        continue
                    try:
                        info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise WorkContextSnapshotError(
                            f"cannot inspect context snapshot entry: {exc}"
                        ) from exc
                    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                        raise WorkContextSnapshotError("unsafe context snapshot entry")
                    count += 1
                    total_bytes += int(info.st_size)
                    if count > self.max_snapshots or total_bytes > self.max_total_bytes:
                        break
        except OSError as exc:
            raise WorkContextSnapshotError(f"cannot enumerate context snapshots: {exc}") from exc
        return count, total_bytes

    def put_json(self, payload: dict[str, Any]) -> str:
        """Persist semantic context bytes and return ``wctx1.<context_sha256>``."""
        if not isinstance(payload, dict):
            raise ValueError("context snapshot payload must be an object")
        digest = self._context_commitment(payload)
        stable = self._stable_payload(payload)
        raw = self._canonical_bytes(stable)
        _verify_snapshot_bytes(raw, digest)
        target = self._path(digest)
        with self.graph_store.lock():
            _private_dir(self.context_dir)
            if target.exists() or target.is_symlink():
                existing = self._read_path_unlocked(target, digest)
                if self._canonical_bytes(existing) != raw:
                    raise WorkContextSnapshotError(
                        "context commitment maps to conflicting stable snapshot bytes"
                    )
                return self.token_for_commitment(digest)
            count, total_bytes = self._snapshot_usage_unlocked()
            if count >= self.max_snapshots:
                raise WorkContextSnapshotError(
                    "context snapshot store reached its bounded entry limit"
                )
            if total_bytes + len(raw) > self.max_total_bytes:
                raise WorkContextSnapshotError(
                    "context snapshot store reached its bounded byte limit"
                )
            fd, name = tempfile.mkstemp(
                prefix=".context-", suffix=".tmp", dir=self.context_dir
            )
            temp_path = Path(name)
            try:
                if os.name == "posix":
                    os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb", closefd=True) as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                fd = -1
                if target.exists() or target.is_symlink():
                    existing = self._read_path_unlocked(target, digest)
                    if self._canonical_bytes(existing) != raw:
                        raise WorkContextSnapshotError(
                            "context commitment maps to conflicting stable snapshot bytes"
                        )
                else:
                    os.replace(temp_path, target)
                    if os.name == "posix":
                        target.chmod(0o600)
                    _fsync_dir(self.context_dir)
            finally:
                if fd >= 0:
                    os.close(fd)
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return self.token_for_commitment(digest)

    def get_json(self, token: str) -> dict[str, Any]:
        """Load only canonical bytes whose inner verified commitment matches."""
        digest = self._digest_from_token(token)
        with self.graph_store.lock():
            _private_dir(self.context_dir)
            return self._read_path_unlocked(self._path(digest), digest)


__all__ = [
    "CONTEXT_SNAPSHOT_TOKEN_PREFIX",
    "DEFAULT_MAX_CONTEXT_BYTES",
    "DEFAULT_MAX_SNAPSHOTS",
    "DEFAULT_MAX_TOTAL_BYTES",
    "WorkContextSnapshotError",
    "WorkContextSnapshotStore",
]
