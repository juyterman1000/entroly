"""Hardened host-byte store for Work Graph context snapshots.

Semantic truth stays elsewhere:
- ContextReceipt/RecoveryHandle meaning is Rust-owned.
- Work Graph state stores only bounded receipt/handle references.
- This module stores the exact host context JSON needed to service a later
  page fault without placing those source bytes inside the graph or MCP token.

The store deliberately reuses ``WorkGraphStore``'s repository directory and
cross-process lock so context snapshots follow the same local isolation and
persistence boundary without inventing another coordination protocol.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .work_graph_store import WorkGraphStore, WorkGraphStateError, _fsync_dir, _private_dir

CONTEXT_SNAPSHOT_TOKEN_PREFIX = "wctx1."
DEFAULT_MAX_CONTEXT_BYTES = 512 * 1024
DEFAULT_MAX_SNAPSHOTS = 1_024


class WorkContextSnapshotError(WorkGraphStateError):
    """A context snapshot is malformed, unsafe, missing, or exceeds bounds."""


class WorkContextSnapshotStore:
    """Content-addressed exact context bytes adjacent to one Work Graph store."""

    def __init__(
        self,
        graph_store: WorkGraphStore,
        *,
        max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
    ) -> None:
        if not isinstance(graph_store, WorkGraphStore):
            raise TypeError("graph_store must be a WorkGraphStore")
        if isinstance(max_context_bytes, bool) or int(max_context_bytes) < 1_024:
            raise ValueError("max_context_bytes must be an integer >= 1024")
        if isinstance(max_snapshots, bool) or not 1 <= int(max_snapshots) <= 100_000:
            raise ValueError("max_snapshots must be an integer between 1 and 100000")
        self.graph_store = graph_store
        self.max_context_bytes = int(max_context_bytes)
        self.max_snapshots = int(max_snapshots)
        self.context_dir = graph_store.repo_dir / "context-snapshots"
        _private_dir(self.context_dir)

    @staticmethod
    def _blob_digest(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def _token(cls, digest: str) -> str:
        return CONTEXT_SNAPSHOT_TOKEN_PREFIX + digest

    @staticmethod
    def _digest_from_token(token: str) -> str:
        if not isinstance(token, str) or not token.startswith(CONTEXT_SNAPSHOT_TOKEN_PREFIX):
            raise WorkContextSnapshotError("unsupported context snapshot token")
        digest = token[len(CONTEXT_SNAPSHOT_TOKEN_PREFIX):]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise WorkContextSnapshotError("context snapshot token has an invalid digest")
        return digest

    def _path(self, digest: str) -> Path:
        return self.context_dir / f"{digest}.json"

    def _read_path_unlocked(self, path: Path, expected_digest: str) -> bytes:
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
        payload = bytes(raw)
        if len(payload) > self.max_context_bytes:
            raise WorkContextSnapshotError("context snapshot exceeds its byte bound")
        actual = self._blob_digest(payload)
        if actual != expected_digest:
            raise WorkContextSnapshotError(
                "context snapshot content does not match its content address"
            )
        return payload

    def _snapshot_count_unlocked(self) -> int:
        count = 0
        try:
            with os.scandir(self.context_dir) as entries:
                for entry in entries:
                    if entry.name.startswith(".context-") and entry.name.endswith(".tmp"):
                        continue
                    if entry.name.endswith(".json"):
                        count += 1
                        if count > self.max_snapshots:
                            break
        except OSError as exc:
            raise WorkContextSnapshotError(f"cannot enumerate context snapshots: {exc}") from exc
        return count

    def put_json(self, payload: dict[str, Any]) -> str:
        """Persist exact canonical JSON bytes and return a short content token."""
        if not isinstance(payload, dict):
            raise ValueError("context snapshot payload must be an object")
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
        digest = self._blob_digest(raw)
        target = self._path(digest)
        with self.graph_store.lock():
            _private_dir(self.context_dir)
            if target.exists() or target.is_symlink():
                existing = self._read_path_unlocked(target, digest)
                if existing != raw:
                    raise WorkContextSnapshotError(
                        "content-addressed context snapshot bytes disagree"
                    )
                return self._token(digest)
            if self._snapshot_count_unlocked() >= self.max_snapshots:
                raise WorkContextSnapshotError(
                    "context snapshot store reached its bounded entry limit"
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
                    if existing != raw:
                        raise WorkContextSnapshotError(
                            "content-addressed context snapshot bytes disagree"
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
        return self._token(digest)

    def get_json(self, token: str) -> dict[str, Any]:
        """Load a token only when its exact stored bytes still hash correctly."""
        digest = self._digest_from_token(token)
        with self.graph_store.lock():
            _private_dir(self.context_dir)
            raw = self._read_path_unlocked(self._path(digest), digest)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkContextSnapshotError("context snapshot is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise WorkContextSnapshotError("context snapshot root must be an object")
        return payload


__all__ = [
    "CONTEXT_SNAPSHOT_TOKEN_PREFIX",
    "DEFAULT_MAX_CONTEXT_BYTES",
    "DEFAULT_MAX_SNAPSHOTS",
    "WorkContextSnapshotError",
    "WorkContextSnapshotStore",
]
