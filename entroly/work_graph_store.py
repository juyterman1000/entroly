"""Durable cross-process storage for Entroly's shared Rust AI Work Graph.

Persistence mechanics live here; graph semantics remain in ``entroly-engine``.
Python and Node use the same repo-keyed layout, exclusive-create lock protocol,
and canonical Rust JSON so independent agent processes converge on one graph.
"""

from __future__ import annotations

import hashlib
import math
import os
import socket
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .work_graph import WorkGraph
from .work_graph_repo import discover_repository_observation

DEFAULT_LOCK_TIMEOUT_SECONDS = 5.0
DEFAULT_STALE_LOCK_SECONDS = 120.0
DEFAULT_LOCK_SETTLE_SECONDS = 1.0
DEFAULT_MAX_STATE_BYTES = 64 * 1024 * 1024


class WorkGraphStoreError(RuntimeError):
    pass


class WorkGraphLockTimeout(WorkGraphStoreError):
    pass


class WorkGraphStateError(WorkGraphStoreError):
    pass


def _store_root() -> Path:
    configured = os.environ.get("ENTROLY_DIR")
    base = Path(configured).expanduser() if configured else Path.home() / ".entroly"
    return base / "work-graphs"


def _repo_key(repo_id: str) -> str:
    if not repo_id:
        raise WorkGraphStateError("repo_id must not be empty")
    return hashlib.sha256(repo_id.encode("utf-8")).hexdigest()[:32]


def _finite_nonnegative(value: float | int, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkGraphStateError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise WorkGraphStateError(f"{name} must be a finite non-negative number")
    return number


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise WorkGraphStateError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WorkGraphStateError(f"{name} must be a positive integer") from exc
    if number < 1:
        raise WorkGraphStateError(f"{name} must be a positive integer")
    return number


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.lstat()
    except OSError as exc:
        raise WorkGraphStoreError(f"cannot inspect Work Graph directory {path}: {exc}") from exc
    if path.is_symlink() or not path.is_dir():
        raise WorkGraphStateError(f"unsafe Work Graph directory: {path}")
    if os.name == "posix":
        try:
            path.chmod(0o700)
        except OSError as exc:
            raise WorkGraphStoreError(f"cannot secure Work Graph directory {path}: {exc}") from exc


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(path, getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class WorkGraphStore:
    """Atomic persistence and advisory multi-agent coordination for one repo."""

    def __init__(
        self,
        repo_id: str,
        *,
        root: str | os.PathLike[str] | None = None,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
        stale_lock_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
        max_state_bytes: int = DEFAULT_MAX_STATE_BYTES,
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root).expanduser() if root is not None else _store_root()
        self.repo_dir = self.root / _repo_key(repo_id)
        self.state_path = self.repo_dir / "state.json"
        self.lock_path = self.repo_dir / ".lock"
        self.lock_timeout_seconds = _finite_nonnegative(
            lock_timeout_seconds, "lock_timeout_seconds"
        )
        self.stale_lock_seconds = max(
            1.0, _finite_nonnegative(stale_lock_seconds, "stale_lock_seconds")
        )
        self.max_state_bytes = max(1024, _positive_int(max_state_bytes, "max_state_bytes"))
        _private_dir(self.root)
        _private_dir(self.repo_dir)

    @classmethod
    def for_repository(
        cls, path: str | os.PathLike[str] = ".", **options: Any
    ) -> "WorkGraphStore":
        observation = discover_repository_observation(path, observed_at_ms=0)
        return cls(observation["repo_id"], **options)

    def _lock_token(self) -> str:
        try:
            if self.lock_path.is_symlink():
                raise WorkGraphStateError(f"unsafe Work Graph lock path: {self.lock_path}")
            return self.lock_path.read_text(encoding="utf-8").split("\n", 1)[0]
        except FileNotFoundError:
            return ""
        except OSError:
            return ""

    def _filesystem_now(self) -> float:
        probe = self.repo_dir / f".clock-{uuid.uuid4().hex}"
        fd: int | None = None
        try:
            fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            fd = None
            return probe.stat().st_mtime
        except OSError as exc:
            raise WorkGraphStoreError(f"cannot sample Work Graph filesystem clock: {exc}") from exc
        finally:
            if fd is not None:
                os.close(fd)
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass

    def _stale_lock(self) -> bool:
        try:
            stat = self.lock_path.lstat()
        except FileNotFoundError:
            return False
        except OSError:
            return False
        if self.lock_path.is_symlink() or not self.lock_path.is_file():
            raise WorkGraphStateError(f"unsafe Work Graph lock path: {self.lock_path}")
        first = stat.st_mtime
        if self._filesystem_now() - first < self.stale_lock_seconds:
            return False
        time.sleep(DEFAULT_LOCK_SETTLE_SECONDS)
        try:
            second = self.lock_path.stat().st_mtime
        except OSError:
            return False
        return second == first and self._filesystem_now() - second >= self.stale_lock_seconds

    def _try_lock(self, token: str) -> bool:
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        except OSError as exc:
            raise WorkGraphStoreError(f"cannot acquire Work Graph lock: {exc}") from exc
        try:
            os.write(fd, f"{token}\n{time.time():.6f}\n".encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return True

    def _break_stale_lock(self) -> bool:
        if not self._stale_lock():
            return False
        try:
            self.lock_path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    @contextmanager
    def lock(self) -> Iterator[None]:
        deadline = time.monotonic() + self.lock_timeout_seconds
        token = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        delay = 0.001
        while not self._try_lock(token):
            if self._break_stale_lock():
                continue
            if time.monotonic() >= deadline:
                raise WorkGraphLockTimeout(
                    f"timed out acquiring Work Graph lock for {self.repo_id}"
                )
            jitter = 0.5 + int.from_bytes(os.urandom(2), "big") / 131070.0
            time.sleep(delay * jitter)
            delay = min(delay * 2.0, 0.05)
        try:
            yield
        finally:
            if self._lock_token() == token:
                try:
                    self.lock_path.unlink()
                except OSError:
                    pass

    def _load_unlocked(self) -> WorkGraph:
        if not self.state_path.exists():
            return WorkGraph(self.repo_id)
        if self.state_path.is_symlink():
            raise WorkGraphStateError(f"refusing symlink Work Graph state: {self.state_path}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.state_path, flags)
        except FileNotFoundError:
            return WorkGraph(self.repo_id)
        except OSError as exc:
            raise WorkGraphStateError(f"cannot open Work Graph state safely: {exc}") from exc
        try:
            stat = os.fstat(fd)
            if stat.st_size > self.max_state_bytes:
                raise WorkGraphStateError(
                    f"Work Graph state is {stat.st_size} bytes; limit is {self.max_state_bytes}"
                )
            payload = bytearray()
            while len(payload) <= self.max_state_bytes:
                chunk = os.read(fd, min(1024 * 1024, self.max_state_bytes + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) > self.max_state_bytes:
                raise WorkGraphStateError(f"Work Graph state exceeds {self.max_state_bytes} bytes")
        finally:
            os.close(fd)
        try:
            graph = WorkGraph.from_json(bytes(payload).decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RuntimeError) as exc:
            raise WorkGraphStateError(f"cannot load Work Graph state: {exc}") from exc
        if graph.repo_id != self.repo_id:
            raise WorkGraphStateError(
                f"stored Work Graph repo mismatch: expected {self.repo_id}, got {graph.repo_id}"
            )
        return graph

    def _save_unlocked(self, graph: WorkGraph) -> None:
        if graph.repo_id != self.repo_id:
            raise WorkGraphStateError(
                f"cannot persist foreign Work Graph: expected {self.repo_id}, got {graph.repo_id}"
            )
        payload = graph.export_json().encode("utf-8")
        if len(payload) > self.max_state_bytes:
            raise WorkGraphStateError(
                f"Work Graph state is {len(payload)} bytes; limit is {self.max_state_bytes}"
            )
        _private_dir(self.repo_dir)
        fd, name = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=self.repo_dir)
        temp_path = Path(name)
        try:
            if os.name == "posix":
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.state_path)
            if os.name == "posix":
                self.state_path.chmod(0o600)
            _fsync_dir(self.repo_dir)
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def load(self) -> WorkGraph:
        with self.lock():
            return self._load_unlocked()

    def save(self, graph: WorkGraph) -> WorkGraph:
        with self.lock():
            current = self._load_unlocked()
            current.merge(graph)
            self._save_unlocked(current)
            return current

    def submit_observation(self, observation: dict[str, Any]) -> WorkGraph:
        if observation.get("repo_id") != self.repo_id:
            raise WorkGraphStateError(
                f"repository identity changed: expected {self.repo_id}, got {observation.get('repo_id')}"
            )
        with self.lock():
            graph = self._load_unlocked()
            graph.observe_repository(observation)
            self._save_unlocked(graph)
            return graph

    def update_repository(self, path: str | os.PathLike[str] = ".", **options: Any) -> WorkGraph:
        return self.submit_observation(discover_repository_observation(path, **options))

    def claim_work(
        self,
        path: str | os.PathLike[str],
        *,
        agent_id: str,
        task_title: str,
        task_id: str = "",
        session_id: str = "",
        scope_paths: list[str] | tuple[str, ...] = (),
        scope_symbols: list[str] | tuple[str, ...] = (),
        ttl_seconds: float = 900.0,
        lease_id: str | None = None,
        observed_at_ms: int | None = None,
    ) -> tuple[WorkGraph, str]:
        if not agent_id.strip() or not task_title.strip():
            raise WorkGraphStateError("agent_id and task_title must not be empty")
        now_ms = int(observed_at_ms if observed_at_ms is not None else time.time() * 1000)
        ttl_ms = max(1, int(_finite_nonnegative(ttl_seconds, "ttl_seconds") * 1000))
        selected = lease_id or uuid.uuid4().hex
        observation = discover_repository_observation(
            path,
            agent_id=agent_id,
            session_id=session_id,
            task_hint={
                "task_id": task_id,
                "title": task_title,
                "trust": "observed",
                "explicit_status": "in_progress",
                "remaining_work": [],
                "source_kind": "user_statement",
                "source_ref": f"work-claim:{selected}",
            },
            observed_at_ms=now_ms,
        )
        observation["leases"] = [{
            "lease_id": selected,
            "agent_id": agent_id,
            "task_id": task_id,
            "scope_paths": sorted(set(scope_paths)),
            "scope_symbols": sorted(set(scope_symbols)),
            "expires_at_ms": now_ms + ttl_ms,
            "source_ref": f"work-lease:{selected}",
        }]
        return self.submit_observation(observation), selected

    def coordination(self, *, now_ms: int | None = None) -> dict[str, Any]:
        timestamp = int(now_ms if now_ms is not None else time.time() * 1000)
        return self.load().coordination(timestamp)

    def resume(self, workstream_id: str | None = None, *, max_evidence: int = 128) -> dict[str, Any]:
        return self.load().resume(workstream_id, max_evidence=max_evidence)

    def handoff(
        self,
        workstream_id: str,
        from_agent: str,
        to_agent: str,
        *,
        generated_at_ms: int | None = None,
    ) -> dict[str, Any]:
        timestamp = int(generated_at_ms if generated_at_ms is not None else time.time() * 1000)
        return self.load().handoff(workstream_id, from_agent, to_agent, timestamp)


__all__ = [
    "WorkGraphLockTimeout",
    "WorkGraphStateError",
    "WorkGraphStore",
    "WorkGraphStoreError",
]
