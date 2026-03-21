"""
Checkpoint & Resume System
===========================

Serializes the full agent state to disk so that multi-step tasks
can resume from the last checkpoint instead of restarting from scratch.

The Problem:
  An agent working on a 10-step refactoring task fails at step 7
  (API timeout, context overflow, rate limit). Today, the developer
  must restart the entire task — re-reading files, re-planning,
  re-executing steps 1-6 — wasting time and tokens.

The Solution:
  Entroly automatically checkpoints after every N tool calls:
    - All tracked context fragments (with scores)
    - The dedup index state
    - Co-access patterns from the pre-fetcher
    - Custom metadata (task plan, current step, etc.)

  On resume, the full state is restored in <100ms, and the agent
  picks up exactly where it left off.

Multi-Instance Support:
  Each CheckpointManager has an instance_id (hostname + PID).
  Checkpoints include the instance_id in metadata so multiple
  entroly instances on the same project can share learned state.
  merge_from_peers() scans for other instances' checkpoints and
  merges fragments using most-recent-writer-wins with access_count
  tiebreaker. Inspired by agentOS Pitman-Yor lifecycle management.

Storage Format:
  JSON for human readability and debuggability. Gzipped for
  space efficiency. Typical checkpoint: 50-200 KB compressed.

References:
  - Agentic Plan Caching (arXiv 2025) -- reusing structured plans
  - SagaLLM (arXiv 2025) -- transactional guarantees for multi-agent planning
"""

from __future__ import annotations

import gzip
import json
import math
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from entroly_core import ContextFragment

# Checkpoint schema version — increment when serialization format changes.
# Migration functions handle loading older versions.
CHECKPOINT_SCHEMA_VERSION = 2


@dataclass
class Checkpoint:
    """A serialized snapshot of the Entroly state."""

    checkpoint_id: str
    """Unique ID for this checkpoint (timestamp-based)."""

    timestamp: float
    """Unix timestamp when this checkpoint was created."""

    current_turn: int
    """The turn number at checkpoint time."""

    fragments: List[Dict[str, Any]]
    """Serialized context fragments."""

    dedup_fingerprints: Dict[str, int]
    """fragment_id -> SimHash fingerprint mapping."""

    co_access_data: Dict[str, Dict[str, int]]
    """Pre-fetcher co-access counts."""

    metadata: Dict[str, Any]
    """Custom metadata (task plan, current step, instance_id, etc.)."""

    stats: Dict[str, Any]
    """Performance stats at checkpoint time."""


def _fragment_to_dict(frag: ContextFragment) -> Dict[str, Any]:
    """Serialize a ContextFragment to a JSON-safe dict."""
    return {
        "fragment_id": frag.fragment_id,
        "content": frag.content,
        "token_count": frag.token_count,
        "source": frag.source,
        "recency_score": round(frag.recency_score, 6),
        "frequency_score": round(frag.frequency_score, 6),
        "semantic_score": round(frag.semantic_score, 6),
        "entropy_score": round(frag.entropy_score, 6),
        "turn_created": frag.turn_created,
        "turn_last_accessed": frag.turn_last_accessed,
        "access_count": frag.access_count,
        "is_pinned": frag.is_pinned,
        "simhash": frag.simhash,
    }


def _dict_to_fragment(d: Dict[str, Any]) -> ContextFragment:
    """Deserialize a dict back to a ContextFragment."""
    frag = ContextFragment(
        fragment_id=d["fragment_id"],
        content=d["content"],
        token_count=d["token_count"],
        source=d.get("source", ""),
    )
    frag.recency_score = d.get("recency_score", 0.0)
    frag.frequency_score = d.get("frequency_score", 0.0)
    frag.semantic_score = d.get("semantic_score", 0.0)
    frag.entropy_score = d.get("entropy_score", 0.5)
    frag.turn_created = d.get("turn_created", 0)
    frag.turn_last_accessed = d.get("turn_last_accessed", 0)
    frag.access_count = d.get("access_count", 0)
    frag.is_pinned = d.get("is_pinned", False)
    frag.simhash = d.get("simhash", 0)
    return frag


def _merge_fragments(
    local: List[Dict[str, Any]], remote: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge fragment lists from different instances.

    Conflict resolution: most-recent-writer-wins with access_count tiebreaker.
    Inspired by Pitman-Yor lifecycle management from agentOS/persona_manifold.rs.
    """
    merged: Dict[str, Dict[str, Any]] = {f["fragment_id"]: f for f in local}

    for rf in remote:
        fid = rf["fragment_id"]
        if fid not in merged:
            merged[fid] = rf
        else:
            local_f = merged[fid]
            # Prefer the version with more recent access
            if rf.get("turn_last_accessed", 0) > local_f.get("turn_last_accessed", 0):
                merged[fid] = rf
            elif rf.get("turn_last_accessed", 0) == local_f.get("turn_last_accessed", 0):
                # Tiebreak: higher access_count wins
                if rf.get("access_count", 0) > local_f.get("access_count", 0):
                    merged[fid] = rf

    return list(merged.values())


_LOCK_SIZE = 1024  # Lock region size for Windows msvcrt


def _acquire_file_lock(lock_path: Path) -> Any:
    """Acquire a file lock for distributed checkpoint safety.

    Uses fcntl on POSIX, msvcrt on Windows.
    Returns the lock file handle (caller must close to release).
    """
    lock_file = open(lock_path, "w")
    try:
        import fcntl
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (ImportError, OSError):
        try:
            import msvcrt
            # Seek to start and lock a meaningful region
            lock_file.seek(0)
            lock_file.write(" " * _LOCK_SIZE)
            lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, _LOCK_SIZE)
        except (ImportError, OSError):
            pass  # Best-effort: if locking unavailable, proceed without
    return lock_file


def _release_file_lock(lock_file: Any) -> None:
    """Release a file lock."""
    try:
        import fcntl
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        try:
            import msvcrt
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, _LOCK_SIZE)
        except (ImportError, OSError):
            pass
    lock_file.close()


# Ebbinghaus decay constant (ported from agentOS/persona_manifold.rs)
_EBBINGHAUS_LAMBDA = 0.01


class CheckpointManager:
    """
    Manages saving and restoring Entroly state.

    Checkpoints are stored as gzipped JSON files in the checkpoint
    directory. Each checkpoint includes the full state needed to
    resume a session without any data loss.

    Multi-instance: Each manager has an instance_id. Checkpoints from
    peer instances can be merged via merge_from_peers().

    Auto-checkpoint:
      If auto_interval is set, the manager automatically creates
      a checkpoint every N tool calls. This provides crash recovery
      without explicit save calls.

    Retention:
      Keeps the last `max_checkpoints` checkpoints and deletes older
      ones to prevent unbounded disk usage.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        auto_interval: int = 5,
        max_checkpoints: int = 10,
        instance_id: str | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_interval = auto_interval
        self.max_checkpoints = max_checkpoints
        self.instance_id = instance_id or f"{socket.gethostname()}_{os.getpid()}"

        self._tool_calls_since_checkpoint = 0
        self._total_checkpoints_created = 0

        # Ensure directory exists with restricted permissions (0700)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(self.checkpoint_dir), 0o700)
        except OSError:
            pass  # Best-effort on platforms without chmod

    def should_auto_checkpoint(self) -> bool:
        """Check if an auto-checkpoint is due."""
        self._tool_calls_since_checkpoint += 1
        return self._tool_calls_since_checkpoint >= self.auto_interval

    def save(
        self,
        fragments: List[ContextFragment],
        dedup_fingerprints: Dict[str, int],
        co_access_data: Dict[str, Dict[str, int]],
        current_turn: int,
        metadata: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint to disk with distributed file locking.

        Returns the checkpoint file path.
        """
        checkpoint_id = f"ckpt_{self.instance_id}_{int(time.time())}_{self._total_checkpoints_created}"

        meta = metadata or {}
        meta["instance_id"] = self.instance_id

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            current_turn=current_turn,
            fragments=[_fragment_to_dict(f) for f in fragments],
            dedup_fingerprints={k: v for k, v in dedup_fingerprints.items()},
            co_access_data={
                k: dict(v) for k, v in co_access_data.items()
            },
            metadata=meta,
            stats=stats or {},
        )

        # Disk-space check: bail early if < 10MB free (prevents silent corruption)
        try:
            import shutil as _shutil
            disk_usage = _shutil.disk_usage(str(self.checkpoint_dir))
            free_bytes = disk_usage.free
            if free_bytes < 10 * 1024 * 1024:  # 10 MB minimum
                import logging
                logging.getLogger("entroly").warning(
                    f"Checkpoint skipped: only {free_bytes // 1024}KB free on disk"
                )
                return ""
        except (OSError, AttributeError):
            pass  # statvfs not available on Windows; proceed best-effort

        # Serialize to gzipped JSON
        filepath = self.checkpoint_dir / f"{checkpoint_id}.json.gz"
        data = json.dumps({
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_id": checkpoint.checkpoint_id,
            "timestamp": checkpoint.timestamp,
            "current_turn": checkpoint.current_turn,
            "fragments": checkpoint.fragments,
            "dedup_fingerprints": checkpoint.dedup_fingerprints,
            "co_access_data": checkpoint.co_access_data,
            "metadata": checkpoint.metadata,
            "stats": checkpoint.stats,
        }, separators=(",", ":"))

        # Distributed file lock + atomic write
        lock_path = self.checkpoint_dir / ".checkpoint.lock"
        lock_file = _acquire_file_lock(lock_path)
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.checkpoint_dir), suffix=".json.gz.tmp"
            )
            try:
                os.close(tmp_fd)
                with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                # Restrict permissions before moving into place (0600 = owner-only)
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    pass  # Windows may not support chmod
                os.replace(tmp_path, str(filepath))  # atomic on POSIX + Windows
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        finally:
            _release_file_lock(lock_file)

        self._tool_calls_since_checkpoint = 0
        self._total_checkpoints_created += 1

        # Enforce retention policy
        self._prune_old_checkpoints()

        return str(filepath)

    def load_latest(self) -> Optional[Checkpoint]:
        """
        Load the most recent checkpoint (from this instance).

        Returns None if no checkpoints exist or all are unreadable.
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for cp in checkpoints:
            result = self._load_file(cp)
            if result is not None:
                return result

        return None

    def load_by_id(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a specific checkpoint by its ID."""
        filepath = self.checkpoint_dir / f"{checkpoint_id}.json.gz"
        if not filepath.exists():
            return None
        return self._load_file(filepath)

    def merge_from_peers(self, local_fragments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scan for checkpoints from peer instances and merge their fragments.

        Returns the merged fragment list combining local + peer knowledge.
        Uses most-recent-writer-wins conflict resolution.
        """
        all_checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        merged = list(local_fragments)
        seen_instances = {self.instance_id}

        for cp_path in all_checkpoints:
            loaded = self._load_file(cp_path)
            if loaded is None:
                continue

            peer_id = loaded.metadata.get("instance_id", "")
            if peer_id in seen_instances or peer_id == self.instance_id:
                continue

            seen_instances.add(peer_id)
            # Merge this peer's fragments (one checkpoint per peer, most recent)
            merged = _merge_fragments(merged, loaded.fragments)

        return merged

    def apply_ebbinghaus_decay(
        self, fragments: List[Dict[str, Any]], current_tick: int
    ) -> List[Dict[str, Any]]:
        """Apply Ebbinghaus forgetting curve to fragment health scores.

        Ported from agentOS/persona_manifold.rs lifecycle_tick().
        Fragments with health < 0.05 are evicted.
        """
        result = []
        for frag in fragments:
            dt = max(0, current_tick - frag.get("turn_last_accessed", 0))
            health = math.exp(-_EBBINGHAUS_LAMBDA * dt)
            if health >= 0.05:
                result.append(frag)
        return result

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json.gz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        result = []
        for cp_path in checkpoints:
            try:
                stat = cp_path.stat()
                result.append({
                    "checkpoint_id": cp_path.stem.replace(".json", ""),
                    "path": str(cp_path),
                    "size_bytes": stat.st_size,
                    "created": stat.st_mtime,
                })
            except OSError:
                continue

        return result

    def restore_fragments(self, checkpoint: Checkpoint) -> List[ContextFragment]:
        """Extract ContextFragment objects from a checkpoint."""
        return [_dict_to_fragment(d) for d in checkpoint.fragments]

    def _load_file(self, filepath: Path) -> Optional[Checkpoint]:
        """Load and parse a checkpoint file with schema migration. Returns None if corrupted."""
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.loads(f.read())
        except (EOFError, gzip.BadGzipFile, json.JSONDecodeError, OSError):
            return None

        # Schema migration
        version = data.get("schema_version", 1)
        if version < 2:
            # v1 -> v2: add instance_id to metadata
            if "metadata" not in data:
                data["metadata"] = {}
            if "instance_id" not in data.get("metadata", {}):
                data["metadata"]["instance_id"] = "migrated_v1"

        return Checkpoint(
            checkpoint_id=data["checkpoint_id"],
            timestamp=data["timestamp"],
            current_turn=data["current_turn"],
            fragments=data["fragments"],
            dedup_fingerprints=data.get("dedup_fingerprints", {}),
            co_access_data=data.get("co_access_data", {}),
            metadata=data.get("metadata", {}),
            stats=data.get("stats", {}),
        )

    def _prune_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the retention limit (own instance only)."""
        # Only prune this instance's checkpoints, leave peers' alone
        own_pattern = f"ckpt_{self.instance_id}_*.json.gz"
        checkpoints = sorted(
            self.checkpoint_dir.glob(own_pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_cp in checkpoints[self.max_checkpoints:]:
            try:
                old_cp.unlink()
            except OSError:
                pass

    def stats(self) -> dict:
        checkpoints = list(self.checkpoint_dir.glob("ckpt_*.json.gz"))
        own_checkpoints = [
            cp for cp in checkpoints
            if cp.name.startswith(f"ckpt_{self.instance_id}_")
        ]
        peer_checkpoints = len(checkpoints) - len(own_checkpoints)
        total_size = sum(cp.stat().st_size for cp in checkpoints)
        return {
            "instance_id": self.instance_id,
            "total_checkpoints": len(checkpoints),
            "own_checkpoints": len(own_checkpoints),
            "peer_checkpoints": peer_checkpoints,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "tool_calls_since_last": self._tool_calls_since_checkpoint,
            "auto_interval": self.auto_interval,
        }
