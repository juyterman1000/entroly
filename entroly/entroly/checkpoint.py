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

Storage Format:
  JSON for human readability and debuggability. Gzipped for
  space efficiency. Typical checkpoint: 50-200 KB compressed.

References:
  - Agentic Plan Caching (arXiv 2025) — reusing structured plans
  - SagaLLM (arXiv 2025) — transactional guarantees for multi-agent planning
"""

from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from entroly_core import ContextFragment



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
    """fragment_id → SimHash fingerprint mapping."""

    co_access_data: Dict[str, Dict[str, int]]
    """Pre-fetcher co-access counts."""

    metadata: Dict[str, Any]
    """Custom metadata (task plan, current step, etc.)."""

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


class CheckpointManager:
    """
    Manages saving and restoring Entroly state.

    Checkpoints are stored as gzipped JSON files in the checkpoint
    directory. Each checkpoint includes the full state needed to
    resume a session without any data loss.

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
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_interval = auto_interval
        self.max_checkpoints = max_checkpoints

        self._tool_calls_since_checkpoint = 0
        self._total_checkpoints_created = 0

        # Ensure directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        Save a checkpoint to disk.

        Returns the checkpoint file path.
        """
        checkpoint_id = f"ckpt_{int(time.time())}_{self._total_checkpoints_created}"

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            current_turn=current_turn,
            fragments=[_fragment_to_dict(f) for f in fragments],
            dedup_fingerprints={k: v for k, v in dedup_fingerprints.items()},
            co_access_data={
                k: dict(v) for k, v in co_access_data.items()
            },
            metadata=metadata or {},
            stats=stats or {},
        )

        # Serialize to gzipped JSON
        filepath = self.checkpoint_dir / f"{checkpoint_id}.json.gz"
        data = json.dumps({
            "checkpoint_id": checkpoint.checkpoint_id,
            "timestamp": checkpoint.timestamp,
            "current_turn": checkpoint.current_turn,
            "fragments": checkpoint.fragments,
            "dedup_fingerprints": checkpoint.dedup_fingerprints,
            "co_access_data": checkpoint.co_access_data,
            "metadata": checkpoint.metadata,
            "stats": checkpoint.stats,
        }, separators=(",", ":"))

        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            f.write(data)

        self._tool_calls_since_checkpoint = 0
        self._total_checkpoints_created += 1

        # Enforce retention policy
        self._prune_old_checkpoints()

        return str(filepath)

    def load_latest(self) -> Optional[Checkpoint]:
        """
        Load the most recent checkpoint.

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
        """Load and parse a checkpoint file. Returns None if corrupted."""
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.loads(f.read())
        except (EOFError, gzip.BadGzipFile, json.JSONDecodeError, OSError):
            return None

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
        """Remove old checkpoints beyond the retention limit."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("ckpt_*.json.gz"),
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
        total_size = sum(cp.stat().st_size for cp in checkpoints)
        return {
            "total_checkpoints": len(checkpoints),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "tool_calls_since_last": self._tool_calls_since_checkpoint,
            "auto_interval": self.auto_interval,
        }
