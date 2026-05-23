"""
Causal Attribution for PRISM feedback
=====================================

Fixes the credit-assignment bug where ``record_outcome(success=True)``
reinforces every fragment the caller passes in — regardless of whether
those fragments actually contributed to the work that was done.

The original contract bound feedback to **retrieval output**. A
positive outcome therefore rewarded fragments that may have been
ignored entirely (e.g. when the user solved the task via Grep, not via
the surfaced context). Over time this drifts PRISM toward
"popular-but-irrelevant" fragments — a self-reinforcing failure mode
identical to filter-bubble collapse in recommender systems.

This module rebinds feedback to **causal evidence of use**, using a
git working-tree diff between optimize-time and outcome-time as the
observable proxy:

    fragment was used  ⇐ source_path(fragment) ∈ modified_files

Statistical claim:
  The legacy estimator
        ŵ_legacy(f) = E[reward | retrieved(f)]
  is a biased estimator of the quantity we actually want, namely
        ŵ_target(f) = E[reward | used(f)]
  because P(used | retrieved) ≪ 1 in practice.

  Conditioning on the modification proxy yields the corrected estimator
        ŵ_causal(f) = E[reward · 1{source(f) ∈ modified_files}
                        | retrieved(f), modified_files ≠ ∅]
  which is unbiased w.r.t. ŵ_target up to the false-negative rate of
  the proxy (fragments whose insight was used without modifying the
  source file — read-only consumption). For those we abstain rather
  than punish, preserving Robbins-Monro convergence while bounding
  bias.

Three signal classes are emitted:

  verified_hits        — retrieved AND source modified  → strong +/-
  unverified           — retrieved but source untouched → ABSTAIN
  should_have_retrieved — modified but not retrieved   → PRISM blind spot

The third class is the most valuable learning signal and was discarded
entirely by the legacy contract.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("entroly.causal_attribution")

# ── Tunables ──────────────────────────────────────────────────────────

# Max snapshots held in the LRU. Beyond this, the oldest is evicted.
# Each snapshot is small (~few KB), so we can afford a generous bound.
DEFAULT_MAX_SNAPSHOTS = 256

# Max age before a snapshot is considered stale. A typical
# optimize→outcome round is < 5 minutes; anything older is unlikely
# to be a real causal pair.
DEFAULT_SNAPSHOT_TTL_SECONDS = 60 * 60  # 1 hour

# Cap on `should_have_retrieved` files. Big refactors touch hundreds of
# files, almost none of which are causally related to one query.
# Capping prevents one mega-PR from dominating the learning signal.
MAX_MISS_FILES_PER_OUTCOME = 20

# Env switch: set ENTROLY_CAUSAL_CREDIT=0 to fall back to legacy
# (every-passed-id reinforces blindly) behavior. Default ON.
ENV_FLAG = "ENTROLY_CAUSAL_CREDIT"


def causal_credit_enabled() -> bool:
    """Read the env switch lazily so tests can flip it per call."""
    return os.environ.get(ENV_FLAG, "1").strip() not in ("0", "false", "no", "")


# ── Snapshot dataclass ────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievedFragment:
    """Minimal view of a retrieved fragment for attribution purposes."""

    fragment_id: str
    source_path: str  # repo-relative, normalized to forward slashes


@dataclass
class RetrievalSnapshot:
    """Captured at ``optimize_context`` time, consumed at ``record_outcome``."""

    request_id: str
    repo_root: str
    git_head: str | None  # None if not in a git repo
    dirty_at_start: frozenset[str]
    retrieved: tuple[RetrievedFragment, ...]
    created_at: float = field(default_factory=time.time)


@dataclass
class CausalCredit:
    """Result of attributing an outcome to a snapshot."""

    verified_hits: list[str]            # fragment ids whose source was modified
    unverified: list[str]               # passed ids without modification evidence
    should_have_retrieved: list[str]    # files modified but not retrieved
    modified_files: list[str]           # the full diff (capped)
    fallback_reason: str | None = None  # set when causal path could not run

    @property
    def is_causal(self) -> bool:
        return self.fallback_reason is None

    def summary(self) -> dict:
        return {
            "verified_hits": len(self.verified_hits),
            "unverified": len(self.unverified),
            "should_have_retrieved": len(self.should_have_retrieved),
            "modified_files": len(self.modified_files),
            "is_causal": self.is_causal,
            "fallback_reason": self.fallback_reason,
        }


# ── Snapshot store (bounded LRU, thread-safe) ─────────────────────────


class SnapshotStore:
    """Bounded LRU keyed by request_id with a TTL."""

    def __init__(
        self,
        max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
        ttl_seconds: float = DEFAULT_SNAPSHOT_TTL_SECONDS,
    ) -> None:
        self._max = max_snapshots
        self._ttl = ttl_seconds
        self._data: "OrderedDict[str, RetrievalSnapshot]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, snap: RetrievalSnapshot) -> None:
        with self._lock:
            self._data[snap.request_id] = snap
            self._data.move_to_end(snap.request_id)
            self._evict_locked()

    def get(self, request_id: str) -> RetrievalSnapshot | None:
        with self._lock:
            snap = self._data.get(request_id)
            if snap is None:
                return None
            if time.time() - snap.created_at > self._ttl:
                # expired
                self._data.pop(request_id, None)
                return None
            self._data.move_to_end(request_id)
            return snap

    def latest(self) -> RetrievalSnapshot | None:
        """Return the most recently inserted live snapshot, if any."""
        with self._lock:
            now = time.time()
            for rid in reversed(list(self._data.keys())):
                snap = self._data[rid]
                if now - snap.created_at <= self._ttl:
                    return snap
                self._data.pop(rid, None)
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def _evict_locked(self) -> None:
        # TTL eviction first
        now = time.time()
        stale = [
            rid for rid, snap in self._data.items()
            if now - snap.created_at > self._ttl
        ]
        for rid in stale:
            self._data.pop(rid, None)
        # Size cap
        while len(self._data) > self._max:
            self._data.popitem(last=False)


# ── Git interrogation (best-effort, never raises) ─────────────────────


def _git_available() -> bool:
    return shutil.which("git") is not None


def _run_git(repo_root: str, *args: str, timeout: float = 5.0) -> str | None:
    """Run a git command, return stdout or None on any failure.

    We never raise from here — the entire causal layer is best-effort.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _normalize_path(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def git_head(repo_root: str) -> str | None:
    if not _git_available():
        return None
    out = _run_git(repo_root, "rev-parse", "HEAD")
    if out is None:
        return None
    return out.strip() or None


def git_dirty_files(repo_root: str) -> set[str]:
    """Files currently differing from HEAD (staged + unstaged + untracked)."""
    out = _run_git(repo_root, "status", "--porcelain", "-uall")
    if out is None:
        return set()
    files: set[str] = set()
    for line in out.splitlines():
        if len(line) < 4:
            continue
        # Porcelain v1 layout:  XY␠filename  (rename: orig -> new)
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        # Strip surrounding quotes that git uses for special chars
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        files.add(_normalize_path(path))
    return files


def git_files_changed_since(
    repo_root: str,
    base_sha: str | None,
    dirty_at_start: Iterable[str],
) -> set[str]:
    """All files modified between snapshot time and now.

    Combines:
      • committed deltas since ``base_sha`` (covers user committing mid-task)
      • current working tree dirtiness (uncommitted edits)
    Then subtracts ``dirty_at_start`` so pre-existing edits don't get credit.
    """
    changed: set[str] = set()

    if base_sha:
        out = _run_git(repo_root, "diff", "--name-only", f"{base_sha}..HEAD")
        if out:
            for line in out.splitlines():
                line = line.strip()
                if line:
                    changed.add(_normalize_path(line))

    changed.update(git_dirty_files(repo_root))
    changed.difference_update({_normalize_path(p) for p in dirty_at_start})
    return changed


# ── The actual attribution algorithm ──────────────────────────────────


def attribute(
    snapshot: RetrievalSnapshot,
    passed_ids: Iterable[str],
    *,
    repo_root_override: str | None = None,
) -> CausalCredit:
    """Bind an outcome to causal evidence of fragment use.

    Args:
        snapshot: captured at retrieval time.
        passed_ids: fragment ids the caller asserted led to the outcome.
        repo_root_override: optional override (used by tests to point at
            a fixture repo without depending on cwd).

    Returns:
        A ``CausalCredit`` separating verified / unverified / missed signals.
        Falls back to "legacy" mode (treats every passed id as verified)
        only when git is unavailable or the repo can't be inspected.
    """
    passed_id_list = [pid for pid in passed_ids if pid]
    repo_root = repo_root_override or snapshot.repo_root

    if not _git_available() or snapshot.git_head is None:
        return CausalCredit(
            verified_hits=passed_id_list,
            unverified=[],
            should_have_retrieved=[],
            modified_files=[],
            fallback_reason="git_unavailable_or_no_head",
        )

    try:
        modified = git_files_changed_since(
            repo_root, snapshot.git_head, snapshot.dirty_at_start
        )
    except Exception as e:  # defensive: never break callers
        logger.debug("causal attribution diff failed: %s", e)
        return CausalCredit(
            verified_hits=passed_id_list,
            unverified=[],
            should_have_retrieved=[],
            modified_files=[],
            fallback_reason=f"git_diff_failed:{type(e).__name__}",
        )

    if not modified:
        # Read-only / question task. Don't fabricate signal — abstain.
        # All passed ids land in `unverified`, which the engine will treat
        # as "no update". This is the critical safety property: success
        # without modification produces no PRISM update at all.
        return CausalCredit(
            verified_hits=[],
            unverified=passed_id_list,
            should_have_retrieved=[],
            modified_files=[],
            fallback_reason=None,
        )

    retrieved_by_path: dict[str, list[str]] = {}
    for frag in snapshot.retrieved:
        if not frag.source_path:
            continue
        retrieved_by_path.setdefault(_normalize_path(frag.source_path), []) \
            .append(frag.fragment_id)
    retrieved_paths = set(retrieved_by_path.keys())

    # Verified = retrieved fragments whose source is in the modified set
    verified_set: set[str] = set()
    for path in modified & retrieved_paths:
        verified_set.update(retrieved_by_path[path])

    passed_set = set(passed_id_list)
    # Unverified = passed by caller but not causally implicated
    unverified = sorted(passed_set - verified_set)
    # Verified credit only goes to ids the caller actually asserted —
    # we don't silently expand the credit set beyond the caller's claim.
    verified = sorted(verified_set & passed_set) if passed_set else sorted(verified_set)

    # Should-have = modified files that retrieval missed entirely.
    # Cap to prevent mega-refactor noise.
    misses = sorted(modified - retrieved_paths)
    if len(misses) > MAX_MISS_FILES_PER_OUTCOME:
        misses = misses[:MAX_MISS_FILES_PER_OUTCOME]

    return CausalCredit(
        verified_hits=verified,
        unverified=unverified,
        should_have_retrieved=misses,
        modified_files=sorted(modified),
        fallback_reason=None,
    )


# ── Helper: build a snapshot from an engine result dict ───────────────


def build_snapshot(
    request_id: str,
    repo_root: str | Path,
    selected_fragments: Iterable[dict],
) -> RetrievalSnapshot:
    """Construct a RetrievalSnapshot from the engine's optimize_context output.

    The selected_fragments list is the canonical Python representation
    used by both the Rust and pure-Python engine paths. We tolerate
    multiple key spellings (``id`` / ``fragment_id`` and
    ``source`` / ``source_path`` / ``path``) so callers don't need to
    normalize first.
    """
    repo_root_str = str(repo_root)
    frags: list[RetrievedFragment] = []
    for f in selected_fragments or ():
        if not isinstance(f, dict):
            continue
        fid = str(
            f.get("id")
            or f.get("fragment_id")
            or ""
        ).strip()
        src = str(
            f.get("source")
            or f.get("source_path")
            or f.get("path")
            or ""
        ).strip()
        if not fid:
            continue
        frags.append(RetrievedFragment(fragment_id=fid, source_path=_normalize_path(src)))

    return RetrievalSnapshot(
        request_id=request_id,
        repo_root=repo_root_str,
        git_head=git_head(repo_root_str),
        dirty_at_start=frozenset(git_dirty_files(repo_root_str)),
        retrieved=tuple(frags),
    )


# ── Module-level singleton for the running server ─────────────────────

_GLOBAL_STORE: SnapshotStore | None = None
_GLOBAL_STORE_LOCK = threading.Lock()


def global_store() -> SnapshotStore:
    """Return the process-wide SnapshotStore (created lazily)."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        with _GLOBAL_STORE_LOCK:
            if _GLOBAL_STORE is None:
                _GLOBAL_STORE = SnapshotStore()
    return _GLOBAL_STORE


def reset_global_store_for_tests() -> None:
    """Test-only: drop the global store so each test starts clean."""
    global _GLOBAL_STORE
    with _GLOBAL_STORE_LOCK:
        _GLOBAL_STORE = None
