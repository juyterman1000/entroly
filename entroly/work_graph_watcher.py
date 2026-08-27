"""Record worktree modifications between observations.

Detection previously happened only when something asked: a resume, an
observation, a verification. Byte changes do invalidate prior graph state, but
only at the moment someone looked. An agent that edited a file, waited, and then
resumed would see the new bytes; an agent that edited a file and kept working on
a stale plan would not learn anything until its next refresh.

This closes the window by sampling the worktree on an interval and keeping a
bounded log of what changed and when.

Deliberately polling rather than an OS notification API. ``watchdog`` is not a
dependency, and adding one to a local-first tool for this is a poor trade:
inotify degrades on network mounts, ReadDirectoryChangesW has its own failure
modes, and both need a fallback path that would be exercised rarely enough to
rot. Sampling ``git status`` plus the digests already computed for observations
reuses machinery that is tested, and the cost is bounded by the interval.

The thread never writes to the Work Graph. Appending events from a background
thread would race the Rust store's single-writer lock and could interleave a
sample with a resume mid-update. It records into memory; the next observation
consumes it. What the watcher provides is the *timeline* -- that a file changed
at 14:02 and again at 14:07 -- which a single point-in-time refresh cannot show
however often it runs.

Off by default. A local-first tool should not start background threads that
touch the user's repository because a library was imported. Enable with
``ENTROLY_WORK_GRAPH_WATCH=1`` or by constructing one directly.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_DEFAULT_INTERVAL_SECONDS = 2.0
_MIN_INTERVAL_SECONDS = 0.25
_DEFAULT_MAX_RECORDS = 512


@dataclass(frozen=True)
class Modification:
    """One observed transition of a path between two samples."""

    path: str
    change: str          # appeared | vanished | modified
    observed_at_ms: int
    digest: str = ""
    previous_digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "change": self.change,
            "observed_at_ms": self.observed_at_ms,
            "digest": self.digest,
            "previous_digest": self.previous_digest,
        }


def _interval_from_env() -> float:
    raw = os.environ.get("ENTROLY_WORK_GRAPH_WATCH_INTERVAL", "").strip()
    try:
        value = float(raw) if raw else _DEFAULT_INTERVAL_SECONDS
    except ValueError:
        return _DEFAULT_INTERVAL_SECONDS
    return max(_MIN_INTERVAL_SECONDS, value)


def watch_enabled() -> bool:
    return os.environ.get("ENTROLY_WORK_GRAPH_WATCH", "0").strip() not in {"", "0", "false", "no"}


class WorkspaceModificationWatcher:
    """Sample a repository on an interval and log what changed between samples.

    ``sampler`` returns ``{path: digest}`` for the currently changed paths. It is
    injected so this class can be tested without a repository and so the caller
    controls which observation path is used -- in particular so the watcher
    inherits the same secret/generated-file policy as everything else rather
    than re-deriving it and drifting.
    """

    def __init__(
        self,
        sampler: Callable[[], dict[str, str]],
        *,
        interval_seconds: float | None = None,
        max_records: int = _DEFAULT_MAX_RECORDS,
    ) -> None:
        self._sampler = sampler
        self._interval = max(
            _MIN_INTERVAL_SECONDS,
            _interval_from_env() if interval_seconds is None else interval_seconds,
        )
        self._records: deque[Modification] = deque(maxlen=max(1, int(max_records)))
        self._previous: dict[str, str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._errors = 0
        self._samples = 0
        self._started = False

    # -- lifecycle -----------------------------------------------------

    def start(self) -> "WorkspaceModificationWatcher":
        if self._thread is not None:
            return self
        # Seed synchronously so the first interval reports transitions rather
        # than announcing every already-modified file as newly appeared.
        self._sample(seed=True)
        self._thread = threading.Thread(
            target=self._run, name="entroly-work-graph-watch", daemon=True
        )
        self._started = True
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    def __enter__(self) -> "WorkspaceModificationWatcher":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # -- sampling ------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self, *, seed: bool = False) -> None:
        try:
            current = self._sampler()
        except Exception:
            # A sampler failure is a gap in the timeline, not a reason to take
            # the process down or to stop watching. It is counted so the gap is
            # visible rather than silently absorbed.
            with self._lock:
                self._errors += 1
            return

        observed_at_ms = int(time.time() * 1000)
        with self._lock:
            self._samples += 1
            if seed:
                self._previous = dict(current)
                return
            previous = self._previous
            for path, digest in current.items():
                before = previous.get(path)
                if before is None:
                    self._records.append(
                        Modification(path, "appeared", observed_at_ms, digest)
                    )
                elif before != digest:
                    self._records.append(
                        Modification(path, "modified", observed_at_ms, digest, before)
                    )
            for path, before in previous.items():
                if path not in current:
                    self._records.append(
                        Modification(path, "vanished", observed_at_ms, "", before)
                    )
            self._previous = dict(current)

    def poll_once(self) -> None:
        """Take one sample on the calling thread. Used by tests and by callers
        that want a deterministic checkpoint without waiting an interval."""
        self._sample()

    # -- reporting -----------------------------------------------------

    def modifications(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.as_dict() for record in self._records]

    def drain(self) -> list[dict[str, Any]]:
        """Return and clear. The next observation consumes the timeline."""
        with self._lock:
            drained = [record.as_dict() for record in self._records]
            self._records.clear()
            return drained

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": bool(self._thread is not None and self._thread.is_alive()),
                "started": self._started,
                "interval_seconds": self._interval,
                "samples": self._samples,
                "sampler_errors": self._errors,
                "pending_modifications": len(self._records),
                # A dropped record is a hole in the timeline, so say so rather
                # than letting the deque quietly discard the oldest.
                "record_capacity": self._records.maxlen,
                "at_capacity": len(self._records) == self._records.maxlen,
            }


def git_status_sampler(
    repo_path: str | os.PathLike[str],
) -> Callable[[], dict[str, str]]:
    """Sampler over the same observation path everything else uses.

    Reusing ``discover_repository_observation`` means the watcher inherits the
    secret-withholding and generated-file policy automatically. A watcher with
    its own scan would be a second place for that policy to be forgotten.
    """
    root = Path(repo_path)

    def sample() -> dict[str, str]:
        from .work_graph_content_digest import enrich_worktree_content_digests
        from .work_graph_repo import discover_repository_observation

        observation = enrich_worktree_content_digests(
            root, discover_repository_observation(root)
        )
        changes = observation.get("changes")
        if not isinstance(changes, list):
            return {}
        sampled: dict[str, str] = {}
        for change in changes:
            if not isinstance(change, dict):
                continue
            path = str(change.get("path", ""))
            if not path:
                continue
            # Fall back to the change kind when no digest is available, so a
            # staged/conflicted path still registers a transition instead of
            # looking permanently unchanged.
            digest = (
                str(change.get("worktree_digest", ""))
                or str(change.get("content_digest", ""))
                or f"kind:{change.get('kind', '')}"
            )
            sampled[path] = digest
        return sampled

    return sample


__all__ = [
    "Modification",
    "WorkspaceModificationWatcher",
    "git_status_sampler",
    "watch_enabled",
]
