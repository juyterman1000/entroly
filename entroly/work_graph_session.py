"""Automatic takeover when an agent opens a repository.

``work_resume`` reconstructs work state correctly, but something had to call it.
An agent that never learned the tool existed, or simply started working, got no
takeover at all -- which made the feature semi-automatic in the only sense that
matters to a user who did not read the docs.

Opening the MCP server *is* an agent opening the repository, so that is where
takeover belongs. The session start runs the same resume path the tool exposes
and arms the same trust gate, so an agent that begins working without
acknowledging is refused by ``work_claim`` exactly as if it had called
``work_resume`` itself.

Two properties this has to preserve:

*It must not block the handshake.* Warm-start already learned this lesson -- a
blocking cold start made the server look hung. Discovery runs once, bounded, and
a failure degrades to "no takeover" rather than a server that will not start.

*It must not fail open on the gate.* A discovery failure means no state was
recovered, so there is nothing to acknowledge and no gate is armed. That is
different from recovering state and failing to arm, which
``work_graph_recovery_ack.arm`` refuses to do quietly.

Idempotent per process and repository: an agent that also calls ``work_resume``
explicitly gets the same token for the same state, not a second demand for the
same facts.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_started: dict[str, dict[str, Any]] = {}


def autostart_enabled() -> bool:
    """Takeover is on unless explicitly disabled.

    Default-on is the point: the previous behaviour was effectively default-off
    and that is the gap. ``ENTROLY_WORK_GRAPH_AUTOSTART=0`` opts out for
    embedders that drive resume themselves.
    """
    return os.environ.get("ENTROLY_WORK_GRAPH_AUTOSTART", "1").strip() not in {
        "0", "false", "no",
    }


def session_state(project: str = "") -> dict[str, Any] | None:
    """What the automatic takeover found, if it ran."""
    with _lock:
        return _started.get(str(project))


def reset_for_tests() -> None:
    with _lock:
        _started.clear()


def start_session(project: str = "", *, force: bool = False) -> dict[str, Any]:
    """Perform automatic takeover once per process and repository.

    Returns a summary describing what happened. Never raises: a repository that
    cannot be observed is a repository with no recoverable work, and the caller
    is a server handshake.
    """
    key = str(project)
    with _lock:
        if not force and key in _started:
            return dict(_started[key], reused=True)

    summary: dict[str, Any] = {
        "attempted": True,
        "recovered": False,
        "gate_armed": False,
        "watcher_started": False,
        "reused": False,
    }

    if not autostart_enabled():
        summary["attempted"] = False
        summary["reason"] = "disabled by ENTROLY_WORK_GRAPH_AUTOSTART"
        with _lock:
            _started[key] = summary
        return summary

    view: Any = None
    try:
        from . import work_graph_mcp as mcp_adapter

        path = mcp_adapter._project_path(project)
        store = mcp_adapter._store_for_path(path)
        store.submit_repository_observation(
            mcp_adapter._passive_observation(path), repository_path=path
        )
        view = store.resume(None, max_evidence=128)
    except Exception as exc:  # noqa: BLE001 - handshake must survive anything
        # "no unfinished workstream" is the documented null control, not a
        # failure: a clean checkout has nothing to continue, and resume refuses
        # rather than manufacturing a task. Recording it as an error made
        # correct fail-closed behaviour look broken -- and, worse, returned
        # early so the watcher never started on a clean repository, which is
        # precisely where a modification timeline begins to be useful.
        if isinstance(exc, ValueError) and "no unfinished workstream" in str(exc):
            summary["null_control"] = "no unfinished work to recover"
        else:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            with _lock:
                _started[key] = summary
            return summary

    if view is not None and _has_recoverable_work(view):
        from .work_graph_recovery_ack import arm, recovery_token

        token = recovery_token(view)
        # Not wrapped in a second try/except. If state was recovered and the
        # gate cannot be armed, the honest outcome is a loud failure, not a
        # session that reports takeover while nothing enforces it.
        acknowledgement = arm(
            store.repo_dir, token, mcp_adapter._recovery_unknowns(view)
        )
        summary["recovered"] = True
        summary["gate_armed"] = True
        summary["acknowledgement"] = acknowledgement

    started, watcher_note = _maybe_start_watcher(key, path)
    summary["watcher_started"] = started
    if watcher_note:
        # A watcher that failed to start is a hole in the modification
        # timeline. Returning a bare False would make "disabled" and "tried and
        # broke" look identical, which is the kind of silence that makes a
        # missing feature look like a working one.
        summary["watcher_note"] = watcher_note
    with _lock:
        _started[key] = summary
    return summary


def _has_recoverable_work(view: Any) -> bool:
    """Whether the reconstruction found anything worth acknowledging.

    A clean checkout is a documented null control: it must not arm a gate and
    demand acknowledgement of nothing, or every fresh clone would greet its
    agent with a refusal.
    """
    if not isinstance(view, dict):
        return False
    for field in ("changed_paths", "evidence", "claims", "commits", "failures"):
        value = view.get(field)
        if isinstance(value, list) and value:
            return True
    selected = view.get("selected_workstream")
    return isinstance(selected, dict) and bool(selected.get("node_id"))


_watchers: dict[str, Any] = {}


def _maybe_start_watcher(key: str, path: Path) -> tuple[bool, str]:
    from .work_graph_watcher import (
        WorkspaceModificationWatcher,
        git_status_sampler,
        watch_enabled,
    )

    if not watch_enabled():
        return False, "disabled; set ENTROLY_WORK_GRAPH_WATCH=1"
    if key in _watchers:
        return False, "already watching this project"
    try:
        watcher = WorkspaceModificationWatcher(git_status_sampler(path)).start()
    except Exception as exc:  # noqa: BLE001 - a missing timeline must not stop a session
        return False, f"{type(exc).__name__}: {exc}"
    _watchers[key] = watcher
    return True, ""


def session_watcher(project: str = "") -> Any:
    return _watchers.get(str(project))


def stop_session_watchers() -> None:
    for watcher in list(_watchers.values()):
        try:
            watcher.stop()
        except Exception:  # noqa: BLE001
            pass
    _watchers.clear()


__all__ = [
    "autostart_enabled",
    "reset_for_tests",
    "session_state",
    "session_watcher",
    "start_session",
    "stop_session_watchers",
]
