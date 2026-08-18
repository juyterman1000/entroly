"""CLI orchestration for the shared Rust AI Work Graph."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .work_graph import WorkGraphUnavailableError
from .work_graph_store import (
    WorkGraphLockTimeout,
    WorkGraphStateError,
    WorkGraphStore,
    WorkGraphStoreError,
)


def _project_path(project: str) -> Path:
    return Path(project or ".").expanduser().resolve()


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    status = payload.get("status", "ok")
    if status == "error":
        print(f"Work Graph error: {payload.get('detail', payload.get('error', 'unknown'))}", file=sys.stderr)
        return
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, WorkGraphUnavailableError):
        code = "native_work_graph_unavailable"
    elif isinstance(exc, WorkGraphLockTimeout):
        code = "work_graph_lock_timeout"
    elif isinstance(exc, WorkGraphStateError):
        code = "work_graph_state_error"
    elif isinstance(exc, WorkGraphStoreError):
        code = "work_graph_store_error"
    elif isinstance(exc, ValueError):
        code = "invalid_work_graph_request"
    else:
        code = "work_graph_error"
    return {"status": "error", "error": code, "detail": str(exc)[:2000]}


def run(args: Any) -> int:
    action = getattr(args, "work_action", None)
    json_output = bool(getattr(args, "json_output", False))
    try:
        project = _project_path(getattr(args, "project", "."))
        store = WorkGraphStore.for_repository(project)
        if action == "state":
            graph = store.load()
            payload = {
                "status": "ok",
                "repo_id": graph.repo_id,
                "summary": graph.summary(),
                "unfinished": graph.unfinished(),
                "coordination": graph.coordination(int(time.time() * 1000)),
            }
        elif action == "claim":
            graph, lease_id = store.claim_work(
                project,
                agent_id=args.agent,
                task_title=args.task,
                task_id=args.task_id,
                session_id=args.session,
                scope_paths=args.path or [],
                scope_symbols=args.symbol or [],
                ttl_seconds=args.ttl,
                source_kind="user_statement",
            )
            payload = {
                "status": "ok",
                "lease_id": lease_id,
                "summary": graph.summary(),
                "coordination": graph.coordination(int(time.time() * 1000)),
            }
        elif action == "resume":
            payload = {
                "status": "ok",
                "resume": store.resume(args.workstream or None, max_evidence=args.max_evidence),
            }
        elif action == "handoff":
            payload = {
                "status": "ok",
                "handoff": store.handoff(args.workstream, args.from_agent, args.to_agent),
            }
        else:
            payload = {"status": "error", "error": "missing_work_action", "detail": "choose state, claim, resume, or handoff"}
    except Exception as exc:
        payload = _error(exc)
    _emit(payload, json_output=json_output)
    return 0 if payload.get("status") == "ok" else 1


__all__ = ["run"]
