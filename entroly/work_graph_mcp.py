"""MCP-facing orchestration for Entroly's shared Rust AI Work Graph.

This module contains no Work Graph semantics. It resolves the configured
project, delegates persistence/coordination to ``WorkGraphStore``, and renders
recovered work state as fenced untrusted data before an agent consumes it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .hardening import sanitize_injected_context
from .work_graph import WorkGraphUnavailableError
from .work_graph_store import (
    WorkGraphLockTimeout,
    WorkGraphStateError,
    WorkGraphStore,
    WorkGraphStoreError,
)

_MAX_RENDER_BYTES = 512 * 1024
_MAX_SCOPE_ITEMS = 256
_MAX_EVIDENCE = 4096


def _project_root() -> Path:
    return Path(os.environ.get("ENTROLY_SOURCE", os.getcwd())).expanduser().resolve()


def _project_path(project: str = "") -> Path:
    root = _project_root()
    candidate = root if not project.strip() else (root / project).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("project must stay inside ENTROLY_SOURCE") from exc
    return candidate


def _bounded_strings(values: list[str] | tuple[str, ...] | None, name: str) -> list[str]:
    if not values:
        return []
    if len(values) > _MAX_SCOPE_ITEMS:
        raise ValueError(f"{name} may contain at most {_MAX_SCOPE_ITEMS} items")
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            continue
        if len(value) > 2048:
            raise ValueError(f"{name} entries may not exceed 2048 characters")
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _render_untrusted(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    encoded = raw.encode("utf-8")
    if len(encoded) > _MAX_RENDER_BYTES:
        raise WorkGraphStateError(
            f"{kind} output exceeds {_MAX_RENDER_BYTES} bytes; narrow the request"
        )
    fenced, report = sanitize_injected_context(raw, fence=True)
    return {
        "status": "ok",
        "kind": kind,
        "trust": "untrusted_recovered_work_state",
        "context": fenced,
        "injection_scan": {
            "matches": list(report.matches),
            "invisible_chars_stripped": report.invisible_chars_stripped,
        },
    }


def _error(kind: str, exc: Exception) -> dict[str, Any]:
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
    return {"status": "error", "kind": kind, "error": code, "detail": str(exc)[:2000]}


def work_state(*, project: str = "", now_ms: int | None = None) -> dict[str, Any]:
    """Return persisted Work Graph state without creating a polling event."""
    try:
        path = _project_path(project)
        store = WorkGraphStore.for_repository(path)
        graph = store.load()
        timestamp = int(now_ms if now_ms is not None else time.time() * 1000)
        payload = {
            "summary": graph.summary(),
            "unfinished": graph.unfinished(),
            "coordination": graph.coordination(timestamp),
        }
        return _render_untrusted("work_state", payload)
    except Exception as exc:  # MCP boundary must return structured errors
        return _error("work_state", exc)


def work_claim(
    *,
    agent_id: str,
    task_title: str,
    project: str = "",
    task_id: str = "",
    session_id: str = "",
    scope_paths: list[str] | None = None,
    scope_symbols: list[str] | None = None,
    ttl_seconds: float = 900.0,
) -> dict[str, Any]:
    """Record explicit agent work and an advisory lease in the shared graph."""
    try:
        bounded_paths = _bounded_strings(scope_paths, "scope_paths")
        bounded_symbols = _bounded_strings(scope_symbols, "scope_symbols")
        path = _project_path(project)
        store = WorkGraphStore.for_repository(path)
        graph, lease_id = store.claim_work(
            path,
            agent_id=agent_id,
            task_title=task_title,
            task_id=task_id,
            session_id=session_id,
            scope_paths=bounded_paths,
            scope_symbols=bounded_symbols,
            ttl_seconds=ttl_seconds,
            source_kind="agent_statement",
        )
        payload = {
            "lease_id": lease_id,
            "summary": graph.summary(),
            "coordination": graph.coordination(int(time.time() * 1000)),
        }
        return _render_untrusted("work_claim", payload)
    except Exception as exc:
        return _error("work_claim", exc)


def work_resume(
    *,
    project: str = "",
    workstream_id: str = "",
    max_evidence: int = 128,
) -> dict[str, Any]:
    """Recover one unfinished workstream from persisted graph state."""
    try:
        if not isinstance(max_evidence, int) or isinstance(max_evidence, bool) or not 0 <= max_evidence <= _MAX_EVIDENCE:
            raise ValueError(f"max_evidence must be an integer between 0 and {_MAX_EVIDENCE}")
        path = _project_path(project)
        store = WorkGraphStore.for_repository(path)
        view = store.resume(workstream_id or None, max_evidence=max_evidence)
        return _render_untrusted("work_resume", view)
    except Exception as exc:
        return _error("work_resume", exc)


def work_handoff(
    *,
    from_agent: str,
    to_agent: str,
    workstream_id: str,
    project: str = "",
) -> dict[str, Any]:
    """Create a graph-bound, tamper-evident cross-agent handoff receipt."""
    try:
        path = _project_path(project)
        store = WorkGraphStore.for_repository(path)
        receipt = store.handoff(workstream_id, from_agent, to_agent)
        return _render_untrusted("work_handoff", receipt)
    except Exception as exc:
        return _error("work_handoff", exc)
