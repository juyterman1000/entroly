"""MCP-facing orchestration for Entroly's shared Rust AI Work Graph.

This module contains no Work Graph semantics. It resolves the configured
project, gathers bounded local observations, delegates all work-state meaning to
Rust through ``WorkGraphStore``, and renders recovered state as fenced untrusted
data before an agent consumes it.
"""
from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .hardening import sanitize_injected_context
from .work_graph import WorkGraphUnavailableError
from .work_graph_content_digest import enrich_worktree_content_digests
from .work_graph_repo import discover_repository_identity, discover_repository_observation
from .work_graph_store import (
    WorkGraphLockTimeout,
    WorkGraphStateError,
    WorkGraphStore,
    WorkGraphStoreError,
)

_MAX_RENDER_BYTES = 512 * 1024
_MAX_SCOPE_ITEMS = 256
_MAX_EVIDENCE = 4096
_MAX_TTL_SECONDS = 30 * 24 * 60 * 60
_MAX_ID_CHARS = 512


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


def _store_for_path(path: Path) -> WorkGraphStore:
    identity = discover_repository_identity(path)
    return WorkGraphStore(identity["repo_id"])


def _passive_observation(path: Path) -> dict[str, Any]:
    """Capture a passive repo snapshot with fail-closed content identity."""
    observation = discover_repository_observation(path)
    return enrich_worktree_content_digests(path, observation)


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


def _bounded_id(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > _MAX_ID_CHARS:
        raise ValueError(f"{name} may not exceed {_MAX_ID_CHARS} characters")
    if "\x00" in text:
        raise ValueError(f"{name} may not contain NUL")
    return text


def _ttl_ms(ttl_seconds: float) -> int:
    if isinstance(ttl_seconds, bool):
        raise ValueError("ttl_seconds must be a finite positive number")
    try:
        value = float(ttl_seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("ttl_seconds must be a finite positive number") from exc
    if not math.isfinite(value) or not 0 < value <= _MAX_TTL_SECONDS:
        raise ValueError(f"ttl_seconds must be > 0 and <= {_MAX_TTL_SECONDS}")
    return max(1, int(value * 1000))


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
        store = _store_for_path(path)
        graph = store.load()
        timestamp = int(now_ms if now_ms is not None else time.time() * 1000)
        payload = {
            "summary": graph.summary(),
            "unfinished": graph.unfinished(),
            "coordination": graph.coordination(timestamp),
        }
        return _render_untrusted("work_state", payload)
    except Exception as exc:
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
        agent = _bounded_id(agent_id, "agent_id")
        title = str(task_title).strip()
        if not title:
            raise ValueError("task_title must not be empty")
        if len(title) > 8192:
            raise ValueError("task_title may not exceed 8192 characters")
        bounded_paths = _bounded_strings(scope_paths, "scope_paths")
        bounded_symbols = _bounded_strings(scope_symbols, "scope_symbols")
        ttl = _ttl_ms(ttl_seconds)
        path = _project_path(project)
        store = _store_for_path(path)
        now_ms = int(time.time() * 1000)
        lease_id = uuid.uuid4().hex
        observation = discover_repository_observation(
            path,
            agent_id=agent,
            session_id=str(session_id),
            task_hint={
                "task_id": str(task_id),
                "title": title,
                "trust": "observed",
                "explicit_status": "in_progress",
                "remaining_work": [],
                "source_kind": "agent_statement",
                "source_ref": f"work-claim:{lease_id}",
            },
            observed_at_ms=now_ms,
        )
        observation["leases"] = [{
            "lease_id": lease_id,
            "agent_id": agent,
            "task_id": str(task_id),
            "scope_paths": bounded_paths,
            "scope_symbols": bounded_symbols,
            "expires_at_ms": now_ms + ttl,
            "source_ref": f"work-lease:{lease_id}",
        }]
        graph = store.submit_observation(observation)
        payload = {
            "lease_id": lease_id,
            "summary": graph.summary(),
            "coordination": graph.coordination(now_ms),
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
    """Refresh durable repo facts, then recover one unfinished workstream."""
    try:
        if (
            not isinstance(max_evidence, int)
            or isinstance(max_evidence, bool)
            or not 0 <= max_evidence <= _MAX_EVIDENCE
        ):
            raise ValueError(
                f"max_evidence must be an integer between 0 and {_MAX_EVIDENCE}"
            )
        selected_workstream = str(workstream_id).strip()
        if len(selected_workstream) > _MAX_ID_CHARS or "\x00" in selected_workstream:
            raise ValueError(f"workstream_id may not exceed {_MAX_ID_CHARS} characters or contain NUL")
        path = _project_path(project)
        store = _store_for_path(path)
        store.submit_observation(_passive_observation(path))
        view = store.resume(selected_workstream or None, max_evidence=max_evidence)
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
    """Refresh durable facts, then create a graph-bound cross-agent receipt."""
    try:
        source_agent = _bounded_id(from_agent, "from_agent")
        target_agent = _bounded_id(to_agent, "to_agent")
        selected_workstream = _bounded_id(workstream_id, "workstream_id")
        path = _project_path(project)
        store = _store_for_path(path)
        # Explicit handoff is a state-sealing operation. Capture the latest
        # bounded Git/checkpoint facts plus exact worktree content identity first
        # so the receipt is bound to what is actually on disk.
        store.submit_observation(_passive_observation(path))
        receipt = store.handoff(selected_workstream, source_agent, target_agent)
        return _render_untrusted("work_handoff", receipt)
    except Exception as exc:
        return _error("work_handoff", exc)


__all__ = ["work_claim", "work_handoff", "work_resume", "work_state"]
