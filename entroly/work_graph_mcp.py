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
_MAX_CONTRACT_BYTES = 1024 * 1024


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


def _bounded_contract(value: str | dict[str, Any], name: str) -> str | dict[str, Any]:
    if not isinstance(value, (str, dict)):
        raise ValueError(f"{name} must be a JSON object or JSON text")
    raw = value if isinstance(value, str) else json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    if not raw.strip():
        raise ValueError(f"{name} must not be empty")
    if len(raw.encode("utf-8")) > _MAX_CONTRACT_BYTES:
        raise ValueError(f"{name} exceeds {_MAX_CONTRACT_BYTES} bytes")
    return value


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
    to_agent: str = "",
) -> dict[str, Any]:
    """Refresh durable facts and recover work, optionally with a no-handoff proof."""
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
        target_agent = str(to_agent).strip()
        if target_agent:
            target_agent = _bounded_id(target_agent, "to_agent")
        path = _project_path(project)
        store = _store_for_path(path)
        store.submit_observation(_passive_observation(path))
        view = store.resume(selected_workstream or None, max_evidence=max_evidence)
        payload: dict[str, Any] = {"resume": view}
        if target_agent:
            payload["continuation_proof"] = store.reconstructed_continuation_proof(
                str(view["selected_workstream"]["node_id"]),
                target_agent,
                outstanding_work_refs=list(view.get("changed_paths", []))
                + list(view.get("failures", [])),
                created_at_ms=int(time.time() * 1000),
            )
        return _render_untrusted("work_resume", payload)
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
        view = store.resume(selected_workstream, max_evidence=128)
        proof = store.continuation_proof(
            receipt,
            outstanding_work_refs=list(view.get("changed_paths", []))
            + list(view.get("failures", [])),
            created_at_ms=int(time.time() * 1000),
        )
        return _render_untrusted(
            "work_handoff",
            {"handoff": receipt, "continuation_proof": proof},
        )
    except Exception as exc:
        return _error("work_handoff", exc)


def work_record_context(
    *,
    receipt: str | dict[str, Any],
    project: str = "",
    agent_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Persist one verified canonical ContextReceipt in the shared graph."""
    try:
        bounded_receipt = _bounded_contract(receipt, "receipt")
        path = _project_path(project)
        store = _store_for_path(path)
        graph, event_id = store.record_context_receipt(
            bounded_receipt,
            agent_id=str(agent_id),
            session_id=str(session_id),
        )
        return _render_untrusted(
            "work_record_context",
            {"event_id": event_id, "summary": graph.summary()},
        )
    except Exception as exc:
        return _error("work_record_context", exc)


def work_record_memory(
    *,
    memory: str | dict[str, Any],
    project: str = "",
    now_ms: int = 0,
    superseded_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Persist one provenance-bearing canonical MemoryRecord."""
    try:
        bounded_memory = _bounded_contract(memory, "memory")
        timestamp = int(now_ms or time.time() * 1000)
        if timestamp < 0:
            raise ValueError("now_ms must be non-negative")
        superseded = _bounded_strings(superseded_ids, "superseded_ids")
        path = _project_path(project)
        store = _store_for_path(path)
        graph, event_id = store.record_memory(
            bounded_memory,
            now_ms=timestamp,
            superseded_ids=superseded,
        )
        return _render_untrusted(
            "work_record_memory",
            {"event_id": event_id, "summary": graph.summary()},
        )
    except Exception as exc:
        return _error("work_record_memory", exc)


def work_record_execution(
    *,
    route: str | dict[str, Any],
    outcome: str | dict[str, Any],
    verification: str | dict[str, Any],
    project: str = "",
    invalidated_commitments: list[str] | None = None,
) -> dict[str, Any]:
    """Atomically close route, execution and verification into Work Graph state."""
    try:
        bounded_route = _bounded_contract(route, "route")
        bounded_outcome = _bounded_contract(outcome, "outcome")
        bounded_verification = _bounded_contract(verification, "verification")
        invalidated = _bounded_strings(
            invalidated_commitments, "invalidated_commitments"
        )
        path = _project_path(project)
        store = _store_for_path(path)
        graph, event_id = store.record_execution_chain(
            bounded_route,
            bounded_outcome,
            bounded_verification,
            invalidated_commitments=invalidated,
        )
        return _render_untrusted(
            "work_record_execution",
            {"event_id": event_id, "summary": graph.summary()},
        )
    except Exception as exc:
        return _error("work_record_execution", exc)


__all__ = [
    "work_claim",
    "work_handoff",
    "work_record_context",
    "work_record_execution",
    "work_record_memory",
    "work_resume",
    "work_state",
]
