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
from .work_context_snapshot_store import (
    CONTEXT_SNAPSHOT_TOKEN_PREFIX,
    WorkContextSnapshotStore,
)
from .repository_intelligence import RepositoryIntelligenceService
from .repository_intelligence.verified_context import verify_context_commitment
from .work_graph import (
    WorkGraphUnavailableError,
    create_recovery_handle,
    create_work_context_receipt,
    verify_recovered_bytes,
    verify_recovery_handle,
)
from .work_graph_content_digest import enrich_worktree_content_digests
from .work_graph_repo import discover_repository_identity, discover_repository_observation
from .work_graph_store import (
    continuation_outstanding_refs,
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
_MAX_REF_BYTES = 4096
_MAX_CONTRACT_BYTES = 1024 * 1024
_CONTEXT_TOKEN_PREFIX = CONTEXT_SNAPSHOT_TOKEN_PREFIX
_CONTEXT_SELECTION_POLICY = "repository-intelligence/verified-context-v1"


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


def _head_sha(observation: dict[str, Any]) -> str:
    branch = observation.get("branch")
    head = branch.get("head_sha") if isinstance(branch, dict) else None
    if not isinstance(head, str) or not head:
        raise ValueError("repository must have a committed HEAD for context receipts")
    return head


def _context_parts(context: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not verify_context_commitment(context):
        raise ValueError("verified context commitment is invalid")
    fragments = context.get("fragments")
    descriptors = context.get("recoverable_fragments")
    if (
        not isinstance(fragments, list)
        or not all(isinstance(item, dict) for item in fragments)
        or not isinstance(descriptors, list)
        or not all(isinstance(item, dict) for item in descriptors)
    ):
        raise ValueError("verified context has an invalid fragment shape")
    return fragments, descriptors


def _host_receipt_id(context: dict[str, Any]) -> str:
    receipt = context.get("receipt")
    commitment = receipt.get("context_sha256") if isinstance(receipt, dict) else None
    if not isinstance(commitment, str) or len(commitment) != 64:
        raise ValueError("verified context is missing its host commitment")
    return f"vctx_{commitment}"


def _context_contracts(
    *,
    context: dict[str, Any],
    scope: dict[str, Any],
    head_sha: str,
    observed_at_ms: int,
    pinned_refs: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project host context into existing canonical Rust contracts."""
    fragments, descriptors = _context_parts(context)
    for field in ("repo_id", "graph_commitment", "workstream_id"):
        if not isinstance(scope.get(field), str):
            raise ValueError(f"scope.{field} must be a string")
    repository_id = _bounded_id(scope["repo_id"], "scope.repo_id")
    graph_commitment = _bounded_id(
        scope["graph_commitment"], "scope.graph_commitment"
    )
    workstream_id = _bounded_id(
        scope["workstream_id"], "scope.workstream_id"
    )
    host_receipt_id = _host_receipt_id(context)
    handles = [
        create_recovery_handle(
            repository_id=repository_id,
            receipt_id=host_receipt_id,
            disposition="omitted_but_recoverable",
            source_ref=str(descriptor["path"]),
            source_commitment=str(descriptor["source_sha256"]),
            fragment_commitment=str(descriptor["fragment_sha256"]),
            byte_start=int(descriptor["start_byte"]),
            byte_end=int(descriptor["end_byte"]),
            version=head_sha,
            observed_at_ms=observed_at_ms,
        )
        for descriptor in descriptors
    ]
    retrieval = context.get("retrieval")
    if not isinstance(retrieval, dict):
        raise ValueError("verified context retrieval metadata is invalid")
    evidence_ids = [
        *(
            str(item)
            for item in scope.get("evidence_ids", [])
            if isinstance(item, str) and item
        ),
        *(str(fragment["symbol_id"]) for fragment in fragments),
    ]
    canonical = create_work_context_receipt(
        repository_id=repository_id,
        repository_commitment=head_sha,
        graph_commitment=graph_commitment,
        work_scope_id=workstream_id,
        source_commitment=str(context["receipt"]["context_sha256"]),
        selected_refs=[str(fragment["context_ref"]) for fragment in fragments],
        omitted_refs=[str(descriptor["context_ref"]) for descriptor in descriptors],
        pinned_refs=pinned_refs or [],
        recoverable_refs=[str(descriptor["context_ref"]) for descriptor in descriptors],
        recovery_handles=[str(handle["handle_id"]) for handle in handles],
        evidence_ids=evidence_ids,
        budget_tokens=int(retrieval.get("token_budget", 0)),
        selection_policy=_CONTEXT_SELECTION_POLICY,
        created_at_ms=observed_at_ms,
    )
    return canonical, handles


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


def _bounded_ref(value: object, name: str) -> str:
    """Bound canonical source/reference strings by UTF-8 bytes, not ID length."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in text:
        raise ValueError(f"{name} may not contain NUL")
    if len(text.encode("utf-8")) > _MAX_REF_BYTES:
        raise ValueError(f"{name} may not exceed {_MAX_REF_BYTES} UTF-8 bytes")
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


def _snapshot_store_for_graph(store: WorkGraphStore) -> WorkContextSnapshotStore:
    """Use the same repository store/lock as Work Graph persistence."""
    return WorkContextSnapshotStore(store)


def _snapshot_token_from_receipt_id(receipt_id: object) -> str:
    """Derive the parent snapshot locator committed by a RecoveryHandle."""
    value = str(receipt_id or "")
    prefix = "vctx_"
    if not value.startswith(prefix):
        raise ValueError("recovery handle receipt_id is not a verified context id")
    return WorkContextSnapshotStore.token_for_commitment(value[len(prefix):])


def _encode_context_token(context: dict[str, Any], store: WorkGraphStore) -> str:
    """Persist exact committed context and return a short content-addressed token."""
    return _snapshot_store_for_graph(store).put_json(context)


def _decode_context_token(
    token: str, store: WorkGraphStore | None = None
) -> dict[str, Any]:
    """Resolve a repository-scoped snapshot and re-verify its inner commitment."""
    graph_store = store if store is not None else _store_for_path(_project_path())
    return _snapshot_store_for_graph(graph_store).get_json(token)


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


def _render_context_result(
    kind: str,
    *,
    context: dict[str, Any],
    context_token: str,
    canonical_receipt: dict[str, Any],
    recovery_handles: list[dict[str, Any]],
    work_event_id: Any,
    work_summary: dict[str, Any],
    integrity_state: str = "",
) -> dict[str, Any]:
    """Separate exact machine context from sanitized model-facing source data."""
    raw = json.dumps(
        context,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(raw.encode("utf-8")) > _MAX_RENDER_BYTES:
        raise WorkGraphStateError(
            f"{kind} model context exceeds {_MAX_RENDER_BYTES} bytes; narrow the request"
        )
    fenced, report = sanitize_injected_context(raw, fence=True)
    result: dict[str, Any] = {
        "status": "ok",
        "kind": kind,
        "trust": "untrusted_retrieved_source_data",
        "context_token": context_token,
        "context_block": fenced,
        "canonical_receipt": canonical_receipt,
        "recovery_handles": recovery_handles,
        "work_event_id": work_event_id,
        "work_summary": work_summary,
        "injection_scan": {
            "matches": list(report.matches),
            "invisible_chars_stripped": report.invisible_chars_stripped,
        },
    }
    if integrity_state:
        result["integrity_state"] = integrity_state
    return result


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
    """Record explicit agent work plus a bounded advisory scope lease."""
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
        enrich_worktree_content_digests(path, observation)
        graph = store.submit_repository_observation(
            observation, repository_path=path
        )
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
        store.submit_repository_observation(
            _passive_observation(path), repository_path=path
        )
        view = store.resume(selected_workstream or None, max_evidence=max_evidence)
        payload: dict[str, Any] = {"resume": view}
        if target_agent:
            payload["continuation_proof"] = store.reconstructed_continuation_proof(
                str(view["selected_workstream"]["node_id"]),
                target_agent,
                outstanding_work_refs=continuation_outstanding_refs(view),
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
        store.submit_repository_observation(
            _passive_observation(path), repository_path=path
        )
        receipt = store.handoff(selected_workstream, source_agent, target_agent)
        view = store.resume(selected_workstream, max_evidence=128)
        proof = store.continuation_proof(
            receipt,
            outstanding_work_refs=continuation_outstanding_refs(view),
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


def work_compile_context(
    *,
    query: str,
    project: str = "",
    workstream_id: str = "",
    agent_id: str = "",
    session_id: str = "",
    token_budget: int = 2_000,
    max_hops: int = 2,
    max_fragments: int = 24,
) -> dict[str, Any]:
    """Compile verified source context and record its canonical graph receipt."""
    try:
        selected_workstream = str(workstream_id).strip()
        if selected_workstream:
            selected_workstream = _bounded_id(
                selected_workstream, "workstream_id"
            )
        path = _project_path(project)
        observation = _passive_observation(path)
        store = _store_for_path(path)
        graph = store.submit_repository_observation(
            observation, repository_path=path
        )
        scope = graph.context_scope(selected_workstream or None, max_evidence=128)
        service = RepositoryIntelligenceService(path)
        proposals = service.work_scope_proposals(scope)
        context = service.context(
            query,
            token_budget=token_budget,
            max_hops=max_hops,
            max_fragments=max_fragments,
            proposal_scores=proposals,
            proposal_provider="rust-work-scope",
        )
        service.validate_context(context)
        confirmation = _passive_observation(path)
        confirmed_graph = store.submit_repository_observation(
            confirmation, repository_path=path
        )
        if confirmed_graph.graph_commitment != graph.graph_commitment:
            raise ValueError("repository changed during context compilation; retry")
        now_ms = int(time.time() * 1000)
        canonical, handles = _context_contracts(
            context=context,
            scope=scope,
            head_sha=_head_sha(confirmation),
            observed_at_ms=now_ms,
        )
        source_commitment = str(context["receipt"]["context_sha256"])
        if canonical.get("source_commitment") != source_commitment:
            raise WorkGraphStateError("canonical receipt lost source commitment")

        # Persist exact host bytes before publishing their bounded receipt.
        # A failed graph mutation can leave only a bounded content-addressed
        # orphan; the graph can never point at missing recovery state.
        context_token = _encode_context_token(context, store)
        recorded_graph, event_id = store.record_context_receipt(
            canonical,
            agent_id=str(agent_id),
            session_id=str(session_id),
        )
        return _render_context_result(
            "work_compile_context",
            context=context,
            context_token=context_token,
            canonical_receipt=canonical,
            recovery_handles=handles,
            work_event_id=event_id,
            work_summary=recorded_graph.summary(),
        )
    except Exception as exc:
        return _error("work_compile_context", exc)


def work_context_fault(
    *,
    context: dict[str, Any] | str,
    context_ref: str,
    recovery_handle: dict[str, Any],
    project: str = "",
    workstream_id: str = "",
    agent_id: str = "",
    session_id: str = "",
    token_budget: int | None = None,
) -> dict[str, Any]:
    """Verify one recovery handle, fault in exact bytes, and record the receipt."""
    try:
        bounded_handle = _bounded_contract(recovery_handle, "recovery_handle")
        if not isinstance(bounded_handle, dict):
            raise ValueError("recovery_handle must be a JSON object")
        handle = verify_recovery_handle(bounded_handle)
        selected_ref = _bounded_ref(context_ref, "context_ref")
        selected_workstream = str(workstream_id).strip()
        if selected_workstream:
            selected_workstream = _bounded_id(
                selected_workstream, "workstream_id"
            )

        path = _project_path(project)
        store = _store_for_path(path)
        observation = _passive_observation(path)
        if handle.get("version") != _head_sha(observation):
            raise ValueError("recovery handle repository version is stale")
        graph = store.submit_repository_observation(
            observation, repository_path=path
        )
        scope = graph.context_scope(selected_workstream or None, max_evidence=128)
        if handle.get("repository_id") != scope.get("repo_id"):
            raise ValueError("recovery handle belongs to another repository")

        if isinstance(context, str):
            expected_token = _snapshot_token_from_receipt_id(handle.get("receipt_id"))
            if context != expected_token:
                raise ValueError("context token does not match recovery handle parent")
            bounded_context = _decode_context_token(context, store)
        else:
            bounded_context = _bounded_contract(context, "context")
            if not isinstance(bounded_context, dict):
                raise ValueError("context must be a JSON object or context token")

        _fragments, descriptors = _context_parts(bounded_context)
        matches = [
            item for item in descriptors if item.get("context_ref") == selected_ref
        ]
        if len(matches) != 1:
            raise ValueError("context_ref is not a unique committed omission")
        descriptor = matches[0]
        expected_handle_fields = {
            "receipt_id": _host_receipt_id(bounded_context),
            "disposition": "omitted_but_recoverable",
            "source_ref": descriptor["path"],
            "source_commitment": descriptor["source_sha256"],
            "fragment_commitment": descriptor["fragment_sha256"],
            "byte_start": descriptor["start_byte"],
            "byte_end": descriptor["end_byte"],
        }
        if any(handle.get(key) != value for key, value in expected_handle_fields.items()):
            raise ValueError("recovery handle does not match the committed omission")

        service = RepositoryIntelligenceService(path)
        recovered = service.context_fault(
            bounded_context,
            selected_ref,
            token_budget=token_budget,
        )
        target = next(
            fragment for fragment in recovered["fragments"]
            if fragment["context_ref"] == selected_ref
        )
        recovered_bytes = str(target["content"]).encode(
            "utf-8", errors="surrogateescape"
        )
        if verify_recovered_bytes(handle, recovered_bytes) != "verified":
            raise ValueError("recovered bytes do not match the recovery handle")
        service.validate_context(recovered)

        # Re-observe after byte recovery so a concurrent edit/rebase cannot
        # publish a receipt for a workspace different from the one recovered.
        confirmation = _passive_observation(path)
        if handle.get("version") != _head_sha(confirmation):
            raise ValueError("recovery handle repository version became stale")
        confirmed_graph = store.submit_repository_observation(
            confirmation, repository_path=path
        )
        if confirmed_graph.graph_commitment != graph.graph_commitment:
            raise ValueError("repository changed during context recovery; retry")
        confirmed_scope = confirmed_graph.context_scope(
            selected_workstream or None, max_evidence=128
        )
        if handle.get("repository_id") != confirmed_scope.get("repo_id"):
            raise ValueError("recovery handle belongs to another repository")

        now_ms = int(time.time() * 1000)
        canonical, handles = _context_contracts(
            context=recovered,
            scope=confirmed_scope,
            head_sha=_head_sha(confirmation),
            observed_at_ms=now_ms,
            pinned_refs=[selected_ref],
        )
        context_token = _encode_context_token(recovered, store)
        recorded_graph, event_id = store.record_context_receipt(
            canonical,
            agent_id=str(agent_id),
            session_id=str(session_id),
        )
        return _render_context_result(
            "work_context_fault",
            context=recovered,
            context_token=context_token,
            canonical_receipt=canonical,
            recovery_handles=handles,
            work_event_id=event_id,
            work_summary=recorded_graph.summary(),
            integrity_state="verified",
        )
    except Exception as exc:
        return _error("work_context_fault", exc)


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
    "work_compile_context",
    "work_context_fault",
    "work_handoff",
    "work_record_context",
    "work_record_execution",
    "work_record_memory",
    "work_resume",
    "work_state",
]
