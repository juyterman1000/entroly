"""Python orchestration for Entroly's shared Rust AI Work Graph.

The authoritative Work Graph implementation lives in ``entroly-engine``. This
module intentionally contains no task-state inference, trust upgrades,
coordination rules, or handoff verification logic; it only provides ergonomic
Python conversion and local observation around the PyO3 boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .native_status import (
    WORK_GRAPH_SYMBOLS,
    native_status,
    native_status_message,
)

# Resolved through the shared gate rather than a bare ``import entroly_core``.
# The gate is the single answer to "may this process call into the native core",
# and it refuses a core below MIN_ENTROLY_CORE_VERSION. A bare import here would
# accept a stale core that the rest of the package refuses, producing the mixed
# process `usable_core` exists to prevent -- the failure that once surfaced as
# ``ContextFragment.__new__() got an unexpected keyword argument
# 'recency_score'`` when one module used a stale core and another fell back.
#
# This deliberately does NOT add a pure-Python fallback. Work Graph semantics
# are Rust-owned; a Python re-implementation would be a second source of truth
# for status inference and commitments. Missing or stale core fails closed with
# an actionable message instead.
_NATIVE_STATUS = native_status(WORK_GRAPH_SYMBOLS)
_RustWorkGraph = (
    getattr(_NATIVE_STATUS.module, "WorkGraph", None) if _NATIVE_STATUS.ok else None
)

# Kept so callers and tests can inspect *why* the binding is unavailable. The
# gate reports "absent", "incomplete", or "below the required version" through
# native_status; all three arrive here as an actionable reason rather than a
# bare ImportError from a direct import.
_NATIVE_IMPORT_ERROR: Exception | None = None
if _RustWorkGraph is None:
    _NATIVE_IMPORT_ERROR = ImportError(
        native_status_message(
            _NATIVE_STATUS, feature="the Entroly Work Graph"
        )
    )


class WorkGraphUnavailableError(RuntimeError):
    """Raised when the native Rust Work Graph binding is unavailable."""


def _require_native() -> type:
    if _RustWorkGraph is None:
        detail = f": {_NATIVE_IMPORT_ERROR}" if _NATIVE_IMPORT_ERROR else ""
        raise WorkGraphUnavailableError(
            "Entroly Work Graph requires the native entroly_core extension. "
            'Install it with `pip install "entroly[native]"` and retry' + detail
        )
    return _RustWorkGraph


def _require_native_module():
    """The native module itself, for the free functions rather than the class."""
    if _RustWorkGraph is None:
        detail = f": {_NATIVE_IMPORT_ERROR}" if _NATIVE_IMPORT_ERROR else ""
        raise WorkGraphUnavailableError(
            "Entroly Work Graph requires the native entroly_core extension. "
            'Install it with `pip install "entroly[native]"` and retry' + detail
        )
    return _NATIVE_STATUS.module

def _json_text(value: str | Mapping[str, Any] | list[Any]) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_value(text: str) -> Any:
    return json.loads(text)


class WorkGraph:
    """Evidence-backed temporal state shared across agents and runtimes."""

    __slots__ = ("_inner",)

    def __init__(self, repo_id: str) -> None:
        native = _require_native()
        self._inner = native(repo_id)

    @classmethod
    def from_json(cls, serialized: str | Mapping[str, Any]) -> "WorkGraph":
        native = _require_native()
        obj = cls.__new__(cls)
        obj._inner = native.from_json(_json_text(serialized))
        return obj

    @classmethod
    def from_repository(
        cls,
        path: str = ".",
        *,
        agent_id: str = "",
        session_id: str = "",
        task_hint: dict[str, Any] | None = None,
        default_branch: str | None = None,
        max_commits: int = 20,
        observed_at_ms: int | None = None,
        include_checkpoint: bool = True,
        checkpoint_dir: str | None = None,
    ) -> "WorkGraph":
        """Build a graph from durable repository facts without guessing intent."""
        from .work_graph_repo import discover_repository_observation

        observation = discover_repository_observation(
            path,
            agent_id=agent_id,
            session_id=session_id,
            task_hint=task_hint,
            default_branch=default_branch,
            max_commits=max_commits,
            observed_at_ms=observed_at_ms,
            include_checkpoint=include_checkpoint,
            checkpoint_dir=checkpoint_dir,
        )
        graph = cls(observation["repo_id"])
        graph.observe_repository(observation)
        return graph

    @staticmethod
    def verify_handoff_integrity(receipt: str | Mapping[str, Any]) -> bool:
        native = _require_native()
        return bool(native.verify_handoff_integrity_json(_json_text(receipt)))

    @property
    def repo_id(self) -> str:
        return str(self._inner.repo_id)

    @property
    def revision(self) -> int:
        return int(self._inner.revision)

    @property
    def graph_commitment(self) -> str:
        return str(self._inner.graph_commitment)

    @property
    def event_count(self) -> int:
        return int(self._inner.event_count)

    def apply_event(self, event: str | Mapping[str, Any]) -> str:
        return str(self._inner.apply_event_json(_json_text(event)))

    def observe_repository(self, observation: str | Mapping[str, Any]) -> str:
        return str(self._inner.observe_repository_json(_json_text(observation)))

    def refresh_repository(
        self,
        path: str = ".",
        *,
        agent_id: str = "",
        session_id: str = "",
        task_hint: dict[str, Any] | None = None,
        default_branch: str | None = None,
        max_commits: int = 20,
        observed_at_ms: int | None = None,
        include_checkpoint: bool = True,
        checkpoint_dir: str | None = None,
    ) -> str:
        """Refresh current Git/checkpoint facts using the shared Rust semantics."""
        from .work_graph_repo import discover_repository_observation

        observation = discover_repository_observation(
            path,
            agent_id=agent_id,
            session_id=session_id,
            task_hint=task_hint,
            default_branch=default_branch,
            max_commits=max_commits,
            observed_at_ms=observed_at_ms,
            include_checkpoint=include_checkpoint,
            checkpoint_dir=checkpoint_dir,
        )
        if observation["repo_id"] != self.repo_id:
            raise ValueError(
                f"repository identity changed: expected {self.repo_id}, "
                f"got {observation['repo_id']}"
            )
        return self.observe_repository(observation)

    def merge(self, other: "WorkGraph" | str | Mapping[str, Any]) -> int:
        payload = other.export_json() if isinstance(other, WorkGraph) else _json_text(other)
        return int(self._inner.merge_json(payload))

    def export_json(self, *, pretty: bool = False) -> str:
        return str(self._inner.export_json(pretty))

    def export_state(self) -> dict[str, Any]:
        return _json_value(self.export_json())

    def summary(self) -> dict[str, Any]:
        return _json_value(str(self._inner.summary_json()))

    def snapshot(self, *, pretty: bool = False) -> dict[str, Any]:
        return _json_value(str(self._inner.snapshot_json(pretty)))

    def unfinished(self, *, pretty: bool = False) -> list[dict[str, Any]]:
        return _json_value(str(self._inner.unfinished_json(pretty)))

    def resume(
        self,
        workstream_id: str | None = None,
        *,
        max_evidence: int = 128,
        pretty: bool = False,
    ) -> dict[str, Any]:
        return _json_value(str(self._inner.resume_json(workstream_id, max_evidence, pretty)))

    def context_scope(
        self,
        workstream_id: str | None = None,
        *,
        max_evidence: int = 128,
        pretty: bool = False,
    ) -> dict[str, Any]:
        """Bounded Rust-owned scope for Context/Trust decisions."""
        return _json_value(
            str(self._inner.context_scope_json(workstream_id, max_evidence, pretty))
        )

    def record_context_receipt(
        self,
        receipt: str | Mapping[str, Any],
        *,
        agent_id: str = "",
        session_id: str = "",
    ) -> str:
        """Attach a canonical exact-graph Context Receipt as evidence."""
        return str(
            self._inner.record_context_receipt_json(
                _json_text(receipt), agent_id, session_id
            )
        )

    def record_memory(
        self,
        memory: str | Mapping[str, Any],
        *,
        now_ms: int,
        superseded_ids: list[str] | None = None,
    ) -> str:
        """Attach a canonical provenance-bearing memory record."""
        return str(
            self._inner.record_memory_json(
                _json_text(memory), now_ms, _json_text(superseded_ids or [])
            )
        )

    def record_execution_chain(
        self,
        route: str | Mapping[str, Any],
        outcome: str | Mapping[str, Any],
        verification: str | Mapping[str, Any],
        *,
        invalidated_commitments: list[str] | None = None,
    ) -> str:
        """Append a committed route/execution/verification chain atomically."""
        return str(
            self._inner.record_execution_chain_json(
                _json_text(route),
                _json_text(outcome),
                _json_text(verification),
                _json_text(invalidated_commitments or []),
            )
        )

    def continuation_proof(
        self,
        handoff: str | Mapping[str, Any],
        *,
        context_receipt_commitments: list[str] | None = None,
        routing_commitments: list[str] | None = None,
        execution_outcome_commitments: list[str] | None = None,
        verification_commitments: list[str] | None = None,
        memory_commitments: list[str] | None = None,
        outstanding_work_refs: list[str] | None = None,
        recovery_handle_ids: list[str] | None = None,
        created_at_ms: int,
    ) -> dict[str, Any]:
        """Seal a graph-bound proof spanning context, execution and trust."""
        manifest = {
            "context_receipt_commitments": context_receipt_commitments or [],
            "routing_commitments": routing_commitments or [],
            "execution_outcome_commitments": execution_outcome_commitments or [],
            "verification_commitments": verification_commitments or [],
            "memory_commitments": memory_commitments or [],
            "outstanding_work_refs": outstanding_work_refs or [],
            "recovery_handle_ids": recovery_handle_ids or [],
            "created_at_ms": created_at_ms,
        }
        return _json_value(
            str(
                self._inner.continuation_proof_json(
                    _json_text(handoff), _json_text(manifest)
                )
            )
        )

    def reconstructed_continuation_proof(
        self,
        workstream_id: str,
        to_agent: str,
        *,
        context_receipt_commitments: list[str] | None = None,
        routing_commitments: list[str] | None = None,
        execution_outcome_commitments: list[str] | None = None,
        verification_commitments: list[str] | None = None,
        memory_commitments: list[str] | None = None,
        outstanding_work_refs: list[str] | None = None,
        recovery_handle_ids: list[str] | None = None,
        created_at_ms: int,
    ) -> dict[str, Any]:
        """Reconstruct a graph-bound proof without inventing prior-agent intent."""
        manifest = {
            "context_receipt_commitments": context_receipt_commitments or [],
            "routing_commitments": routing_commitments or [],
            "execution_outcome_commitments": execution_outcome_commitments or [],
            "verification_commitments": verification_commitments or [],
            "memory_commitments": memory_commitments or [],
            "outstanding_work_refs": outstanding_work_refs or [],
            "recovery_handle_ids": recovery_handle_ids or [],
            "created_at_ms": created_at_ms,
        }
        return _json_value(
            str(
                self._inner.reconstructed_continuation_proof_json(
                    workstream_id, to_agent, _json_text(manifest)
                )
            )
        )

    def coordination(self, now_ms: int, *, pretty: bool = False) -> dict[str, Any]:
        return _json_value(str(self._inner.coordination_json(now_ms, pretty)))

    def handoff(
        self,
        workstream_id: str,
        from_agent: str,
        to_agent: str,
        generated_at_ms: int,
        *,
        pretty: bool = False,
    ) -> dict[str, Any]:
        return _json_value(
            str(
                self._inner.handoff_json(
                    workstream_id,
                    from_agent,
                    to_agent,
                    generated_at_ms,
                    pretty,
                )
            )
        )

    def verify_handoff(self, receipt: str | Mapping[str, Any]) -> bool:
        return bool(self._inner.verify_handoff_json(_json_text(receipt)))


def stable_node_id(kind: str, repo_id: str, key: str) -> str:
    """Canonical identity for a graph-addressable artifact.

    Computed by `entroly_engine::work_graph::stable_node_id`, the same function
    the graph uses when it materializes a node, and the same one the WASM
    binding exposes to Node as `workGraphNodeId`. One artifact therefore has one
    id in every runtime.

    This exists because the function was previously unreachable outside Rust.
    `entroly/repository_intelligence` — the highest-fan-in module set in the
    package — grew its own free-form ``symbol_id``, so a ``File`` in the work
    graph and a ``FileRecord`` in repository intelligence described the same
    artifact and could not be matched. Identity is a shared semantic; deriving
    it twice guarantees two graphs that never join.

    ``kind`` is a node-kind token such as ``repository``, ``file``, ``symbol``,
    ``task`` or ``commit``; unknown tokens are rejected rather than hashed.
    """
    native = _require_native_module()
    return str(native.work_graph_node_id(kind, repo_id, key))


def stable_edge_id(from_id: str, kind: str, to_id: str) -> str:
    """Canonical identity for an edge between two nodes.

    See :func:`stable_node_id`. ``kind`` is an edge-kind token such as
    ``contains``, ``defines``, ``imports`` or ``depends_on``.
    """
    native = _require_native_module()
    return str(native.work_graph_edge_id(from_id, kind, to_id))


def create_work_context_receipt(
    *,
    repository_id: str,
    repository_commitment: str,
    graph_commitment: str,
    work_scope_id: str,
    created_at_ms: int,
    source_commitment: str = "",
    selected_refs: list[str] | None = None,
    omitted_refs: list[str] | None = None,
    pinned_refs: list[str] | None = None,
    recoverable_refs: list[str] | None = None,
    recovery_handles: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    budget_tokens: int = 0,
    selection_policy: str = "",
    execution_id: str = "",
) -> dict[str, Any]:
    """Create the canonical Rust-owned ContextReceipt envelope."""
    native = _require_native_module()
    return _json_value(str(native.context_receipt_build_json(
        repository_id,
        repository_commitment,
        graph_commitment,
        work_scope_id,
        source_commitment,
        selected_refs or [],
        omitted_refs or [],
        pinned_refs or [],
        recoverable_refs or [],
        recovery_handles or [],
        evidence_ids or [],
        budget_tokens,
        selection_policy,
        execution_id,
        created_at_ms,
    )))


def verify_work_context_receipt(
    receipt: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and canonicalize a ContextReceipt through the Rust contract."""
    native = _require_native_module()
    return _json_value(str(native.context_receipt_verify_json(_json_text(receipt))))


def create_recovery_handle(
    *,
    repository_id: str,
    receipt_id: str,
    disposition: str,
    observed_at_ms: int,
    source_ref: str = "",
    source_commitment: str = "",
    fragment_commitment: str = "",
    byte_start: int = 0,
    byte_end: int = 0,
    version: str = "",
    storage_locator: str = "",
) -> dict[str, Any]:
    """Create a canonical recovery promise through the Rust contract."""
    native = _require_native_module()
    return _json_value(str(native.recovery_handle_build_json(
        repository_id,
        receipt_id,
        disposition,
        source_ref,
        source_commitment,
        fragment_commitment,
        byte_start,
        byte_end,
        version,
        storage_locator,
        observed_at_ms,
    )))


def verify_recovery_handle(
    handle: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and canonicalize a recovery handle through Rust."""
    native = _require_native_module()
    return _json_value(str(native.recovery_handle_verify_json(_json_text(handle))))


def verify_recovered_bytes(
    handle: str | Mapping[str, Any],
    payload: bytes,
) -> str:
    """Return Rust's integrity state for bytes recovered through a handle."""
    native = _require_native_module()
    return str(native.recovery_handle_verify_bytes(_json_text(handle), payload))


def create_routing_decision(
    *,
    repository_id: str,
    task_id: str,
    workstream_id: str,
    provider: str,
    model: str,
    runtime: str,
    context_budget_tokens: int,
    policy_version: str,
    decided_at_ms: int,
    reason_codes: list[str] | None = None,
    feature_commitments: list[str] | None = None,
    fallback_route_ids: list[str] | None = None,
    receipt_id: str = "",
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create the canonical inspectable route contract in Rust."""
    native = _require_native_module()
    return _json_value(
        str(
            native.routing_decision_build_json(
                repository_id,
                task_id,
                workstream_id,
                provider,
                model,
                runtime,
                context_budget_tokens,
                policy_version,
                reason_codes or [],
                feature_commitments or [],
                fallback_route_ids or [],
                receipt_id,
                evidence_ids or [],
                decided_at_ms,
            )
        )
    )


def create_model_execution_outcome(
    *,
    routing_id: str,
    repository_id: str,
    task_id: str,
    workstream_id: str,
    provider: str,
    model: str,
    runtime: str,
    state: str,
    verification_state: str,
    completed_at_ms: int,
    receipt_id: str = "",
    request_commitment: str = "",
    response_commitment: str = "",
    latency_ms: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_micro_usd: int = 0,
    error_code: str = "",
    evidence_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create the canonical result of one routed model execution."""
    native = _require_native_module()
    return _json_value(
        str(
            native.model_execution_outcome_build_json(
                routing_id,
                repository_id,
                task_id,
                workstream_id,
                provider,
                model,
                runtime,
                receipt_id,
                request_commitment,
                response_commitment,
                state,
                verification_state,
                latency_ms,
                input_tokens,
                output_tokens,
                cost_micro_usd,
                error_code,
                evidence_ids or [],
                completed_at_ms,
            )
        )
    )


def create_verification_record(
    *,
    repository_id: str,
    subject_id: str,
    subject_commitment: str,
    verified_repository_commitment: str,
    verdict: str,
    observed_at_ms: int,
    valid_until_ms: int = 0,
    evidence_ids: list[str] | None = None,
    dependency_commitments: list[str] | None = None,
) -> dict[str, Any]:
    """Create exact-version verification with transitive dependencies."""
    native = _require_native_module()
    return _json_value(
        str(
            native.verification_record_build_json(
                repository_id,
                subject_id,
                subject_commitment,
                verified_repository_commitment,
                verdict,
                evidence_ids or [],
                dependency_commitments or [],
                observed_at_ms,
                valid_until_ms,
            )
        )
    )


def verification_freshness(
    record: str | Mapping[str, Any],
    *,
    current_repository_commitment: str,
    now_ms: int,
    invalidated_commitments: list[str] | None = None,
) -> str:
    """Return current, stale, invalidated, or unknown from Rust semantics."""
    native = _require_native_module()
    return str(
        native.verification_record_freshness(
            _json_text(record),
            current_repository_commitment,
            now_ms,
            invalidated_commitments or [],
        )
    )


__all__ = [
    "create_model_execution_outcome",
    "create_recovery_handle",
    "create_routing_decision",
    "create_verification_record",
    "create_work_context_receipt",
    "stable_edge_id",
    "stable_node_id",
    "verification_freshness",
    "verify_recovered_bytes",
    "verify_recovery_handle",
    "verify_work_context_receipt",
    "WorkGraph",
    "WorkGraphUnavailableError",
]
