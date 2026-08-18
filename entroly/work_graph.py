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


__all__ = [
    "stable_edge_id",
    "stable_node_id","WorkGraph", "WorkGraphUnavailableError"]
