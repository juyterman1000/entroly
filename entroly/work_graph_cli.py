"""CLI orchestration for the shared Rust AI Work Graph."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from .work_graph import WorkGraphUnavailableError
from .work_graph_content_digest import enrich_worktree_content_digests
from .work_graph_repo import discover_repository_identity, discover_repository_observation
from .work_graph_store import (
    continuation_outstanding_refs,
    WorkGraphLockTimeout,
    WorkGraphStateError,
    WorkGraphStore,
    WorkGraphStoreError,
)

_MAX_TTL_SECONDS = 30 * 24 * 60 * 60
_MAX_EVIDENCE = 4096
_MAX_ID_CHARS = 512
_MAX_LABEL_CHARS = 8192


def _project_path(project: str) -> Path:
    return Path(project or ".").expanduser().resolve()


def _store_for_path(project: Path) -> WorkGraphStore:
    identity = discover_repository_identity(project)
    return WorkGraphStore(identity["repo_id"])


def _passive_observation(project: Path) -> dict[str, Any]:
    observation = discover_repository_observation(project)
    return enrich_worktree_content_digests(project, observation)


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    return None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _add_field(lines: list[str], label: str, value: object) -> None:
    if value in (None, "", [], {}):
        return
    if isinstance(value, (list, tuple, set)):
        value = len(value)
    elif isinstance(value, dict):
        value = len(value)
    text = str(value)
    if len(text) > 160:
        text = text[:157] + "..."
    lines.append(f"  {label:<12} {text}")


def _coordination_conflicts(coordination: object) -> int | None:
    data = _mapping(coordination)
    for key in ("conflicts", "active_conflicts", "overlaps"):
        count = _count(data.get(key))
        if count is not None:
            return count
    return None


def _summary_lines(summary: object) -> list[str]:
    data = _mapping(summary)
    lines: list[str] = []
    metrics = (
        ("Events", ("event_count", "events")),
        ("Nodes", ("node_count", "nodes")),
        ("Edges", ("edge_count", "edges")),
        ("Tasks", ("task_count", "tasks")),
        ("Workstreams", ("workstream_count", "workstreams")),
        ("Evidence", ("evidence_count", "evidence")),
    )
    for label, keys in metrics:
        value = _first(data, *keys)
        count = _count(value)
        _add_field(lines, label, count if count is not None else value)
    return lines


def _human_lines(payload: dict[str, Any]) -> list[str]:
    lines = ["Entroly Work Graph"]

    if "repo_id" in payload:
        _add_field(lines, "Repository", payload.get("repo_id"))
        lines.extend(_summary_lines(payload.get("summary")))
        unfinished = _count(payload.get("unfinished"))
        if unfinished is not None:
            _add_field(lines, "Unfinished", unfinished)
        conflicts = _coordination_conflicts(payload.get("coordination"))
        if conflicts is not None:
            _add_field(lines, "Conflicts", conflicts)
        if unfinished == 0:
            lines.append("  State        No unfinished work is currently recorded.")

    elif "lease_id" in payload:
        lines.append("  Status       Work claim recorded.")
        _add_field(lines, "Lease", payload.get("lease_id"))
        lines.extend(_summary_lines(payload.get("summary")))
        conflicts = _coordination_conflicts(payload.get("coordination"))
        if conflicts is not None:
            _add_field(lines, "Conflicts", conflicts)

    elif "resume" in payload:
        view = _mapping(payload.get("resume"))
        lines.append("  Status       Resume package ready.")
        _add_field(lines, "Workstream", _first(view, "workstream_id", "workstream", "id"))
        _add_field(lines, "Title", _first(view, "title", "task_title", "label"))
        _add_field(lines, "Work status", _first(view, "status", "work_status"))
        _add_field(lines, "Tasks", _first(view, "task_ids", "tasks"))
        _add_field(lines, "Agents", _first(view, "agent_ids", "agents"))
        _add_field(lines, "Changes", _first(view, "changed_paths", "changes"))
        _add_field(lines, "Evidence", _first(view, "evidence_ids", "evidence"))
        _add_field(lines, "Remaining", _first(view, "remaining_work", "remaining"))
        if len(lines) == 2:
            lines.append("  State        Recovery succeeded; use --json for the complete view.")

    elif "handoff" in payload:
        receipt = _mapping(payload.get("handoff"))
        lines.append("  Status       Handoff sealed to the current Work Graph.")
        _add_field(lines, "Workstream", _first(receipt, "workstream_id", "workstream"))
        _add_field(lines, "From", _first(receipt, "source_agent", "from_agent"))
        _add_field(lines, "To", _first(receipt, "target_agent", "to_agent"))
        _add_field(lines, "Receipt", _first(receipt, "receipt_id", "handoff_id", "id"))
        _add_field(lines, "Commitment", _first(receipt, "graph_commitment", "commitment"))
        if len(lines) == 2:
            lines.append("  State        Handoff created; use --json for the complete receipt.")

    else:
        lines.append("  Status       Operation completed.")

    lines.append("  Detail       Use --json for the complete machine-readable result.")
    return lines


def _emit(payload: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    status = payload.get("status", "ok")
    if status == "error":
        code = payload.get("error", "work_graph_error")
        detail = payload.get("detail", "unknown")
        print(f"Entroly Work Graph error [{code}]: {detail}", file=sys.stderr)
        return
    print("\n".join(_human_lines(payload)))


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


def _required_id(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > _MAX_ID_CHARS or "\x00" in text:
        raise ValueError(f"{name} may not exceed {_MAX_ID_CHARS} characters or contain NUL")
    return text


def _claim(project: Path, args: Any) -> tuple[Any, str]:
    ttl = float(args.ttl)
    if not math.isfinite(ttl) or not 0 < ttl <= _MAX_TTL_SECONDS:
        raise ValueError(f"ttl must be > 0 and <= {_MAX_TTL_SECONDS}")
    agent = _required_id(args.agent, "--agent")
    title = str(args.task).strip()
    if not title:
        raise ValueError("--task must not be empty")
    if len(title) > _MAX_LABEL_CHARS:
        raise ValueError(f"--task may not exceed {_MAX_LABEL_CHARS} characters")
    task_id = str(args.task_id or "").strip()
    session_id = str(args.session or "").strip()
    if len(task_id) > _MAX_ID_CHARS or "\x00" in task_id:
        raise ValueError(f"--task-id may not exceed {_MAX_ID_CHARS} characters or contain NUL")
    if len(session_id) > _MAX_ID_CHARS or "\x00" in session_id:
        raise ValueError(f"--session may not exceed {_MAX_ID_CHARS} characters or contain NUL")
    store = _store_for_path(project)
    now_ms = int(time.time() * 1000)
    lease_id = uuid.uuid4().hex
    observation = discover_repository_observation(
        project,
        agent_id=agent,
        session_id=session_id,
        task_hint={
            "task_id": task_id,
            "title": title,
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": [],
            "source_kind": "user_statement",
            "source_ref": f"work-claim:{lease_id}",
        },
        observed_at_ms=now_ms,
    )
    observation["leases"] = [{
        "lease_id": lease_id,
        "agent_id": agent,
        "task_id": task_id,
        "scope_paths": sorted(set(args.path or [])),
        "scope_symbols": sorted(set(args.symbol or [])),
        "expires_at_ms": now_ms + max(1, int(ttl * 1000)),
        "source_ref": f"work-lease:{lease_id}",
    }]
    enrich_worktree_content_digests(project, observation)
    return store.submit_repository_observation(
        observation, repository_path=project
    ), lease_id


def run(args: Any) -> int:
    action = getattr(args, "work_action", None)
    json_output = bool(getattr(args, "json_output", False))
    try:
        project = _project_path(getattr(args, "project", "."))
        if action == "state":
            store = _store_for_path(project)
            graph = store.load()
            payload = {
                "status": "ok",
                "repo_id": graph.repo_id,
                "summary": graph.summary(),
                "unfinished": graph.unfinished(),
                "coordination": graph.coordination(int(time.time() * 1000)),
            }
        elif action == "claim":
            graph, lease_id = _claim(project, args)
            payload = {
                "status": "ok",
                "lease_id": lease_id,
                "summary": graph.summary(),
                "coordination": graph.coordination(int(time.time() * 1000)),
            }
        elif action == "resume":
            max_evidence = args.max_evidence
            if (
                not isinstance(max_evidence, int)
                or isinstance(max_evidence, bool)
                or not 0 <= max_evidence <= _MAX_EVIDENCE
            ):
                raise ValueError(
                    f"max_evidence must be an integer between 0 and {_MAX_EVIDENCE}"
                )
            workstream = str(args.workstream or "").strip()
            if len(workstream) > _MAX_ID_CHARS or "\x00" in workstream:
                raise ValueError(
                    f"--workstream may not exceed {_MAX_ID_CHARS} characters or contain NUL"
                )
            store = _store_for_path(project)
            # Resume is an explicit recovery action: refresh bounded durable
            # repository/checkpoint facts plus exact unstaged worktree identity
            # before Rust reconstructs the workstream.
            store.submit_repository_observation(
                _passive_observation(project), repository_path=project
            )
            resume_view = store.resume(workstream or None, max_evidence=max_evidence)
            payload = {
                "status": "ok",
                "resume": resume_view,
            }
            target_agent = str(getattr(args, "to_agent", "") or "").strip()
            if target_agent:
                target_agent = _required_id(target_agent, "--to-agent")
                payload["continuation_proof"] = store.reconstructed_continuation_proof(
                    str(resume_view["selected_workstream"]["node_id"]),
                    target_agent,
                    outstanding_work_refs=continuation_outstanding_refs(resume_view),
                    created_at_ms=int(time.time() * 1000),
                )
        elif action == "handoff":
            workstream = _required_id(args.workstream, "--workstream")
            source_agent = _required_id(args.from_agent, "--from-agent")
            target_agent = _required_id(args.to_agent, "--to-agent")
            store = _store_for_path(project)
            # A handoff receipt must commit to the latest durable worktree and
            # checkpoint facts, including exact unstaged worktree content identity.
            store.submit_repository_observation(
                _passive_observation(project), repository_path=project
            )
            handoff = store.handoff(workstream, source_agent, target_agent)
            resume_view = store.resume(workstream, max_evidence=128)
            payload = {
                "status": "ok",
                "handoff": handoff,
                "continuation_proof": store.continuation_proof(
                    handoff,
                    outstanding_work_refs=continuation_outstanding_refs(resume_view),
                    created_at_ms=int(time.time() * 1000),
                ),
            }
        else:
            payload = {
                "status": "error",
                "error": "missing_work_action",
                "detail": "choose state, claim, resume, or handoff",
            }
    except Exception as exc:
        payload = _error(exc)
    _emit(payload, json_output=json_output)
    return 0 if payload.get("status") == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entroly-work",
        description="Inspect, resume, coordinate, and hand off evidence-backed repository work.",
    )
    parser.add_argument("--project", default=".", help="Git worktree to inspect")
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit the complete machine-readable result",
    )
    actions = parser.add_subparsers(dest="work_action", required=True)
    actions.add_parser("state", help="Show shared repository work state")
    claim = actions.add_parser("claim", help="Claim work with an advisory lease")
    claim.add_argument("--agent", required=True)
    claim.add_argument("--task", required=True)
    claim.add_argument("--task-id", default="")
    claim.add_argument("--session", default="")
    claim.add_argument("--path", action="append", default=[])
    claim.add_argument("--symbol", action="append", default=[])
    claim.add_argument("--ttl", type=float, default=900.0)
    resume = actions.add_parser("resume", help="Recover an unfinished workstream")
    resume.add_argument("--workstream", default="")
    resume.add_argument("--max-evidence", type=int, default=128)
    resume.add_argument(
        "--to-agent",
        default="",
        help="Also seal an evidence-bounded no-handoff continuation proof",
    )
    handoff = actions.add_parser("handoff", help="Seal a graph-bound cross-agent handoff")
    handoff.add_argument("--workstream", required=True)
    handoff.add_argument("--from-agent", required=True)
    handoff.add_argument("--to-agent", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run"]
