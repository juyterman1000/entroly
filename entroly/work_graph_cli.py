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
    return store.submit_observation(observation), lease_id


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
            store.submit_observation(_passive_observation(project))
            payload = {
                "status": "ok",
                "resume": store.resume(workstream or None, max_evidence=max_evidence),
            }
        elif action == "handoff":
            workstream = _required_id(args.workstream, "--workstream")
            source_agent = _required_id(args.from_agent, "--from-agent")
            target_agent = _required_id(args.to_agent, "--to-agent")
            store = _store_for_path(project)
            # A handoff receipt must commit to the latest durable worktree and
            # checkpoint facts, including exact unstaged worktree content identity.
            store.submit_observation(_passive_observation(project))
            payload = {
                "status": "ok",
                "handoff": store.handoff(workstream, source_agent, target_agent),
            }
        else:
            payload = {"status": "error", "error": "missing_work_action", "detail": "choose state, claim, resume, or handoff"}
    except Exception as exc:
        payload = _error(exc)
    _emit(payload, json_output=json_output)
    return 0 if payload.get("status") == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m entroly.work_graph_cli", description="Entroly AI Work Graph")
    parser.add_argument("--project", default=".", help="Git worktree to inspect")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit JSON")
    actions = parser.add_subparsers(dest="work_action", required=True)
    actions.add_parser("state", help="Show persisted shared work state")
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
    handoff = actions.add_parser("handoff", help="Create a fresh graph-bound handoff receipt")
    handoff.add_argument("--workstream", required=True)
    handoff.add_argument("--from-agent", required=True)
    handoff.add_argument("--to-agent", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run"]
