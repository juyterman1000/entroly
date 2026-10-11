"""Compact, source-bound agent review records on the existing Work Graph.

The review verdict is an agent statement. The graph proves which repository
state and handoff were reviewed; it does not prove that the verdict is correct.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .work_graph_content_digest import enrich_worktree_content_digests
from .work_graph import WorkGraph, stable_edge_id, stable_node_id
from .work_graph_mcp import _project_path, _store_for_path
from .work_graph_repo import discover_repository_observation

_MAX_FINDINGS = 32
_MAX_TEXT = 1024
_MAX_FILE_BYTES = 4 * 1024 * 1024
_VERDICTS = {"approve", "changes_requested", "comment"}
_SEVERITIES = {"critical", "high", "medium", "low", "note"}


def _bounded(value: Any, name: str, limit: int = _MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return value.strip()


def _observation(path: Path) -> dict[str, Any]:
    observed = discover_repository_observation(path)
    enrich_worktree_content_digests(path, observed)
    return observed


def _source_state(repo_id: str, observed: dict[str, Any]) -> str:
    branch = observed["branch"]
    changes = {
        str(item.get("path", "")): str(item.get("content_digest", ""))
        for item in observed.get("changes", [])
    }
    return _source_state_parts(repo_id, branch.get("name", ""), branch.get("head_sha", ""), changes)


def _source_state_parts(repo_id: str, branch: Any, head: Any, changes: dict[str, str]) -> str:
    lines = [repo_id, str(branch), str(head)]
    for path, digest in sorted(changes.items()):
        lines.append(f"{path}\t{digest}")
    return "sha256:" + hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _graph_source_state(graph: Any, repo_id: str) -> str:
    """Read the last repository observation already sealed by a handoff."""
    for event in reversed(graph.export_state()["events"]):
        operations = event.get("operations", [])
        repository = next((
            op["node"]["attributes"] for op in operations
            if op.get("op") == "upsert_node"
            and op.get("node", {}).get("kind") == "repository"
            and "head_sha" in op["node"].get("attributes", {})
        ), None)
        if repository is None:
            continue
        changes = {}
        for op in operations:
            evidence = op.get("evidence", {}) if op.get("op") == "add_evidence" else {}
            if evidence.get("kind") == "git_status":
                changes[str(evidence["locator"])] = str(evidence.get("digest", ""))
        return _source_state_parts(
            repo_id, repository.get("branch", ""), repository.get("head_sha", ""), changes
        )
    raise ValueError("handoff graph contains no repository observation")


def _handoff_graph(graph: WorkGraph, handoff: dict[str, Any]) -> WorkGraph:
    """Verify a receipt against its sealed event prefix after later graph writes."""
    revision = handoff.get("graph_revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or not 0 < revision <= graph.revision:
        raise ValueError("handoff is stale or has an invalid graph revision")
    if revision == graph.revision:
        historical = graph
    else:
        historical = WorkGraph(graph.repo_id)
        for event in graph.export_state()["events"][:revision]:
            historical.apply_event(event)
    if not historical.verify_handoff(handoff):
        raise ValueError("handoff is stale or fails graph integrity verification")
    return historical


def _findings(root: Path, values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list) or len(values) > _MAX_FINDINGS:
        raise ValueError(f"findings must be a list of at most {_MAX_FINDINGS} items")
    result = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"findings[{index}] must be an object")
        rel = _bounded(item.get("path"), "finding path", 4096).replace("\\", "/")
        candidate = root / rel
        resolved = candidate.resolve()
        if resolved == root or not resolved.is_relative_to(root) or candidate.is_symlink():
            raise ValueError("finding path must be a regular file inside the repository")
        if not resolved.is_file() or resolved.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError("finding path must be a readable file of at most 4 MiB")
        line = item.get("line")
        if not isinstance(line, int) or isinstance(line, bool) or line < 1:
            raise ValueError("finding line must be a positive integer")
        content = resolved.read_bytes()
        if line > len(content.splitlines()):
            raise ValueError("finding line is outside the current file")
        severity = _bounded(item.get("severity"), "finding severity", 16)
        if severity not in _SEVERITIES:
            raise ValueError("finding severity is invalid")
        result.append({
            "path": resolved.relative_to(root).as_posix(),
            "line": line,
            "severity": severity,
            "message": _bounded(item.get("message"), "finding message"),
            "source_digest": "sha256:" + hashlib.sha256(content).hexdigest(),
        })
    return result


def _checks(values: Any) -> list[dict[str, Any]]:
    if values is None:
        return []
    if not isinstance(values, list) or len(values) > 16:
        raise ValueError("checks must be a list of at most 16 items")
    result = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"checks[{index}] must be an object")
        code = item.get("exit_code")
        if not isinstance(code, int) or isinstance(code, bool) or not -255 <= code <= 255:
            raise ValueError("check exit_code must be an integer between -255 and 255")
        result.append({"name": _bounded(item.get("name"), "check name", 256), "exit_code": code})
    return result


def record_review(
    *, handoff: dict[str, Any], reviewer_id: str, verdict: str,
    findings: list[dict[str, Any]], summary: str, project: str = "",
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record one independent agent statement after verifying exact source state."""
    reviewer = _bounded(reviewer_id, "reviewer_id", 512)
    choice = _bounded(verdict, "verdict", 32)
    if choice not in _VERDICTS:
        raise ValueError("verdict must be approve, changes_requested, or comment")
    if not isinstance(handoff, dict) or handoff.get("to_agent") != reviewer:
        raise ValueError("handoff must target the reviewer")
    if handoff.get("from_agent") == reviewer:
        raise ValueError("writer and reviewer must differ")
    root = _project_path(project)
    store = _store_for_path(root)
    summary_text = _bounded(summary, "summary")
    checked_checks = _checks(checks)

    def append(graph: Any) -> dict[str, Any]:
        sealed = _handoff_graph(graph, handoff)
        observed = _observation(root)
        state = _source_state(store.repo_id, observed)
        if state != _graph_source_state(sealed, store.repo_id):
            raise ValueError("handoff is stale against current repository source")
        checked_findings = _findings(root, findings)
        if _source_state(store.repo_id, _observation(root)) != state:
            raise ValueError("repository source changed while recording review")
        if choice == "approve" and any(
            finding["severity"] in {"critical", "high"} for finding in checked_findings
        ):
            raise ValueError("approve conflicts with blocking findings")
        payload = {
            "schema_version": 1,
            "record_kind": "agent_review",
            "repo_id": store.repo_id,
            "workstream_id": handoff["workstream_id"],
            "handoff_commitment": handoff["payload_commitment"],
            "writer_id": handoff["from_agent"],
            "reviewer_id": reviewer,
            "source_state": state,
            "verdict": choice,
            "summary": summary_text,
            "findings": checked_findings,
            "checks": checked_checks,
            "trust_boundary": "agent_statement; source digests observed; verdict unverified",
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        review_key = "agent-review:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        review_id = stable_node_id("decision", store.repo_id, review_key)
        now_ms = int(time.time() * 1000)
        event = {
            "observed_at_ms": now_ms,
            "source_kind": "agent_statement",
            "source_ref": review_id,
            "actor_id": reviewer,
            "operations": [{
                "op": "upsert_node",
                "node": {
                    "node_id": review_id,
                    "kind": "decision",
                    "label": "agent review recorded",
                    "trust": "untrusted",
                    "status": "completed",
                    "status_trust": "untrusted",
                    "attributes": payload,
                    "updated_at_ms": now_ms,
                },
            }, {
                "op": "upsert_edge",
                "edge": {
                    "edge_id": stable_edge_id(review_id, "part_of", handoff["workstream_id"]),
                    "from_node": review_id,
                    "to_node": handoff["workstream_id"],
                    "kind": "part_of",
                    "trust": "untrusted",
                    "updated_at_ms": now_ms,
                },
            }],
        }
        graph.apply_event(event)
        return {"review_id": review_id, **payload}

    _, record = store._mutate(append)
    return {"status": "ok", "kind": "work_record_review", "review": record}


def reviews(*, project: str = "", workstream_id: str = "") -> dict[str, Any]:
    """Read compact review records and report source staleness without mutation."""
    root = _project_path(project)
    store = _store_for_path(root)
    current = _source_state(store.repo_id, _observation(root))
    records = []
    for node in store.load().snapshot().get("nodes", []):
        attrs = node.get("attributes", {})
        if attrs.get("record_kind") != "agent_review":
            continue
        if workstream_id and attrs.get("workstream_id") != workstream_id:
            continue
        records.append({"review_id": node["node_id"], "current": attrs["source_state"] == current, **attrs})
    return {"status": "ok", "kind": "work_reviews", "reviews": records}
