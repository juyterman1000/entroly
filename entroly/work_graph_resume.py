"""Shared, evidence-bounded selection for recovering unfinished work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_CONTINUATION_CONTENT_FIELDS = (
    "remaining_work",
    "decision_ids",
    "verification_ids",
    "failure_ids",
    "agent_ids",
)


def resumable_workstream_id(store: Any) -> str:
    """Prefer recorded work over a newer, empty passive-observation placeholder.

    Refreshing repository facts can create a hintless branch workstream. The
    Work Graph orders unfinished work by recency, so a plain resume(None) can
    select that placeholder instead of the task the previous agent was doing.
    Preserve recency within the contentful and placeholder groups.
    """
    unfinished = store.load().unfinished()
    if not unfinished:
        return ""
    for item in unfinished:
        if isinstance(item, dict) and any(
            item.get(field) for field in _CONTINUATION_CONTENT_FIELDS
        ):
            return str(item.get("node_id") or "")
    return str(unfinished[0].get("node_id") or "")


def recovery_unknowns(view: Any) -> list[str]:
    """Carry uncertainty forward without inferring the previous agent's intent."""
    unknowns = {"unknown:previous-agent-intent"}
    if isinstance(view, dict):
        recorded = view.get("unknowns")
        if isinstance(recorded, list):
            unknowns.update(str(item) for item in recorded if isinstance(item, str))
        if view.get("current_worktree_changes_truncated"):
            unknowns.add("unknown:current-worktree-changes-truncated")
    return sorted(unknowns)


def attach_current_worktree_changes(
    view: dict[str, Any], observation: dict[str, Any], *, max_paths: int = 128
) -> dict[str, Any]:
    """Keep live changed paths distinct from a workstream's historical paths."""
    if not 0 <= max_paths <= 4096:
        raise ValueError("max_paths must be between 0 and 4096")
    changes = observation.get("changes")
    paths: list[str] = []
    if isinstance(changes, list):
        paths = sorted({
            item["path"]
            for item in changes
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and item["path"]
        })
    view["current_worktree_changed_paths"] = paths[:max_paths]
    view["current_worktree_change_count"] = len(paths)
    view["current_worktree_changes_truncated"] = len(paths) > max_paths
    view["current_workspace_commitment"] = workspace_commitment(observation)
    return view


def workspace_commitment(observation: dict[str, Any]) -> str:
    """Bind the branch and observed change digests, excluding clock time."""
    stable = {
        "repo_id": observation.get("repo_id"),
        "branch": observation.get("branch"),
        "changes": observation.get("changes"),
        "path_disclosure": observation.get("path_disclosure"),
    }
    raw = json.dumps(stable, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"workspace:{digest}"


def recover_workstream(
    store: Any,
    project: Path,
    observation: dict[str, Any],
    *,
    workstream_id: str = "",
    max_evidence: int = 128,
) -> dict[str, Any]:
    """Refresh durable facts, select recorded work, and expose live changes."""
    store.submit_repository_observation(observation, repository_path=project)
    selected = workstream_id or resumable_workstream_id(store)
    view = store.resume(selected or None, max_evidence=max_evidence)
    return attach_current_worktree_changes(view, observation, max_paths=max_evidence)
