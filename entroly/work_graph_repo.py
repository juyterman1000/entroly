"""Conservative repository observation adapter for Entroly AI Work Graph.

This module is orchestration-only. It observes durable local Git/checkpoint
facts and normalizes them into the schema owned by ``entroly-engine``. It never
decides work status, upgrades trust, or infers task intent from filenames,
commit prose, or branch names; those semantics stay in Rust.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

_MAX_GIT_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_CHANGES = 512
_MAX_COMMITS = 20


class RepositoryDiscoveryError(RuntimeError):
    """Raised when a repository cannot be observed safely and completely."""


def _run_git(
    cwd: Path,
    *args: str,
    timeout: float = 5.0,
    check: bool = True,
) -> str:
    """Run one read-only Git observation with hostile-repo safeguards."""
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "LC_ALL": "C",
            "LANG": "C",
        }
    )
    command = [
        "git",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "submodule.recurse=false",
        "-C",
        str(cwd),
        *args,
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RepositoryDiscoveryError(f"git {' '.join(args)} failed: {exc}") from exc

    stdout_bytes = len(result.stdout.encode("utf-8", "replace"))
    stderr_bytes = len(result.stderr.encode("utf-8", "replace"))
    if stdout_bytes > _MAX_GIT_OUTPUT_BYTES or stderr_bytes > _MAX_GIT_OUTPUT_BYTES:
        raise RepositoryDiscoveryError(
            f"git {' '.join(args)} produced more than {_MAX_GIT_OUTPUT_BYTES} bytes; "
            "refusing a partial Work Graph observation"
        )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or f"exit {result.returncode}"
        raise RepositoryDiscoveryError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _try_git(cwd: Path, *args: str, timeout: float = 5.0) -> str:
    try:
        return _run_git(cwd, *args, timeout=timeout, check=True).strip()
    except RepositoryDiscoveryError:
        return ""


def _normalize_remote(remote: str) -> str:
    value = remote.strip().rstrip("/")
    if not value:
        return ""
    if value.endswith(".git"):
        value = value[:-4]
    if "://" in value:
        parsed = urlsplit(value)
        host = (parsed.hostname or "").lower()
        if parsed.port:
            host = f"{host}:{parsed.port}"
        path = parsed.path.strip("/")
        return f"{host}/{path}" if host and path else ""
    if ":" in value and "@" in value.split(":", 1)[0]:
        lhs, rhs = value.split(":", 1)
        host = lhs.rsplit("@", 1)[-1].lower()
        return f"{host}/{rhs.strip('/')}"
    return ""


def _repository_id(root: Path) -> str:
    remote = _normalize_remote(_try_git(root, "config", "--get", "remote.origin.url"))
    if remote:
        digest = hashlib.sha256(remote.encode("utf-8")).hexdigest()[:32]
        return f"git:{digest}"
    roots = _try_git(root, "rev-list", "--max-parents=0", "HEAD").splitlines()
    if roots:
        material = f"{roots[0]}\0{root.name}".encode("utf-8", "surrogatepass")
        return f"git-root:{hashlib.sha256(material).hexdigest()[:32]}"
    canonical = os.path.normcase(str(root.resolve())).encode("utf-8", "surrogatepass")
    return f"git-local:{hashlib.sha256(canonical).hexdigest()[:32]}"


def _resolve_root(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    root = _try_git(candidate, "rev-parse", "--show-toplevel")
    if not root:
        raise RepositoryDiscoveryError(f"not a Git worktree: {candidate}")
    return Path(root).resolve()


def _validated_branch_override(root: Path, override: str | None) -> str:
    if not override:
        return ""
    name = override.removeprefix("origin/").strip()
    if not name or name.startswith("-") or "\x00" in name or "\n" in name or "\r" in name:
        raise RepositoryDiscoveryError(f"invalid default branch override: {override!r}")
    if not _try_git(root, "check-ref-format", f"refs/heads/{name}"):
        raise RepositoryDiscoveryError(f"invalid default branch override: {override!r}")
    return name


def _default_branch(root: Path, override: str | None) -> str:
    validated = _validated_branch_override(root, override)
    if validated:
        return validated
    remote_head = _try_git(root, "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD")
    if remote_head.startswith("origin/"):
        return remote_head[len("origin/") :]
    for candidate in ("main", "master"):
        if _try_git(root, "rev-parse", "--verify", "--quiet", f"refs/heads/{candidate}"):
            return candidate
    return ""


def _base_ref(root: Path, default_branch: str) -> str:
    if not default_branch:
        return ""
    for candidate in (f"refs/remotes/origin/{default_branch}", f"refs/heads/{default_branch}"):
        if _try_git(root, "rev-parse", "--verify", "--quiet", candidate):
            return candidate
    return ""


def _ahead_behind(root: Path, base_ref: str) -> tuple[int, int]:
    if not base_ref:
        return 0, 0
    output = _try_git(root, "rev-list", "--left-right", "--count", f"{base_ref}...HEAD")
    try:
        behind, ahead = (int(part) for part in output.split())
    except (TypeError, ValueError):
        return 0, 0
    return ahead, behind


def _git_dir(root: Path) -> Path | None:
    value = _try_git(root, "rev-parse", "--absolute-git-dir")
    return Path(value).resolve() if value else None


def _parse_status(root: Path) -> list[dict[str, Any]]:
    raw = _run_git(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=all",
    )
    tokens = raw.split("\0")
    changes: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        entry = tokens[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 3:
            raise RepositoryDiscoveryError("malformed porcelain status record")
        xy = entry[:2]
        path = entry[3:].replace("\\", "/")
        old_path = ""
        if ("R" in xy or "C" in xy) and index < len(tokens):
            old_path = tokens[index].replace("\\", "/")
            index += 1

        conflicted = "U" in xy or xy in {"DD", "AA"}
        if xy == "??":
            kind = "untracked"
        elif conflicted:
            kind = "unmerged"
        elif "R" in xy:
            kind = "renamed"
        elif "C" in xy:
            kind = "copied"
        elif "D" in xy:
            kind = "deleted"
        elif "A" in xy:
            kind = "added"
        elif "M" in xy or "T" in xy:
            kind = "modified"
        else:
            kind = "unknown"

        changes.append(
            {
                "path": path,
                "kind": kind,
                "staged": xy[0] not in {" ", "?", "!"},
                "conflicted": conflicted,
                "old_path": old_path,
            }
        )
        if len(changes) > _MAX_CHANGES:
            raise RepositoryDiscoveryError(
                f"repository has more than {_MAX_CHANGES} changed/untracked paths; "
                "refusing a partial Work Graph observation"
            )
    changes.sort(key=lambda item: (item["path"], item["old_path"], item["kind"]))
    return changes


def _branch_commits(root: Path, base_ref: str, ahead_by: int, max_commits: int) -> list[dict[str, Any]]:
    if max_commits < 0:
        raise RepositoryDiscoveryError("max_commits must be >= 0")
    if max_commits > _MAX_COMMITS:
        raise RepositoryDiscoveryError(f"max_commits must be <= {_MAX_COMMITS}")
    if ahead_by <= 0 or not base_ref or max_commits == 0:
        return []
    output = _run_git(
        root,
        "log",
        "--no-decorate",
        f"-n{max_commits}",
        "--format=%H%x00%ct%x00%P%x00%s%x00",
        f"{base_ref}..HEAD",
    )
    fields = output.split("\0")
    commits: list[dict[str, Any]] = []
    for offset in range(0, len(fields) - 3, 4):
        sha, timestamp, parents, subject = fields[offset : offset + 4]
        sha = sha.strip()
        if not sha:
            continue
        try:
            timestamp_ms = int(timestamp.strip()) * 1000
        except ValueError:
            timestamp_ms = 0
        commits.append(
            {
                "sha": sha,
                "subject": subject.strip(),
                "timestamp_ms": timestamp_ms,
                "parent_shas": [item for item in parents.split() if item],
                "changed_paths": [],
            }
        )
    commits.sort(key=lambda item: (item["timestamp_ms"], item["sha"]))
    return commits


def _checkpoint_metadata(
    root: Path,
    checkpoint_dir: str | os.PathLike[str] | None,
) -> tuple[str, dict[str, Any]]:
    """Read latest project checkpoint without creating/pruning storage."""
    if checkpoint_dir is None:
        if root != Path.cwd().resolve():
            return "", {}
        from .config import _project_checkpoint_dir
        directory = _project_checkpoint_dir()
    else:
        directory = Path(checkpoint_dir).expanduser()
    if not directory.exists() or not directory.is_dir():
        return "", {}

    from .checkpoint import CheckpointManager

    manager = CheckpointManager(
        directory,
        auto_interval=0,
        max_checkpoints=10,
        peer_retention_seconds=0,
        max_total_checkpoints=0,
    )
    checkpoint = manager.load_latest()
    if checkpoint is None:
        return "", {}
    metadata = checkpoint.metadata if isinstance(checkpoint.metadata, dict) else {}
    return checkpoint.checkpoint_id, dict(metadata)


def discover_repository_observation(
    path: str | os.PathLike[str] = ".",
    *,
    agent_id: str = "",
    session_id: str = "",
    task_hint: dict[str, Any] | None = None,
    default_branch: str | None = None,
    max_commits: int = _MAX_COMMITS,
    observed_at_ms: int | None = None,
    include_checkpoint: bool = True,
    checkpoint_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Observe durable Git/checkpoint facts without deciding what they mean.

    A clean default-branch checkout produces no changes/commits-ahead and stays
    a Work Graph null control. Checkpoint task/decision metadata is considered
    only when Git independently proves that work exists, preventing an old
    checkpoint from resurrecting a completed task in a clean repository.
    """
    root = _resolve_root(path)
    branch_name = _try_git(root, "symbolic-ref", "--quiet", "--short", "HEAD")
    head_sha = _try_git(root, "rev-parse", "--verify", "HEAD")
    default = _default_branch(root, default_branch)
    base = _base_ref(root, default)
    ahead, behind = _ahead_behind(root, base)
    git_dir = _git_dir(root)

    merge_in_progress = bool(git_dir and (git_dir / "MERGE_HEAD").exists())
    rebase_in_progress = bool(
        git_dir
        and ((git_dir / "rebase-merge").exists() or (git_dir / "rebase-apply").exists())
    )
    changes = _parse_status(root)
    commits = _branch_commits(root, base, ahead, int(max_commits))
    meaningful_git = bool(changes or ahead or merge_in_progress or rebase_in_progress)

    decisions: list[dict[str, Any]] = []
    effective_task_hint = dict(task_hint) if task_hint else None
    if include_checkpoint and meaningful_git:
        checkpoint_id, metadata = _checkpoint_metadata(root, checkpoint_dir)
        if checkpoint_id:
            source_ref = f"checkpoint:{checkpoint_id}"
            if effective_task_hint is None:
                task = str(metadata.get("task", "")).strip()
                if task:
                    remaining: list[str] = []
                    step = str(metadata.get("step", "")).strip()
                    if step:
                        remaining.append(step)
                    effective_task_hint = {
                        "task_id": f"checkpoint:{checkpoint_id}",
                        "title": task,
                        "trust": "observed",
                        "explicit_status": "unknown",
                        "remaining_work": remaining,
                        "source_kind": "checkpoint",
                        "source_ref": source_ref,
                    }
            raw_decisions = metadata.get("decisions", [])
            if isinstance(raw_decisions, list):
                for index, value in enumerate(raw_decisions[:20]):
                    text = str(value).strip()
                    if text:
                        decisions.append(
                            {
                                "decision_id": f"{checkpoint_id}:{index}",
                                "text": text,
                                "source_ref": source_ref,
                                "source_kind": "checkpoint",
                                "trust": "observed",
                            }
                        )

    return {
        "repo_id": _repository_id(root),
        "observed_at_ms": int(observed_at_ms if observed_at_ms is not None else time.time() * 1000),
        "repository_label": root.name,
        "agent_id": agent_id,
        "session_id": session_id,
        "task_hint": effective_task_hint,
        "branch": {
            "name": branch_name,
            "head_sha": head_sha,
            "base_ref": base,
            "default_branch": default,
            "ahead_by": ahead,
            "behind_by": behind,
            "merge_in_progress": merge_in_progress,
            "rebase_in_progress": rebase_in_progress,
            "detached": not bool(branch_name),
        },
        "changes": changes,
        "commits": commits,
        "verifications": [],
        "decisions": decisions,
        "claims": [],
        "leases": [],
        "model_executions": [],
    }


__all__ = ["RepositoryDiscoveryError", "discover_repository_observation"]
