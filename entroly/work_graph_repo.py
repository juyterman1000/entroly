"""Conservative Git observation adapter for Entroly AI Work Graph.

This module is orchestration-only. It observes durable repository facts and
normalizes them into the schema owned by ``entroly-engine``. It intentionally
never guesses task intent from filenames, commit prose, or branch names.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class RepositoryDiscoveryError(RuntimeError):
    """Raised when a repository cannot be observed safely."""


def _run_git(
    cwd: Path,
    *args: str,
    timeout: float = 5.0,
    check: bool = True,
) -> str:
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C", "LANG": "C"})
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
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


def _default_branch(root: Path, override: str | None) -> str:
    if override:
        return override.removeprefix("origin/")
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
    for candidate in (f"origin/{default_branch}", default_branch):
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
    raw = _run_git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    tokens = raw.split("\0")
    changes: list[dict[str, Any]] = []
    index = 0
    while index < len(tokens):
        entry = tokens[index]
        index += 1
        if not entry or len(entry) < 3:
            continue
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
    changes.sort(key=lambda item: (item["path"], item["old_path"], item["kind"]))
    return changes


def _branch_commits(root: Path, base_ref: str, ahead_by: int, max_commits: int) -> list[dict[str, Any]]:
    if ahead_by <= 0 or not base_ref or max_commits <= 0:
        return []
    limit = min(int(max_commits), 100)
    output = _run_git(
        root,
        "log",
        "--no-decorate",
        f"-n{limit}",
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
            }
        )
    commits.sort(key=lambda item: (item["timestamp_ms"], item["sha"]))
    return commits


def discover_repository_observation(
    path: str | os.PathLike[str] = ".",
    *,
    agent_id: str = "",
    session_id: str = "",
    task_hint: dict[str, Any] | None = None,
    default_branch: str | None = None,
    max_commits: int = 20,
    observed_at_ms: int | None = None,
) -> dict[str, Any]:
    """Observe durable Git facts without inferring task intent.

    A clean default-branch checkout produces no changes/commits-ahead and therefore
    remains a Work Graph null control: Rust will not invent an unfinished task.
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

    return {
        "repo_id": _repository_id(root),
        "observed_at_ms": int(observed_at_ms if observed_at_ms is not None else time.time() * 1000),
        "repository_label": root.name,
        "agent_id": agent_id,
        "session_id": session_id,
        "task_hint": task_hint,
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
        "changes": _parse_status(root),
        "commits": _branch_commits(root, base, ahead, max_commits),
        "verifications": [],
        "decisions": [],
        "claims": [],
        "leases": [],
        "model_executions": [],
    }


__all__ = ["RepositoryDiscoveryError", "discover_repository_observation"]
