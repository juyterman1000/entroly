"""Bounded, local-only Git history evidence for selected repository paths."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


def collect_git_history(
    root: Path,
    paths: Iterable[str],
    *,
    max_commits: int = 20,
    timeout_seconds: float = 5.0,
) -> dict[str, object]:
    """Return commit evidence without fetching, hooks, or working-tree writes."""
    selected = tuple(sorted(dict.fromkeys(str(path) for path in paths)))[:50]
    if not selected:
        return {
            "available": False,
            "commits": [],
            "diagnostic": "no-selected-paths",
            "remote_calls": 0,
        }
    limit = max(1, min(int(max_commits), 100))
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    command = [
        "git",
        "-c",
        "core.quotepath=false",
        "log",
        "--no-decorate",
        f"--max-count={limit}",
        "--format=%H%x1f%ct%x1f%s%x1e",
        "--",
        *selected,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(0.1, min(float(timeout_seconds), 30.0)),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": False,
            "commits": [],
            "diagnostic": type(exc).__name__,
            "remote_calls": 0,
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "commits": [],
            "diagnostic": "not-a-git-worktree-or-log-failed",
            "remote_calls": 0,
        }

    commits: list[dict[str, object]] = []
    for record in completed.stdout.split("\x1e"):
        clean = record.strip("\r\n")
        if not clean:
            continue
        fields = clean.split("\x1f", 2)
        if len(fields) != 3:
            continue
        commit, timestamp, subject = fields
        if len(commit) != 40 or not all(char in "0123456789abcdef" for char in commit):
            continue
        commits.append({
            "commit": commit,
            "committed_unix": int(timestamp) if timestamp.isdigit() else 0,
            "subject": subject.replace("\x1e", " ").replace("\x1f", " "),
            "trust": "untrusted-git-metadata",
        })
    return {
        "available": True,
        "paths": list(selected),
        "commits": commits,
        "diagnostic": None,
        "remote_calls": 0,
    }
