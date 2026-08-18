"""Content fingerprints for passive Work Graph repository observations.

This is orchestration-only. It does not decide whether two observations are the
same work state; Rust owns that semantic decision. The helper merely supplies a
content identity when the current worktree bytes can be represented exactly.

Staged/conflicted paths deliberately remain without a digest because a single
worktree hash cannot represent both index and worktree state. Hash failures also
fail closed by leaving the digest empty, which makes the snapshot ineligible for
future Rust passive-snapshot deduplication rather than risking lost work.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

_MAX_HASH_OUTPUT_BYTES = 4 * 1024 * 1024


def _root(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "submodule.recurse=false",
            "-C",
            str(candidate),
            "rev-parse",
            "--show-toplevel",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
        env={
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "LC_ALL": "C",
            "LANG": "C",
        },
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError(f"not a Git worktree: {candidate}")
    return Path(result.stdout.strip()).resolve()


def enrich_worktree_content_digests(
    repo_path: str | os.PathLike[str], observation: dict[str, Any]
) -> dict[str, Any]:
    """Return *observation* with exact worktree digests where safely available.

    The input object is updated in place to avoid copying potentially large
    bounded observations. Digests use Git's canonical blob hashing with filters
    disabled, so Python and Node can produce the same identity for the same
    bytes. Unrepresentable changes retain an empty ``content_digest``.
    """

    root = _root(repo_path)
    changes = observation.get("changes")
    if not isinstance(changes, list):
        return observation

    pending: list[tuple[dict[str, Any], str]] = []
    for raw in changes:
        if not isinstance(raw, dict):
            continue
        raw.setdefault("content_digest", "")
        if raw.get("staged") or raw.get("conflicted"):
            continue
        kind = str(raw.get("kind", ""))
        if kind == "deleted":
            raw["content_digest"] = "worktree:deleted"
            continue
        repo_rel = str(raw.get("path", ""))
        # --stdin-paths is line delimited. Rather than invent escaping rules,
        # leave newline-bearing paths non-dedupeable.
        if not repo_rel or "\n" in repo_rel or "\r" in repo_rel or "\x00" in repo_rel:
            continue
        pending.append((raw, repo_rel))

    if not pending:
        return observation

    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "LC_ALL": "C",
        "LANG": "C",
    }
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                "-c",
                "submodule.recurse=false",
                "-C",
                str(root),
                "hash-object",
                "--no-filters",
                "--stdin-paths",
            ],
            input="".join(f"{repo_rel}\n" for _change, repo_rel in pending),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return observation

    if result.returncode != 0:
        return observation
    if len(result.stdout.encode("utf-8", "replace")) > _MAX_HASH_OUTPUT_BYTES:
        return observation
    hashes = result.stdout.splitlines()
    if len(hashes) != len(pending):
        return observation
    lowered = [digest.strip().lower() for digest in hashes]
    if any(
        len(digest) not in {40, 64}
        or any(ch not in "0123456789abcdef" for ch in digest)
        for digest in lowered
    ):
        return observation

    for (change, _repo_rel), digest in zip(pending, lowered, strict=True):
        change["content_digest"] = f"git-blob:{digest}"
    return observation


__all__ = ["enrich_worktree_content_digests"]
