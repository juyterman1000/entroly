"""Content fingerprints for passive Work Graph repository observations.

This is orchestration-only. It does not decide whether two observations are the
same work state; Rust owns that semantic decision. The helper merely supplies a
content identity when the current worktree bytes can be represented exactly.

Security and honesty rules:
- never follow symlinks or read special files;
- never read outside the repository root;
- bound every file read and the aggregate bytes of one observation;
- staged/conflicted paths remain non-dedupeable because one worktree digest
  cannot represent both index and worktree state;
- detect path/file races before and after reading and fail closed;
- preserve Git's SHA-1/SHA-256 blob identity exactly.
"""
from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

_MAX_HASH_FILE_BYTES = 64 * 1024 * 1024
_MAX_HASH_TOTAL_BYTES = 128 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024


def _git_env() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "LC_ALL": "C",
        "LANG": "C",
    }


def _git_text(root: Path, *args: str) -> str:
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
                *args,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            env=_git_env(),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _root(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    output = _git_text(candidate, "rev-parse", "--show-toplevel")
    if not output:
        raise ValueError(f"not a Git worktree: {candidate}")
    return Path(output).resolve()


def _object_hash_algorithm(root: Path) -> str | None:
    value = _git_text(root, "rev-parse", "--show-object-format").lower()
    return value if value in {"sha1", "sha256"} else None


def _relative_regular_path(root: Path, repo_rel: str) -> tuple[Path, os.stat_result] | None:
    value = repo_rel.replace("\\", "/")
    if not value or value.startswith("/") or "\x00" in value or "\n" in value or "\r" in value:
        return None
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    candidate = root.joinpath(*parts)
    try:
        before = os.lstat(candidate)
    except OSError:
        return None
    if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_HASH_FILE_BYTES:
        return None
    return candidate, before


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(left.st_mode)
        and stat.S_ISREG(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and getattr(left, "st_mtime_ns", None) == getattr(right, "st_mtime_ns", None)
        and getattr(left, "st_ctime_ns", None) == getattr(right, "st_ctime_ns", None)
    )


def _git_blob_digest(candidate: Path, expected: os.stat_result, algorithm: str) -> str:
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        # O_NOFOLLOW prevents the critical Unix symlink swap. Platforms without
        # it still get pre/open/post identity checks before any bytes are read.
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(candidate, flags)
    except OSError:
        return ""

    try:
        opened = os.fstat(descriptor)
        try:
            path_after_open = os.lstat(candidate)
        except OSError:
            return ""
        if not _same_file(expected, opened) or not _same_file(opened, path_after_open):
            return ""
        if opened.st_size > _MAX_HASH_FILE_BYTES:
            return ""

        try:
            hasher = hashlib.new(algorithm)
        except (ValueError, TypeError):
            return ""
        hasher.update(f"blob {opened.st_size}\0".encode("ascii"))

        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(_HASH_CHUNK_BYTES, remaining))
            if not chunk:
                return ""
            hasher.update(chunk)
            remaining -= len(chunk)

        after = os.fstat(descriptor)
        try:
            path_after_read = os.lstat(candidate)
        except OSError:
            return ""
        if not _same_file(opened, after) or not _same_file(after, path_after_read):
            return ""
        return f"git-blob:{hasher.hexdigest()}"
    except OSError:
        return ""
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def enrich_worktree_content_digests(
    repo_path: str | os.PathLike[str], observation: dict[str, Any]
) -> dict[str, Any]:
    """Add exact, bounded worktree blob identities where safely available.

    The input object is updated in place to avoid copying a potentially large
    bounded observation. Any uncertainty leaves ``content_digest`` empty, which
    merely disables Rust passive-snapshot deduplication for that observation.

    The aggregate budget is all-or-nothing for non-deleted paths. This prevents
    a partial subset from masquerading as content-complete semantic identity.
    """

    root = _root(repo_path)
    changes = observation.get("changes")
    if not isinstance(changes, list):
        return observation
    algorithm = _object_hash_algorithm(root)

    pending: list[tuple[dict[str, Any], Path, os.stat_result]] = []
    total_bytes = 0
    for raw in changes:
        if not isinstance(raw, dict):
            continue
        raw["content_digest"] = ""
        if raw.get("staged") or raw.get("conflicted"):
            continue
        kind = str(raw.get("kind", ""))
        if kind == "deleted":
            raw["content_digest"] = "worktree:deleted"
            continue
        if algorithm is None:
            continue
        safe = _relative_regular_path(root, str(raw.get("path", "")))
        if safe is None:
            continue
        candidate, metadata = safe
        total_bytes += metadata.st_size
        if total_bytes > _MAX_HASH_TOTAL_BYTES:
            # Keep every non-deleted digest empty; Rust will fail closed and
            # retain the observation as distinct audit history.
            for pending_raw, _candidate, _metadata in pending:
                pending_raw["content_digest"] = ""
            return observation
        pending.append((raw, candidate, metadata))

    for raw, candidate, metadata in pending:
        raw["content_digest"] = _git_blob_digest(candidate, metadata, algorithm)
    return observation


__all__ = ["enrich_worktree_content_digests"]
