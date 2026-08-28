"""Content fingerprints for passive Work Graph repository observations.

This is orchestration-only. It does not decide whether two observations are the
same work state; Rust owns that semantic decision. The helper merely supplies a
content identity when the current worktree bytes can be represented exactly.

Security and honesty rules:
- never follow symlinks or read special files;
- never read outside the repository root;
- bound every file read;
- staged/conflicted paths remain non-dedupeable via ``content_digest``, because
  one worktree digest cannot represent both index and worktree state -- but
  they are no longer left unbound: ``index_digest`` and ``worktree_digest``
  commit to each state separately, and a conflicted path additionally records
  its base/ours/theirs stage digests;
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


def _relative_regular_path(root: Path, repo_rel: str) -> Path | None:
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
    return candidate


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


def _git_blob_digest(root: Path, repo_rel: str, algorithm: str) -> str:
    candidate = _relative_regular_path(root, repo_rel)
    if candidate is None:
        return ""
    try:
        before = os.lstat(candidate)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        # O_NOFOLLOW prevents the critical Unix symlink swap. Platforms without
        # it still get the pre/open/post identity checks before any bytes are read.
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
        if not _same_file(before, opened) or not _same_file(opened, path_after_open):
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


_CONFLICT_STAGE_NAMES = {"1": "base", "2": "ours", "3": "theirs"}


def _index_entries(root: Path, repo_rel: str) -> dict[str, str]:
    """Blob identities Git already holds in its index, keyed by stage name.

    Read from the index rather than recomputed from disk. For a staged path the
    index is the only place the staged bytes exist -- the worktree may have
    moved on since ``git add`` -- and for a conflicted path the three stages
    have no on-disk representation at all: what is in the worktree is merged
    text with markers, which is not any of them.

    ``ls-files -s`` reports ``<mode> <sha> <stage>\\t<path>``. Stage 0 is an
    ordinary index entry; 1/2/3 are base/ours/theirs of an unresolved merge.
    """
    if not repo_rel or repo_rel.startswith("sensitive:"):
        return {}
    raw = _git_text(root, "ls-files", "-s", "-z", "--", repo_rel)
    entries: dict[str, str] = {}
    for record in raw.split("\0"):
        if not record or "\t" not in record:
            continue
        meta, _, path = record.partition("\t")
        if path.replace("\\", "/") != repo_rel:
            continue
        fields = meta.split()
        if len(fields) < 3:
            continue
        _mode, sha, stage = fields[0], fields[1], fields[2]
        if not sha or set(sha) <= {"0"}:
            continue
        # Same ``git-blob:`` prefix the worktree digests carry. These are the
        # same kind of thing -- a Git blob identity -- and a caller comparing
        # index against worktree should not have to strip one side first.
        entries[_CONFLICT_STAGE_NAMES.get(stage, "index")] = f"git-blob:{sha}"
    return entries


def enrich_worktree_content_digests(
    repo_path: str | os.PathLike[str], observation: dict[str, Any]
) -> dict[str, Any]:
    """Add exact, bounded worktree blob identities where safely available.

    The input object is updated in place to avoid copying a potentially large
    bounded observation. Any uncertainty leaves ``content_digest`` empty, which
    merely disables Rust passive-snapshot deduplication for that observation.
    """

    root = _root(repo_path)
    changes = observation.get("changes")
    if not isinstance(changes, list):
        return observation
    algorithm = _object_hash_algorithm(root)

    for raw in changes:
        if not isinstance(raw, dict):
            continue
        raw["content_digest"] = ""
        raw["index_digest"] = ""
        raw["worktree_digest"] = ""

        # A withheld path carries an opaque id, not a filename, so there is
        # nothing to hash and nothing that should be hashed: a content digest
        # of a short credential file is a confirmation oracle for a guessed
        # value.
        if raw.get("redacted"):
            continue

        repo_rel = str(raw.get("path", ""))
        kind = str(raw.get("kind", ""))
        staged = bool(raw.get("staged"))
        conflicted = bool(raw.get("conflicted"))

        if kind == "deleted":
            raw["worktree_digest"] = "worktree:deleted"
            if not (staged or conflicted):
                raw["content_digest"] = "worktree:deleted"
        elif algorithm is not None:
            worktree = _git_blob_digest(root, repo_rel, algorithm)
            raw["worktree_digest"] = worktree
            # content_digest keeps its original meaning -- a single identity
            # that stands for the whole path -- so it stays empty exactly when
            # one digest cannot honestly represent the path's state. Rust's
            # dedup semantics are unchanged by this commit.
            if not (staged or conflicted):
                raw["content_digest"] = worktree

        if staged or conflicted:
            entries = _index_entries(root, repo_rel)
            if "index" in entries:
                raw["index_digest"] = entries["index"]
            stages = {
                name: sha for name, sha in entries.items() if name != "index"
            }
            if stages:
                raw["conflict_stage_digests"] = stages
    return observation


__all__ = ["enrich_worktree_content_digests"]
