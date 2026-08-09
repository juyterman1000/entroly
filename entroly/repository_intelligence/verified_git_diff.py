"""Git-object-backed semantic architecture diffs without checkout mutation."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

from ..tree_sitter_support import language_for_path
from .architecture_diff import build_verified_architecture_diff
from .models import RepositoryIndex, RepositoryLimits
from .verified_architecture import build_verified_architecture

VERIFIED_GIT_ARCHITECTURE_DIFF_SCHEMA_VERSION = (
    "entroly.verified-git-architecture-diff.v1"
)
_MAX_GIT_OUTPUT = 64 * 1024 * 1024


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _git_env() -> dict[str, str]:
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {
            "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
            "COMSPEC", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
        }
    }
    allowed.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    })
    return allowed


def _startupinfo():
    if os.name != "nt":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = subprocess.SW_HIDE
    return info


def _run_git(
    root: Path,
    arguments: Sequence[str],
    *,
    timeout_seconds: float = 30.0,
    max_output: int = _MAX_GIT_OUTPUT,
) -> bytes:
    command = ["git", "--no-pager", *arguments]
    try:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=_git_env(),
            startupinfo=_startupinfo(),
        )
    except OSError as exc:
        raise ValueError("git executable is unavailable") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise ValueError("bounded git operation timed out") from None
    if len(stdout) > max_output or len(stderr) > 1024 * 1024:
        raise ValueError("bounded git output limit exceeded")
    if process.returncode != 0:
        raise ValueError(
            "git operation failed: "
            + stderr[:4096].decode("utf-8", errors="replace").strip()
        )
    return stdout


def _safe_ref(value: str) -> str:
    ref = value.strip()
    if (
        not ref
        or len(ref) > 512
        or ref.startswith("-")
        or any(ord(character) < 32 for character in ref)
    ):
        raise ValueError("git ref must be a bounded non-option string")
    return ref


def _safe_tree_path(value: str) -> str | None:
    path = value.replace("\\", "/")
    if (
        not path
        or path.startswith(("/", "//"))
        or (len(path) >= 2 and path[1] == ":")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        return None
    return path


def _resolve_repository(root: Path, ref: str) -> tuple[str, str]:
    top = _run_git(root, ["rev-parse", "--show-toplevel"], max_output=64 * 1024)
    top_path = Path(top.decode("utf-8", errors="surrogateescape").strip()).resolve()
    if top_path != root:
        raise ValueError("git-aware architecture requires the repository top-level root")
    commit = _run_git(
        root,
        ["rev-parse", "--verify", f"{_safe_ref(ref)}^{{commit}}"],
        max_output=64 * 1024,
    ).decode("ascii", errors="strict").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("git did not resolve a full commit identity")
    head = _run_git(
        root, ["rev-parse", "--verify", "HEAD^{commit}"], max_output=64 * 1024
    ).decode("ascii", errors="strict").strip()
    return commit, head


def _tree_entries(
    root: Path,
    commit: str,
    limits: RepositoryLimits,
) -> tuple[list[tuple[str, str, int]], list[str]]:
    output = _run_git(
        root,
        ["ls-tree", "-r", "-z", "--long", "--full-tree", commit],
    )
    selected: list[tuple[str, str, int]] = []
    diagnostics: list[str] = []
    total = 0
    for raw_entry in output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, kind, oid, raw_size = metadata.split(b" ", 3)
            path = raw_path.decode("utf-8", errors="surrogateescape")
            size = int(raw_size)
        except (ValueError, UnicodeError):
            diagnostics.append("malformed-git-tree-entry-omitted")
            continue
        safe = _safe_tree_path(path)
        if safe is None:
            diagnostics.append("unsafe-git-tree-path-omitted")
            continue
        if kind != b"blob" or mode not in {b"100644", b"100755"}:
            continue
        if language_for_path(safe) is None:
            continue
        if size > limits.max_file_bytes:
            diagnostics.append(f"oversized-baseline-file-omitted:{safe}")
            continue
        if len(selected) >= limits.max_files or total + size > limits.max_total_bytes:
            diagnostics.append("baseline-repository-limits-reached")
            break
        selected.append((safe, oid.decode("ascii", errors="strict"), size))
        total += size
    return selected, diagnostics


def _materialize_blobs(
    root: Path,
    destination: Path,
    entries: Sequence[tuple[str, str, int]],
    *,
    timeout_seconds: float = 60.0,
) -> None:
    process = subprocess.Popen(
        ["git", "--no-pager", "cat-file", "--batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        env=_git_env(),
        startupinfo=_startupinfo(),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        for path, oid, expected_size in entries:
            process.stdin.write(oid.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline(1024)
            parts = header.rstrip(b"\n").split(b" ")
            if len(parts) != 3 or parts[0].decode("ascii") != oid or parts[1] != b"blob":
                raise ValueError("git cat-file returned an invalid blob header")
            size = int(parts[2])
            if size != expected_size:
                raise ValueError("git tree and blob sizes disagree")
            remaining = size
            chunks: list[bytes] = []
            while remaining:
                chunk = process.stdout.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError("git cat-file ended before the blob completed")
                chunks.append(chunk)
                remaining -= len(chunk)
            if process.stdout.read(1) != b"\n":
                raise ValueError("git cat-file blob framing is invalid")
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"".join(chunks))
        process.stdin.close()
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise ValueError("git cat-file operation timed out") from None
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.returncode != 0:
            raise ValueError(
                "git cat-file failed: "
                + stderr[:4096].decode("utf-8", errors="replace")
            )
    except Exception:
        process.kill()
        process.wait()
        raise


def _name_status(root: Path, commit: str, *, limit: int) -> tuple[list[dict[str, str]], int]:
    output = _run_git(
        root,
        ["diff", "--name-status", "-z", "--find-renames", commit, "--"],
    )
    fields = output.split(b"\0")
    changes: list[dict[str, str]] = []
    position = 0
    total = 0
    while position < len(fields) and fields[position]:
        status = fields[position].decode("ascii", errors="replace")
        position += 1
        if position >= len(fields):
            break
        old_path = fields[position].decode("utf-8", errors="surrogateescape")
        position += 1
        item = {"status": status[:1], "path": old_path}
        if status.startswith(("R", "C")) and position < len(fields):
            new_path = fields[position].decode("utf-8", errors="surrogateescape")
            position += 1
            item = {"status": status[:1], "old_path": old_path, "path": new_path}
        total += 1
        if len(changes) < limit:
            changes.append(item)
    return changes, max(0, total - limit)


def build_verified_git_architecture_diff(
    root: Path,
    current_index: RepositoryIndex,
    *,
    current_index_digest: str,
    ref: str,
    limits: RepositoryLimits,
    build_index,
    max_changes: int = 10_000,
    max_components: int = 5_000,
    max_communities: int = 1_000,
    max_cycles: int = 1_000,
    max_dependency_edges: int = 100_000,
    max_hotspots: int = 100,
    max_routes: int = 100,
) -> dict[str, object]:
    """Compare a verified Git object graph with the current verified worktree."""
    root = root.expanduser().resolve(strict=True)
    selected_ref = _safe_ref(ref)
    commit, head = _resolve_repository(root, selected_ref)
    entries, materialization_diagnostics = _tree_entries(root, commit, limits)
    with tempfile.TemporaryDirectory(prefix="entroly-git-architecture-") as directory:
        baseline_root = Path(directory).resolve()
        _materialize_blobs(root, baseline_root, entries)
        baseline_index = build_index(baseline_root, limits=limits)
        portable = baseline_index.to_dict()
        portable["root"] = "."
        baseline_digest = "sha256:" + hashlib.sha256(_canonical(portable)).hexdigest()
        bounds = {
            "max_components": max_components,
            "max_communities": max_communities,
            "max_cycles": max_cycles,
            "max_dependency_edges": max_dependency_edges,
            "max_hotspots": max_hotspots,
            "max_routes": max_routes,
        }
        baseline_architecture = build_verified_architecture(
            baseline_root,
            baseline_index,
            index_digest=baseline_digest,
            **bounds,
        )
    current_architecture = build_verified_architecture(
        root,
        current_index,
        index_digest=current_index_digest,
        **bounds,
    )
    diff = build_verified_architecture_diff(
        baseline_architecture,
        current_architecture,
        limit=max(1, min(int(max_changes), 50_000)),
    )
    name_status, git_changes_omitted = _name_status(
        root, commit, limit=max(1, min(int(max_changes), 50_000))
    )
    payload: dict[str, object] = {
        "schema_version": VERIFIED_GIT_ARCHITECTURE_DIFF_SCHEMA_VERSION,
        "base_ref": selected_ref,
        "base_commit": commit,
        "head_commit": head,
        "current_basis": "verified-worktree-index-including-untracked-source-files",
        "baseline_materialization": {
            "source": "local-git-commit-objects",
            "checkout_mutated": False,
            "regular_source_blobs": len(entries),
            "diagnostics": sorted(materialization_diagnostics),
            "git_filters_or_worktree_conversion_applied": False,
        },
        "git_name_status": name_status,
        "architecture_diff": diff,
        "baseline_architecture_receipt": baseline_architecture["receipt"],
        "current_architecture_receipt": current_architecture["receipt"],
        "truncation": {"git_name_status_omitted": git_changes_omitted},
        "receipt": {
            "remote_calls": 0,
            "git_commands": ["rev-parse", "ls-tree", "cat-file", "diff"],
            "subprocess_shell": False,
            "commitment_scope": (
                "payload-excluding-command-generation-and-git-architecture-diff-sha256"
            ),
        },
    }
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["git_architecture_diff_sha256"] = hashlib.sha256(
        _canonical(payload)
    ).hexdigest()
    return payload


def verify_git_architecture_diff_commitment(payload: Mapping[str, object]) -> bool:
    try:
        candidate = copy.deepcopy(dict(payload))
        candidate.pop("command", None)
        candidate.pop("generation", None)
        if candidate.get("schema_version") != VERIFIED_GIT_ARCHITECTURE_DIFF_SCHEMA_VERSION:
            return False
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("git_architecture_diff_sha256"))
        return hashlib.sha256(_canonical(candidate)).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "VERIFIED_GIT_ARCHITECTURE_DIFF_SCHEMA_VERSION",
    "build_verified_git_architecture_diff",
    "verify_git_architecture_diff_commitment",
]
