"""Filesystem containment helpers for repository-scoped reads."""
from __future__ import annotations

from pathlib import Path


def resolve_file_within(root: str | Path, candidate: str | Path) -> Path | None:
    """Return a resolved file path only when it remains inside ``root``."""
    root_path = Path(root).resolve()
    return resolve_file_within_resolved(root_path, candidate)


def resolve_file_within_resolved(root: Path, candidate: str | Path) -> Path | None:
    """Resolve a file against an already-resolved root for recursive scanners."""
    root_path = root
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = root_path / candidate_path
    try:
        resolved = candidate_path.resolve(strict=True)
        resolved.relative_to(root_path)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def resolve_dir_within(root: str | Path, candidate: str | Path = ".") -> Path | None:
    """Return a resolved directory path only when it remains inside ``root``."""
    try:
        root_path = Path(root).resolve(strict=True)
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = root_path / candidate_path
        resolved = candidate_path.resolve(strict=True)
        resolved.relative_to(root_path)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_dir() else None


def resolve_output_within(root: str | Path, candidate: str | Path) -> Path | None:
    """Return an output file path only when it resolves inside ``root``.

    The candidate is resolved whether or not it exists. Checking the parent and
    then returning the unresolved candidate is not enough: ``Path.exists``
    follows symlinks, so a link inside ``root`` whose target is missing reports
    ``False`` and skips the containment check, while a write through it still
    follows the link. Measured against a temporary tree, a dangling
    ``root/link -> outside/target`` was returned as safe and the write landed in
    ``outside``. A live symlink was already blocked, so only the dangling case
    escaped -- which is the one an attacker plants, since the target not
    existing yet is the point.
    """
    try:
        root_path = Path(root).resolve(strict=True)
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = root_path / candidate_path
        parent = candidate_path.parent.resolve(strict=True)
        parent.relative_to(root_path)
        # Non-strict: resolves symlinks and ``..`` as far as the path exists,
        # which is what a write will follow, and tolerates a target that has
        # not been created yet.
        resolved = candidate_path.resolve()
        resolved.relative_to(root_path)
    except (OSError, RuntimeError, ValueError):
        return None
    if candidate_path.exists() and not resolved.is_file():
        return None
    return resolved
