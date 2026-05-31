"""Filesystem containment helpers for repository-scoped reads."""
from __future__ import annotations

from pathlib import Path


def resolve_file_within(root: str | Path, candidate: str | Path) -> Path | None:
    """Return a resolved file path only when it remains inside ``root``."""
    root_path = Path(root).resolve()
    candidate_path = Path(candidate)
    if not candidate_path.is_absolute():
        candidate_path = root_path / candidate_path
    try:
        resolved = candidate_path.resolve(strict=True)
        resolved.relative_to(root_path)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None
