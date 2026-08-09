"""Deterministic bounded retention for repository-intelligence caches.

Content-addressed entries remain immutable; retention only removes old complete
files after successful builds. Cleanup is deliberately amortized (once per
index build), fail-open, symlink-safe, and race-tolerant so cache maintenance can
never make repository intelligence unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

CACHE_RETENTION_SCHEMA_VERSION = "entroly.cache-retention.v1"


@dataclass(frozen=True)
class CacheRetentionReport:
    scanned_files: int
    scanned_bytes: int
    removed_files: int
    removed_bytes: int
    remaining_files: int
    remaining_bytes: int
    errors: int
    bounded: bool

    def diagnostic(self, label: str) -> str:
        return (
            f"{label}-retention "
            f"files={self.remaining_files} bytes={self.remaining_bytes} "
            f"removed_files={self.removed_files} "
            f"removed_bytes={self.removed_bytes} errors={self.errors} "
            f"bounded={str(self.bounded).lower()}"
        )


@dataclass(frozen=True)
class _Candidate:
    path: Path
    size: int
    mtime_ns: int
    relative: str


def _safe_json_candidates(
    root: Path,
    *,
    excluded_top_level: frozenset[str],
) -> tuple[list[_Candidate], int]:
    candidates: list[_Candidate] = []
    errors = 0
    try:
        iterator = root.rglob("*.json")
    except OSError:
        return [], 1
    try:
        for candidate in iterator:
            try:
                relative_path = candidate.relative_to(root)
                if not relative_path.parts:
                    continue
                if relative_path.parts[0] in excluded_top_level:
                    continue
                # Never follow or delete symlinked cache artifacts. The cache
                # writers themselves create ordinary files atomically.
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                stat = candidate.stat()
                relative = relative_path.as_posix()
                candidates.append(_Candidate(
                    path=candidate,
                    size=max(0, int(stat.st_size)),
                    mtime_ns=max(0, int(stat.st_mtime_ns)),
                    relative=relative,
                ))
            except (OSError, RuntimeError, ValueError):
                errors += 1
    except OSError:
        errors += 1
    return candidates, errors


def prune_cache_tree(
    directory: Path,
    *,
    max_total_bytes: int,
    max_files: int,
    protected: Iterable[Path] = (),
    excluded_top_level: Iterable[str] = (),
) -> CacheRetentionReport:
    """Prune oldest immutable JSON entries until both hard bounds are met.

    Ordering is deterministic: oldest mtime first, then workspace-relative path.
    ``protected`` entries are never deleted during this pass (typically the
    snapshot just produced for the current repository state).
    """
    root = directory.expanduser().resolve()
    byte_limit = max(0, int(max_total_bytes))
    file_limit = max(0, int(max_files))
    excluded = frozenset(str(item) for item in excluded_top_level)
    protected_paths: set[Path] = set()
    for item in protected:
        try:
            value = item.expanduser().resolve(strict=False)
            value.relative_to(root)
            protected_paths.add(value)
        except (OSError, RuntimeError, ValueError):
            continue

    if not root.exists():
        return CacheRetentionReport(0, 0, 0, 0, 0, 0, 0, True)
    candidates, errors = _safe_json_candidates(
        root,
        excluded_top_level=excluded,
    )
    scanned_files = len(candidates)
    scanned_bytes = sum(item.size for item in candidates)
    remaining_files = scanned_files
    remaining_bytes = scanned_bytes
    removed_files = 0
    removed_bytes = 0

    ordered = sorted(candidates, key=lambda item: (
        item.mtime_ns,
        item.relative,
    ))
    for candidate in ordered:
        if remaining_files <= file_limit and remaining_bytes <= byte_limit:
            break
        try:
            resolved = candidate.path.resolve(strict=True)
            resolved.relative_to(root)
            if resolved in protected_paths or candidate.path.is_symlink():
                continue
            current_size = candidate.path.stat().st_size
            candidate.path.unlink()
        except (FileNotFoundError,):
            # Another process won the race; count it as gone using the size we
            # observed during enumeration so our local accounting converges.
            current_size = candidate.size
        except (OSError, RuntimeError, ValueError):
            errors += 1
            continue
        removed_files += 1
        removed_bytes += max(0, int(current_size))
        remaining_files = max(0, remaining_files - 1)
        remaining_bytes = max(0, remaining_bytes - max(0, int(current_size)))

    # Empty two-level hash directories are harmless, but remove them best-effort
    # to keep cache trees inspectable. Never remove the root itself.
    try:
        directories = sorted(
            (path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()),
            key=lambda path: (-len(path.parts), path.as_posix()),
        )
        for path in directories:
            try:
                path.rmdir()
            except OSError:
                pass
    except OSError:
        errors += 1

    bounded = remaining_files <= file_limit and remaining_bytes <= byte_limit
    return CacheRetentionReport(
        scanned_files=scanned_files,
        scanned_bytes=scanned_bytes,
        removed_files=removed_files,
        removed_bytes=removed_bytes,
        remaining_files=remaining_files,
        remaining_bytes=remaining_bytes,
        errors=errors,
        bounded=bounded,
    )


__all__ = [
    "CACHE_RETENTION_SCHEMA_VERSION",
    "CacheRetentionReport",
    "prune_cache_tree",
]
