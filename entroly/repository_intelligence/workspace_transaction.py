"""Recoverable multi-file workspace transactions for verified refactors.

A filesystem cannot provide true multi-file atomic rename semantics across an
arbitrary edit set.  Entroly therefore uses a bounded journaled transaction:
recheck every preimage, stage new bytes beside each target, keep durable backups,
apply mutations deterministically, and roll back in reverse order on failure.

The critical invariant is recovery preservation: if any rollback step fails,
remaining backups/stages are *not* deleted.  The raised error carries a bounded
machine-readable report naming workspace-relative recovery artifacts.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

WORKSPACE_TRANSACTION_SCHEMA_VERSION = "entroly.workspace-transaction.v1"
_MAX_MUTATIONS = 10_000
_MAX_RECOVERY_ARTIFACTS = 10_000


@dataclass(frozen=True)
class WorkspaceTransactionReport:
    mutation_count: int
    rollback_performed: bool
    rollback_complete: bool
    completed_mutations: tuple[str, ...]
    recovery_artifacts: tuple[str, ...]
    rollback_errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORKSPACE_TRANSACTION_SCHEMA_VERSION,
            "mutation_count": self.mutation_count,
            "rollback_performed": self.rollback_performed,
            "rollback_complete": self.rollback_complete,
            "completed_mutations": list(self.completed_mutations),
            "recovery_artifacts": list(self.recovery_artifacts),
            "rollback_errors": list(self.rollback_errors),
        }


class WorkspaceTransactionError(ValueError):
    """Raised when commit fails; ``report`` describes rollback/recovery state."""

    def __init__(self, message: str, report: WorkspaceTransactionReport) -> None:
        self.report = report
        super().__init__(message)


def _safe_target(root: Path, relative: str, *, must_exist: bool) -> Path:
    raw = str(relative).replace("\\", "/")
    if (
        not raw
        or "\x00" in raw
        or raw.startswith(("/", "//"))
        or (len(raw) >= 2 and raw[1] == ":")
        or any(part == ".." for part in raw.split("/"))
    ):
        raise ValueError("workspace transaction path is unsafe")
    candidate = (root / raw).resolve(strict=must_exist)
    candidate.relative_to(root)
    if not must_exist:
        parent = candidate.parent.resolve(strict=True)
        parent.relative_to(root)
        candidate = parent / candidate.name
    return candidate


def _write_durable_temp(
    directory: Path,
    *,
    prefix: str,
    content: bytes,
    mode: int,
) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=prefix, dir=directory)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, mode)
        return path
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _replace(source: Path, target: Path) -> None:
    source.replace(target)


def _unlink(path: Path) -> None:
    path.unlink()


def _read_exact(path: Path) -> bytes:
    return path.read_bytes()


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Temporary artifacts are created below workspace directories and should
        # always be relative. Keep the report non-secret even under corruption.
        return path.name


def apply_workspace_transaction(
    root: Path,
    *,
    replacements: Mapping[str, bytes] = {},
    creations: Mapping[str, bytes] = {},
    deletions: Mapping[str, bytes] = {},
    expected_originals: Mapping[str, bytes] = {},
    creation_modes: Mapping[str, int] = {},
) -> WorkspaceTransactionReport:
    """Apply a deterministic recoverable transaction.

    ``expected_originals`` must contain every replacement/deletion path and is
    rechecked immediately before each destructive mutation.  Creation targets
    must remain absent.  On incomplete rollback the function raises
    :class:`WorkspaceTransactionError` and intentionally leaves recovery files.
    """
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    replacement_map = {str(k): bytes(v) for k, v in replacements.items()}
    creation_map = {str(k): bytes(v) for k, v in creations.items()}
    deletion_map = {str(k): bytes(v) for k, v in deletions.items()}
    mutation_count = len(replacement_map) + len(creation_map) + len(deletion_map)
    if mutation_count <= 0 or mutation_count > _MAX_MUTATIONS:
        raise ValueError(f"workspace transaction requires 1 to {_MAX_MUTATIONS} mutations")
    overlap = (
        set(replacement_map) & set(creation_map)
        | set(replacement_map) & set(deletion_map)
        | set(creation_map) & set(deletion_map)
    )
    if overlap:
        raise ValueError("workspace transaction path appears in multiple mutation classes")
    required = set(replacement_map) | set(deletion_map)
    if set(expected_originals) != required:
        raise ValueError("expected_originals must exactly cover replacements and deletions")

    targets: dict[str, Path] = {}
    originals: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    for path in sorted(required):
        target = _safe_target(root, path, must_exist=True)
        if not target.is_file() or target.is_symlink():
            raise ValueError("workspace transaction existing target must be a regular file")
        current = _read_exact(target)
        expected = bytes(expected_originals[path])
        if current != expected:
            raise ValueError(f"workspace transaction preimage changed: {path}")
        targets[path] = target
        originals[path] = current
        modes[path] = target.stat().st_mode
    for path in sorted(creation_map):
        target = _safe_target(root, path, must_exist=False)
        if target.exists() or target.is_symlink() or not target.parent.is_dir():
            raise ValueError(f"workspace transaction creation target unavailable: {path}")
        targets[path] = target
        modes[path] = int(creation_modes.get(path, 0o600))

    stages: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    completed: list[tuple[str, str]] = []
    rollback_errors: list[str] = []
    rollback_performed = False
    commit_error: BaseException | None = None

    try:
        for path in sorted(replacement_map):
            target = targets[path]
            stages[path] = _write_durable_temp(
                target.parent,
                prefix=f".{target.name}.entroly-stage.",
                content=replacement_map[path],
                mode=modes[path],
            )
            backups[path] = _write_durable_temp(
                target.parent,
                prefix=f".{target.name}.entroly-backup.",
                content=originals[path],
                mode=modes[path],
            )
        for path in sorted(deletion_map):
            target = targets[path]
            backups[path] = _write_durable_temp(
                target.parent,
                prefix=f".{target.name}.entroly-backup.",
                content=originals[path],
                mode=modes[path],
            )
        for path in sorted(creation_map):
            target = targets[path]
            stages[path] = _write_durable_temp(
                target.parent,
                prefix=f".{target.name}.entroly-stage.",
                content=creation_map[path],
                mode=modes[path],
            )

        for path in sorted(replacement_map):
            target = targets[path]
            if _read_exact(target) != originals[path]:
                raise OSError(f"preimage changed during commit: {path}")
            _replace(stages[path], target)
            completed.append(("replace", path))
        for path in sorted(creation_map):
            target = targets[path]
            if target.exists() or target.is_symlink():
                raise OSError(f"creation target appeared during commit: {path}")
            _replace(stages[path], target)
            completed.append(("create", path))
        for path in sorted(deletion_map):
            target = targets[path]
            if _read_exact(target) != originals[path]:
                raise OSError(f"deletion preimage changed during commit: {path}")
            _unlink(target)
            completed.append(("delete", path))
    except (OSError, RuntimeError) as exc:
        commit_error = exc
        rollback_performed = bool(completed)
        for operation, path in reversed(completed):
            target = targets[path]
            try:
                if operation in {"replace", "delete"}:
                    backup = backups[path]
                    if not backup.exists():
                        raise OSError("backup missing")
                    _replace(backup, target)
                elif operation == "create" and target.exists():
                    _unlink(target)
            except (OSError, RuntimeError, ValueError) as rollback_exc:
                rollback_errors.append(
                    f"{operation}:{path}:{type(rollback_exc).__name__}"
                )

    rollback_complete = not rollback_errors
    recovery_paths: list[Path] = []
    if commit_error is not None and not rollback_complete:
        # Preserve every remaining artifact. Even a stage can be useful for
        # comparing intended postimage vs. current workspace state.
        recovery_paths.extend(
            path for path in (*backups.values(), *stages.values()) if path.exists()
        )
    else:
        for temporary in (*backups.values(), *stages.values()):
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                # Cleanup failure after a successful commit does not invalidate
                # workspace contents; retain/report the artifact for janitorial
                # cleanup rather than claiming it vanished.
                if temporary.exists():
                    recovery_paths.append(temporary)

    report = WorkspaceTransactionReport(
        mutation_count=mutation_count,
        rollback_performed=rollback_performed,
        rollback_complete=rollback_complete,
        completed_mutations=tuple(
            f"{operation}:{path}" for operation, path in completed
        ),
        recovery_artifacts=tuple(
            sorted({_rel(root, path) for path in recovery_paths})[:_MAX_RECOVERY_ARTIFACTS]
        ),
        rollback_errors=tuple(rollback_errors[:_MAX_RECOVERY_ARTIFACTS]),
    )
    if commit_error is not None:
        if rollback_complete:
            raise WorkspaceTransactionError(
                "workspace transaction failed; rollback completed",
                report,
            ) from commit_error
        raise WorkspaceTransactionError(
            "workspace transaction failed; rollback incomplete; recovery artifacts preserved",
            report,
        ) from commit_error
    return report


__all__ = [
    "WORKSPACE_TRANSACTION_SCHEMA_VERSION",
    "WorkspaceTransactionError",
    "WorkspaceTransactionReport",
    "apply_workspace_transaction",
]
