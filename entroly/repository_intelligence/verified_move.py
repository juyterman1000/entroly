"""Verified file-move public surface using the shared recoverable transaction.

Move planning remains byte-identical in :mod:`verified_move_impl`. This facade
replaces only the filesystem apply path so source deletion, target creation, and
import rewrites share the same rollback/recovery guarantees as other refactors.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from .models import RepositoryIndex
from .verified_move_impl import _safe_path, build_verified_file_move_plan
from .verified_refactor_impl import (
    REFACTOR_APPLY_SCHEMA_VERSION,
    _syntax_status,
    _verified_source,
    verify_refactor_plan_commitment,
)
from .workspace_transaction import apply_workspace_transaction


def apply_verified_file_move_plan(
    root: Path,
    index: RepositoryIndex,
    plan: Mapping[str, object],
    *,
    index_digest: str,
    expected_plan_sha256: str,
    acknowledge_incomplete: bool = False,
) -> dict[str, object]:
    """Apply an exact module-move plan through a recoverable transaction."""
    root = root.expanduser().resolve(strict=True)
    candidate = copy.deepcopy(dict(plan))
    if not verify_refactor_plan_commitment(candidate):
        raise ValueError("refactor plan commitment is invalid")
    receipt = candidate.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("expected_plan_sha256 does not match the plan")
    if candidate.get("index_digest") != index_digest:
        raise ValueError("refactor plan index is stale")
    if candidate.get("operation") != "file-move" or candidate.get("safe_to_apply") is not True:
        raise ValueError("file-move plan has blockers or is not applicable")
    risk = candidate.get("risk")
    if not isinstance(risk, dict) or (
        risk.get("requires_incomplete_acknowledgement") and not acknowledge_incomplete
    ):
        raise ValueError("refactor completeness is unproven; explicit acknowledgement required")
    source = _safe_path(str(candidate.get("source_path", "")))
    target = _safe_path(str(candidate.get("target_path", "")))
    if source is None or target is None or source not in index.files:
        raise ValueError("file-move paths are invalid")
    source_file = (root / source).resolve(strict=True)
    source_file.relative_to(root)
    target_file = (root / target).resolve(strict=False)
    target_file.relative_to(root)
    if target_file.exists() or not target_file.parent.is_dir():
        raise ValueError("file-move target is no longer available")
    source_mode = source_file.stat().st_mode
    source_raw, status = _verified_source(root, index, source)
    if source_raw is None or hashlib.sha256(source_raw).hexdigest() != candidate.get(
        "source_sha256"
    ):
        raise ValueError(f"file-move source preimage failed: {status}")

    raw_changes = candidate.get("changes")
    if not isinstance(raw_changes, list):
        raise ValueError("file-move changes must be an array")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in raw_changes:
        if not isinstance(item, dict) or str(item.get("path", "")) not in index.files:
            raise ValueError("invalid file-move change")
        grouped[str(item["path"])].append(item)

    originals: dict[str, bytes] = {}
    updated: dict[str, bytes] = {}
    syntax: dict[str, str] = {}
    for path, changes in sorted(grouped.items()):
        raw, status = _verified_source(root, index, path)
        if raw is None:
            raise ValueError(f"file-move preimage failed: {status}")
        originals[path] = raw
        result = raw
        cursor = len(raw)
        for item in sorted(
            changes,
            key=lambda value: int(value["start_byte"]),
            reverse=True,
        ):
            start = int(item["start_byte"])
            end = int(item["end_byte"])
            old = str(item["old_identifier"]).encode("utf-8")
            new = str(item["new_identifier"]).encode("utf-8")
            if not (0 <= start < end <= cursor) or raw[start:end] != old:
                raise ValueError("file-move preimage range changed or overlaps")
            if hashlib.sha256(raw[start:end]).hexdigest() != item.get("evidence_sha256"):
                raise ValueError("file-move preimage evidence hash mismatch")
            result = result[:start] + new + result[end:]
            cursor = start
        syntax[path] = _syntax_status(path, result)
        if syntax[path].startswith("invalid-"):
            raise ValueError(f"file-move staged syntax failed for {path}")
        updated[path] = result

    moved_content = updated.pop(source, source_raw)
    originals.pop(source, None)
    moved_syntax = _syntax_status(target, moved_content)
    if moved_syntax.startswith("invalid-"):
        raise ValueError("file-move target syntax validation failed")

    expected_originals = dict(originals)
    expected_originals[source] = source_raw
    transaction = apply_workspace_transaction(
        root,
        replacements=updated,
        creations={target: moved_content},
        deletions={source: source_raw},
        expected_originals=expected_originals,
        creation_modes={target: source_mode},
    )

    result: dict[str, object] = {
        "schema_version": REFACTOR_APPLY_SCHEMA_VERSION,
        "operation": "file-move",
        "plan_sha256": expected_plan_sha256,
        "index_digest_before": index_digest,
        "source_path": source,
        "target_path": target,
        "source_sha256": hashlib.sha256(source_raw).hexdigest(),
        "target_sha256": hashlib.sha256(moved_content).hexdigest(),
        "updated_files": [
            {
                "path": path,
                "before_sha256": hashlib.sha256(originals[path]).hexdigest(),
                "after_sha256": hashlib.sha256(updated[path]).hexdigest(),
                "syntax_validation": syntax[path],
            }
            for path in sorted(updated)
        ],
        "target_syntax_validation": moved_syntax,
        "change_count": len(raw_changes),
        "acknowledged_incomplete": bool(acknowledge_incomplete),
        "rollback_performed": transaction.rollback_performed,
        "rollback_complete": transaction.rollback_complete,
        "recovery_artifacts": list(transaction.recovery_artifacts),
        "workspace_transaction": transaction.to_dict(),
        "remote_calls": 0,
    }
    result["apply_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return result


__all__ = [
    "apply_verified_file_move_plan",
    "build_verified_file_move_plan",
]
