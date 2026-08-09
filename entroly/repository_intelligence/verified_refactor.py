"""Verified refactor public surface with recoverable filesystem transactions.

Planning, evidence classification, and receipt verification remain implemented
in :mod:`verified_refactor_impl`.  This facade replaces only the destructive
apply path so rename/safe-delete/inline-local operations inherit the shared
workspace transaction's rollback and recovery guarantees.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from .models import RepositoryIndex
from .verified_refactor_impl import (
    REFACTOR_APPLY_SCHEMA_VERSION,
    REFACTOR_PLAN_SCHEMA_VERSION,
    _syntax_status,
    _verified_source,
    build_verified_inline_local_plan,
    build_verified_rename_plan,
    build_verified_safe_delete_plan,
    verify_refactor_apply_commitment,
    verify_refactor_plan_commitment,
)
from .workspace_transaction import apply_workspace_transaction


def apply_verified_rename_plan(
    root: Path,
    index: RepositoryIndex,
    plan: Mapping[str, object],
    *,
    index_digest: str,
    expected_plan_sha256: str,
    acknowledge_incomplete: bool = False,
) -> dict[str, object]:
    """Apply an exact verified plan through a recoverable transaction."""
    root = root.expanduser().resolve(strict=True)
    candidate_plan = copy.deepcopy(dict(plan))
    if not verify_refactor_plan_commitment(candidate_plan):
        raise ValueError("refactor plan commitment is invalid")
    receipt = candidate_plan.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("plan_sha256") != expected_plan_sha256:
        raise ValueError("expected_plan_sha256 does not match the plan")
    if candidate_plan.get("index_digest") != index_digest:
        raise ValueError("refactor plan index is stale")
    operation = str(candidate_plan.get("operation", ""))
    if operation not in {"rename", "safe-delete", "inline-local"}:
        raise ValueError("unsupported refactor operation")
    if candidate_plan.get("resolution") != "resolved":
        raise ValueError("only a resolved refactor plan can be applied")
    if operation == "safe-delete" and candidate_plan.get("safe_to_apply") is not True:
        raise ValueError("safe-delete plan has blockers or incomplete evidence")
    if operation == "inline-local" and candidate_plan.get("safe_to_apply") is not True:
        raise ValueError("inline-local plan has blockers or incomplete evidence")
    risk = candidate_plan.get("risk")
    if not isinstance(risk, dict) or (
        risk.get("requires_incomplete_acknowledgement") and not acknowledge_incomplete
    ):
        raise ValueError("refactor completeness is unproven; explicit acknowledgement required")
    raw_changes = candidate_plan.get("changes")
    if not isinstance(raw_changes, list) or not raw_changes:
        raise ValueError("refactor plan contains no changes")

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in raw_changes:
        if not isinstance(item, dict):
            raise ValueError("invalid refactor change")
        path = str(item.get("path", ""))
        if path not in index.files:
            raise ValueError("refactor change path is not indexed")
        grouped[path].append(item)

    originals: dict[str, bytes] = {}
    updated: dict[str, bytes] = {}
    syntax: dict[str, str] = {}
    for path, changes in sorted(grouped.items()):
        raw, status = _verified_source(root, index, path)
        if raw is None:
            raise ValueError(f"refactor preimage failed: {status}")
        originals[path] = raw
        cursor = len(raw)
        result = raw
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
                raise ValueError("refactor preimage range changed or overlaps")
            if hashlib.sha256(raw[start:end]).hexdigest() != item.get("evidence_sha256"):
                raise ValueError("refactor preimage evidence hash mismatch")
            result = result[:start] + new + result[end:]
            cursor = start
        syntax[path] = _syntax_status(path, result)
        if syntax[path].startswith("invalid-"):
            raise ValueError(f"refactor staged syntax failed for {path}")
        updated[path] = result

    transaction = apply_workspace_transaction(
        root,
        replacements=updated,
        expected_originals=originals,
    )
    result: dict[str, object] = {
        "schema_version": REFACTOR_APPLY_SCHEMA_VERSION,
        "operation": operation,
        "plan_sha256": expected_plan_sha256,
        "index_digest_before": index_digest,
        "files": [
            {
                "path": path,
                "before_sha256": hashlib.sha256(originals[path]).hexdigest(),
                "after_sha256": hashlib.sha256(updated[path]).hexdigest(),
                "syntax_validation": syntax[path],
            }
            for path in sorted(updated)
        ],
        "change_count": len(raw_changes),
        "file_count": len(updated),
        "acknowledged_incomplete": bool(acknowledge_incomplete),
        "rollback_performed": transaction.rollback_performed,
        "rollback_complete": transaction.rollback_complete,
        "recovery_artifacts": list(transaction.recovery_artifacts),
        "workspace_transaction": transaction.to_dict(),
        "remote_calls": 0,
    }
    canonical = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    result["apply_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


apply_verified_refactor_plan = apply_verified_rename_plan


__all__ = [
    "REFACTOR_APPLY_SCHEMA_VERSION",
    "REFACTOR_PLAN_SCHEMA_VERSION",
    "apply_verified_rename_plan",
    "apply_verified_refactor_plan",
    "build_verified_inline_local_plan",
    "build_verified_safe_delete_plan",
    "build_verified_rename_plan",
    "verify_refactor_apply_commitment",
    "verify_refactor_plan_commitment",
]
