"""Two-phase, source-verified Python module move transactions."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

from .models import RepositoryIndex, normalize_relative
from .parsers import module_name
from .verified_refactor import (
    REFACTOR_APPLY_SCHEMA_VERSION,
    REFACTOR_PLAN_SCHEMA_VERSION,
    _finish_plan,
    _identifier_occurrences,
    _syntax_status,
    _verified_source,
    verify_refactor_plan_commitment,
)


def _safe_path(value: str) -> str | None:
    raw = str(value).replace("\\", "/")
    if (
        not raw
        or "\x00" in raw
        or raw.startswith(("/", "//"))
        or (len(raw) >= 2 and raw[1] == ":")
        or any(part in {"", ".", ".."} for part in raw.split("/"))
    ):
        return None
    return normalize_relative(raw)


def _line_offsets(text: str) -> list[int]:
    offsets = [0]
    current = 0
    for line in text.splitlines(keepends=True):
        current += len(line.encode("utf-8", errors="surrogateescape"))
        offsets.append(current)
    return offsets


def _node_range(text: str, node: ast.AST) -> tuple[int, int]:
    offsets = _line_offsets(text)
    line = max(1, int(getattr(node, "lineno", 1)))
    end_line = max(line, int(getattr(node, "end_lineno", line)))
    start = offsets[min(line - 1, len(offsets) - 1)] + int(
        getattr(node, "col_offset", 0)
    )
    end = offsets[min(end_line - 1, len(offsets) - 1)] + int(
        getattr(node, "end_col_offset", 0)
    )
    return start, end


def _resolved_from_module(path: str, node: ast.ImportFrom) -> str:
    current = module_name(path).split(".")
    if Path(path).name != "__init__.py" and current:
        current.pop()
    if node.level:
        current = current[:max(0, len(current) - node.level + 1)]
    else:
        current = []
    if node.module:
        current.extend(part for part in node.module.split(".") if part)
    return ".".join(current)


def _replacement(
    *,
    path: str,
    raw: bytes,
    start: int,
    end: int,
    old: str,
    new: str,
    source_sha256: str,
    kind: str,
) -> dict[str, object]:
    before = raw[start:end]
    if before.decode("utf-8", errors="surrogateescape") != old:
        raise ValueError("module-move import preimage is not exact")
    return {
        "change_id": hashlib.sha256(
            f"{path}\0{start}\0{end}\0{old}\0{new}".encode("utf-8")
        ).hexdigest()[:24],
        "path": path,
        "start_byte": start,
        "end_byte": end,
        "old_identifier": old,
        "new_identifier": new,
        "evidence_sha256": hashlib.sha256(before).hexdigest(),
        "source_sha256": source_sha256,
        "kind": kind,
        "confidence": "python-ast-import",
    }


def _python_import_changes(
    path: str,
    raw: bytes,
    *,
    old_module: str,
    new_module: str,
    source_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    text = raw.decode("utf-8", errors="surrogateescape")
    try:
        tree = ast.parse(text, filename=path, type_comments=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
        return [], [{"kind": "python-parse-error", "path": path}]
    changes: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != old_module:
                    continue
                node_start, node_end = _node_range(text, node)
                occurrences = _identifier_occurrences(
                    raw, node_start, node_end, old_module
                )
                if len(occurrences) != 1:
                    blockers.append({
                        "kind": "ambiguous-import-range", "path": path,
                        "line": node.lineno,
                    })
                    continue
                start, end = occurrences[0]
                if alias.asname:
                    replacement = new_module
                elif "." not in old_module:
                    replacement = f"{new_module} as {old_module}"
                else:
                    blockers.append({
                        "kind": "dotted-import-binding-change",
                        "path": path,
                        "line": node.lineno,
                        "detail": "use an explicit import alias before moving",
                    })
                    continue
                changes.append(_replacement(
                    path=path, raw=raw, start=start, end=end,
                    old=old_module, new=replacement,
                    source_sha256=source_sha256, kind="python-import-module",
                ))
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolved_from_module(path, node)
            if resolved != old_module:
                continue
            node_start, node_end = _node_range(text, node)
            statement = raw[node_start:node_end].decode(
                "utf-8", errors="surrogateescape"
            )
            match = re.search(r"\bfrom\s+([.A-Za-z_][.A-Za-z0-9_]*)\s+import\b", statement)
            if match is None:
                blockers.append({
                    "kind": "unsupported-from-import", "path": path,
                    "line": node.lineno,
                })
                continue
            prefix = statement[: match.start(1)].encode(
                "utf-8", errors="surrogateescape"
            )
            old = match.group(1)
            start = node_start + len(prefix)
            end = start + len(old.encode("utf-8", errors="surrogateescape"))
            changes.append(_replacement(
                path=path, raw=raw, start=start, end=end, old=old,
                new=new_module, source_sha256=source_sha256,
                kind="python-from-module",
            ))
    return changes, blockers


def build_verified_file_move_plan(
    root: Path,
    index: RepositoryIndex,
    source_path: str,
    target_path: str,
    *,
    index_digest: str,
    max_changes: int = 10_000,
    max_blockers: int = 10_000,
) -> dict[str, object]:
    """Preview a Python module move without an IDE or filesystem writes."""
    root = root.expanduser().resolve(strict=True)
    source = _safe_path(source_path)
    target = _safe_path(target_path)
    if source is None or target is None:
        raise ValueError("source and target must be safe workspace-relative paths")
    if source == target:
        raise ValueError("source and target paths are identical")
    if source not in index.files:
        raise ValueError("source path is not indexed")
    if not source.lower().endswith((".py", ".pyi", ".pyw")) or not target.lower().endswith(
        (".py", ".pyi", ".pyw")
    ):
        raise ValueError("headless file move currently supports Python modules only")
    if Path(source).name == "__init__.py" or Path(target).name == "__init__.py":
        raise ValueError("package initializer moves require compiler-aware review")
    destination = (root / target).resolve(strict=False)
    destination.relative_to(root)
    if destination.exists() or target in index.files:
        raise ValueError("target path already exists")
    if not destination.parent.is_dir():
        raise ValueError("target parent directory must already exist")
    old_module = module_name(source)
    new_module = module_name(target)
    change_limit = max(1, min(int(max_changes), 100_000))
    blocker_limit = max(1, min(int(max_blockers), 100_000))
    omissions: Counter[str] = Counter()
    verified: dict[str, bytes] = {}
    for path in sorted(index.files):
        raw, status = _verified_source(root, index, path)
        if raw is None:
            omissions[status] += 1
        else:
            verified[path] = raw
    source_raw = verified.get(source)
    if source_raw is None:
        raise ValueError("source changed or became unreadable after indexing")

    blockers: list[dict[str, object]] = []
    if Path(source).parent != Path(target).parent:
        try:
            source_tree = ast.parse(
                source_raw.decode("utf-8", errors="surrogateescape"),
                filename=source,
                type_comments=True,
            )
        except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
            blockers.append({"kind": "source-parse-error", "path": source})
        else:
            for node in ast.walk(source_tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    blockers.append({
                        "kind": "relative-import-package-change",
                        "path": source,
                        "line": node.lineno,
                    })

    changes: dict[tuple[str, int, int], dict[str, object]] = {}
    for path, raw in sorted(verified.items()):
        if not path.lower().endswith((".py", ".pyi", ".pyw")):
            continue
        found, found_blockers = _python_import_changes(
            path,
            raw,
            old_module=old_module,
            new_module=new_module,
            source_sha256=index.files[path].sha256,
        )
        blockers.extend(found_blockers)
        for change in found:
            key = (path, int(change["start_byte"]), int(change["end_byte"]))
            if len(changes) < change_limit:
                changes[key] = change
            else:
                omissions["change-limit"] += 1

    changed_paths = {str(item["path"]) for item in changes.values()}
    for dependent in index.dependents_of(source):
        if dependent not in changed_paths:
            blockers.append({
                "kind": "dependency-without-safe-import-rewrite",
                "path": dependent,
            })

    planned_ranges = set(changes)
    old_path_literal = source.replace("\\", "/")
    for path, raw in sorted(verified.items()):
        for needle in (old_module, old_path_literal):
            encoded = needle.encode("utf-8")
            cursor = 0
            while encoded:
                start = raw.find(encoded, cursor)
                if start < 0:
                    break
                end = start + len(encoded)
                cursor = end
                if (path, start, end) in planned_ranges:
                    continue
                if path == source and needle == old_path_literal:
                    continue
                blockers.append({
                    "kind": "unclassified-module-or-path-reference",
                    "path": path,
                    "line": raw.count(b"\n", 0, start) + 1,
                    "start_byte": start,
                    "end_byte": end,
                    "value": needle,
                })
    unique_blockers = {
        json.dumps(item, sort_keys=True, separators=(",", ":")): item
        for item in blockers
    }
    ordered_blockers = [unique_blockers[key] for key in sorted(unique_blockers)]
    ordered_changes = sorted(changes.values(), key=lambda item: (
        str(item["path"]), int(item["start_byte"]), int(item["end_byte"])
    ))
    safe_to_apply = not ordered_blockers and not omissions
    base: dict[str, object] = {
        "schema_version": REFACTOR_PLAN_SCHEMA_VERSION,
        "index_digest": index_digest,
        "operation": "file-move",
        "resolution": "resolved",
        "source_path": source,
        "target_path": target,
        "old_module": old_module,
        "new_module": new_module,
        "source_sha256": index.files[source].sha256,
        "safe_to_apply": safe_to_apply,
        "changes": ordered_changes if safe_to_apply else [],
        "blockers": ordered_blockers[:blocker_limit],
        "risk": {
            "reference_completeness": "not-proven",
            "requires_incomplete_acknowledgement": True,
            "dynamic_reflection_generated_and_external_references": "not-indexed",
            "supported_rewrites": "python-import-and-from-import",
        },
        "truncation": {
            "blockers_omitted": max(0, len(ordered_blockers) - blocker_limit),
        },
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "change_count": len(ordered_changes) if safe_to_apply else 0,
            "blocker_count_before_output_limit": len(ordered_blockers),
            "omissions_by_reason": dict(sorted(omissions.items())),
            "remote_calls": 0,
            "writes_performed": 0,
            "commitment_scope": "payload-excluding-generation-command-and-plan-sha256",
        },
    }
    return _finish_plan(base)


def apply_verified_file_move_plan(
    root: Path,
    index: RepositoryIndex,
    plan: Mapping[str, object],
    *,
    index_digest: str,
    expected_plan_sha256: str,
    acknowledge_incomplete: bool = False,
) -> dict[str, object]:
    """Apply an exact module-move plan with rollback-aware filesystem writes."""
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
        for item in sorted(changes, key=lambda value: int(value["start_byte"]), reverse=True):
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
    stages: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    completed: list[str] = []
    target_stage: Path | None = None
    target_created = False
    source_deleted = False
    rollback_performed = False
    try:
        for path, content in sorted(updated.items()):
            current = (root / path).resolve(strict=True)
            stage_fd, stage_name = tempfile.mkstemp(
                prefix=f".{current.name}.entroly-stage.", dir=current.parent
            )
            stages[path] = Path(stage_name)
            with os.fdopen(stage_fd, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            backup_fd, backup_name = tempfile.mkstemp(
                prefix=f".{current.name}.entroly-backup.", dir=current.parent
            )
            backups[path] = Path(backup_name)
            with os.fdopen(backup_fd, "wb") as handle:
                handle.write(originals[path])
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(stage_name, current.stat().st_mode)
            os.chmod(backup_name, current.stat().st_mode)
        target_fd, target_name = tempfile.mkstemp(
            prefix=f".{target_file.name}.entroly-stage.", dir=target_file.parent
        )
        target_stage = Path(target_name)
        with os.fdopen(target_fd, "wb") as handle:
            handle.write(moved_content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target_name, source_file.stat().st_mode)
        for path in sorted(updated):
            stages[path].replace((root / path).resolve(strict=True))
            completed.append(path)
        target_stage.replace(target_file)
        target_created = True
        source_file.unlink()
        source_deleted = True
    except OSError as exc:
        rollback_performed = bool(completed or target_created or source_deleted)
        if source_deleted:
            source_file.write_bytes(source_raw)
        if target_created:
            try:
                target_file.unlink()
            except OSError:
                pass
        for path in reversed(completed):
            try:
                backups[path].replace((root / path).resolve(strict=True))
            except OSError:
                pass
        raise ValueError("file-move filesystem transaction failed; rollback attempted") from exc
    finally:
        temporaries = [*stages.values(), *backups.values()]
        if target_stage is not None:
            temporaries.append(target_stage)
        for temporary in temporaries:
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass
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
        "rollback_performed": rollback_performed,
        "remote_calls": 0,
    }
    result["apply_sha256"] = hashlib.sha256(
        json.dumps(
            result, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return result


__all__ = [
    "apply_verified_file_move_plan",
    "build_verified_file_move_plan",
]
