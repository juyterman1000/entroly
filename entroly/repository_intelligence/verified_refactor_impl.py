"""Two-phase, source-verified repository refactor transactions."""
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
from typing import Iterable, Mapping

from entroly.tree_sitter_support import validate_structural_syntax

from .models import RepositoryIndex, Symbol
from .parsers import module_name
from .semantic_overlay import build_verified_semantic_overlay

REFACTOR_PLAN_SCHEMA_VERSION = "entroly.verified-refactor-plan.v1"
REFACTOR_APPLY_SCHEMA_VERSION = "entroly.verified-refactor-apply.v1"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _identifier_occurrences(raw: bytes, start: int, end: int, name: str) -> list[tuple[int, int]]:
    needle = name.encode("utf-8")
    if not needle or not (0 <= start <= end <= len(raw)):
        return []
    result: list[tuple[int, int]] = []
    cursor = start
    while True:
        offset = raw.find(needle, cursor, end)
        if offset < 0:
            break
        finish = offset + len(needle)
        left = raw[offset - 1:offset] if offset > 0 else b""
        right = raw[finish:finish + 1]
        left_ok = not left or not (left.isalnum() or left in {b"_", b"$"})
        right_ok = not right or not (right.isalnum() or right in {b"_", b"$"})
        if left_ok and right_ok:
            result.append((offset, finish))
        cursor = offset + 1
    return result


def _resolve(index: RepositoryIndex, query: str) -> tuple[str, list[Symbol]]:
    lowered = query.strip().lower()
    matches = sorted(
        (
            symbol
            for symbol in index.symbols.values()
            if symbol.symbol_id.lower() == lowered
            or symbol.qualified_name.lower() == lowered
            or symbol.name.lower() == lowered
        ),
        key=lambda symbol: symbol.symbol_id,
    )
    status = "resolved" if len(matches) == 1 else "ambiguous" if matches else "not-found"
    return status, matches


def _verified_source(root: Path, index: RepositoryIndex, path: str) -> tuple[bytes | None, str]:
    record = index.files.get(path)
    if record is None:
        return None, "unknown-path"
    try:
        candidate = (root / path).resolve(strict=True)
        candidate.relative_to(root)
        raw = candidate.read_bytes()
    except (OSError, RuntimeError, ValueError):
        return None, "unsafe-or-unreadable"
    if hashlib.sha256(raw).hexdigest() != record.sha256:
        return None, "stale-index"
    return raw, "verified"


def _change(
    *,
    path: str,
    start: int,
    end: int,
    raw: bytes,
    old_name: str,
    new_name: str,
    kind: str,
    source_sha256: str,
    confidence: str,
) -> dict[str, object] | None:
    before = raw[start:end]
    if before != old_name.encode("utf-8"):
        return None
    change_id = hashlib.sha256(
        f"{path}\0{start}\0{end}\0{old_name}\0{new_name}".encode("utf-8")
    ).hexdigest()[:24]
    return {
        "change_id": change_id,
        "path": path,
        "start_byte": start,
        "end_byte": end,
        "old_identifier": old_name,
        "new_identifier": new_name,
        "evidence_sha256": hashlib.sha256(before).hexdigest(),
        "source_sha256": source_sha256,
        "kind": kind,
        "confidence": confidence,
    }


def _finish_plan(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["plan_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _python_import_changes(
    path: str,
    raw: bytes,
    symbol: Symbol,
    new_name: str,
    source_sha256: str,
) -> list[dict[str, object]]:
    if not path.lower().endswith((".py", ".pyi", ".pyw")):
        return []
    text = raw.decode("utf-8", errors="surrogateescape")
    try:
        tree = ast.parse(text, filename=path, type_comments=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
        return []
    line_offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line.encode("utf-8", errors="surrogateescape"))
        line_offsets.append(running)
    target_module = module_name(symbol.path)
    changes: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        current = module_name(path).split(".")
        if Path(path).name != "__init__.py" and current:
            current.pop()
        if node.level:
            current = current[:max(0, len(current) - node.level + 1)]
        else:
            current = []
        if node.module:
            current.extend(part for part in node.module.split(".") if part)
        imported_module = ".".join(current)
        if imported_module != target_module:
            continue
        for alias in node.names:
            if alias.name != symbol.name:
                continue
            line = max(1, int(getattr(alias, "lineno", 1)))
            start = line_offsets[min(line - 1, len(line_offsets) - 1)]
            start += max(0, int(getattr(alias, "col_offset", 0)))
            end = start + len(symbol.name.encode("utf-8"))
            item = _change(
                path=path, start=start, end=end, raw=raw,
                old_name=symbol.name, new_name=new_name,
                kind="python-import-binding", source_sha256=source_sha256,
                confidence="python-ast-import",
            )
            if item is not None:
                changes.append(item)
    return changes


def build_verified_rename_plan(
    root: Path,
    index: RepositoryIndex,
    symbol_query: str,
    new_name: str,
    *,
    index_digest: str,
    semantic_relationships: Iterable[Mapping[str, object]] = (),
    provider: str = "none",
    max_changes: int = 10_000,
) -> dict[str, object]:
    """Preview a bounded rename without claiming reference completeness."""
    root = root.expanduser().resolve(strict=True)
    query = symbol_query.strip()
    clean_new_name = new_name.strip()
    if not query or len(query) > 1_000:
        raise ValueError("symbol_query must contain 1 to 1000 characters")
    if not _IDENTIFIER.fullmatch(clean_new_name) or len(clean_new_name) > 128:
        raise ValueError("new_name must be a conservative ASCII identifier")
    resolution, matches = _resolve(index, query)
    candidates = [symbol.to_dict() for symbol in matches[:100]]
    base: dict[str, object] = {
        "schema_version": REFACTOR_PLAN_SCHEMA_VERSION,
        "index_digest": index_digest,
        "operation": "rename",
        "symbol_query": query,
        "new_name": clean_new_name,
        "resolution": resolution,
        "candidates": candidates,
        "changes": [],
        "risk": {
            "reference_completeness": "not-proven",
            "requires_incomplete_acknowledgement": True,
            "unresolved_same_name_calls": 0,
            "non_call_references_indexed": False,
            "external_provider": str(provider).strip()[:128] or "none",
        },
        "receipt": {
            "freshness": "not-evaluated" if resolution != "resolved" else "pending",
            "change_count": 0,
            "omissions_by_reason": {},
            "remote_calls": 0,
            "writes_performed": 0,
            "commitment_scope": "payload-excluding-generation-command-and-plan-sha256",
        },
    }
    if resolution != "resolved":
        return _finish_plan(base)
    symbol = matches[0]
    if clean_new_name == symbol.name:
        raise ValueError("new_name is identical to the current symbol name")
    limit = max(1, min(int(max_changes), 100_000))
    omissions: Counter[str] = Counter()
    source_cache: dict[str, tuple[bytes | None, str]] = {}

    def source(path: str) -> tuple[bytes | None, str]:
        if path not in source_cache:
            source_cache[path] = _verified_source(root, index, path)
        return source_cache[path]

    changes: dict[tuple[str, int, int], dict[str, object]] = {}
    raw, status = source(symbol.path)
    if raw is None:
        base["resolution"] = status
        base["receipt"]["freshness"] = status  # type: ignore[index]
        base["receipt"]["omissions_by_reason"] = {status: 1}  # type: ignore[index]
        return _finish_plan(base)
    signature = symbol.signature.encode("utf-8", errors="surrogateescape")
    signature_start = raw.find(signature, symbol.start_byte, symbol.end_byte)
    if signature_start < 0:
        omissions["definition-signature-unverified"] += 1
    else:
        occurrences = _identifier_occurrences(
            raw, signature_start, signature_start + len(signature), symbol.name
        )
        if not occurrences:
            omissions["definition-identifier-unverified"] += 1
        else:
            start, end = occurrences[0]
            item = _change(
                path=symbol.path, start=start, end=end, raw=raw,
                old_name=symbol.name, new_name=clean_new_name,
                kind="definition", source_sha256=index.files[symbol.path].sha256,
                confidence="parser-definition",
            )
            if item is not None:
                changes[(symbol.path, start, end)] = item

    for edge in index.call_edges:
        if edge.callee_id != symbol.symbol_id:
            continue
        if len(changes) >= limit:
            omissions["change-limit"] += 1
            break
        call_raw, call_status = source(edge.path)
        if call_raw is None:
            omissions[call_status] += 1
            continue
        if not (0 <= edge.start_byte < edge.end_byte <= len(call_raw)):
            omissions["invalid-call-range"] += 1
            continue
        call_end = call_raw.find(b"(", edge.start_byte, edge.end_byte)
        if call_end < 0:
            omissions["call-identifier-unverified"] += 1
            continue
        occurrences = _identifier_occurrences(
            call_raw, edge.start_byte, call_end, symbol.name
        )
        if not occurrences:
            omissions["call-identifier-unverified"] += 1
            continue
        start, end = occurrences[-1]
        item = _change(
            path=edge.path, start=start, end=end, raw=call_raw,
            old_name=symbol.name, new_name=clean_new_name,
            kind="resolved-call", source_sha256=index.files[edge.path].sha256,
            confidence=edge.confidence,
        )
        if item is not None:
            changes[(edge.path, start, end)] = item

    for path in sorted(index.files):
        if len(changes) >= limit:
            omissions["change-limit"] += 1
            break
        import_raw, import_status = source(path)
        if import_raw is None:
            omissions[import_status] += 1
            continue
        for item in _python_import_changes(
            path,
            import_raw,
            symbol,
            clean_new_name,
            index.files[path].sha256,
        ):
            if len(changes) >= limit:
                omissions["change-limit"] += 1
                break
            changes[(path, int(item["start_byte"]), int(item["end_byte"]))] = item

    relationship_limit = max(limit * 4, 100)
    relationships: list[Mapping[str, object]] = []
    for position, relationship in enumerate(semantic_relationships):
        if position >= relationship_limit:
            omissions["semantic-input-limit"] += 1
            break
        relationships.append(relationship)
    if relationships:
        overlay = build_verified_semantic_overlay(
            root,
            index,
            relationships,
            index_digest=index_digest,
            provider=provider,
            max_relationships=relationship_limit,
        )
        for relationship in overlay["relationships"]:
            if not isinstance(relationship, dict):
                continue
            target = relationship.get("target")
            location = relationship.get("source")
            if not isinstance(target, dict) or not isinstance(location, dict):
                continue
            if target.get("symbol_id") != symbol.symbol_id:
                continue
            path = str(location.get("path", ""))
            start = int(location.get("start_byte", -1))
            end = int(location.get("end_byte", -1))
            external_raw, external_status = source(path)
            if external_raw is None:
                omissions[external_status] += 1
                continue
            item = _change(
                path=path, start=start, end=end, raw=external_raw,
                old_name=symbol.name, new_name=clean_new_name,
                kind="external-semantic-reference",
                source_sha256=index.files[path].sha256,
                confidence="externally-reported-source-verified",
            )
            if item is None:
                omissions["external-range-not-exact-identifier"] += 1
            elif len(changes) < limit:
                changes[(path, start, end)] = item
            else:
                omissions["change-limit"] += 1
        base["semantic_overlay_receipt"] = overlay["receipt"]

    same_name_unresolved = sum(
        1
        for call in index.unresolved_calls
        if call.target.rpartition(".")[2] == symbol.name
    )
    planned_ranges = set(changes)
    lexical_review_candidates = 0
    for path in sorted(index.files):
        lexical_raw, _status = source(path)
        if lexical_raw is None:
            continue
        lexical_review_candidates += sum(
            (path, start, end) not in planned_ranges
            for start, end in _identifier_occurrences(
                lexical_raw, 0, len(lexical_raw), symbol.name
            )
        )
    ordered = sorted(changes.values(), key=lambda item: (
        str(item["path"]), int(item["start_byte"]), int(item["end_byte"])
    ))
    base["root_symbol_id"] = symbol.symbol_id
    base["old_name"] = symbol.name
    base["changes"] = ordered
    base["risk"] = {
        "reference_completeness": "not-proven",
        "requires_incomplete_acknowledgement": True,
        "unresolved_same_name_calls": same_name_unresolved,
        "unindexed_lexical_occurrences": lexical_review_candidates,
        "non_call_references_indexed": bool(relationships),
        "external_provider": str(provider).strip()[:128] or "none",
        "dynamic_and_string_references": "not-indexed",
    }
    base["receipt"] = {
        "freshness": "verified-against-indexed-source-sha256",
        "change_count": len(ordered),
        "file_count": len({str(item["path"]) for item in ordered}),
        "omissions_by_reason": dict(sorted(omissions.items())),
        "remote_calls": 0,
        "writes_performed": 0,
        "commitment_scope": "payload-excluding-generation-command-and-plan-sha256",
    }
    return _finish_plan(base)


def _python_node_range(
    node: ast.AST,
    line_offsets: list[int],
) -> tuple[int, int] | None:
    line = int(getattr(node, "lineno", 0))
    end_line = int(getattr(node, "end_lineno", 0))
    column = int(getattr(node, "col_offset", -1))
    end_column = int(getattr(node, "end_col_offset", -1))
    if (
        line < 1
        or end_line < line
        or column < 0
        or end_column < 0
        or end_line >= len(line_offsets)
    ):
        return None
    return line_offsets[line - 1] + column, line_offsets[end_line - 1] + end_column


def _literal_inline_expression(node: ast.AST) -> bool:
    """Accept expressions whose evaluation has no user-code or name lookup."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_literal_inline_expression(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None
            and _literal_inline_expression(key)
            and _literal_inline_expression(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Invert)
    ):
        return isinstance(node.operand, ast.Constant) and isinstance(
            node.operand.value, (int, float, complex)
        )
    return False


def _python_statement_deletion_range(
    raw: bytes,
    node: ast.stmt,
    line_offsets: list[int],
) -> tuple[int, int] | None:
    span = _python_node_range(node, line_offsets)
    if span is None:
        return None
    start, end = span
    line_start = raw.rfind(b"\n", 0, start) + 1
    if raw[line_start:start].strip():
        return None
    line_end = raw.find(b"\n", end)
    content_end = len(raw) if line_end < 0 else line_end
    if raw[end:content_end].strip():
        return None
    return line_start, len(raw) if line_end < 0 else line_end + 1


def build_verified_inline_local_plan(
    root: Path,
    index: RepositoryIndex,
    symbol_query: str,
    binding_name: str,
    *,
    index_digest: str,
) -> dict[str, object]:
    """Preview a semantics-conservative Python single-use local inline."""
    root = root.expanduser().resolve(strict=True)
    query = symbol_query.strip()
    binding = binding_name.strip()
    if not query or len(query) > 1_000:
        raise ValueError("symbol_query must contain 1 to 1000 characters")
    if not _IDENTIFIER.fullmatch(binding) or len(binding) > 128:
        raise ValueError("binding_name must be a conservative ASCII identifier")
    resolution, matches = _resolve(index, query)
    base: dict[str, object] = {
        "schema_version": REFACTOR_PLAN_SCHEMA_VERSION,
        "index_digest": index_digest,
        "operation": "inline-local",
        "symbol_query": query,
        "binding_name": binding,
        "resolution": resolution,
        "candidates": [symbol.to_dict() for symbol in matches[:100]],
        "safe_to_apply": False,
        "blockers": [],
        "changes": [],
        "risk": {
            "supported_subset": "python-single-use-local-literal",
            "semantic_preservation": "under-documented-no-runtime-reflection-assumption",
            "requires_incomplete_acknowledgement": True,
            "dynamic_frame_and_trace_observation": "not-proven-absent",
        },
        "receipt": {
            "freshness": "not-evaluated" if resolution != "resolved" else "pending",
            "change_count": 0,
            "blocker_count": 0,
            "remote_calls": 0,
            "writes_performed": 0,
            "commitment_scope": "payload-excluding-generation-command-and-plan-sha256",
        },
    }
    if resolution != "resolved":
        return _finish_plan(base)
    symbol = matches[0]
    blockers: list[dict[str, object]] = []
    if symbol.language != "python":
        blockers.append({"kind": "unsupported-language", "language": symbol.language})
    raw, freshness = _verified_source(root, index, symbol.path)
    if raw is None:
        base["resolution"] = freshness
        base["receipt"]["freshness"] = freshness  # type: ignore[index]
        return _finish_plan(base)
    text = raw.decode("utf-8", errors="surrogateescape")
    try:
        tree = ast.parse(text, filename=symbol.path, type_comments=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
        blockers.append({"kind": "python-ast-parse-failed"})
        tree = ast.Module(body=[], type_ignores=[])
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == symbol.name
            and int(getattr(node, "lineno", -1)) == symbol.line_start
        ),
        None,
    )
    if function is None:
        blockers.append({"kind": "unsupported-symbol", "required": "python-function"})

    line_offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line.encode("utf-8", errors="surrogateescape"))
        line_offsets.append(running)

    assignment: ast.Assign | ast.AnnAssign | None = None
    load: ast.Name | None = None
    expression: ast.expr | None = None
    if function is not None:
        parameters = {
            argument.arg
            for argument in (
                *function.args.posonlyargs,
                *function.args.args,
                *function.args.kwonlyargs,
                *((function.args.vararg,) if function.args.vararg else ()),
                *((function.args.kwarg,) if function.args.kwarg else ()),
            )
        }
        if binding in parameters:
            blockers.append({"kind": "binding-is-parameter"})
        direct_assignments: list[ast.Assign | ast.AnnAssign] = []
        for statement in function.body:
            target: ast.expr | None = None
            value: ast.expr | None = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target, value = statement.targets[0], statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                target, value = statement.target, statement.value
            if isinstance(target, ast.Name) and target.id == binding and value is not None:
                direct_assignments.append(statement)
                expression = value
        if len(direct_assignments) != 1:
            blockers.append({
                "kind": "binding-definition-count",
                "count": len(direct_assignments),
                "required": 1,
            })
        else:
            assignment = direct_assignments[0]

        parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(function):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        names = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Name) and node.id == binding
        ]
        loads = [node for node in names if isinstance(node.ctx, ast.Load)]
        stores = [node for node in names if isinstance(node.ctx, ast.Store)]
        deletes = [node for node in names if isinstance(node.ctx, ast.Del)]
        if len(loads) != 1:
            blockers.append({
                "kind": "binding-load-count", "count": len(loads), "required": 1
            })
        else:
            load = loads[0]
        if len(stores) != 1:
            blockers.append({
                "kind": "binding-store-count", "count": len(stores), "required": 1
            })
        if deletes:
            blockers.append({"kind": "binding-delete", "count": len(deletes)})

        if load is not None:
            ancestor = parent.get(load)
            while ancestor is not None and ancestor is not function:
                if isinstance(ancestor, (
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                    ast.Lambda,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                )):
                    blockers.append({
                        "kind": "potential-repeated-or-deferred-use",
                        "ancestor": type(ancestor).__name__,
                    })
                    break
                ancestor = parent.get(ancestor)

        reflective_names = {"locals", "globals", "vars", "eval", "exec", "dir"}
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in reflective_names
            for node in ast.walk(function)
        ) or any(
            isinstance(node, ast.Attribute)
            and node.attr in {"f_locals", "currentframe", "_getframe"}
            for node in ast.walk(function)
        ):
            blockers.append({"kind": "runtime-local-reflection"})

    if expression is not None:
        expression_span = _python_node_range(expression, line_offsets)
        if not _literal_inline_expression(expression):
            blockers.append({"kind": "expression-not-side-effect-free-literal"})
        elif expression_span is None:
            blockers.append({"kind": "expression-range-unavailable"})
        elif int(getattr(expression, "lineno", 0)) != int(
            getattr(expression, "end_lineno", -1)
        ):
            blockers.append({"kind": "multiline-expression-unsupported"})
        elif expression_span[1] - expression_span[0] > 16_384:
            blockers.append({"kind": "expression-too-large", "max_bytes": 16_384})

    assignment_span = (
        _python_statement_deletion_range(raw, assignment, line_offsets)
        if assignment is not None
        else None
    )
    load_span = _python_node_range(load, line_offsets) if load is not None else None
    expression_span = (
        _python_node_range(expression, line_offsets) if expression is not None else None
    )
    if assignment is not None and assignment_span is None:
        blockers.append({"kind": "assignment-not-an-isolated-statement"})
    if load is not None and load_span is None:
        blockers.append({"kind": "load-range-unavailable"})
    if assignment_span is not None and load_span is not None and load_span[0] < assignment_span[1]:
        blockers.append({"kind": "use-does-not-follow-definition"})

    changes: list[dict[str, object]] = []
    if not blockers and assignment_span and load_span and expression_span:
        source_sha256 = index.files[symbol.path].sha256
        expression_bytes = raw[expression_span[0]:expression_span[1]]
        replacement = b"(" + expression_bytes + b")"
        for start, end, new, kind in (
            (assignment_span[0], assignment_span[1], b"", "inline-binding-delete"),
            (load_span[0], load_span[1], replacement, "inline-use"),
        ):
            before = raw[start:end]
            changes.append({
                "change_id": hashlib.sha256(
                    f"{symbol.path}\0{start}\0{end}\0{kind}".encode("utf-8")
                ).hexdigest()[:24],
                "path": symbol.path,
                "start_byte": start,
                "end_byte": end,
                "old_identifier": before.decode("utf-8", errors="surrogateescape"),
                "new_identifier": new.decode("utf-8", errors="surrogateescape"),
                "evidence_sha256": hashlib.sha256(before).hexdigest(),
                "source_sha256": source_sha256,
                "kind": kind,
                "confidence": "python-ast-conservative-inline",
            })
        staged = raw
        for item in sorted(changes, key=lambda value: int(value["start_byte"]), reverse=True):
            start = int(item["start_byte"])
            end = int(item["end_byte"])
            staged = staged[:start] + str(item["new_identifier"]).encode(
                "utf-8", errors="surrogateescape"
            ) + staged[end:]
        syntax = _syntax_status(symbol.path, staged)
        if syntax.startswith("invalid-"):
            blockers.append({"kind": "staged-syntax-invalid", "status": syntax})
            changes = []
        else:
            base["staged_syntax"] = syntax

    base["root_symbol_id"] = symbol.symbol_id
    base["safe_to_apply"] = not blockers and bool(changes)
    base["blockers"] = blockers
    base["changes"] = changes
    base["receipt"] = {
        "freshness": "verified-against-indexed-source-sha256",
        "change_count": len(changes),
        "file_count": int(bool(changes)),
        "blocker_count": len(blockers),
        "remote_calls": 0,
        "writes_performed": 0,
        "commitment_scope": "payload-excluding-generation-command-and-plan-sha256",
    }
    return _finish_plan(base)


def _python_top_level_deletion_range(raw: bytes, symbol: Symbol) -> tuple[int, int] | None:
    text = raw.decode("utf-8", errors="surrogateescape")
    try:
        tree = ast.parse(text, filename=symbol.path, type_comments=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
        return None
    node = next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and item.name == symbol.name
            and int(getattr(item, "lineno", -1)) == symbol.line_start
        ),
        None,
    )
    if node is None:
        return None
    decorated = getattr(node, "decorator_list", ())
    first_line = min(
        [int(getattr(item, "lineno", symbol.line_start)) for item in decorated]
        or [symbol.line_start]
    )
    lines = text.splitlines(keepends=True)
    start = sum(
        len(item.encode("utf-8", errors="surrogateescape"))
        for item in lines[: first_line - 1]
    )
    end_line = int(getattr(node, "end_lineno", symbol.line_end))
    end = sum(
        len(item.encode("utf-8", errors="surrogateescape"))
        for item in lines[:end_line]
    )
    while end < len(raw) and raw[end:end + 1] in {b"\r", b"\n"}:
        end += 1
    return start, end


def _deletion_range(raw: bytes, symbol: Symbol) -> tuple[int, int] | None:
    if symbol.parent_id is not None:
        return None
    if symbol.language == "python":
        return _python_top_level_deletion_range(raw, symbol)
    start = symbol.start_byte
    end = symbol.end_byte
    if not (0 <= start < end <= len(raw)):
        return None
    line_start = raw.rfind(b"\n", 0, start) + 1
    if raw[line_start:start].strip():
        line_start = start
    line_end = raw.find(b"\n", end)
    return line_start, len(raw) if line_end < 0 else line_end + 1


def build_verified_safe_delete_plan(
    root: Path,
    index: RepositoryIndex,
    symbol_query: str,
    *,
    index_digest: str,
    max_blockers: int = 10_000,
) -> dict[str, object]:
    """Preview a conservative headless delete and expose every known blocker."""
    root = root.expanduser().resolve(strict=True)
    query = symbol_query.strip()
    if not query or len(query) > 1_000:
        raise ValueError("symbol_query must contain 1 to 1000 characters")
    resolution, matches = _resolve(index, query)
    base: dict[str, object] = {
        "schema_version": REFACTOR_PLAN_SCHEMA_VERSION,
        "index_digest": index_digest,
        "operation": "safe-delete",
        "symbol_query": query,
        "resolution": resolution,
        "candidates": [symbol.to_dict() for symbol in matches[:100]],
        "safe_to_apply": False,
        "blockers": [],
        "changes": [],
        "risk": {
            "reference_completeness": "not-proven",
            "requires_incomplete_acknowledgement": True,
            "dynamic_reflection_and_generated_code": "not-indexed",
            "policy": "any-unexplained-lexical-occurrence-blocks",
        },
        "receipt": {
            "freshness": "not-evaluated" if resolution != "resolved" else "pending",
            "change_count": 0,
            "blocker_count_before_output_limit": 0,
            "omissions_by_reason": {},
            "remote_calls": 0,
            "writes_performed": 0,
            "commitment_scope": "payload-excluding-generation-command-and-plan-sha256",
        },
    }
    if resolution != "resolved":
        return _finish_plan(base)
    symbol = matches[0]
    bound = max(1, min(int(max_blockers), 100_000))
    omissions: Counter[str] = Counter()
    verified: dict[str, bytes] = {}
    for path in sorted(index.files):
        raw, status = _verified_source(root, index, path)
        if raw is None:
            omissions[status] += 1
        else:
            verified[path] = raw
    raw = verified.get(symbol.path)
    if raw is None:
        base["resolution"] = "stale-or-unreadable"
        base["receipt"]["freshness"] = "failed"  # type: ignore[index]
        base["receipt"]["omissions_by_reason"] = dict(sorted(omissions.items()))  # type: ignore[index]
        return _finish_plan(base)
    deletion = _deletion_range(raw, symbol)
    if deletion is None:
        omissions["unsupported-nested-or-unverified-declaration"] += 1
        base["receipt"]["freshness"] = "verified-against-indexed-source-sha256"  # type: ignore[index]
        base["receipt"]["omissions_by_reason"] = dict(sorted(omissions.items()))  # type: ignore[index]
        return _finish_plan(base)
    start, end = deletion
    blockers: dict[tuple[str, int, int, str], dict[str, object]] = {}
    for edge in index.call_edges:
        if edge.callee_id == symbol.symbol_id:
            blockers[(edge.path, edge.start_byte, edge.end_byte, "resolved-call")] = {
                "kind": "resolved-call",
                "path": edge.path,
                "line": edge.line,
                "start_byte": edge.start_byte,
                "end_byte": edge.end_byte,
                "evidence_sha256": edge.evidence_sha256,
            }
    for call in index.unresolved_calls:
        if symbol.symbol_id in call.candidates or (
            call.target.rpartition(".")[2] == symbol.name
        ):
            blockers[(call.path, call.start_byte, call.end_byte, "unresolved-call")] = {
                "kind": "unresolved-call-candidate",
                "path": call.path,
                "line": call.line,
                "start_byte": call.start_byte,
                "end_byte": call.end_byte,
                "reason": call.reason,
            }
    for path, source in verified.items():
        for found_start, found_end in _identifier_occurrences(
            source, 0, len(source), symbol.name
        ):
            if path == symbol.path and start <= found_start < found_end <= end:
                continue
            key = (path, found_start, found_end, "lexical")
            blockers.setdefault(key, {
                "kind": "unclassified-lexical-reference",
                "path": path,
                "line": source.count(b"\n", 0, found_start) + 1,
                "start_byte": found_start,
                "end_byte": found_end,
                "evidence_sha256": hashlib.sha256(
                    source[found_start:found_end]
                ).hexdigest(),
            })
    ordered_blockers = sorted(blockers.values(), key=lambda item: (
        str(item["path"]), int(item["start_byte"]), str(item["kind"])
    ))
    before = raw[start:end]
    staged = raw[:start] + raw[end:]
    syntax = _syntax_status(symbol.path, staged)
    if syntax.startswith("invalid-"):
        omissions["staged-syntax-invalid"] += 1
    safe_to_apply = not blockers and not omissions and not syntax.startswith("invalid-")
    change = {
        "change_id": hashlib.sha256(
            f"{symbol.path}\0{start}\0{end}\0safe-delete".encode("utf-8")
        ).hexdigest()[:24],
        "path": symbol.path,
        "start_byte": start,
        "end_byte": end,
        "old_identifier": before.decode("utf-8", errors="surrogateescape"),
        "new_identifier": "",
        "evidence_sha256": hashlib.sha256(before).hexdigest(),
        "source_sha256": index.files[symbol.path].sha256,
        "kind": "definition-delete",
        "confidence": "parser-definition-with-conservative-reference-scan",
        "staged_syntax": syntax,
    }
    base["root_symbol_id"] = symbol.symbol_id
    base["safe_to_apply"] = safe_to_apply
    base["blockers"] = ordered_blockers[:bound]
    base["changes"] = [change] if safe_to_apply else []
    base["truncation"] = {
        "blockers_omitted": max(0, len(ordered_blockers) - bound)
    }
    base["receipt"] = {
        "freshness": "verified-against-indexed-source-sha256",
        "change_count": int(safe_to_apply),
        "file_count": int(safe_to_apply),
        "blocker_count_before_output_limit": len(ordered_blockers),
        "omissions_by_reason": dict(sorted(omissions.items())),
        "remote_calls": 0,
        "writes_performed": 0,
        "commitment_scope": "payload-excluding-generation-command-and-plan-sha256",
    }
    return _finish_plan(base)


def verify_refactor_plan_commitment(payload: dict[str, object]) -> bool:
    try:
        candidate = copy.deepcopy(payload)
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("plan_sha256"))
        canonical = json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


def _syntax_status(path: str, raw: bytes) -> str:
    text = raw.decode("utf-8", errors="surrogateescape")
    if path.lower().endswith((".py", ".pyi", ".pyw")):
        try:
            ast.parse(text, filename=path, type_comments=True)
        except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
            return "invalid-python-syntax"
        return "verified-python-ast"
    valid = validate_structural_syntax(text, path)
    if valid is False:
        return "invalid-parser-syntax"
    return "verified-tree-sitter" if valid is True else "parser-unavailable"


def apply_verified_rename_plan(
    root: Path,
    index: RepositoryIndex,
    plan: Mapping[str, object],
    *,
    index_digest: str,
    expected_plan_sha256: str,
    acknowledge_incomplete: bool = False,
) -> dict[str, object]:
    """Apply an exact rename plan with preimage checks and rollback attempts."""
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
        for item in sorted(changes, key=lambda value: int(value["start_byte"]), reverse=True):
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

    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    completed: list[str] = []
    rollback_performed = False
    try:
        for path in sorted(updated):
            target = (root / path).resolve(strict=True)
            target.relative_to(root)
            stage_fd, stage_name = tempfile.mkstemp(
                prefix=f".{target.name}.entroly-stage.", dir=target.parent
            )
            staged[path] = Path(stage_name)
            with os.fdopen(stage_fd, "wb") as handle:
                handle.write(updated[path])
                handle.flush()
                os.fsync(handle.fileno())
            backup_fd, backup_name = tempfile.mkstemp(
                prefix=f".{target.name}.entroly-backup.", dir=target.parent
            )
            backups[path] = Path(backup_name)
            with os.fdopen(backup_fd, "wb") as handle:
                handle.write(originals[path])
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(stage_name, target.stat().st_mode)
            os.chmod(backup_name, target.stat().st_mode)
        for path in sorted(updated):
            target = (root / path).resolve(strict=True)
            staged[path].replace(target)
            completed.append(path)
    except OSError as exc:
        rollback_performed = bool(completed)
        for path in reversed(completed):
            try:
                target = (root / path).resolve(strict=True)
                backups[path].replace(target)
            except OSError:
                pass
        raise ValueError("refactor filesystem transaction failed; rollback attempted") from exc
    finally:
        for temporary in (*staged.values(), *backups.values()):
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError:
                pass

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
        "rollback_performed": rollback_performed,
        "remote_calls": 0,
    }
    canonical = json.dumps(
        result, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    result["apply_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


apply_verified_refactor_plan = apply_verified_rename_plan


def verify_refactor_apply_commitment(payload: Mapping[str, object]) -> bool:
    try:
        candidate = copy.deepcopy(dict(payload))
        expected = str(candidate.pop("apply_sha256"))
        canonical = json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


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
