"""Bounded explicit Python re-export discovery, not a completeness certificate."""
from __future__ import annotations

import ast
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Callable

from .models import RepositoryIndex, Symbol
from .parsers import module_name
from .rename_binding_review import MAX_REVIEW_SOURCE_BYTES, python_module_binding_evidence

MAX_EXPORT_FILES = 512
MAX_EXPORT_BYTES = 32 * 1024 * 1024
MAX_EXPORT_IMPORTS = 16_384


def _import_module(path: str, node: ast.ImportFrom) -> str | None:
    if not node.level:
        return node.module or ""
    package = module_name(path).split(".")
    if Path(path).stem != "__init__":
        package = package[:-1]
    if node.level > len(package):
        return None
    package = package[:len(package) - node.level + 1]
    return ".".join([*package, *(node.module or "").split(".")]).rstrip(".")


def discover_python_export_changes(
    index: RepositoryIndex,
    symbol: Symbol,
    source: Callable[[str], tuple[bytes | None, str]],
    *,
    max_changes: int,
) -> dict:
    """Follow only uniquely bound, unaliased, module-level imported names.

    Explicit aliases end propagation, even ``from x import name as name``.
    Every proposed span is subsequently checked against source by the planner.
    Unknown effects never become inferred import edges or guessed references.
    """
    unknown: Counter[str] = Counter()
    proposed: dict[tuple[str, int, int], dict] = {}
    protected: set[tuple[str, int, int]] = set()
    modules: dict[str, list[str]] = defaultdict(list)
    files: dict[str, dict] = {}
    edges: dict[str, list[tuple[str, ast.ImportFrom, ast.alias, bool]]] = defaultdict(list)
    imported = 0
    total_bytes = 0
    affected: set[str] = set()

    def result(status: str) -> dict:
        return {
            "changes": [proposed[key] for key in sorted(proposed)],
            "protected_loads": protected,
            "risk": {
                "status": status,
                "scope": "explicit-module-imports-and-compiler-owned-global-loads",
                "files_examined": len(files),
                "affected_module_count": len(affected),
                "unknown_reasons": dict(sorted(unknown.items())),
            },
        }

    if symbol.language != "python" or symbol.parent_id is not None:
        return result("not-applicable")
    paths = [path for path in sorted(index.files) if path.endswith((".py", ".pyi", ".pyw"))]
    for path in paths:
        modules[module_name(path)].append(path)
    ordered = [symbol.path, *(path for path in paths if path != symbol.path)]
    if len(ordered) > MAX_EXPORT_FILES:
        unknown["file-limit"] += len(ordered) - MAX_EXPORT_FILES
    for path in ordered[:MAX_EXPORT_FILES]:
        raw, status = source(path)
        if raw is None:
            unknown[status] += 1
            continue
        if len(raw) > MAX_REVIEW_SOURCE_BYTES:
            unknown["source-size-limit"] += 1
            continue
        if total_bytes + len(raw) > MAX_EXPORT_BYTES:
            unknown["total-source-size-limit"] += 1
            continue
        total_bytes += len(raw)
        try:
            tree = ast.parse(raw, filename=path)
            evidence = python_module_binding_evidence(path, raw, tree, symbol.name)
        except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
            unknown["compiler-unavailable"] += 1
            continue
        offsets = [0]
        for line in raw.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        info = {"tree": tree, "raw": raw, "offsets": offsets, "evidence": evidence,
                "wildcard": False, "reflective": False}
        files[path] = info
        direct = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"exec", "eval", "globals", "locals", "vars"}:
                    info["reflective"] = True
            if not isinstance(node, ast.ImportFrom):
                continue
            if any(alias.name == "*" for alias in node.names):
                info["wildcard"] = True
            target = _import_module(path, node)
            for alias in node.names:
                if alias.name != symbol.name and alias.name != "*":
                    continue
                if imported >= MAX_EXPORT_IMPORTS:
                    unknown["import-limit"] += 1
                    continue
                imported += 1
                if target is None:
                    unknown["invalid-relative-import"] += 1
                    continue
                edges[target].append((path, node, alias, id(node) in direct))

    def offset(info: dict, node: ast.AST) -> int:
        return info["offsets"][node.lineno - 1] + node.col_offset

    def propose(path: str, start: int, kind: str) -> None:
        end = start + len(symbol.name.encode("utf-8"))
        if files[path]["raw"][start:end] != symbol.name.encode("utf-8"):
            unknown["unverified-identifier-span"] += 1
            return
        key = path, start, end
        if key not in proposed and len(proposed) >= max_changes:
            unknown["change-limit"] += 1
            return
        proposed[key] = {"path": path, "start_byte": start, "end_byte": end, "kind": kind}

    def export_contract(info: dict) -> tuple[list[int], Counter[str]]:
        reasons: Counter[str] = Counter()
        positions = []
        direct_export_targets = {
            id(target)
            for statement in info["tree"].body
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for target in (statement.targets if isinstance(statement, ast.Assign) else [statement.target])
        }
        for node in ast.walk(info["tree"]):
            if (isinstance(node, ast.Name) and node.id == "__all__"
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    and id(node) not in direct_export_targets):
                reasons["nested-export-contract"] += 1
            if (isinstance(node, (ast.Attribute, ast.Subscript))
                    and isinstance(node.value, ast.Name) and node.value.id == "__all__"):
                reasons["nonliteral-export-contract"] += 1
        for node in info["tree"].body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                continue
            if not isinstance(node, ast.Assign) or not isinstance(node.value, (ast.List, ast.Tuple)):
                reasons["nonliteral-export-contract"] += 1
                continue
            for value in node.value.elts:
                if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    reasons["nonliteral-export-contract"] += 1
                elif value.value == symbol.name:
                    start, end = offset(info, value), info["offsets"][value.end_lineno - 1] + value.end_col_offset
                    text = info["raw"][start:end]
                    if re.fullmatch(rb"(['\"])" + re.escape(symbol.name.encode()) + rb"\1", text):
                        positions.append(start + 1)
                    else:
                        reasons["noncanonical-export-string"] += 1
        return positions, reasons

    def stable(path: str, expected: tuple[int, int]) -> bool:
        info = files[path]
        writes = info["evidence"]["writes"]
        exports, export_unknown = export_contract(info)
        info["export_positions"] = exports
        reasons = Counter(info["evidence"]["unknown"]) + export_unknown
        if info["wildcard"] or info["reflective"]:
            reasons["dynamic-module-binding"] += 1
        if len(writes) != 1 or writes[0]["span"] != expected:
            reasons["ambiguous-module-binding"] += 1
        unknown.update(reasons)
        return not reasons

    anchor_module = module_name(symbol.path)
    if len(modules[anchor_module]) != 1 or symbol.path not in files:
        unknown["ambiguous-or-unavailable-defining-module"] += 1
        return result("unknown")
    anchor_info = files[symbol.path]
    definitions = [node for node in anchor_info["tree"].body
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                   and node.name == symbol.name]
    writes = anchor_info["evidence"]["writes"]
    if len(definitions) != 1 or len(writes) != 1 or not stable(symbol.path, writes[0]["span"]):
        unknown["defining-binding-unproven"] += 1
        return result("unknown")
    queue = deque([anchor_module])
    affected.add(anchor_module)
    while queue:
        current = queue.popleft()
        path = modules[current][0]
        info = files[path]
        for start, _end in info["evidence"]["loads"]:
            propose(path, start, "python-module-reference")
        for start in info["export_positions"]:
            propose(path, start, "python-literal-export")
        for consumer, node, alias, direct in edges.get(current, ()):
            consumer_info = files[consumer]
            if alias.name == "*":
                unknown["wildcard-export-frontier"] += 1
                continue
            if not direct:
                unknown["non-module-level-import-frontier"] += 1
                continue
            start = offset(consumer_info, alias)
            propose(consumer, start, "python-import-binding")
            if alias.asname is not None:
                if alias.asname == symbol.name:
                    protected.update((consumer, begin, end)
                                     for begin, end in consumer_info["evidence"]["loads"])
                continue
            expected = start, start + len(symbol.name.encode())
            target = module_name(consumer)
            if len(modules[target]) != 1:
                unknown["ambiguous-module-name"] += 1
            elif target not in affected and stable(consumer, expected):
                affected.add(target)
                queue.append(target)
    return result("unknown" if unknown else "resolved-static-subset")
