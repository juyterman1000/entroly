"""Dependency/call graph construction, impact analysis, and test localization."""
from __future__ import annotations

import posixpath
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Mapping

from .models import (
    CallEdge,
    ImpactReport,
    RepositoryIndex,
    RepositoryLimits,
    Symbol,
    TestCandidate,
    normalize_relative,
)
from .parsers import ParsedFile, module_name


def _module_paths(parsed: Mapping[str, ParsedFile]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(parsed):
        name = module_name(path)
        if name:
            grouped[name].append(path)
    return {name: tuple(sorted(paths)) for name, paths in sorted(grouped.items())}


def _unique_module(paths: Mapping[str, tuple[str, ...]], name: str) -> str | None:
    matches = paths.get(name, ())
    return matches[0] if len(matches) == 1 else None


def resolve_imports(parsed: Mapping[str, ParsedFile]) -> dict[str, tuple[str, ...]]:
    modules = _module_paths(parsed)
    result: dict[str, tuple[str, ...]] = {}
    for path, item in sorted(parsed.items()):
        found: set[str] = set()
        for imported in item.imports:
            candidate = imported
            while candidate:
                target = _unique_module(modules, candidate)
                if target and target != path:
                    found.add(target)
                    break
                candidate = candidate.rpartition(".")[0]
            if imported.startswith(("./", "../")):
                base = posixpath.normpath(posixpath.join(posixpath.dirname(path), imported))
                for suffix in (
                    ".ts", ".tsx", ".js", ".jsx",
                    "/index.ts", "/index.tsx", "/index.js", "/index.jsx",
                ):
                    target = normalize_relative(base + suffix)
                    if target in parsed and target != path:
                        found.add(target)
                        break
        result[path] = tuple(sorted(found))
    return result


def resolve_calls(
    parsed: Mapping[str, ParsedFile],
    symbols: Mapping[str, Symbol],
    limits: RepositoryLimits,
) -> tuple[CallEdge, ...]:
    by_name: dict[str, list[Symbol]] = defaultdict(list)
    by_path_name: dict[tuple[str, str], list[Symbol]] = defaultdict(list)
    for symbol in symbols.values():
        by_name[symbol.name].append(symbol)
        by_path_name[(symbol.path, symbol.name)].append(symbol)
    modules = _module_paths(parsed)
    edges: set[CallEdge] = set()

    def imported(item: ParsedFile, owner: str, name: str) -> list[Symbol]:
        target = item.import_aliases.get(owner or name)
        if not target:
            return []
        if owner:
            module, symbol_name = target, name
        else:
            module, separator, symbol_name = target.rpartition(".")
            if not separator:
                return []
        path = _unique_module(modules, module)
        return list(by_path_name.get((path, symbol_name), ())) if path else []

    limit_reached = False
    for path, item in sorted(parsed.items()):
        if limit_reached:
            break
        for caller_id, target, line in item.calls:
            owner, dot, name = target.rpartition(".")
            if not dot:
                owner, name = "", target
            candidates = list(by_path_name.get((path, name), ()))
            candidates.extend(imported(item, owner, name))
            if not candidates and not item.import_aliases.get(owner or name):
                if len(by_name.get(name, ())) == 1:
                    candidates.extend(by_name[name])
            unique = {item.symbol_id: item for item in candidates}
            if len(unique) != 1:
                continue
            callee = next(iter(unique.values()))
            caller = caller_id or f"{path}::<module>::module"
            edges.add(CallEdge(caller, callee.symbol_id, path, line))
            if len(edges) >= limits.max_edges:
                limit_reached = True
                break
    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.caller_id, edge.callee_id, edge.path, edge.line
            ),
        )
    )


def analyze_change_impact(
    index: RepositoryIndex,
    changed_paths: Iterable[str],
    *,
    max_depth: int = 4,
    max_impacted_paths: int = 5_000,
) -> ImpactReport:
    if max_depth < 0 or max_impacted_paths <= 0:
        raise ValueError("max_depth must be non-negative and max_impacted_paths positive")
    requested = tuple(
        sorted(
            path
            for raw in changed_paths
            if (path := normalize_relative(raw)) in index.files
        )
    )
    truncated = len(requested) > max_impacted_paths
    seeds = requested[:max_impacted_paths]
    impacted = set(seeds)
    reasons: dict[str, set[str]] = defaultdict(set)
    queue = deque((path, 0) for path in seeds)
    for path in seeds:
        reasons[path].add("changed")
    while queue:
        path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for dependent in index.dependents_of(path):
            if dependent not in impacted:
                if len(impacted) >= max_impacted_paths:
                    truncated = True
                    continue
                impacted.add(dependent)
                queue.append((dependent, depth + 1))
            reasons[dependent].add(f"imports:{path}")
    changed_symbols = {
        symbol.symbol_id
        for path in seeds
        for symbol in index.symbols_for_path(path)
    }
    impacted_symbols = set(changed_symbols)
    symbol_queue = deque((symbol_id, 0) for symbol_id in sorted(changed_symbols))
    while symbol_queue:
        symbol_id, depth = symbol_queue.popleft()
        if depth >= max_depth:
            continue
        for caller in index.callers_of(symbol_id):
            if caller.path not in impacted:
                if len(impacted) >= max_impacted_paths:
                    truncated = True
                    continue
                impacted.add(caller.path)
            reasons[caller.path].add(f"calls:{symbol_id}")
            if caller.symbol_id not in impacted_symbols:
                impacted_symbols.add(caller.symbol_id)
                symbol_queue.append((caller.symbol_id, depth + 1))
    return ImpactReport(
        seeds, tuple(sorted(impacted)), tuple(sorted(impacted_symbols)),
        {path: tuple(sorted(values)) for path, values in sorted(reasons.items())},
        truncated,
    )


TOKEN_STOPWORDS = frozenset(
    {
        "app", "src", "lib", "test", "tests", "spec", "specs",
        "py", "pyi", "rs", "js", "jsx", "ts", "tsx", "index",
    }
)


def _tokens(value: str) -> set[str]:
    expanded = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", expanded.replace("-", "_"))
        if len(token) > 1 and token.lower() not in TOKEN_STOPWORDS
    }


def localize_tests(
    index: RepositoryIndex,
    changed_paths: Iterable[str],
    *,
    limit: int = 20,
) -> tuple[TestCandidate, ...]:
    if limit <= 0:
        return ()
    report = analyze_change_impact(index, changed_paths)
    changed = set(report.changed_paths)
    changed_symbols = [symbol for path in changed for symbol in index.symbols_for_path(path)]
    changed_tokens = set().union(
        *(_tokens(path) for path in changed),
        *(_tokens(symbol.name) for symbol in changed_symbols),
    ) if changed or changed_symbols else set()
    result: list[TestCandidate] = []
    for test_path in index.test_paths:
        score = 0.0
        reasons: list[str] = []
        direct = sorted(set(index.file_dependencies.get(test_path, ())) & changed)
        if direct:
            score += 100.0 + 10.0 * len(direct)
            reasons.append("direct-import:" + ",".join(direct))
        if test_path in report.impacted_paths:
            score += 50.0
            reasons.append("reverse-impact")
        called = {
            edge.callee_id for edge in index.call_edges
            if edge.path == test_path and edge.callee_id in report.impacted_symbols
        }
        if called:
            score += 80.0 + 5.0 * len(called)
            reasons.append("calls-changed-symbol")
        overlap = _tokens(test_path) & changed_tokens
        if overlap:
            score += min(20.0, 4.0 * len(overlap))
            reasons.append("name-overlap:" + ",".join(sorted(overlap)))
        if any(Path(test_path).parent == Path(path).parent for path in changed):
            score += 5.0
            reasons.append("co-located")
        if score > 0:
            result.append(TestCandidate(test_path, score, tuple(reasons)))
    result.sort(key=lambda item: (-item.score, item.path))
    return tuple(result[:limit])
