"""Freshness-checked structural code health and navigability evidence.

This module deliberately separates parser facts from heuristic policy.  Exact
source ranges, hashes, graph edges, and ambiguity counts are facts.  Thresholds,
grades, and recommendations are transparent policy and are labelled as such.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from entroly.tree_sitter_support import extract_structural_profiles

from .models import RepositoryIndex, Symbol

VERIFIED_HEALTH_SCHEMA_VERSION = "entroly.verified-code-health.v1"

_THRESHOLDS = {
    "cyclomatic_complexity": 10,
    "cognitive_complexity": 15,
    "max_control_nesting": 4,
    "parameter_count": 6,
    "symbol_lines": 80,
    "file_lines": 800,
}


@dataclass(frozen=True)
class _Metrics:
    decision_points: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    max_control_nesting: int
    parameter_count: int
    return_points: int


class _PythonProfiles:
    """Build qualified Python profiles without charging nested functions twice."""

    def __init__(self) -> None:
        self.items: dict[str, _Metrics] = {}
        self._parents: list[str] = []
        self._active: list[dict[str, int]] = []

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._function(node)
            return
        if isinstance(node, ast.ClassDef):
            self._parents.append(node.name)
            for child in node.body:
                self.visit(child)
            self._parents.pop()
            return
        increments = 0
        if self._active:
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp)):
                increments = 1
            elif isinstance(node, ast.ExceptHandler):
                increments = 1
            elif isinstance(node, ast.match_case):
                increments = 1
            elif isinstance(node, ast.comprehension):
                increments = 1 + len(node.ifs)
            elif isinstance(node, ast.BoolOp):
                increments = max(1, len(node.values) - 1)
            state = self._active[-1]
            if increments:
                state["decisions"] += increments
                state["cognitive"] += increments * (1 + state["nesting"])
                state["nesting"] += 1
                state["max_nesting"] = max(state["max_nesting"], state["nesting"])
            if isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Raise)):
                state["returns"] += 1
        for child in ast.iter_child_nodes(node):
            self.visit(child)
        if increments:
            self._active[-1]["nesting"] -= 1

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = ".".join((*self._parents, node.name))
        args = node.args
        parameter_count = (
            len(args.posonlyargs)
            + len(args.args)
            + len(args.kwonlyargs)
            + int(args.vararg is not None)
            + int(args.kwarg is not None)
        )
        if self._parents and parameter_count and args.args and args.args[0].arg in {"self", "cls"}:
            parameter_count -= 1
        state = {
            "decisions": 0,
            "cognitive": 0,
            "max_nesting": 0,
            "nesting": 0,
            "returns": 0,
        }
        self._active.append(state)
        self._parents.append(node.name)
        for child in node.body:
            self.visit(child)
        self._parents.pop()
        self._active.pop()
        self.items[qualified] = _Metrics(
            decision_points=state["decisions"],
            cyclomatic_complexity=1 + state["decisions"],
            cognitive_complexity=state["cognitive"],
            max_control_nesting=state["max_nesting"],
            parameter_count=parameter_count,
            return_points=state["returns"],
        )


def _python_profiles(text: str) -> dict[str, _Metrics]:
    try:
        tree = ast.parse(text, type_comments=True)
    except (SyntaxError, ValueError, MemoryError, RecursionError, UnicodeError):
        return {}
    visitor = _PythonProfiles()
    visitor.visit(tree)
    return visitor.items


def _source_files(
    root: Path,
    index: RepositoryIndex,
) -> tuple[dict[str, bytes], Counter[str]]:
    verified: dict[str, bytes] = {}
    omissions: Counter[str] = Counter()
    for path, record in sorted(index.files.items()):
        try:
            candidate = (root / path).resolve(strict=True)
            candidate.relative_to(root)
            raw = candidate.read_bytes()
        except (OSError, RuntimeError, ValueError):
            omissions["unsafe-or-unreadable"] += 1
            continue
        if hashlib.sha256(raw).hexdigest() != record.sha256:
            omissions["stale-index"] += 1
            continue
        verified[path] = raw
    return verified, omissions


def _tree_profiles(text: str, path: str) -> dict[tuple[int, int, str], _Metrics]:
    profiles = extract_structural_profiles(text, path) or ()
    return {
        (item.start_byte, item.end_byte, item.name): _Metrics(
            item.decision_points,
            item.cyclomatic_complexity,
            item.cognitive_complexity,
            item.max_control_nesting,
            item.parameter_count,
            item.return_points,
        )
        for item in profiles
    }


def _strongly_connected_components(
    graph: dict[str, tuple[str, ...]],
) -> list[tuple[str, ...]]:
    """Deterministic Tarjan SCCs over known file-import edges."""
    index_counter = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def connect(node: str) -> None:
        nonlocal index_counter
        indexes[node] = index_counter
        lowlinks[node] = index_counter
        index_counter += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(graph.get(node, ())):
            if target not in graph:
                continue
            if target not in indexes:
                connect(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[target])
        if lowlinks[node] == indexes[node]:
            members: list[str] = []
            while stack:
                member = stack.pop()
                on_stack.remove(member)
                members.append(member)
                if member == node:
                    break
            has_self_loop = len(members) == 1 and node in graph.get(node, ())
            if len(members) > 1 or has_self_loop:
                components.append(tuple(sorted(members)))

    for node in sorted(graph):
        if node not in indexes:
            connect(node)
    return sorted(components, key=lambda item: (-len(item), item))


def _finding(
    symbol: Symbol,
    record_sha256: str,
    metric: str,
    value: int,
    threshold: int,
) -> dict[str, object]:
    ratio = value / max(1, threshold)
    severity = "high" if ratio >= 2 else "medium" if ratio >= 1.35 else "low"
    return {
        "finding_id": hashlib.sha256(
            f"{symbol.symbol_id}\0{metric}\0{value}\0{threshold}".encode("utf-8")
        ).hexdigest()[:20],
        "kind": "structural-threshold",
        "severity": severity,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "symbol_id": symbol.symbol_id,
        "qualified_name": symbol.qualified_name,
        "path": symbol.path,
        "line_start": symbol.line_start,
        "line_end": symbol.line_end,
        "start_byte": symbol.start_byte,
        "end_byte": symbol.end_byte,
        "source_sha256": record_sha256,
        "evidence_sha256": symbol.evidence_sha256,
        "analysis_backend": symbol.parse_backend,
        "confidence": "parser-exact" if symbol.parse_backend in {"python-ast", "tree-sitter"} else "unsupported",
        "policy": "configurable-threshold-v1",
    }


def _commit(payload: dict[str, object]) -> dict[str, object]:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    receipt = payload["receipt"]
    assert isinstance(receipt, dict)
    receipt["code_health_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def build_verified_code_health(
    root: Path,
    index: RepositoryIndex,
    *,
    index_digest: str,
    max_findings: int = 500,
    max_symbols: int = 2_000,
) -> dict[str, object]:
    """Analyze exact parser spans and graph navigability without remote calls."""
    finding_limit = max(1, min(int(max_findings), 10_000))
    symbol_limit = max(1, min(int(max_symbols), 20_000))
    verified, omissions = _source_files(root, index)
    findings: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    languages: dict[str, Counter[str]] = defaultdict(Counter)
    high_complexity = 0
    oversized = 0
    eligible = 0

    symbols_by_path: dict[str, list[Symbol]] = defaultdict(list)
    for symbol in index.symbols.values():
        symbols_by_path[symbol.path].append(symbol)

    for path, raw in sorted(verified.items()):
        record = index.files[path]
        text = raw.decode("utf-8", errors="surrogateescape")
        python = _python_profiles(text) if record.language == "python" else {}
        structural = _tree_profiles(text, path) if record.language != "python" else {}
        for symbol in sorted(symbols_by_path.get(path, ()), key=lambda item: item.symbol_id):
            if symbol.kind in {"class", "struct", "enum", "interface", "trait", "module", "namespace", "type"}:
                continue
            eligible += 1
            metrics = python.get(symbol.qualified_name)
            if metrics is None:
                metrics = structural.get((symbol.start_byte, symbol.end_byte, symbol.name))
            if metrics is None:
                languages[record.language]["unprofiled"] += 1
                continue
            if not (0 <= symbol.start_byte < symbol.end_byte <= len(raw)):
                languages[record.language]["invalid-span"] += 1
                continue
            evidence_sha256 = hashlib.sha256(raw[symbol.start_byte:symbol.end_byte]).hexdigest()
            if symbol.evidence_sha256 and evidence_sha256 != symbol.evidence_sha256:
                languages[record.language]["stale-symbol-span"] += 1
                continue
            languages[record.language]["profiled"] += 1
            line_count = max(1, symbol.line_end - symbol.line_start + 1)
            values = {
                "cyclomatic_complexity": metrics.cyclomatic_complexity,
                "cognitive_complexity": metrics.cognitive_complexity,
                "max_control_nesting": metrics.max_control_nesting,
                "parameter_count": metrics.parameter_count,
                "symbol_lines": line_count,
            }
            if metrics.cyclomatic_complexity > _THRESHOLDS["cyclomatic_complexity"]:
                high_complexity += 1
            if line_count > _THRESHOLDS["symbol_lines"]:
                oversized += 1
            for metric, value in values.items():
                threshold = _THRESHOLDS[metric]
                if value > threshold:
                    findings.append(_finding(symbol, record.sha256, metric, value, threshold))
            profiles.append({
                "symbol_id": symbol.symbol_id,
                "qualified_name": symbol.qualified_name,
                "path": path,
                "language": record.language,
                "line_start": symbol.line_start,
                "line_end": symbol.line_end,
                "start_byte": symbol.start_byte,
                "end_byte": symbol.end_byte,
                "source_sha256": record.sha256,
                "evidence_sha256": evidence_sha256,
                "analysis_backend": symbol.parse_backend,
                "decision_points": metrics.decision_points,
                "cyclomatic_complexity": metrics.cyclomatic_complexity,
                "cognitive_complexity": metrics.cognitive_complexity,
                "max_control_nesting": metrics.max_control_nesting,
                "parameter_count": metrics.parameter_count,
                "return_points": metrics.return_points,
                "symbol_lines": line_count,
            })

        if record.line_count > _THRESHOLDS["file_lines"]:
            findings.append({
                "finding_id": hashlib.sha256(f"{path}\0file_lines".encode()).hexdigest()[:20],
                "kind": "structural-threshold",
                "severity": "medium",
                "metric": "file_lines",
                "value": record.line_count,
                "threshold": _THRESHOLDS["file_lines"],
                "path": path,
                "line_start": 1,
                "line_end": record.line_count,
                "start_byte": 0,
                "end_byte": len(raw),
                "source_sha256": record.sha256,
                "evidence_sha256": record.sha256,
                "analysis_backend": "file-record",
                "confidence": "hash-exact",
                "policy": "configurable-threshold-v1",
            })

    cycles = _strongly_connected_components(index.file_dependencies)
    cyclic_paths = {path for cycle in cycles for path in cycle}
    reverse_dependencies: Counter[str] = Counter()
    for dependencies in index.file_dependencies.values():
        reverse_dependencies.update(dependencies)
    coupling = sorted(
        (
            {
                "path": path,
                "fan_in": reverse_dependencies[path],
                "fan_out": len(index.file_dependencies.get(path, ())),
                "source_sha256": index.files[path].sha256,
            }
            for path in index.files
        ),
        key=lambda item: (-(int(item["fan_in"]) + int(item["fan_out"])), str(item["path"])),
    )
    unresolved_by_reason = Counter(call.reason for call in index.unresolved_calls)
    resolved_count = len(index.call_edges)
    unresolved_count = len(index.unresolved_calls)
    call_total = resolved_count + unresolved_count
    unresolved_rate = unresolved_count / max(1, call_total)
    profiled_count = len(profiles)
    parser_error_files = sum(record.parse_error is not None for record in index.files.values())
    cyclic_fraction = len(cyclic_paths) / max(1, len(index.files))
    complexity_fraction = high_complexity / max(1, profiled_count)
    oversized_fraction = oversized / max(1, profiled_count)
    parse_error_fraction = parser_error_files / max(1, len(index.files))
    score = max(0.0, 100.0 - (
        40.0 * unresolved_rate
        + 20.0 * cyclic_fraction
        + 20.0 * complexity_fraction
        + 10.0 * oversized_fraction
        + 10.0 * parse_error_fraction
    ))
    grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "D" if score >= 60 else "F"

    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda item: (
        severity_order.get(str(item.get("severity")), 3),
        -float(item.get("value", 0)) / max(1.0, float(item.get("threshold", 1))),
        str(item.get("path", "")),
        str(item.get("finding_id", "")),
    ))
    omitted_findings = max(0, len(findings) - finding_limit)
    omitted_profiles = max(0, len(profiles) - symbol_limit)
    profiles.sort(key=lambda item: (
        -int(item["cognitive_complexity"]),
        -int(item["cyclomatic_complexity"]),
        str(item["symbol_id"]),
    ))
    source_set_sha256 = hashlib.sha256(
        "\n".join(
            f"{path}\0{index.files[path].sha256}" for path in sorted(verified)
        ).encode("utf-8")
    ).hexdigest()
    payload: dict[str, object] = {
        "schema_version": VERIFIED_HEALTH_SCHEMA_VERSION,
        "index_digest": index_digest,
        "policy": {
            "name": "transparent-structural-health-v1",
            "thresholds": dict(_THRESHOLDS),
            "score_formula": (
                "100 - 40*unresolved_call_rate - 20*cyclic_file_fraction - "
                "20*high_complexity_symbol_fraction - 10*oversized_symbol_fraction - "
                "10*parse_error_file_fraction"
            ),
            "interpretation": "ranking-and-review-aid-not-a-proof-of-defect",
        },
        "summary": {
            "code_health_score": round(score, 2),
            "health_grade": grade,
            "verified_files": len(verified),
            "indexed_files": len(index.files),
            "eligible_symbols": eligible,
            "profiled_symbols": profiled_count,
            "finding_count": len(findings),
            "architecture_cycle_count": len(cycles),
            "cyclic_file_count": len(cyclic_paths),
            "resolved_calls": resolved_count,
            "unresolved_calls": unresolved_count,
            "unresolved_call_rate": round(unresolved_rate, 6),
            "parser_error_files": parser_error_files,
        },
        "coverage_by_language": {
            language: dict(sorted(counts.items()))
            for language, counts in sorted(languages.items())
        },
        "unresolved_calls_by_reason": dict(sorted(unresolved_by_reason.items())),
        "architecture_cycles": [
            {
                "cycle_id": hashlib.sha256("\n".join(cycle).encode()).hexdigest()[:20],
                "paths": list(cycle),
                "member_count": len(cycle),
                "source_sha256": {path: index.files[path].sha256 for path in cycle},
            }
            for cycle in cycles
        ],
        "coupling_hotspots": coupling[:100],
        "symbol_profiles": profiles[:symbol_limit],
        "findings": findings[:finding_limit],
        "receipt": {
            "freshness": "verified-against-indexed-source-sha256",
            "source_set_sha256": source_set_sha256,
            "verified_source_count": len(verified),
            "source_omissions_by_reason": dict(sorted(omissions.items())),
            "omitted_findings": omitted_findings,
            "omitted_symbol_profiles": omitted_profiles,
            "remote_calls": 0,
            "metric_semantics": "python-ast-or-tree-sitter-structural",
            "commitment_scope": "payload-excluding-generation-command-and-code-health-sha256",
        },
    }
    return _commit(payload)


def verify_code_health_commitment(payload: dict[str, object]) -> bool:
    """Verify the report commitment without accessing the workspace."""
    try:
        candidate = copy.deepcopy(payload)
        candidate.pop("generation", None)
        candidate.pop("command", None)
        receipt = candidate["receipt"]
        if not isinstance(receipt, dict):
            return False
        expected = str(receipt.pop("code_health_sha256"))
        canonical = json.dumps(
            candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest() == expected
    except (KeyError, TypeError, ValueError):
        return False


__all__ = [
    "VERIFIED_HEALTH_SCHEMA_VERSION",
    "build_verified_code_health",
    "verify_code_health_commitment",
]
