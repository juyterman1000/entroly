"""Exhaustive small-graph proof for atomic dependency-closure selection.

This benchmark measures one narrow safety property: whenever Entroly selects a
fragment, every transitively reachable *resolved* dependency must be selected
with it.  It does not measure answer quality, retrieval recall, model cost, or
superiority over another memory system.

The matrix is exhaustive over the declared graph families, node counts, token
costs, and budgets.  A copy of Entroly's former partial-add policy is retained
only as a regression control so the benchmark proves it can detect the defect
that motivated the production change.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

from entroly.context_receipts.models import (
    ContextIndex,
    DependencyLink,
    DocumentChunk,
    RankedChunk,
)
from entroly.context_receipts.selection import select_context

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "dependency_closure_integrity.json"
SCHEMA_VERSION = "entroly.dependency-closure-integrity.v1"
GRAPH_FAMILIES = ("chain", "star", "diamond", "cycle")
TOKEN_COSTS = (1, 2, 3)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _graph_edges(family: str, node_count: int) -> tuple[tuple[int, int], ...]:
    if family == "chain":
        edges = {(index, index + 1) for index in range(node_count - 1)}
    elif family == "star":
        edges = {(0, index) for index in range(1, node_count)}
    elif family == "diamond":
        if node_count < 4:
            edges = {(index, index + 1) for index in range(node_count - 1)}
        else:
            edges = {(0, 1), (0, 2), (1, 3), (2, 3)}
            edges.update((index, index + 1) for index in range(3, node_count - 1))
    elif family == "cycle":
        edges = {(index, (index + 1) % node_count) for index in range(node_count)}
    else:
        raise ValueError(f"unknown graph family: {family}")
    return tuple(sorted(edges))


def _unique_graphs(
    max_nodes: int,
) -> Iterable[tuple[int, tuple[tuple[int, int], ...], list[str]]]:
    for node_count in range(2, max_nodes + 1):
        unique: dict[tuple[tuple[int, int], ...], list[str]] = {}
        for family in GRAPH_FAMILIES:
            unique.setdefault(_graph_edges(family, node_count), []).append(family)
        for edges, families in sorted(unique.items()):
            yield node_count, edges, families


def _chunk(node: int, token_count: int) -> DocumentChunk:
    chunk_id = f"n{node}"
    text = f"evidence-{node}"
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{node}",
        source_path=f"node-{node}.md",
        title=f"Node {node}",
        section_heading=None,
        page_number=None,
        chunk_index=0,
        byte_start=0,
        byte_end=len(text),
        token_start=0,
        token_end=token_count,
        token_count=token_count,
        fingerprint=f"fingerprint-{node}",
        text=text,
    )


def _dependency(source: int, target: int) -> DependencyLink:
    return DependencyLink(
        source_chunk_id=f"n{source}",
        target_chunk_id=f"n{target}",
        relation_type="requires",
        evidence=f"n{source} requires n{target}",
        source_document_id=f"doc-{source}",
        target_document_id=f"doc-{target}",
        resolved=True,
    )


def _ranks(node_count: int) -> list[RankedChunk]:
    return [
        RankedChunk(
            chunk_id=f"n{node}",
            lexical_score=float(node_count - node),
            semantic_score=0.0,
            rerank_score=0.0,
            final_score=float(node_count - node),
            reasons=["deterministic benchmark rank"],
        )
        for node in range(node_count)
    ]


def _transitive_closure(
    edges: tuple[tuple[int, int], ...], node_count: int
) -> dict[str, set[str]]:
    outgoing: dict[str, set[str]] = {f"n{node}": set() for node in range(node_count)}
    for source, target in edges:
        outgoing[f"n{source}"].add(f"n{target}")

    closure: dict[str, set[str]] = {}
    for root in outgoing:
        seen: set[str] = set()
        pending = list(outgoing[root])
        while pending:
            target = pending.pop()
            if target in seen:
                continue
            seen.add(target)
            pending.extend(outgoing[target] - seen)
        closure[root] = seen
    return closure


def _partial_closure_count(selected: set[str], closure: dict[str, set[str]]) -> int:
    return sum(1 for chunk_id in selected if not closure[chunk_id].issubset(selected))


def _legacy_partial_select(
    costs: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
    budget: int,
) -> set[str]:
    """Reproduce the former select-root-then-try-direct-dependencies policy."""

    deps_by_source: dict[int, list[int]] = {}
    for source, target in edges:
        deps_by_source.setdefault(source, []).append(target)

    selected: set[int] = set()
    used = 0
    for root in range(len(costs)):
        if root in selected or used + costs[root] > budget:
            continue
        selected.add(root)
        used += costs[root]
        for target in sorted(deps_by_source.get(root, [])):
            if target in selected or used + costs[target] > budget:
                continue
            selected.add(target)
            used += costs[target]
    return {f"n{node}" for node in selected}


def run_matrix(*, max_nodes: int = 6) -> dict[str, Any]:
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least 2")

    cases = 0
    unique_graphs = 0
    legacy_partial_cases = 0
    legacy_partial_fragments = 0
    legacy_budget_violations = 0
    atomic_partial_cases = 0
    atomic_partial_fragments = 0
    atomic_budget_violations = 0
    atomic_bundle_omission_cases = 0

    for node_count, edges, _families in _unique_graphs(max_nodes):
        unique_graphs += 1
        closure = _transitive_closure(edges, node_count)
        dependencies = [_dependency(source, target) for source, target in edges]
        ranks = _ranks(node_count)

        for costs in itertools.product(TOKEN_COSTS, repeat=node_count):
            index = ContextIndex(
                schema_version="context-receipt.v1",
                documents=[],
                chunks=[
                    _chunk(node, token_count) for node, token_count in enumerate(costs)
                ],
                chunk_token_limit=sum(costs),
                chunk_overlap=0,
                source_fingerprints={},
            )
            for budget in range(1, sum(costs) + 1):
                cases += 1

                legacy_selected = _legacy_partial_select(costs, edges, budget)
                legacy_partial = _partial_closure_count(legacy_selected, closure)
                legacy_partial_fragments += legacy_partial
                legacy_partial_cases += int(legacy_partial > 0)
                legacy_tokens = sum(
                    costs[int(chunk_id[1:])] for chunk_id in legacy_selected
                )
                legacy_budget_violations += int(legacy_tokens > budget)

                atomic = select_context(
                    index,
                    ranks,
                    dependencies,
                    token_budget=budget,
                    max_omitted=node_count,
                )
                atomic_selected = {item.chunk_id for item in atomic.selected}
                atomic_partial = _partial_closure_count(atomic_selected, closure)
                atomic_partial_fragments += atomic_partial
                atomic_partial_cases += int(atomic_partial > 0)
                atomic_tokens = sum(item.token_count for item in atomic.selected)
                atomic_budget_violations += int(atomic_tokens > budget)
                atomic_bundle_omission_cases += int(
                    any(
                        "Dependency bundle omitted atomically due to budget" in warning
                        for warning in atomic.warnings
                    )
                )

    report = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": (
            "exhaustive small-graph dependency-closure integrity only; "
            "not answer quality, retrieval recall, model cost, or competitor superiority"
        ),
        "implementation_commit": _git_head(),
        "harness": {
            "path": "benchmarks/dependency_closure_integrity.py",
            "sha256": _sha256_bytes(Path(__file__).read_bytes()),
        },
        "matrix": {
            "graph_families": list(GRAPH_FAMILIES),
            "unique_graphs": unique_graphs,
            "node_count_min": 2,
            "node_count_max": max_nodes,
            "token_costs": list(TOKEN_COSTS),
            "budget_range": "1..sum(token_costs), inclusive",
            "cases": cases,
        },
        "legacy_partial_add_control": {
            "partial_closure_cases": legacy_partial_cases,
            "partial_closure_fragments": legacy_partial_fragments,
            "budget_violations": legacy_budget_violations,
        },
        "atomic_dependency_closure": {
            "partial_closure_cases": atomic_partial_cases,
            "partial_closure_fragments": atomic_partial_fragments,
            "budget_violations": atomic_budget_violations,
            "bundle_omission_cases": atomic_bundle_omission_cases,
            "closure_integrity_pass": atomic_partial_cases == 0,
            "hard_budget_pass": atomic_budget_violations == 0,
        },
        "limitations": [
            "The graphs are synthetic and small.",
            "All edges are resolved and trusted; unresolved references are tested separately.",
            "The benchmark measures a safety invariant, not downstream answer quality.",
            "The legacy control is a local reproduction of Entroly's former selector, not another product.",
            "No LLM, LoCoMo, MuSiQue, HotpotQA, agent trajectory, latency, cache, or billing result is measured.",
        ],
        "reproduce": (
            "python -m benchmarks.dependency_closure_integrity "
            "--max-nodes 6 --output benchmarks/results/dependency_closure_integrity.json"
        ),
    }
    verify_report(report)
    return report


def verify_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema_version")
    matrix = report.get("matrix")
    legacy = report.get("legacy_partial_add_control")
    atomic = report.get("atomic_dependency_closure")
    harness = report.get("harness")
    if not all(isinstance(value, dict) for value in (matrix, legacy, atomic, harness)):
        raise ValueError("report sections must be JSON objects")
    assert isinstance(matrix, dict)
    assert isinstance(legacy, dict)
    assert isinstance(atomic, dict)
    assert isinstance(harness, dict)

    numeric_values = [
        matrix.get("cases"),
        legacy.get("partial_closure_cases"),
        legacy.get("partial_closure_fragments"),
        legacy.get("budget_violations"),
        atomic.get("partial_closure_cases"),
        atomic.get("partial_closure_fragments"),
        atomic.get("budget_violations"),
        atomic.get("bundle_omission_cases"),
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in numeric_values
    ):
        raise ValueError("metrics must be finite non-negative numbers")
    if int(matrix["cases"]) < 1:
        raise ValueError("matrix must contain at least one case")
    if int(legacy["partial_closure_cases"]) < 1:
        raise ValueError("legacy control did not detect the historical defect")
    if int(legacy["budget_violations"]) != 0:
        raise ValueError("legacy control exceeded the hard budget")
    if int(atomic["partial_closure_cases"]) != 0:
        raise ValueError("atomic selector emitted a partial dependency closure")
    if int(atomic["budget_violations"]) != 0:
        raise ValueError("atomic selector exceeded the hard budget")
    if atomic.get("closure_integrity_pass") is not True:
        raise ValueError("closure_integrity_pass must be true")
    if atomic.get("hard_budget_pass") is not True:
        raise ValueError("hard_budget_pass must be true")
    if (
        not isinstance(report.get("implementation_commit"), str)
        or len(report["implementation_commit"]) != 40
    ):
        raise ValueError("implementation_commit must be a full Git commit")
    harness_path = ROOT / str(harness.get("path", ""))
    if not harness_path.is_file():
        raise ValueError("harness path is missing")
    if harness.get("sha256") != _sha256_bytes(harness_path.read_bytes()):
        raise ValueError("harness sha256 mismatch")
    limitations = report.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise ValueError("limitations must be non-empty")


def _write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(content, encoding="utf-8", newline="\n")
    output.with_suffix(output.suffix + ".sha256").write_text(
        _sha256_bytes(content.encode("utf-8")) + "\n",
        encoding="ascii",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nodes", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()

    if args.verify is not None:
        verify_report(json.loads(args.verify.read_text(encoding="utf-8")))
        print(f"verified {args.verify}")
        return 0

    report = run_matrix(max_nodes=args.max_nodes)
    _write_report(report, args.output)
    print(json.dumps(report["atomic_dependency_closure"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
