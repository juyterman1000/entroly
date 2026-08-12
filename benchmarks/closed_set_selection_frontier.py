"""Exhaustive small-graph benchmark for Entroly's closed-set selector.

This benchmark measures one narrow internal objective: maximize the sum of
positive retrieval scores while respecting a hard token budget and transitive
resolved dependency closure. It does not measure answer quality, recall,
latency, cost, or superiority over another memory system.

An independent brute-force oracle evaluates every dependency-closed subset.
The former rank-order atomic selector is retained as a local regression control.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.dependency_closure_integrity import (
    GRAPH_FAMILIES,
    TOKEN_COSTS,
    _chunk,
    _dependency,
    _ranks,
    _transitive_closure,
    _unique_graphs,
)
from entroly.context_receipts.models import ContextIndex, DocumentChunk, stable_hash
from entroly.context_receipts.selection import select_context

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "closed_set_selection_frontier.json"
SCHEMA_VERSION = "entroly.closed-set-selection-frontier.v1"
_EPSILON = 1e-9


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_source(path: Path) -> str:
    """Hash tracked source canonically across Git checkout line endings."""
    return _sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _closed_bundles(
    closure: dict[str, set[str]], node_count: int
) -> dict[str, set[str]]:
    return {
        f"n{node}": {f"n{node}", *closure[f"n{node}"]}
        for node in range(node_count)
    }


def _benchmark_chunk(node: int, token_count: int) -> DocumentChunk:
    chunk = _chunk(node, token_count)
    # The integrity benchmark intentionally ignores retrieval redundancy. This
    # frontier benchmark needs semantically distinct fragments so its oracle
    # and production selector optimize the same declared feasible set.
    chunk.text = f"evidence nodeword{node}"
    chunk.byte_end = len(chunk.text)
    return chunk


def _score(selected: set[str], node_count: int) -> float:
    return float(sum(node_count - int(chunk_id[1:]) for chunk_id in selected))


def _tokens(selected: set[str], costs: tuple[int, ...]) -> int:
    return sum(costs[int(chunk_id[1:])] for chunk_id in selected)


def _rank_order_atomic_control(
    bundles: dict[str, set[str]], costs: tuple[int, ...], budget: int
) -> set[str]:
    selected: set[str] = set()
    for node in range(len(costs)):
        root = f"n{node}"
        missing = bundles[root] - selected
        if _tokens(missing, costs) <= budget - _tokens(selected, costs):
            selected.update(missing)
    return selected


def _exact_oracle(
    bundles: dict[str, set[str]],
    costs: tuple[int, ...],
    budget: int,
    node_count: int,
) -> tuple[set[str], float]:
    roots = [f"n{node}" for node in range(node_count)]
    best_selected: set[str] = set()
    best_score = 0.0
    best_tokens = 0
    for mask in range(1, 1 << node_count):
        selected: set[str] = set()
        for position, root in enumerate(roots):
            if mask & (1 << position):
                selected.update(bundles[root])
        used = _tokens(selected, costs)
        if used > budget:
            continue
        score = _score(selected, node_count)
        if score > best_score + _EPSILON or (
            abs(score - best_score) <= _EPSILON
            and (not best_selected or used < best_tokens)
        ):
            best_selected = selected
            best_score = score
            best_tokens = used
    return best_selected, best_score


def _certificate_hash_valid(certificate: dict[str, Any]) -> bool:
    payload = copy.deepcopy(certificate)
    claimed = payload.pop("certificate_sha256", None)
    return isinstance(claimed, str) and claimed == stable_hash(payload)


def run_matrix(*, max_nodes: int = 6) -> dict[str, Any]:
    if max_nodes < 2:
        raise ValueError("max_nodes must be at least 2")

    cases = 0
    unique_graphs = 0
    rank_suboptimal_cases = 0
    production_strict_improvements = 0
    production_regressions = 0
    production_optimal_cases = 0
    production_budget_violations = 0
    production_partial_closures = 0
    exact_certificates = 0
    invalid_certificates = 0
    aggregate_rank_regret = 0.0
    aggregate_production_regret = 0.0
    maximum_score_improvement = 0.0

    for node_count, edges, _families in _unique_graphs(max_nodes):
        unique_graphs += 1
        closure = _transitive_closure(edges, node_count)
        bundles = _closed_bundles(closure, node_count)
        dependencies = [_dependency(source, target) for source, target in edges]
        ranks = _ranks(node_count)

        for costs in itertools.product(TOKEN_COSTS, repeat=node_count):
            index = ContextIndex(
                schema_version="context-receipt.v1",
                documents=[],
                chunks=[
                    _benchmark_chunk(node, token_count)
                    for node, token_count in enumerate(costs)
                ],
                chunk_token_limit=sum(costs),
                chunk_overlap=0,
                source_fingerprints={},
            )
            for budget in range(1, sum(costs) + 1):
                cases += 1
                rank_selected = _rank_order_atomic_control(bundles, costs, budget)
                rank_score = _score(rank_selected, node_count)
                oracle_selected, oracle_score = _exact_oracle(
                    bundles, costs, budget, node_count
                )
                del oracle_selected

                result = select_context(
                    index,
                    ranks,
                    dependencies,
                    token_budget=budget,
                    max_omitted=node_count,
                )
                production_selected = {item.chunk_id for item in result.selected}
                production_score = _score(production_selected, node_count)
                production_tokens = _tokens(production_selected, costs)
                partial = sum(
                    1
                    for chunk_id in production_selected
                    if not closure[chunk_id].issubset(production_selected)
                )

                rank_regret = max(0.0, oracle_score - rank_score)
                production_regret = max(0.0, oracle_score - production_score)
                improvement = production_score - rank_score
                aggregate_rank_regret += rank_regret
                aggregate_production_regret += production_regret
                maximum_score_improvement = max(maximum_score_improvement, improvement)
                rank_suboptimal_cases += int(rank_regret > _EPSILON)
                production_strict_improvements += int(improvement > _EPSILON)
                production_regressions += int(improvement < -_EPSILON)
                production_optimal_cases += int(production_regret <= _EPSILON)
                production_budget_violations += int(production_tokens > budget)
                production_partial_closures += partial
                exact_certificates += int(
                    result.certificate.get("optimality")
                    == "exact_for_internal_relevance_objective"
                )
                invalid_certificates += int(
                    not _certificate_hash_valid(result.certificate)
                )
                if abs(
                    float(result.certificate["objective"]["selected_score"])
                    - production_score
                ) > _EPSILON:
                    invalid_certificates += 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": (
            "exhaustive small-graph optimality for Entroly's declared internal "
            "retrieval-score objective only; not answer quality, recall, cost, "
            "latency, or competitor superiority"
        ),
        "implementation_commit": _git_head(),
        "harness": {
            "path": "benchmarks/closed_set_selection_frontier.py",
            "sha256": _sha256_source(Path(__file__)),
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
        "rank_order_atomic_control": {
            "suboptimal_cases": rank_suboptimal_cases,
            "aggregate_regret": round(aggregate_rank_regret, 6),
        },
        "certified_closed_set_selector": {
            "strict_improvements_vs_rank_order": production_strict_improvements,
            "regressions_vs_rank_order": production_regressions,
            "optimal_cases": production_optimal_cases,
            "aggregate_regret": round(aggregate_production_regret, 6),
            "maximum_score_improvement": round(maximum_score_improvement, 6),
            "budget_violations": production_budget_violations,
            "partial_dependency_closures": production_partial_closures,
            "exact_certificates": exact_certificates,
            "invalid_certificates": invalid_certificates,
            "no_regression_pass": production_regressions == 0,
            "declared_objective_optimality_pass": production_optimal_cases == cases,
            "hard_budget_pass": production_budget_violations == 0,
            "dependency_closure_pass": production_partial_closures == 0,
            "certificate_integrity_pass": invalid_certificates == 0,
        },
        "limitations": [
            "The graphs are synthetic and contain at most the declared maximum node count.",
            "All retrieval scores are positive integers in deterministic rank order.",
            "All dependency edges are resolved and trusted.",
            "The exact oracle optimizes Entroly's internal score, not downstream answer quality.",
            "The rank-order control is Entroly's prior local policy, not another product.",
            "No LLM, LoCoMo, MuSiQue, HotpotQA, latency, cache, billing, or competitor result is measured.",
        ],
        "reproduce": (
            "python -m benchmarks.closed_set_selection_frontier "
            "--max-nodes 6 --output "
            "benchmarks/results/closed_set_selection_frontier.json"
        ),
    }
    verify_report(report)
    return report


def verify_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected schema_version")
    matrix = report.get("matrix")
    control = report.get("rank_order_atomic_control")
    production = report.get("certified_closed_set_selector")
    harness = report.get("harness")
    if not all(
        isinstance(value, dict) for value in (matrix, control, production, harness)
    ):
        raise ValueError("report sections must be JSON objects")
    assert isinstance(matrix, dict)
    assert isinstance(control, dict)
    assert isinstance(production, dict)
    assert isinstance(harness, dict)

    numeric_values = [
        matrix.get("cases"),
        control.get("suboptimal_cases"),
        control.get("aggregate_regret"),
        production.get("strict_improvements_vs_rank_order"),
        production.get("regressions_vs_rank_order"),
        production.get("optimal_cases"),
        production.get("aggregate_regret"),
        production.get("maximum_score_improvement"),
        production.get("budget_violations"),
        production.get("partial_dependency_closures"),
        production.get("exact_certificates"),
        production.get("invalid_certificates"),
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in numeric_values
    ):
        raise ValueError("metrics must be finite non-negative numbers")
    cases = int(matrix["cases"])
    if cases < 1:
        raise ValueError("matrix must contain at least one case")
    if int(matrix.get("node_count_max", 0)) >= 5 and int(
        control["suboptimal_cases"]
    ) < 1:
        raise ValueError("rank-order control did not expose the optimization gap")
    if int(production["regressions_vs_rank_order"]) != 0:
        raise ValueError("certified selector regressed against rank order")
    if int(production["optimal_cases"]) != cases:
        raise ValueError("certified selector missed the exact oracle")
    if float(production["aggregate_regret"]) != 0.0:
        raise ValueError("certified selector has non-zero aggregate regret")
    if int(production["budget_violations"]) != 0:
        raise ValueError("certified selector exceeded the hard budget")
    if int(production["partial_dependency_closures"]) != 0:
        raise ValueError("certified selector emitted a partial dependency closure")
    if int(production["exact_certificates"]) != cases:
        raise ValueError("bounded matrix did not emit exact certificates")
    if int(production["invalid_certificates"]) != 0:
        raise ValueError("selection certificate integrity failed")
    for key in (
        "no_regression_pass",
        "declared_objective_optimality_pass",
        "hard_budget_pass",
        "dependency_closure_pass",
        "certificate_integrity_pass",
    ):
        if production.get(key) is not True:
            raise ValueError(f"{key} must be true")
    if (
        not isinstance(report.get("implementation_commit"), str)
        or len(report["implementation_commit"]) != 40
    ):
        raise ValueError("implementation_commit must be a full Git commit")
    harness_path = ROOT / str(harness.get("path", ""))
    if not harness_path.is_file():
        raise ValueError("harness path is missing")
    if harness.get("sha256") != _sha256_source(harness_path):
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
    print(json.dumps(report["certified_closed_set_selector"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
