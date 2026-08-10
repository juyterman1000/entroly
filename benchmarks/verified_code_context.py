#!/usr/bin/env python3
"""Reproducible evidence benchmark for verified repository context."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entroly.repository_intelligence import (  # noqa: E402
    RepositoryIntelligenceService,
    verify_symbol_graph_commitment,
)

SCHEMA_VERSION = "entroly.verified-code-context-benchmark.v1"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "verified_code_context.json"

FIXTURES = {
    "python/service.py": (
        "def authorize(card):\n    return bool(card)\n\n"
        "def charge_card(card):\n    return authorize(card)\n"
    ),
    "python/a.py": "def duplicate():\n    return 'a'\n",
    "python/b.py": "def duplicate():\n    return 'b'\n",
    "python/ambiguous.py": "def invoke():\n    return duplicate()\n",
    "rust/lib.rs": (
        "fn helper() {}\nstruct Worker;\n"
        "impl Worker { fn run(&self) { helper(); } }\n"
    ),
    "typescript/index.ts": (
        "function helper(): number { return 1; }\n"
        "export function run(): number { return helper(); }\n"
    ),
    "go/main.go": "package main\nfunc helper() {}\nfunc run() { helper() }\n",
    "java/Sample.java": (
        "class Sample { static void helper() {} "
        "static void run() { helper(); } }\n"
    ),
}

GOLD_EDGES = (
    ("python/service.py::charge_card::function", "python/service.py::authorize::function"),
    ("rust/lib.rs::Worker.run::fn", "rust/lib.rs::helper::fn"),
    ("typescript/index.ts::run::function", "typescript/index.ts::helper::function"),
    ("go/main.go::run::function", "go/main.go::helper::function"),
    ("java/Sample.java::Sample.run::method", "java/Sample.java::Sample.helper::method"),
)

IMPLEMENTATION_FILES = (
    "entroly/tree_sitter_support.py",
    "entroly/repository_intelligence/models.py",
    "entroly/repository_intelligence/parsers.py",
    "entroly/repository_intelligence/graph.py",
    "entroly/repository_intelligence/verified_context.py",
)


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _git_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    return bool(result.stdout.strip())


def _implementation_sha256() -> str:
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _write_fixtures(root: Path) -> None:
    for relative, content in FIXTURES.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _edge_evidence_valid(root: Path, edge: Any) -> bool:
    raw = (root / edge.path).read_bytes()
    evidence = raw[edge.start_byte:edge.end_byte]
    return bool(evidence) and hashlib.sha256(evidence).hexdigest() == edge.evidence_sha256


def _fragment_valid(root: Path, fragment: dict[str, object]) -> bool:
    raw = (root / str(fragment["path"])).read_bytes()
    if hashlib.sha256(raw).hexdigest() != fragment["source_sha256"]:
        return False
    start = int(fragment["start_byte"])
    end = int(fragment["end_byte"])
    content = str(fragment["content"]).encode("utf-8")
    return raw[start:end] == content and hashlib.sha256(content).hexdigest() == fragment["fragment_sha256"]


def _graph_edge_valid(root: Path, edge: dict[str, object]) -> bool:
    raw = (root / str(edge["path"])).read_bytes()
    start = int(edge["start_byte"])
    end = int(edge["end_byte"])
    evidence = raw[start:end]
    return bool(evidence) and hashlib.sha256(evidence).hexdigest() == edge["evidence_sha256"]


def run_benchmark() -> dict[str, object]:
    errors: list[str] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="entroly-vcc-") as directory:
        fixture_root = Path(directory)
        _write_fixtures(fixture_root)
        service = RepositoryIntelligenceService(fixture_root)
        index, digest, _generation = service._snapshot()  # benchmark inspection
        edge_map = {(edge.caller_id, edge.callee_id): edge for edge in index.call_edges}
        found = [pair for pair in GOLD_EDGES if pair in edge_map]
        valid_edges = sum(_edge_evidence_valid(fixture_root, edge_map[pair]) for pair in found)

        ambiguous_id = "python/ambiguous.py::invoke::function"
        truthfulness = int(
            not any(edge.caller_id == ambiguous_id for edge in index.call_edges)
            and any(
                call.caller_id == ambiguous_id and call.reason == "ambiguous"
                for call in index.unresolved_calls
            )
        )

        queries = ("charge_card", "Worker run", "typescript run", "go run", "Sample run")
        query_hits = 0
        fragment_total = 0
        fragment_valid = 0
        for query in queries:
            try:
                payload = service.context(query, token_budget=512, max_hops=2)
                expected = query.split()[-1].lower()
                names = {
                    str(fragment["qualified_name"]).split(".")[-1].lower()
                    for fragment in payload["fragments"]
                }
                query_hits += int(expected in names)
                for fragment in payload["fragments"]:
                    fragment_total += 1
                    fragment_valid += int(_fragment_valid(fixture_root, fragment))
            except Exception as exc:  # noqa: BLE001 - failures stay in matrix
                errors.append(f"query:{query}:{type(exc).__name__}:{exc}")

        first = service.context("charge_card", token_budget=512, max_hops=2)
        second = service.context("charge_card", token_budget=512, max_hops=2)
        deterministic = int(first == second)

        symbol_graph = service.symbol_graph("authorize", direction="callers")
        graph_evidence = int(
            symbol_graph["resolution"] == "resolved"
            and bool(symbol_graph["edges"])
            and all(_graph_edge_valid(fixture_root, edge) for edge in symbol_graph["edges"])
            and verify_symbol_graph_commitment(symbol_graph)
        )
        ambiguous_graph = service.symbol_graph("duplicate")
        graph_ambiguity = int(
            ambiguous_graph["resolution"] == "ambiguous"
            and len(ambiguous_graph["candidates"]) == 2
            and not ambiguous_graph["nodes"]
            and not ambiguous_graph["edges"]
            and verify_symbol_graph_commitment(ambiguous_graph)
        )

        service_path = fixture_root / "python/service.py"
        service_path.write_text("def charge_card(card):\n    return False\n", encoding="utf-8")
        stale = service.context("charge_card", token_budget=512, max_hops=0)
        stale_closed = int(
            all(fragment["path"] != "python/service.py" for fragment in stale["fragments"])
            and stale["receipt"]["omissions_by_reason"].get("stale-index", 0) >= 1
        )
        stale_graph = service.symbol_graph("authorize")
        graph_stale_closed = int(
            stale_graph["resolution"] == "stale-index"
            and not stale_graph["nodes"]
            and not stale_graph["edges"]
            and verify_symbol_graph_commitment(stale_graph)
        )

        metrics = {
            "gold_edge_recall": round(len(found) / len(GOLD_EDGES), 6),
            "edge_evidence_validity": round(valid_edges / max(1, len(found)), 6),
            "ambiguous_call_truthfulness": truthfulness,
            "query_symbol_recall": round(query_hits / len(queries), 6),
            "fragment_evidence_validity": round(fragment_valid / max(1, fragment_total), 6),
            "deterministic_receipt": deterministic,
            "stale_source_fail_closed": stale_closed,
            "symbol_graph_evidence_validity": graph_evidence,
            "symbol_graph_ambiguity_truthfulness": graph_ambiguity,
            "symbol_graph_stale_source_fail_closed": graph_stale_closed,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "git_commit": _git_head(),
            "git_dirty": _git_dirty(),
            "implementation_sha256": _implementation_sha256(),
            "index_digest": digest,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "workload": {
                "languages": ["python", "rust", "typescript", "go", "java"],
                "gold_edges": len(GOLD_EDGES), "queries": len(queries),
                "token_budget": 512, "max_hops": 2,
            },
            "index": {
                "files": len(index.files), "symbols": len(index.symbols),
                "resolved_calls": len(index.call_edges),
                "ambiguous_or_unresolved_calls": len(index.unresolved_calls),
            },
            "metrics": metrics,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "errors": errors,
        }


def verify(payload: dict[str, object], *, require_current_clean_tree: bool = False) -> bool:
    metrics = payload.get("metrics", {})
    metrics_ok = isinstance(metrics, dict) and not payload.get("errors") and all(
        metrics.get(name) == 1 or metrics.get(name) == 1.0
        for name in (
            "gold_edge_recall", "edge_evidence_validity", "ambiguous_call_truthfulness",
            "query_symbol_recall", "fragment_evidence_validity", "deterministic_receipt",
            "stale_source_fail_closed", "symbol_graph_evidence_validity",
            "symbol_graph_ambiguity_truthfulness", "symbol_graph_stale_source_fail_closed",
        )
    )
    if not metrics_ok:
        return False
    if not require_current_clean_tree:
        return True
    return (
        payload.get("git_dirty") is False
        and payload.get("git_commit") == _git_head()
        and payload.get("implementation_sha256") == _implementation_sha256()
        and not _git_dirty()
    )


def _verify_sidecar(path: Path) -> bool:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        expected = sidecar.read_text(encoding="ascii").strip()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return False
    return len(expected) == 64 and expected == actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        payload = run_benchmark()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        args.out.write_text(rendered, encoding="utf-8")
        args.out.with_suffix(args.out.suffix + ".sha256").write_text(
            hashlib.sha256(rendered.encode("utf-8")).hexdigest() + "\n",
            encoding="ascii",
        )
        valid = verify(payload)
    else:
        payload = json.loads(args.out.read_text(encoding="utf-8"))
        valid = _verify_sidecar(args.out) and verify(payload, require_current_clean_tree=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
