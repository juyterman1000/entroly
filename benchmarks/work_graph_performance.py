#!/usr/bin/env python3
"""Measured boundedness gate for the production Work Graph delivery path.

This benchmark exercises the installed PyO3 boundary, not a Python model of
the graph.  Its thresholds are deliberately generous release tripwires: they
detect accidental whole-history scans, passive polling amplification, runaway
state growth and unusable resume/import latency without pretending that one
developer machine establishes a universal speed claim.
"""
from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from entroly.repository_intelligence.graph_projection import project_repository_scope
from entroly.repository_intelligence.incremental import (
    build_repository_index_incremental,
    build_repository_scope_incremental,
)
from entroly.work_graph import WorkGraph
from entroly.work_graph_store import WorkGraphStore

SCHEMA = "entroly.work-graph-performance.v1"
DEFAULT_FILES = 2_000
DEFAULT_EVENTS = 500
DEFAULT_POLLS = 100
THRESHOLDS_MS = {
    "initial_large_observation": 10_000.0,
    "incremental_append_p95": 250.0,
    "export": 5_000.0,
    "import_rebuild": 10_000.0,
    "resume": 1_000.0,
    "context_scope": 1_000.0,
    "coordination": 1_000.0,
    "initial_repository_index": 60_000.0,
    "one_file_incremental_index": 5_000.0,
    "symbol_scope_projection": 2_000.0,
    "pyo3_summary_p95": 100.0,
    "contended_store_writes": 10_000.0,
}
MAX_STATE_BYTES = 64 * 1024 * 1024


def _ms(operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    result = operation()
    return result, (time.perf_counter_ns() - started) / 1_000_000


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _change(index: int, digest_suffix: str) -> dict[str, Any]:
    digest = hashlib.sha1(digest_suffix.encode("utf-8"), usedforsecurity=False).hexdigest()
    return {
        "path": f"src/module_{index:05d}.py",
        "kind": "modified",
        "staged": False,
        "conflicted": False,
        "content_digest": f"git-blob:{digest}",
    }


def _observation(*, files: int, observed_at_ms: int, digest_suffix: str) -> dict[str, Any]:
    return {
        "repo_id": "repo:work-graph-performance",
        "observed_at_ms": observed_at_ms,
        "repository_label": "bounded performance fixture",
        "branch": {
            "name": "feature/performance",
            "head_sha": "head-performance",
            "default_branch": "main",
            "ahead_by": 1,
        },
        "changes": [_change(index, digest_suffix) for index in range(files)],
    }


def _cache_diagnostic(index: Any) -> str:
    return next(
        item for item in index.diagnostics if item.startswith("incremental-parse-cache")
    )


def _repository_delivery_measurements(
    root: Path,
    *,
    files: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    repository = root / "repository"
    cache = root / "repository-cache"
    repository.mkdir()
    for index in range(files):
        previous = f"module_{index - 1:05d}" if index else ""
        imported = f"from {previous} import symbol_{index - 1:05d}\n" if previous else ""
        (repository / f"module_{index:05d}.py").write_text(
            f"{imported}def symbol_{index:05d}():\n    return {index}\n",
            encoding="utf-8",
        )

    cold, initial_index_ms = _ms(
        lambda: build_repository_index_incremental(repository, cache_dir=cache)
    )
    changed_index = files // 2
    (repository / f"module_{changed_index:05d}.py").write_text(
        f"def symbol_{changed_index:05d}():\n    return {changed_index + 1}\n",
        encoding="utf-8",
    )
    changed_path = f"module_{changed_index:05d}.py"
    warm, incremental_index_ms = _ms(
        lambda: build_repository_scope_incremental(
            repository,
            [changed_path],
            cache_dir=cache,
        )
    )

    selected_paths = sorted(cold.files)[: min(files, 256)]
    selected = [cold.files[path] for path in selected_paths]
    symbols = {path: cold.symbols_for_path(path) for path in selected_paths}
    imports = [
        (source, target)
        for source in selected_paths
        for target in cold.file_dependencies.get(source, ())
    ]
    projected, projection_ms = _ms(
        lambda: project_repository_scope(
            "repo:repository-delivery-performance",
            files=selected,
            symbols=symbols,
            imports=imports,
            observed_at_ms=10_000,
        )
    )

    incremental_diagnostic = _cache_diagnostic(warm)
    return (
        {
            "initial_repository_index": initial_index_ms,
            "one_file_incremental_index": incremental_index_ms,
            "symbol_scope_projection": projection_ms,
        },
        {
            "repository_files": len(cold.files),
            "repository_symbols": len(cold.symbols),
            "incremental_scope_files": len(warm.files),
            "incremental_scope_catalog": next(
                item
                for item in warm.diagnostics
                if item.startswith("active-repository-scope")
            ),
            "incremental_cache_diagnostic": incremental_diagnostic,
            "projection_operations": projected["projection"]["operation_count"],
            "projection_symbols_dropped": projected["projection"]["symbols_dropped"],
            "cold_cache_diagnostic": _cache_diagnostic(cold),
        },
    )


def _contended_store_measurement(root: Path, *, writers: int = 8) -> tuple[float, int]:
    repo_id = "repo:work-graph-lock-performance"
    store = WorkGraphStore(repo_id, root=root / "shared-store")

    def write(index: int) -> None:
        store.submit_observation({
            "repo_id": repo_id,
            "observed_at_ms": 20_000 + index,
            "repository_label": "lock contention fixture",
            "branch": {
                "name": "feature/contention",
                "head_sha": "head-contention",
                "default_branch": "main",
                "ahead_by": 1,
            },
            "changes": [_change(index, f"contention-{index}")],
        })

    def run() -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=writers) as pool:
            list(pool.map(write, range(writers)))

    _, elapsed_ms = _ms(run)
    return elapsed_ms, store.load().event_count


def run_benchmark(
    *, files: int = DEFAULT_FILES,
    events: int = DEFAULT_EVENTS,
    polls: int = DEFAULT_POLLS,
) -> dict[str, Any]:
    if files < 1 or events < 1 or polls < 1:
        raise ValueError("files, events and polls must all be positive")

    graph = WorkGraph("repo:work-graph-performance")
    initial = _observation(files=files, observed_at_ms=1_000, digest_suffix="initial")
    _, initial_ms = _ms(lambda: graph.observe_repository(initial))
    workstream_id = str(graph.unfinished()[0]["node_id"])

    before_polls = graph.event_count
    def passive_poll() -> None:
        for index in range(polls):
            polled = copy.deepcopy(initial)
            polled["observed_at_ms"] = 1_001 + index
            graph.observe_repository(polled)

    _, polling_ms = _ms(passive_poll)
    passive_event_growth = graph.event_count - before_polls

    append_ms: list[float] = []
    for index in range(events):
        incremental = _observation(
            files=1,
            observed_at_ms=2_000 + index,
            digest_suffix=f"edit-{index:08d}",
        )
        _, elapsed = _ms(lambda observation=incremental: graph.observe_repository(observation))
        append_ms.append(elapsed)

    exported, export_ms = _ms(graph.export_json)
    restored, import_ms = _ms(lambda: WorkGraph.from_json(exported))
    resume, resume_ms = _ms(lambda: restored.resume(workstream_id, max_evidence=128))
    scope, context_ms = _ms(
        lambda: restored.context_scope(workstream_id, max_evidence=128)
    )
    coordination, coordination_ms = _ms(lambda: restored.coordination(10_000))
    summary_calls = [_ms(restored.summary)[1] for _ in range(100)]

    with tempfile.TemporaryDirectory(prefix="entroly-work-graph-performance-") as temp:
        delivery_measurements, delivery_state = _repository_delivery_measurements(
            Path(temp),
            files=files,
        )
        contended_ms, contended_events = _contended_store_measurement(Path(temp))

    measurements = {
        "initial_large_observation": initial_ms,
        "incremental_append_median": statistics.median(append_ms),
        "incremental_append_p95": _percentile(append_ms, 0.95),
        "incremental_append_max": max(append_ms),
        "passive_poll_total": polling_ms,
        "export": export_ms,
        "import_rebuild": import_ms,
        "resume": resume_ms,
        "context_scope": context_ms,
        "coordination": coordination_ms,
        "pyo3_summary_p95": _percentile(summary_calls, 0.95),
        "contended_store_writes": contended_ms,
        **delivery_measurements,
    }
    failures = [
        f"{name}={measurements[name]:.3f}ms exceeds {limit:.3f}ms"
        for name, limit in THRESHOLDS_MS.items()
        if measurements[name] > limit
    ]
    if passive_event_growth:
        failures.append(f"passive polling appended {passive_event_growth} duplicate event(s)")
    state_bytes = len(exported.encode("utf-8"))
    if state_bytes > MAX_STATE_BYTES:
        failures.append(f"state_bytes={state_bytes} exceeds {MAX_STATE_BYTES}")
    if restored.graph_commitment != graph.graph_commitment:
        failures.append("imported graph commitment differs from exported graph")
    if resume.get("graph_commitment") != graph.graph_commitment:
        failures.append("resume view is not bound to the measured graph")
    if scope.get("graph_commitment") != graph.graph_commitment:
        failures.append("context scope is not bound to the measured graph")
    expected_events = math.ceil(files / 512) + events
    if restored.event_count != expected_events:
        failures.append(
            f"event_count={restored.event_count} differs from expected {expected_events}"
        )
    if scope.get("changed_paths_total", 0) < files:
        failures.append("context scope does not account for every measured changed path")
    if len(scope.get("changed_paths", [])) > 512:
        failures.append("context scope exceeded the bounded changed-path prefix")
    if not str(scope.get("changed_paths_commitment", "")).startswith("sha256:"):
        failures.append("context scope has no full changed-path commitment")
    if scope.get("evidence_ids_total", 0) < len(scope.get("evidence_ids", [])):
        failures.append("context scope evidence total is smaller than its inline prefix")
    if "misses=1 writes=1" not in str(
        delivery_state["incremental_cache_diagnostic"]
    ):
        failures.append("one-file edit did not parse exactly its changed source")
    if delivery_state["incremental_scope_files"] != 1:
        failures.append("one-file edit parsed more than its active source scope")
    if f"catalog={files}" not in str(delivery_state["incremental_scope_catalog"]):
        failures.append("incremental dependency catalog omitted repository source paths")
    if delivery_state["repository_files"] != files:
        failures.append("repository index omitted measured first-party files")
    if delivery_state["repository_symbols"] < files:
        failures.append("repository index omitted measured symbols")
    if delivery_state["projection_symbols_dropped"]:
        failures.append("bounded active scope dropped measured symbols")
    if contended_events != 8:
        failures.append(
            f"contended store preserved {contended_events} events instead of 8"
        )

    return {
        "schema_version": SCHEMA,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "implementation": platform.python_implementation(),
        },
        "inputs": {"files": files, "events": events, "passive_polls": polls},
        "measurements_ms": {key: round(value, 6) for key, value in measurements.items()},
        "state": {
            "bytes": state_bytes,
            "events": restored.event_count,
            "passive_event_growth": passive_event_growth,
            "unfinished": len(restored.unfinished()),
            "resume_evidence": len(resume.get("evidence", [])),
            "scope_changed_paths_inline": len(scope.get("changed_paths", [])),
            "scope_changed_paths_total": scope.get("changed_paths_total", 0),
            "scope_evidence_inline": len(scope.get("evidence_ids", [])),
            "scope_evidence_total": scope.get("evidence_ids_total", 0),
            "coordination_conflicts": len(coordination.get("conflicts", [])),
            "pyo3_summary_samples": len(summary_calls),
            "contended_store_events": contended_events,
            **delivery_state,
        },
        "thresholds_ms": THRESHOLDS_MS,
        "max_state_bytes": MAX_STATE_BYTES,
        "passed": not failures,
        "failures": failures,
    }


def verify_report(report: dict[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA:
        raise ValueError("unsupported Work Graph performance report schema")
    if not report.get("passed") or report.get("failures"):
        raise ValueError("Work Graph performance gate failed: " + "; ".join(report.get("failures", [])))
    if report.get("state", {}).get("passive_event_growth") != 0:
        raise ValueError("passive polling amplified Work Graph events")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=DEFAULT_FILES)
    parser.add_argument("--events", type=int, default=DEFAULT_EVENTS)
    parser.add_argument("--polls", type=int, default=DEFAULT_POLLS)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = run_benchmark(files=args.files, events=args.events, polls=args.polls)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_benchmark", "verify_report"]
