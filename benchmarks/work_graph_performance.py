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
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

from entroly.work_graph import WorkGraph

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
