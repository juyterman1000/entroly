"""Deterministic microbenchmark for audited-selection overhead.

This benchmark does not claim product superiority. It reports compatibility
QCCR versus audited QCCR latency, receipt bytes, determinism, and budget
compliance on the same fragments and process.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from entroly.audited_qccr import select_with_audit
from entroly.qccr import select as legacy_select


@dataclass(frozen=True)
class TimingSummary:
    iterations: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class OverheadReport:
    legacy: TimingSummary
    audited: TimingSummary
    p50_overhead_ms: float
    p95_overhead_ms: float
    p50_ratio: float
    receipt_bytes: int
    deterministic: bool
    budget_compliant: bool
    emitted_tokens: int
    requested_budget: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "legacy": asdict(self.legacy),
            "audited": asdict(self.audited),
            "p50_overhead_ms": self.p50_overhead_ms,
            "p95_overhead_ms": self.p95_overhead_ms,
            "p50_ratio": self.p50_ratio,
            "receipt_bytes": self.receipt_bytes,
            "deterministic": self.deterministic,
            "budget_compliant": self.budget_compliant,
            "emitted_tokens": self.emitted_tokens,
            "requested_budget": self.requested_budget,
        }


def _percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _time(call: Callable[[], Any], iterations: int, warmup: int) -> tuple[TimingSummary, Any]:
    if iterations <= 0 or warmup < 0:
        raise ValueError("iterations must be positive and warmup non-negative")
    for _ in range(warmup):
        call()
    durations: list[float] = []
    latest: Any = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        latest = call()
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    return (
        TimingSummary(
            iterations=iterations,
            p50_ms=_percentile(durations, 0.50),
            p95_ms=_percentile(durations, 0.95),
            mean_ms=statistics.fmean(durations),
            minimum_ms=min(durations),
            maximum_ms=max(durations),
        ),
        latest,
    )


def compare_overhead(
    fragments: Sequence[dict[str, Any]],
    *,
    query: str,
    budget: int,
    iterations: int = 100,
    warmup: int = 10,
) -> OverheadReport:
    legacy_timing, _ = _time(
        lambda: legacy_select(fragments, token_budget=budget, query=query),
        iterations,
        warmup,
    )
    audited_timing, audited = _time(
        lambda: select_with_audit(fragments, budget, query),
        iterations,
        warmup,
    )
    encoded = json.dumps(audited, sort_keys=True, separators=(",", ":"))
    deterministic = encoded == json.dumps(
        select_with_audit(fragments, budget, query),
        sort_keys=True,
        separators=(",", ":"),
    )
    emitted = int(audited.get("emitted_tokens") or 0)
    denominator = max(legacy_timing.p50_ms, 1e-9)
    return OverheadReport(
        legacy=legacy_timing,
        audited=audited_timing,
        p50_overhead_ms=audited_timing.p50_ms - legacy_timing.p50_ms,
        p95_overhead_ms=audited_timing.p95_ms - legacy_timing.p95_ms,
        p50_ratio=audited_timing.p50_ms / denominator,
        receipt_bytes=len(encoded.encode("utf-8")),
        deterministic=deterministic,
        budget_compliant=emitted <= budget,
        emitted_tokens=emitted,
        requested_budget=budget,
    )


def _load_fragments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ValueError("input must be a JSON array of fragment objects")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-p50-overhead-ms", type=float, default=None)
    parser.add_argument("--max-p50-ratio", type=float, default=None)
    args = parser.parse_args()

    report = compare_overhead(
        _load_fragments(args.input),
        query=args.query,
        budget=args.budget,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    failed = not report.deterministic or not report.budget_compliant
    if args.max_p50_overhead_ms is not None:
        failed = failed or report.p50_overhead_ms > args.max_p50_overhead_ms
    if args.max_p50_ratio is not None:
        failed = failed or report.p50_ratio > args.max_p50_ratio
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
