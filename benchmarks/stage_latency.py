"""Per-stage latency and memory for the codec pipeline.

An end-to-end number hides which stage costs what, so a regression in one
stage is invisible until it dominates. This measures each stage separately,
across input sizes, and reports p50/p95 rather than a mean -- a compressor
that is fast on average and occasionally slow is a compressor that
occasionally blocks a request.

Stages measured
---------------

``digest``        SHA-256 over the input. The floor: every other stage costs
                  at least this, and it scales linearly with input size.
``route``         registry.select() -- which codec claims this content.
``represent``     the codec producing its candidate representations. This is
                  where parsing, templating and summarising happen.
``prune``         Pareto pruning of the representation set.
``store``         writing the omitted bytes to the recovery store.
``recover``       reading them back and verifying digest and length.

Honest scope
------------

* Wall-clock on one machine, one Python. Not a cross-platform claim.
* Peak RSS is process-wide and sampled around the stage, so it includes
  interpreter overhead and is an upper bound on the stage's own cost.
* No model, no network, no proxy. This measures the compression pipeline
  only, and says nothing about end-to-end request latency.
* Inputs are synthetic and generated deterministically per size.

Run:
    python benchmarks/stage_latency.py
    python benchmarks/stage_latency.py --json out.json --repeats 7
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("ENTROLY_DISABLE_UPDATE_CHECK", "1")

SCHEMA_VERSION = "entroly.stage-latency.v1"


def _json_input(target_bytes: int) -> str:
    random.seed(5)
    records, size = [], 0
    while size < target_bytes:
        rec = {
            "id": f"ord-{len(records):07d}",
            "customer": f"cust{random.randint(0, 5000)}",
            "amount_cents": random.randint(100, 900000),
            "status": random.choice(["paid", "failed", "refunded"]),
        }
        records.append(rec)
        size += 90
    return json.dumps({"orders": records}, indent=2)


def _log_input(target_bytes: int) -> str:
    lines, size = ["2026-08-02T10:00:00Z ERROR root cause: disk full on /var"], 60
    i = 0
    while size < target_bytes:
        line = f"2026-08-02T10:00:{i % 60:02d}Z ERROR retry failed (attempt {i})"
        lines.append(line)
        size += len(line) + 1
        i += 1
    return "\n".join(lines)


SIZES = [("tiny", 512), ("10KB", 10_000), ("100KB", 100_000), ("1MB", 1_000_000)]
WORKLOADS = {"json": _json_input, "log": _log_input}


@dataclass
class Row:
    workload: str
    size_label: str
    input_bytes: int
    stage: str
    p50_ms: float
    p95_ms: float
    repeats: int


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _peak_rss_mb() -> float | None:
    try:
        import resource  # noqa: PLC0415 - unix only

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        try:
            import ctypes
            import ctypes.wintypes as wt

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wt.DWORD),
                    ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PMC()
            counters.cb = ctypes.sizeof(_PMC)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            ):
                return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
        except Exception:
            return None
    return None


def _time(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-size", default="1MB",
                    help="largest size to run (tiny/10KB/100KB/1MB)")
    args = ap.parse_args(argv[1:])

    import entroly
    from entroly.codec import RecoveryStore, content_digest, pareto_prune
    from entroly.codecs_builtin import default_registry

    limit = [s for s, _ in SIZES].index(args.max_size) + 1
    rows: list[Row] = []

    for workload, make in WORKLOADS.items():
        for label, target in SIZES[:limit]:
            text = make(target)
            registry = default_registry(RecoveryStore())

            rows.append(_row(workload, label, text, "digest",
                             _time(lambda: content_digest(text), args.repeats), args.repeats))

            rows.append(_row(workload, label, text, "route",
                             _time(lambda: registry.select(text), args.repeats), args.repeats))

            codec = registry.select(text)
            if codec is None:
                continue
            rows.append(_row(workload, label, text, "represent",
                             _time(lambda: codec.representations(text, source_id="b"),
                                   args.repeats), args.repeats))

            produced = codec.representations(text, source_id="b")
            rows.append(_row(workload, label, text, "prune",
                             _time(lambda: pareto_prune(produced), args.repeats), args.repeats))

            store = RecoveryStore()
            rows.append(_row(workload, label, text, "store",
                             _time(lambda: store.put(text, item_count=1), args.repeats),
                             args.repeats))

            ref = store.put(text, item_count=1)
            rows.append(_row(workload, label, text, "recover",
                             _time(lambda: store.recover(ref), args.repeats), args.repeats))

    print(f"\n  Entroly per-stage latency  [{SCHEMA_VERSION}]")
    print(f"  entroly {entroly.__version__}  |  Python {platform.python_version()}  "
          f"|  {platform.system()} {platform.machine()}")
    peak = _peak_rss_mb()
    print(f"  repeats per measurement: {args.repeats}"
          + (f"  |  process peak RSS: {peak:.0f} MB" if peak else ""))
    print(f"\n  {'workload':<9}{'size':<8}{'stage':<12}{'p50 ms':>10}{'p95 ms':>10}")
    for r in rows:
        print(f"  {r.workload:<9}{r.size_label:<8}{r.stage:<12}"
              f"{r.p50_ms:>10.3f}{r.p95_ms:>10.3f}")

    print("\n  Cost relative to hashing the same bytes (p50):")
    for workload in WORKLOADS:
        for label, _ in SIZES[:limit]:
            sel = {r.stage: r for r in rows if r.workload == workload and r.size_label == label}
            if "digest" not in sel or "represent" not in sel:
                continue
            ratio = sel["represent"].p50_ms / max(sel["digest"].p50_ms, 1e-9)
            print(f"    {workload:<6} {label:<7} represent = {ratio:>7.1f}x digest")

    report = {
        "schema_version": SCHEMA_VERSION,
        "entroly_version": entroly.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "repeats": args.repeats,
        "process_peak_rss_mb": peak,
        "rows": [asdict(r) for r in rows],
        "caveats": [
            "Wall clock on one machine and one Python; not a cross-platform claim.",
            "Peak RSS is process-wide, so it is an upper bound on any single stage.",
            "No model, network or proxy: this is the compression pipeline only.",
            "Inputs are synthetic and generated deterministically per size.",
        ],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n  wrote {args.json_out}")
    return 0


def _row(workload, label, text, stage, samples, repeats) -> Row:
    return Row(
        workload=workload,
        size_label=label,
        input_bytes=len(text.encode("utf-8")),
        stage=stage,
        p50_ms=round(_percentile(samples, 0.50), 4),
        p95_ms=round(_percentile(samples, 0.95), 4),
        repeats=repeats,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
