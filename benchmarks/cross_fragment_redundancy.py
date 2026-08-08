#!/usr/bin/env python3
"""How much redundancy do we leave on the table by compressing fragments alone?

Entroly compresses every fragment independently. If fragments in one context
pack are correlated -- shared imports, headers, idioms -- independent coding
spends `sum_i H(X_i)` where joint coding needs only `H(X_1..X_n)`. The gap is
the mutual information across fragments, and it is currently unexploited.

This measures an UPPER BOUND on that gap, and is deliberately not a claim about
achievable token savings:

    independent = sum_i |lzma(f_i)|          each fragment coded alone
    joint       = |lzma(concat(f_1..f_n))|   one model across all fragments
    redundancy    = 1 - joint / independent

Why this is only a bound: the delivered context must stay readable by the model,
so a realizable mechanism is *textual* factoring (templates + verbatim slots),
which recovers only part of the statistical redundancy an entropy coder finds.
If this bound is small, textual factoring cannot be worth building and the idea
dies here. If it is large, the bound says how much is worth chasing.

lzma is used rather than zlib because zlib's 32 KB window cannot see across
files in a realistic pack; the redundancy under test is exactly cross-file.
"""

from __future__ import annotations

import argparse
import json
import lzma
import random
import subprocess
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9}]


def _squeeze(data: bytes) -> int:
    return len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=FILTERS))


def _tracked(suffix: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", suffix], cwd=REPO,
        capture_output=True, text=True, check=True,
    )
    paths = []
    for line in out.stdout.splitlines():
        p = REPO / line
        try:
            if p.is_file() and 200 <= p.stat().st_size <= 200_000:
                paths.append(p)
        except OSError:
            continue
    return paths


def measure(paths: list[Path]) -> dict[str, Any]:
    blobs = []
    for p in paths:
        try:
            blobs.append(p.read_bytes())
        except OSError:
            continue
    if len(blobs) < 2:
        return {}
    independent = sum(_squeeze(b) for b in blobs)
    joint = _squeeze(b"\n".join(blobs))
    raw = sum(len(b) for b in blobs)
    return {
        "files": len(blobs),
        "raw_bytes": raw,
        "independent_bytes": independent,
        "joint_bytes": joint,
        "redundancy": 1.0 - (joint / independent) if independent else 0.0,
        "independent_ratio": independent / raw if raw else 0.0,
        "joint_ratio": joint / raw if raw else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack-size", type=int, default=24)
    ap.add_argument("--packs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--out", type=Path,
                    default=REPO / "benchmarks" / "results" / "cross_fragment_redundancy.json")
    args = ap.parse_args()

    results: dict[str, Any] = {"packs": {}}
    for label, glob in (("python", "*.py"), ("rust", "*.rs"), ("markdown", "*.md")):
        pool = _tracked(glob)
        if len(pool) < args.pack_size:
            continue
        rng = random.Random(f"{args.seed}:{label}")
        rows = []
        for i in range(args.packs):
            sample = rng.sample(pool, args.pack_size)
            row = measure(sample)
            if row:
                rows.append(row)
        if not rows:
            continue
        heads = sorted(r["redundancy"] for r in rows)
        results["packs"][label] = {
            "n_packs": len(rows),
            "pack_size": args.pack_size,
            "redundancy_median": heads[len(heads) // 2],
            "redundancy_min": heads[0],
            "redundancy_max": heads[-1],
            "rows": rows,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"{'workload':<12}{'packs':>7}{'median':>10}{'min':>9}{'max':>9}")
    for label, agg in results["packs"].items():
        print(f"{label:<12}{agg['n_packs']:>7}"
              f"{agg['redundancy_median']:>9.1%}{agg['redundancy_min']:>9.1%}"
              f"{agg['redundancy_max']:>9.1%}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
