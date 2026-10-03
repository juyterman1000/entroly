"""Falsify or support: does the 99-module SCC prevent extracting a context-state
interface, or is it architectural debt that leaves algorithms reusable?

The prior claim -- "a 99-module cycle containing every entry point directly
constrains the model-neutral control-plane vision" -- was an interpretation, not
a measurement. This measures the thing that would actually have to be true for
it to bite: **isolated importability**. If importing any single member drags in
the whole component, no subset can be extracted, tested, or reasoned about
alone. If members import cheaply, the cycle is latent and the algorithms remain
reusable.

Run: python research/experiments/cycle_coupling_probe.py
Writes research/ledger/cycle_coupling.json
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "ledger" / "cycle_coupling.json"

# Probe set: SCC members spanning different architectural roles, plus
# non-SCC controls. The controls matter -- without them a large module count
# could just be "importing anything in Python is expensive here".
SCC_PROBES = [
    "entroly.codec",            # hub, PageRank top-3
    "entroly.sdk",              # public SDK
    "entroly.server",           # MCP
    "entroly.proxy",            # proxy
    "entroly.cli",              # CLI
    "entroly.auto_index",       # named in the cycle description
    "entroly.cache_aligner",    # named in the cycle description
    "entroly.context_receipts", # assurance subsystem
]
CONTROL_PROBES = [
    "entroly.tokens",           # hub but OUTSIDE the SCC
    "entroly.path_safety",      # hub, outside
    "entroly.vault",            # outside the SCC (verified earlier)
    "entroly.config",           # hub, outside
]

PROBE_SRC = """
import sys, time, json
t0 = time.perf_counter()
try:
    __import__({mod!r})
    err = None
except Exception as exc:
    err = f"{{type(exc).__name__}}: {{exc}}"
elapsed = time.perf_counter() - t0
mods = [m for m in sys.modules if m.startswith("entroly")]
print(json.dumps({{
    "module": {mod!r},
    "error": err,
    "seconds": round(elapsed, 3),
    "entroly_modules_loaded": len(mods),
}}))
"""


def probe(mod: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    # Keep the engine from doing network/self-heal work during a pure import
    # measurement; we are measuring coupling, not startup policy.
    env["ENTROLY_NO_SELF_HEAL"] = "1"
    env["ENTROLY_NO_DOCKER"] = "1"
    try:
        r = subprocess.run(
            [sys.executable, "-c", PROBE_SRC.format(mod=mod)],
            cwd=ROOT, capture_output=True, text=True, timeout=300, env=env,
        )
    except subprocess.TimeoutExpired:
        return {"module": mod, "error": "TIMEOUT>300s", "seconds": None,
                "entroly_modules_loaded": None}
    for line in reversed((r.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"module": mod, "error": f"no json (rc={r.returncode})",
            "seconds": None, "entroly_modules_loaded": None}


def main() -> int:
    total = len(json.loads((ROOT / "research" / "ledger" / "graph.json").read_text())["adjacency"]) \
        if (ROOT / "research" / "ledger" / "graph.json").exists() else None

    rows = []
    for group, probes in (("SCC", SCC_PROBES), ("CONTROL", CONTROL_PROBES)):
        for m in probes:
            r = probe(m)
            r["group"] = group
            rows.append(r)
            n = r["entroly_modules_loaded"]
            print(f"  {group:<8} {m:<28} {str(n):>5} modules  "
                  f"{r['seconds']}s  {r['error'] or ''}")

    scc = [r for r in rows if r["group"] == "SCC" and r["entroly_modules_loaded"]]
    ctl = [r for r in rows if r["group"] == "CONTROL" and r["entroly_modules_loaded"]]
    summary = {
        "baseline_commit": "e31f082f",
        "total_package_modules": total,
        "scc_median_modules": sorted(r["entroly_modules_loaded"] for r in scc)[len(scc)//2] if scc else None,
        "control_median_modules": sorted(r["entroly_modules_loaded"] for r in ctl)[len(ctl)//2] if ctl else None,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSCC median modules loaded     : {summary['scc_median_modules']}")
    print(f"CONTROL median modules loaded : {summary['control_median_modules']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
