"""Run every benchmark referenced in README's table at lines 169-176.

Captures each benchmark's JSON output and saves to
`benchmarks/results/<name>_accuracy.json`. Uses subprocess so we get
the same exact behaviour as running `python -m bench.accuracy ...` from
the CLI, but with robust per-file save (no /tmp paths, no shell glue).

Usage:
    python bench/_run_readme_benchmarks.py            # all 7 benchmarks
    python bench/_run_readme_benchmarks.py needle     # one specific
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# README table (lines 169-176): (bench_id, samples, budget) — budget MUST
# match the README's "Budget" column or the savings numbers will be wrong.
BENCHMARKS = [
    ("needle",      20,   2_000),
    ("longbench",   50,   2_000),
    ("bfcl",        50,     500),
    ("squad",       50,     100),
    ("gsm8k",      100,  50_000),  # pass-through expected
    ("mmlu",       100,  50_000),  # pass-through expected
    ("truthfulqa", 100,  50_000),  # pass-through expected
]

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_one(name: str, samples: int, budget: int) -> dict | None:
    out_path = RESULTS_DIR / f"{name}_accuracy.json"
    log_path = RESULTS_DIR / f"_{name}_accuracy.log"
    print(f"\n{'=' * 60}\n=== {name} (n={samples}, budget={budget}) -> {out_path.name}\n{'=' * 60}",
          flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-m", "bench.accuracy",
         "--benchmark", name, "--samples", str(samples),
         "--budget", str(budget),
         "--model", "gpt-4o-mini", "--json"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t0
    log_path.write_text(
        f"# exit={proc.returncode}  elapsed={elapsed:.1f}s\n"
        f"# === STDOUT ===\n{proc.stdout}\n"
        f"# === STDERR ===\n{proc.stderr}\n",
        encoding="utf-8",
    )
    # JSON array sits at the end of stdout. Find the LAST `[ { ... } ]` block.
    matches = list(re.finditer(r"(?ms)^(\[\s*\{.*?\}\s*\])\s*$", proc.stdout))
    if not matches:
        print(f"  !! no JSON in stdout (exit={proc.returncode}); see {log_path.name}",
              flush=True)
        return None
    try:
        data = json.loads(matches[-1].group(1))
    except json.JSONDecodeError as e:
        print(f"  !! JSON parse failed: {e}; see {log_path.name}", flush=True)
        return None
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    # Pretty-print headline numbers
    if isinstance(data, list) and data:
        r = data[0]
        print(f"  baseline={r.get('baseline_accuracy', 0):.1%}  "
              f"entroly={r.get('entroly_accuracy', 0):.1%}  "
              f"retention={r.get('retention', 0):.1%}  "
              f"savings={r.get('token_savings_pct', 0):.1f}%  "
              f"({elapsed:.1f}s)", flush=True)
    return data


def main() -> int:
    requested = sys.argv[1:] if len(sys.argv) > 1 else None
    if requested:
        plan = [(n, s, b) for n, s, b in BENCHMARKS if n in requested]
    else:
        plan = BENCHMARKS

    if not os.environ.get("OPENAI_API_KEY"):
        env = REPO / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                m = re.match(r"(?:export\s+)?([A-Z_]+)\s*=\s*"
                             r"['\"]?([^'\"\s]+)", line.strip())
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2)

    failed = []
    for name, samples, budget in plan:
        result = run_one(name, samples, budget)
        if result is None:
            failed.append(name)
        time.sleep(2)  # small breather between benchmarks for rate-limit grace

    print(f"\n{'=' * 60}\nDONE. Results dir: {RESULTS_DIR}")
    files = sorted(RESULTS_DIR.glob("*_accuracy.json"))
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size} bytes)")
    if failed:
        print(f"  FAILED: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
