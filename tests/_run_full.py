"""Full 200-sample benchmark run: SQuAD + LongBench."""
import os
import subprocess
import sys
import json
import time

# Load key
result = subprocess.run(
    ["powershell", "-c",
     '[System.Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")'],
    capture_output=True, text=True,
)
key = result.stdout.strip()
if not key:
    print("ERROR: no key", flush=True)
    sys.exit(1)
os.environ["OPENAI_API_KEY"] = key
print(f"Key loaded. Starting benchmarks...", flush=True)

from bench.accuracy import run_benchmark
results = {}

for bench in ["squad", "longbench"]:
    print(f"\n{'='*60}", flush=True)
    print(f"  BENCHMARK: {bench} (200 samples, gpt-4o-mini)", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    try:
        r = run_benchmark(bench, model="gpt-4o-mini", samples=200, budget=50000, mode="entroly")
        elapsed = time.time() - t0
        results[bench] = {
            "baseline": f"{r.baseline.accuracy:.1%} [{r.baseline.ci_low:.1%}-{r.baseline.ci_high:.1%}]",
            "entroly": f"{r.entroly.accuracy:.1%} [{r.entroly.ci_low:.1%}-{r.entroly.ci_high:.1%}]",
            "retention": f"{r.retention:.1%}",
            "token_savings": f"{r.token_savings_pct:.1f}%",
            "cost_savings": f"{r.cost_savings_pct:.1f}%",
            "elapsed_s": round(elapsed, 1),
        }
        print(f"\n  Baseline: {results[bench]['baseline']}", flush=True)
        print(f"  Entroly:  {results[bench]['entroly']}", flush=True)
        print(f"  Retention: {results[bench]['retention']}", flush=True)
        print(f"  Token savings: {results[bench]['token_savings']}", flush=True)
        print(f"  Cost savings:  {results[bench]['cost_savings']}", flush=True)
        print(f"  Elapsed: {elapsed:.0f}s", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}", flush=True)
print("  FINAL RESULTS", flush=True)
print(f"{'='*60}", flush=True)
print(json.dumps(results, indent=2), flush=True)
