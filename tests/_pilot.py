"""Quick 5-sample pilot to verify benchmark pipeline end-to-end."""
import os
import subprocess
import sys

# Read User-level env var
result = subprocess.run(
    ["powershell", "-c",
     '[System.Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")'],
    capture_output=True, text=True,
)
key = result.stdout.strip()
if not key:
    print("ERROR: OPENAI_API_KEY not found", flush=True)
    sys.exit(1)
os.environ["OPENAI_API_KEY"] = key
print(f"Key loaded (len={len(key)})", flush=True)

from bench.accuracy import run_benchmark
import json

for bench in ["squad"]:
    print(f"\n=== {bench} (5 samples) ===", flush=True)
    try:
        r = run_benchmark(bench, model="gpt-4o-mini", samples=5, budget=50000, mode="entroly")
        print(f"\nBaseline: {r.baseline.accuracy:.1%} [{r.baseline.ci_low:.1%}-{r.baseline.ci_high:.1%}]", flush=True)
        print(f"Entroly:  {r.entroly.accuracy:.1%} [{r.entroly.ci_low:.1%}-{r.entroly.ci_high:.1%}]", flush=True)
        print(f"Retention: {r.retention:.1%}", flush=True)
        print(f"Token savings: {r.token_savings_pct:.1f}%", flush=True)
        print(f"Cost savings:  {r.cost_savings_pct:.1f}%", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
