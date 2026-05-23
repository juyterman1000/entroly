"""Launcher: reads OPENAI_API_KEY from User env, runs benchmark."""
import os
import subprocess
import sys

# Read the User-level env var (survives shell restarts)
result = subprocess.run(
    ["powershell", "-c",
     '[System.Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")'],
    capture_output=True, text=True,
)
key = result.stdout.strip()
if not key:
    print("ERROR: OPENAI_API_KEY not found in User env vars")
    sys.exit(1)

os.environ["OPENAI_API_KEY"] = key
print(f"Key loaded (length={len(key)})")

# Run the benchmark
sys.argv = [
    "bench/accuracy.py",
    "--benchmark", sys.argv[1] if len(sys.argv) > 1 else "squad",
    "--samples", sys.argv[2] if len(sys.argv) > 2 else "200",
    "--model", "gpt-4o-mini",
    "--mode", "entroly",
]

# Import and run
from bench.accuracy import main
main()
