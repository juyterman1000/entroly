"""Create the frozen v5 recovery-resilience revalidation contract.

Temporary stabilization helper. The builder removes this file before creating the
final branch.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

v4_path = ROOT / "benchmarks/recovery_resilience_protocol_v4.json"
v5_path = ROOT / "benchmarks/recovery_resilience_protocol_v5.json"
protocol = json.loads(v4_path.read_text(encoding="utf-8"))
protocol["schema_version"] = "entroly.recovery-resilience-protocol.v5"
protocol["frozen_at"] = "2026-07-25T00:00:00Z"
protocol["comparison"]["entroly"] = "1.0.66 source"
protocol["supersedes"] = {
    "protocol": "benchmarks/recovery_resilience_protocol_v4.json",
    "artifact": "benchmarks/results/recovery_resilience_holdout_revalidation_v4.json",
    "reason": (
        "Re-anchor after the recovery store was corrected to preserve CRLF/LF "
        "and trailing newlines exactly. The implementation hash therefore "
        "changed for a user-visible integrity fix. The v4 protocol and artifact "
        "remain immutable. Re-run the same frozen 66-entry matrix without "
        "changing gates or claim policy; both systems passing remains parity, "
        "not leadership."
    ),
}
protocol["suites"]["recovery_resilience"]["holdout_policy"] = (
    "Run once after the exact line-ending recovery correction to re-anchor the "
    "implementation hash. Both systems passing means parity, not leadership."
)
v5_path.write_text(
    json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

benchmark_path = ROOT / "benchmarks/recovery_resilience.py"
benchmark = benchmark_path.read_text(encoding="utf-8")
old = '''PRIOR_REVALIDATION_PROTOCOL_PATH_V3 = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v3.json"
)
CURRENT_REVALIDATION_PROTOCOL_PATH = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v4.json"
)
KNOWN_PROTOCOL_PATHS = (
    PROTOCOL_PATH,
    REVALIDATION_PROTOCOL_PATH,
    PRIOR_REVALIDATION_PROTOCOL_PATH_V3,
    CURRENT_REVALIDATION_PROTOCOL_PATH,
)
'''
new = '''PRIOR_REVALIDATION_PROTOCOL_PATH_V3 = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v3.json"
)
PRIOR_REVALIDATION_PROTOCOL_PATH_V4 = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v4.json"
)
CURRENT_REVALIDATION_PROTOCOL_PATH = (
    ROOT / "benchmarks" / "recovery_resilience_protocol_v5.json"
)
KNOWN_PROTOCOL_PATHS = (
    PROTOCOL_PATH,
    REVALIDATION_PROTOCOL_PATH,
    PRIOR_REVALIDATION_PROTOCOL_PATH_V3,
    PRIOR_REVALIDATION_PROTOCOL_PATH_V4,
    CURRENT_REVALIDATION_PROTOCOL_PATH,
)
'''
if benchmark.count(old) != 1:
    raise SystemExit("recovery-resilience protocol registry changed")
benchmark_path.write_text(benchmark.replace(old, new, 1), encoding="utf-8")

artifact_old = "recovery_resilience_holdout_revalidation_v4.json"
artifact_new = "recovery_resilience_holdout_revalidation_v5.json"
for relative in (
    "README.md",
    "docs/public-evidence.md",
    "scripts/readme_proof.py",
    "docs/benchmarks/competitive-evidence-matrix.md",
    "tests/test_recovery_resilience.py",
):
    path = ROOT / relative
    text = path.read_text(encoding="utf-8")
    if artifact_old not in text:
        raise SystemExit(f"{relative}: expected v4 artifact reference is missing")
    text = text.replace(artifact_old, artifact_new)
    if relative == "docs/public-evidence.md":
        text = text.replace(
            "The committed v4 recovery-resilience revalidation",
            "The committed v5 recovery-resilience revalidation",
        )
    path.write_text(text, encoding="utf-8")
