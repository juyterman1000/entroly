"""Entroly compression proxy benchmark scoreboard.

This benchmark is local and deterministic. It measures Entroly's Evidence-Locked
Compression proxy surface on the same classes of workloads Headroom publishes:

- build logs,
- JSON arrays,
- SRE incident logs,
- tool payloads.

It does not call an LLM. It tests the part Entroly controls before an API bill:
compression ratio plus preservation of answer-critical evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from entroly.compression_proxy import compress_proxy_payload
from entroly.evidence_locked_compression import estimate_tokens


@dataclass(slots=True)
class ScenarioScore:
    name: str
    original_tokens: int
    compressed_tokens: int
    savings_ratio: float
    evidence_preserved: bool
    receipt_emitted: bool
    passed: bool


@dataclass(slots=True)
class Scoreboard:
    name: str
    passed: bool
    mean_savings_ratio: float
    scenarios: list[ScenarioScore]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "mean_savings_ratio": self.mean_savings_ratio,
            "scenarios": [asdict(s) for s in self.scenarios],
        }


def _run_tool_scenario(name: str, content: str, query: str, evidence: list[str]) -> ScenarioScore:
    body = {
        "model": "scoreboard-model",
        "messages": [
            {"role": "user", "content": query},
            {"role": "tool", "content": content},
        ],
    }
    result = compress_proxy_payload(body, query=query, budget_tokens=900)
    compressed = result.body["messages"][1]["content"]
    original_tokens = estimate_tokens(content)
    compressed_tokens = estimate_tokens(compressed)
    preserved = all(item in compressed for item in evidence)
    passed = result.changed and preserved and result.receipt.receipts and result.receipt.savings_ratio >= 0.50
    return ScenarioScore(
        name=name,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        savings_ratio=result.receipt.savings_ratio,
        evidence_preserved=preserved,
        receipt_emitted=bool(result.receipt.receipts),
        passed=passed,
    )


def build_log_scenario() -> ScenarioScore:
    lines = [f"[build] compiling module_{i} ... ok" for i in range(2200)]
    lines.extend(
        [
            "[build] linking app",
            "src/auth/session.rs:184:9",
            "ERROR: refresh timeout after retry window",
            "hint: increase token refresh slack before retry",
            "Build finished with exit code 1",
        ]
    )
    return _run_tool_scenario(
        "100k-style build log",
        "\n".join(lines),
        "why did the auth build fail",
        ["ERROR: refresh timeout", "src/auth/session.rs:184", "increase token refresh slack"],
    )


def json_array_scenario() -> ScenarioScore:
    rows = []
    for i in range(900):
        rows.append(
            {
                "id": i,
                "service": "billing" if i % 3 else "auth",
                "status": "ok",
                "latency_ms": i % 117,
                "message": "normal request completed",
                "payload": "x" * 60,
            }
        )
    rows[667] = {
        "id": 667,
        "service": "auth",
        "status": "failed",
        "latency_ms": 9912,
        "message": "refresh timeout at retry boundary",
        "payload": "z" * 1000,
    }
    return _run_tool_scenario(
        "deep JSON array",
        json.dumps(rows),
        "find auth refresh timeout failure",
        [
            '"id": 667',
            '"latency_ms": 9912',
            '"message": "refresh timeout at retry boundary"',
            '"status": "failed"',
        ],
    )


def sre_incident_scenario() -> ScenarioScore:
    lines = []
    for i in range(1800):
        lines.append(f"2026-06-28T08:{i % 60:02d}:00Z INFO worker heartbeat shard={i % 32}")
    lines.extend(
        [
            "2026-06-28T08:44:19Z WARN api latency p99 crossed threshold service=auth",
            "2026-06-28T08:44:22Z ERROR upstream refused connection service=auth region=us-west-2",
            "2026-06-28T08:44:25Z FATAL incident INC-9281 auth unavailable",
            "rollback candidate: deploy 2026.06.28.4",
        ]
    )
    return _run_tool_scenario(
        "SRE incident log",
        "\n".join(lines),
        "which incident caused auth outage",
        ["INC-9281", "upstream refused connection", "rollback candidate"],
    )


def run_scoreboard() -> Scoreboard:
    scenarios = [build_log_scenario(), json_array_scenario(), sre_incident_scenario()]
    mean = sum(s.savings_ratio for s in scenarios) / len(scenarios)
    passed = all(s.passed for s in scenarios) and mean >= 0.70
    return Scoreboard(
        name="Entroly Evidence-Locked Compression Proxy Scoreboard",
        passed=passed,
        mean_savings_ratio=round(mean, 4),
        scenarios=scenarios,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Entroly compression proxy scoreboard")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()
    report = run_scoreboard()
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(f"{report.name}: {'PASS' if report.passed else 'FAIL'} mean_savings={report.mean_savings_ratio:.1%}")
        for s in report.scenarios:
            print(
                f"  {'PASS' if s.passed else 'FAIL'} {s.name}: "
                f"{s.original_tokens}->{s.compressed_tokens} "
                f"saved={s.savings_ratio:.1%} evidence={s.evidence_preserved}"
            )
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
