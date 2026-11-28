"""Strict paired evaluation of agent runs (JSON Lines, one record per task/arm).

Run: python benchmarks/competitive_gate.py results.jsonl --candidate entroly --baseline native
Each row requires task, arm, success (bool), input_tokens, output_tokens,
cost_usd, latency_ms, activated (bool). Costs must include retries and tool/model
calls; task success must come from an external evaluator, not model self-report.
"""
import argparse
import json
import math
import statistics


def evaluate(rows, candidate, baseline, min_tasks=30):
    if candidate == baseline or min_tasks < 1:
        raise ValueError("distinct arms and positive minimum task count required")
    arms = {candidate: {}, baseline: {}}
    for row in rows:
        arm = row["arm"]
        if arm not in arms:
            continue
        task = row["task"]
        if not isinstance(task, str) or not task or task in arms[arm]:
            raise ValueError("invalid or duplicate task: " + str(task))
        for key in ("success", "activated"):
            if type(row[key]) is not bool:
                raise ValueError(key + " must be boolean")
        for key in ("input_tokens", "output_tokens", "cost_usd", "latency_ms"):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid metric: " + key)
            if key.endswith("_tokens") and int(value) != value:
                raise ValueError("token counts must be integral")
        arms[arm][task] = row
    c, b = arms[candidate], arms[baseline]
    if c.keys() != b.keys() or len(c) < min_tasks:
        raise ValueError("incomplete pairing or insufficient tasks")
    n = len(c)
    def total(arm, key):
        return sum(row[key] for row in arm.values())
    def reduction(key):
        base = total(b, key)
        return None if base == 0 else 1 - total(c, key) / base
    regressions = sum(b[t]["success"] and not c[t]["success"] for t in c)
    improvements = sum(c[t]["success"] and not b[t]["success"] for t in c)
    activation = total(c, "activated") / n
    cost = reduction("cost_usd")
    tokens = reduction("input_tokens")
    # Conservative screening gate, not a statistical superiority claim.
    passed = (regressions == 0 and activation == 1 and cost is not None
              and cost >= .20 and tokens is not None and tokens >= .30)
    return {"tasks": n, "candidate_success_rate": total(c, "success") / n,
            "baseline_success_rate": total(b, "success") / n,
            "regressions": regressions, "improvements": improvements,
            "activation_rate": activation, "cost_reduction": cost,
            "input_token_reduction": tokens,
            "output_token_reduction": reduction("output_tokens"),
            "candidate_median_latency_ms": statistics.median(r["latency_ms"] for r in c.values()),
            "baseline_median_latency_ms": statistics.median(r["latency_ms"] for r in b.values()),
            "screening_gate_passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--min-tasks", type=int, default=30)
    args = parser.parse_args()
    try:
        with open(args.results, encoding="utf-8") as stream:
            report = evaluate((json.loads(line) for line in stream if line.strip()),
                              args.candidate, args.baseline, args.min_tasks)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["screening_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
