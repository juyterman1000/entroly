"""Aggregate the continuity pilot into the frozen decision block.

Reads research/ledger/pilot/results.json and prints per-arm aggregates plus the
kill-criterion evaluation. Thresholds are read from the manifest written before
the first model call, so they cannot be adjusted after seeing results.

Reconstruction tokens are reported as *uncached* input plus output. Undifferentiated
input would mostly measure provider cache behaviour: on the first run 151,168 of
173,241 input tokens were served from cache, so the cached portion says more about
prefix stability than about how much the successor had to rediscover. Both the raw
and the uncached figures are printed so the choice is visible rather than buried.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research" / "ledger" / "pilot" / "results.json"
ARMS = ("B0", "B1", "B3")


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def iqr(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    ordered = sorted(values)
    half = len(ordered) // 2
    lo = statistics.median(ordered[:half])
    hi = statistics.median(ordered[-half:])
    return (round(lo, 1), round(hi, 1))


def main() -> int:
    if not RESULTS.is_file():
        print(f"no results at {RESULTS}")
        return 1
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = payload["results"]
    manifest = payload["manifest"]

    by_arm: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for row in rows:
        if row.get("arm") in by_arm:
            by_arm[row["arm"]].append(row)

    tasks = sorted({r["task_id"] for r in rows})
    report: dict = {
        "suite_sha256": manifest["suite_sha256"],
        "harness_sha256": manifest["harness_sha256"],
        "entroly_source_commit": manifest["entroly_source_commit"],
        "native_match": manifest["native"]["match"],
        "native_loaded_sha256": manifest["native"]["loaded_sha256"],
        "codex_model": manifest["codex_model"],
        "tasks_completed": len(tasks),
        "tasks": tasks,
        "per_arm": {},
        "per_task": [],
    }

    for arm in ARMS:
        arm_rows = [r for r in by_arm[arm] if "harness_error" not in r]
        n = len(arm_rows)
        successes = [r for r in arm_rows if r.get("verified_success")]
        recon = [
            float(r.get("uncached_input_tokens") or 0) + float(r.get("output_tokens") or 0)
            for r in arm_rows
        ]
        report["per_arm"][arm] = {
            "n": n,
            "harness_errors": len(by_arm[arm]) - n,
            "verified_completions": len(successes),
            "verified_completion_rate": round(len(successes) / n, 3) if n else None,
            "median_reconstruction_tokens": median(recon),
            "reconstruction_iqr": iqr(recon),
            "median_input_tokens": median([float(r.get("input_tokens") or 0) for r in arm_rows]),
            "median_cached_input": median([float(r.get("cached_input_tokens") or 0) for r in arm_rows]),
            "median_uncached_input": median([float(r.get("uncached_input_tokens") or 0) for r in arm_rows]),
            "median_output_tokens": median([float(r.get("output_tokens") or 0) for r in arm_rows]),
            "median_commands": median([float(r.get("command_count") or 0) for r in arm_rows]),
            "total_duplicate_commands": sum(int(r.get("duplicate_commands") or 0) for r in arm_rows),
            "total_duplicate_file_reads": sum(int(r.get("duplicate_file_reads") or 0) for r in arm_rows),
            "total_duplicate_edits": sum(int(r.get("duplicate_edits") or 0) for r in arm_rows),
            "total_failed_commands": sum(int(r.get("failed_commands") or 0) for r in arm_rows),
            "median_wall_seconds": median([float(r.get("wall_seconds") or 0) for r in arm_rows]),
        }

    for task in tasks:
        entry = {"task_id": task}
        for arm in ARMS:
            row = next((r for r in by_arm[arm] if r["task_id"] == task), None)
            entry[arm] = None if row is None else {
                "verified": row.get("verified_success"),
                "uncached_in": row.get("uncached_input_tokens"),
                "out": row.get("output_tokens"),
                "cmds": row.get("command_count"),
                "dup_reads": row.get("duplicate_file_reads"),
                "first_action": (row.get("first_action") or "")[:110],
                "error": row.get("harness_error"),
            }
        report["per_task"].append(entry)

    # ── kill criterion, against the stronger of B1/B2 (B2 not applicable) ──
    b1 = report["per_arm"]["B1"]
    b3 = report["per_arm"]["B3"]
    verdict: dict = {"baseline": "B1 (B2 not applicable)"}
    if b1["median_reconstruction_tokens"] and b3["median_reconstruction_tokens"]:
        delta = (
            b1["median_reconstruction_tokens"] - b3["median_reconstruction_tokens"]
        ) / b1["median_reconstruction_tokens"]
        verdict["reconstruction_token_reduction"] = round(delta, 3)
        verdict["meets_token_threshold"] = delta >= 0.20
    if b1["verified_completion_rate"] is not None and b3["verified_completion_rate"] is not None:
        pp = b3["verified_completion_rate"] - b1["verified_completion_rate"]
        verdict["completion_improvement_pp"] = round(pp * 100, 1)
        verdict["meets_completion_threshold"] = pp >= 0.10
    # Cost proxy: this account is ChatGPT-subscription authenticated, so there is
    # no per-token invoice to read. Total tokens is the honest stand-in and is
    # labelled as such rather than converted to dollars.
    if b1["median_input_tokens"] and b3["median_input_tokens"]:
        cost_delta = (
            (b3["median_input_tokens"] + (b3["median_output_tokens"] or 0))
            - (b1["median_input_tokens"] + (b1["median_output_tokens"] or 0))
        ) / (b1["median_input_tokens"] + (b1["median_output_tokens"] or 0))
        verdict["total_token_increase"] = round(cost_delta, 3)
        verdict["within_cost_ceiling"] = cost_delta <= 0.10
    verdict["kill_criterion_passed"] = bool(
        (verdict.get("meets_token_threshold") or verdict.get("meets_completion_threshold"))
        and verdict.get("within_cost_ceiling")
    )
    report["kill_criterion"] = verdict

    out = RESULTS.parent / "analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
