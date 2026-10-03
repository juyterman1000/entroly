"""Pilot-1 analysis with frozen eligibility rules.

Four corrections are implemented here rather than left to prose.

1. ELIGIBILITY IS A TAXONOMY, NOT A BOOLEAN. Each row is classified, and the
   class decides whether it enters effect calculations. The distinction that
   matters: a failure caused by the agent's own broken implementation is a TASK
   FAILURE and counts against its arm; a failure caused by the environment --
   missing interpreter, sandbox refusal, provider quota -- is an ENVIRONMENT
   FAILURE and is excluded, because it says nothing about continuation quality.
   Where the evidence cannot decide, the row is UNKNOWN and excluded, never
   silently counted as either.

2. VERIFIER EVIDENCE COMES FROM THE COLLECTOR, NOT THE AGENT TRACE. Codex traces
   contain no record of the harness's independent verifier run, so verifier
   outcome is read from the collector row only. A row whose collector record
   lacks a verifier result is UNKNOWN -- distinguishable from a row that
   recorded a genuine failure.

3. DENOMINATORS ARE STATED AND MATCHED. Success rates use matched eligible
   tasks: a task enters only if ALL arms are eligible for it, so every
   comparison is paired on the same checkpoint. The usage guard is computed on
   MEDIANS over those matched tasks, and the "three additional successes =
   12.5pp" arithmetic holds only at n=24; the realised denominator is reported
   next to it so the discreteness is visible rather than implied.

4. THE COMPARISON IS NAMED FOR WHAT IT MEASURES. Not "representation value":
   B3 carries ~5.7x the payload of B1S, so a win cannot be attributed to
   structure. It is PRODUCTION CONTINUATION PAYLOAD VALUE -- does the shipped
   work_resume payload outperform a compact text projection of the same broad
   recorded-state categories. Larger payload is not evidence of more semantic
   information either; it is reported as a confound, not as a mechanism.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PILOT = ROOT / "research" / "ledger" / "pilot1"
ARMS = ("B0", "B1S", "B3")

# Provider/sandbox refusals. These are environment, never task quality.
_ENV_SIGNATURES = (
    r"hit your usage limit",
    r"usage limit",
    r"rate.?limit",
    r"quota",
    r"OAuth session expired",
    r"not recognized as the name of a cmdlet",
    r"requires a newer version of Codex",
    r"is not supported when using Codex",
)

_CONTAMINATION = {
    "entroly_source_tree": r"entroly-codebase-intelligence|entroly[\\/](?!pilot)",
    "prior_pilot_artifacts": r"research[\\/]ledger|pilot_wave|pilot1[\\/]|results\.json",
    "other_arms": r"[\\/](?:B0|B1S|B3)[\\/]",
    "user_documents": r"Documents|Desktop|Downloads|OneDrive",
    "home_config": r"\.ssh|\.aws|\.gitconfig|\.npmrc",
    "agent_session_history": r"\.claude|\.codex[\\/]sessions",
}
_ALLOWED_OUTSIDE = (
    r"Programs\\Python", r"Program Files", r"System32", r"WindowsPowerShell",
    r"site-packages", r"WindowsApps", r"Get-Command", r"where\.exe",
)


def classify(row: dict, manifest: dict) -> tuple[str, list[str]]:
    """Return (class, reasons). Only ELIGIBLE rows enter effect calculations."""
    reasons: list[str] = []

    if row.get("harness_error"):
        return "HARNESS_FAILURE", [row["harness_error"][:160]]

    blob = json.dumps(row.get("errors") or [])
    for pattern in _ENV_SIGNATURES:
        if re.search(pattern, blob, re.I):
            return "ENVIRONMENT_FAILURE", [f"provider/sandbox refusal: {pattern}"]

    # Zero activity with no error recorded: the evidence cannot say why.
    if (row.get("command_count") or 0) == 0 and not (row.get("errors") or []):
        return "UNKNOWN", ["agent performed no action and no error was recorded"]

    # Verifier evidence must come from the collector record.
    if row.get("verified_success") is None or "verifier_tail" not in row:
        return "UNKNOWN", ["verifier outcome not recorded by the collector"]

    if manifest.get("native", {}).get("match") is not True:
        return "HARNESS_FAILURE", ["native binary hash mismatch"]
    if not row.get("checkpoint_head") or not row.get("worktree_digest"):
        return "HARNESS_FAILURE", ["checkpoint identity missing"]

    for label, pattern in _CONTAMINATION.items():
        for command in row.get("commands") or []:
            if re.search(pattern, command, re.I) and not any(
                re.search(ok, command, re.I) for ok in _ALLOWED_OUTSIDE
            ):
                return "CONTAMINATED", [f"{label}: {command[:120]}"]

    # The agent never ran the suite, but no environment cause was found. That is
    # a property of how the agent worked, so it stays eligible and is flagged.
    if not row.get("agent_ran_tests_successfully"):
        reasons.append("agent did not successfully execute the suite "
                       "(eligible: no environment cause found)")

    return "ELIGIBLE", reasons


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 1) if values else None


def iqr(values: list[float]) -> list[float] | None:
    if len(values) < 4:
        return None
    ordered = sorted(values)
    half = len(ordered) // 2
    return [round(statistics.median(ordered[:half]), 1),
            round(statistics.median(ordered[-half:]), 1)]


def wilson(successes: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3)]


def main() -> int:
    results_path = PILOT / "results.json"
    if not results_path.is_file():
        print(f"no results at {results_path}")
        return 1
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload["results"]
    manifest = payload["manifest"]

    for row in rows:
        row["_class"], row["_reasons"] = classify(row, manifest)

    by_class: dict[str, int] = {}
    for row in rows:
        by_class[row["_class"]] = by_class.get(row["_class"], 0) + 1

    # Matched eligible tasks: every arm eligible, so comparisons stay paired.
    tasks = sorted({r["task_id"] for r in rows})
    matched = [
        t for t in tasks
        if all(
            any(r["task_id"] == t and r["arm"] == arm and r["_class"] == "ELIGIBLE"
                for r in rows)
            for arm in ARMS
        )
    ]
    dropped = {
        t: {r["arm"]: {"class": r["_class"], "reasons": r["_reasons"]}
            for r in rows if r["task_id"] == t}
        for t in tasks if t not in matched
    }

    def arm_rows(arm: str) -> list[dict]:
        return [r for r in rows
                if r["arm"] == arm and r["task_id"] in matched
                and r["_class"] == "ELIGIBLE"]

    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        ars = arm_rows(arm)
        n = len(ars)
        successes = sum(1 for r in ars if r.get("verified_success"))
        redis = [float(r.get("avoidable_rediscovery_operations") or 0) for r in ars]
        prog = [float((r.get("first_progress_action") or {}).get("index"))
                for r in ars
                if (r.get("first_progress_action") or {}).get("index") is not None]
        per_arm[arm] = {
            "n": n,
            "verified_successes": successes,
            "verified_rate": round(successes / n, 3) if n else None,
            "wilson95": wilson(successes, n),
            "rediscovery_median": median(redis),
            "rediscovery_iqr": iqr(redis),
            "first_progress_median": median(prog),
            "cached_input_median": median([float(r.get("cached_input_tokens") or 0) for r in ars]),
            "uncached_input_median": median([float(r.get("uncached_input_tokens") or 0) for r in ars]),
            "total_input_median": median([float(r.get("input_tokens") or 0) for r in ars]),
            "output_median": median([float(r.get("output_tokens") or 0) for r in ars]),
            "noncached_usage_proxy_median": median([
                float(r.get("uncached_input_tokens") or 0)
                + float(r.get("output_tokens") or 0) for r in ars
            ]),
            "wall_seconds_median": median([float(r.get("wall_seconds") or 0) for r in ars]),
            "command_count_median": median([float(r.get("command_count") or 0) for r in ars]),
            "failed_commands_total": sum(int(r.get("failed_commands") or 0) for r in ars),
            "handoff_tokens_median": median([float(r.get("handoff_tokens_estimate") or 0) for r in ars]),
        }

    paired = []
    for task in matched:
        entry: dict = {"task_id": task}
        for arm in ARMS:
            r = next(x for x in rows if x["task_id"] == task and x["arm"] == arm)
            entry[arm] = {
                "verified": r.get("verified_success"),
                "redisc": r.get("avoidable_rediscovery_operations"),
                "progress_idx": (r.get("first_progress_action") or {}).get("index"),
                "uncached": r.get("uncached_input_tokens"),
                "output": r.get("output_tokens"),
                "handoff_tok": r.get("handoff_tokens_estimate"),
                "cmds": r.get("command_count"),
            }
            entry["stratum"] = r.get("stratum")
        b1s, b3 = entry["B1S"], entry["B3"]
        entry["delta_B3_minus_B1S"] = {
            "redisc": (b3["redisc"] or 0) - (b1s["redisc"] or 0),
            "uncached": (b3["uncached"] or 0) - (b1s["uncached"] or 0),
            "output": (b3["output"] or 0) - (b1s["output"] or 0),
        }
        paired.append(entry)

    b1s, b3 = per_arm["B1S"], per_arm["B3"]
    decision: dict = {
        "comparison_name": "Production continuation payload value (NOT representation value)",
        "why_not_representation": (
            "B3 carries ~5.7x the payload of B1S, so any benefit cannot be "
            "attributed to structure. Larger payload is also not evidence of "
            "more semantic information; it is a confound, not a mechanism."
        ),
        "matched_eligible_tasks": len(matched),
        "planned_tasks": manifest.get("suite_task_count"),
        "discreteness_note": (
            f"'3 additional successes = 12.5pp' holds at n=24. Realised matched "
            f"n={len(matched)}, so one task = "
            f"{round(100 / len(matched), 2) if matched else 'n/a'}pp."
        ),
    }
    if b1s["n"] and b3["n"]:
        pp = (b3["verified_rate"] - b1s["verified_rate"]) * 100
        decision["success_delta_pp"] = round(pp, 2)
        decision["success_threshold_passed"] = pp >= 10.0
        if (b1s["rediscovery_median"] or 0) == 0 and (b3["rediscovery_median"] or 0) == 0:
            decision["rediscovery_threshold"] = "NOT RESOLVABLE (both medians 0)"
        else:
            red = (b1s["rediscovery_median"] - b3["rediscovery_median"]) / \
                  max(b1s["rediscovery_median"], 1e-9)
            decision["rediscovery_reduction"] = round(red, 3)
            decision["rediscovery_threshold_passed"] = red >= 0.20
        guard_base = b1s["noncached_usage_proxy_median"] or 0
        if guard_base:
            inc = (b3["noncached_usage_proxy_median"] - guard_base) / guard_base
            decision["noncached_usage_increase"] = round(inc, 3)
            decision["usage_guard_passed"] = inc <= 0.10
            decision["usage_guard_basis"] = "medians over matched eligible tasks"

    report = {
        "phase": "Pilot-1 (ContinuityStress-24)",
        "provenance": {
            "suite_sha256": manifest["suite_sha256"],
            "harness_sha256": manifest["harness_sha256"],
            "rediscovery_sha256": manifest["rediscovery_sha256"],
            "entroly_source_commit": manifest["entroly_source_commit"],
            "native_loaded_sha256": manifest["native"]["loaded_sha256"],
            "native_match": manifest["native"]["match"],
            "codex_model": manifest["codex_model"],
            "codex_auth": manifest["codex_auth"],
        },
        "eligibility": {
            "rows_total": len(rows),
            "by_class": by_class,
            "matched_eligible_tasks": matched,
            "dropped_tasks": dropped,
        },
        "per_arm": per_arm,
        "paired": paired,
        "decision": decision,
    }
    out = PILOT / "analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k not in ("paired",)}
    printable["eligibility"] = {
        **report["eligibility"],
        "dropped_tasks": f"{len(dropped)} tasks (detail in analysis.json)",
    }
    print(json.dumps(printable, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
