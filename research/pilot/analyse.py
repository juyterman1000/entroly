"""Pilot-0 analysis: validity gate, contamination audit, paired comparison.

Pilot-0 is harness validation. Six tasks cannot settle a commercial question --
one failure moves the success rate 16.7 points -- so this script deliberately
refuses to emit a paid-wedge verdict. It reports execution validity, metric
validity, and direction of effect only.

Three accounting decisions are fixed here, before results are read:

1. `uncached_input + output` is NOT called reconstruction. Output can be
   productive work and uncached input can be legitimate new information. The
   combined figure is reported as `noncached_token_volume` and nothing is
   inferred from it about rediscovery.

2. Reconstruction is defined behaviourally: duplicate reads, duplicate searches,
   duplicate test invocations, duplicate edits, and failed/wrong-path commands.
   The primary measure is `rediscovery_operations`, the sum of those duplicate
   counts. Declared here rather than chosen after seeing which variant favours
   B3.

3. Cached input is reported, not discarded. Provider prompt caching is a real
   competitor to Entroly: if B0 or B1 is largely cache-served, Entroly's
   economic advantage shrinks, and that is a legitimate finding rather than
   noise to normalise away.

Handoff payload size is derived from the saved prompts rather than by changing
the harness mid-run: each arm's prompt is on disk, and B0's prompt is the task
statement with no handoff, so `len(arm_prompt) - len(B0_prompt)` is the payload
that arm added.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PILOT_DIR = ROOT / "research" / "ledger" / "pilot"
RESULTS = PILOT_DIR / "results.json"
ARMS = ("B0", "B1", "B3")

# ── §3 contamination audit ─────────────────────────────────────────────
# `disk-full-read-access` was required because CPython lives under
# %LOCALAPPDATA% and `workspace-write` blocks reads outside the workspace. It
# also widens what the agent *could* read, so every command is checked against
# these patterns. Toolchain reads are expected and allowed; anything that could
# carry task answers or unrelated user data is not.
_ALLOWED_OUTSIDE = (
    r"Programs\\Python", r"Program Files", r"System32", r"WindowsPowerShell",
    r"site-packages", r"\\Microsoft\\WindowsApps", r"Get-Command", r"where\.exe",
)
_CONTAMINATION_PATTERNS = {
    "entroly_source_tree": r"entroly-codebase-intelligence|entroly\\|entroly/",
    "prior_benchmark_artifacts": r"research[\\/]ledger|pilot_wave|results\.json|analysis\.json",
    "other_pilot_arms": r"entroly_pilot_.*[\\/](B0|B1|B3)[\\/]",
    "user_documents": r"Documents|Desktop|Downloads|OneDrive",
    "home_config": r"\.ssh|\.aws|\.config[\\/](?!git)|\.gitconfig|\.npmrc",
    "claude_project_state": r"\.claude|\.codex[\\/]sessions",
}


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
    """Wilson score interval. At n=6 or n=24 this is wide, and showing it is the
    point: a bare rate invites overreading a pilot."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3)]


def row_validity(row: dict, manifest: dict) -> tuple[bool, list[str]]:
    """§2 gate. A row is reportable only if every condition holds."""
    reasons: list[str] = []
    if row.get("harness_error"):
        reasons.append(f"harness_exception: {row['harness_error'][:120]}")
    if row.get("codex_exit") not in (0, None):
        reasons.append(f"codex_exit={row.get('codex_exit')}")
    if not row.get("agent_ran_tests_successfully"):
        reasons.append("agent_never_executed_the_test_suite")
    if row.get("verified_success") is None:
        reasons.append("verifier_result_missing")
    if not row.get("checkpoint_head"):
        reasons.append("checkpoint_sha_missing")
    if not row.get("worktree_digest"):
        reasons.append("worktree_digest_missing")
    if manifest.get("native", {}).get("match") is not True:
        reasons.append("native_binary_hash_mismatch")
    if row.get("errors"):
        reasons.append(f"codex_stream_errors={len(row['errors'])}")
    return (not reasons), reasons


def contamination(row: dict) -> dict:
    """§3. Did the agent read anything outside the repo and toolchain?"""
    findings: dict[str, list[str]] = {}
    for command in row.get("commands") or []:
        for label, pattern in _CONTAMINATION_PATTERNS.items():
            if re.search(pattern, command, re.IGNORECASE):
                if any(re.search(ok, command, re.IGNORECASE)
                       for ok in _ALLOWED_OUTSIDE):
                    continue
                findings.setdefault(label, []).append(command[:160])
    return {
        "flagged": bool(findings),
        "categories": sorted(findings),
        "examples": {k: v[:2] for k, v in findings.items()},
    }


def rediscovery_operations(row: dict) -> int:
    """§5 / §24 primary reconstruction measure, declared before results.

    Behavioural, not token-based: repeated work the successor should not have had
    to do. `failed_commands` is included as wrong-path tool calls, which is the
    closest observable to "investigated something already resolved".
    """
    return (
        int(row.get("duplicate_file_reads") or 0)
        + int(row.get("duplicate_commands") or 0)
        + int(row.get("duplicate_edits") or 0)
        + int(row.get("failed_commands") or 0)
    )


def payload_sizes(rows: list[dict]) -> dict[str, dict]:
    """§29. How much state did each arm actually send?

    Taken from the saved prompts. B0's prompt is the task with no handoff, so the
    difference is the handoff payload. Reported so a B3 win by sheer volume is
    visible rather than hidden.
    """
    by_task: dict[str, dict[str, int]] = {}
    for row in rows:
        chars = row.get("prompt_chars")
        if isinstance(chars, int):
            by_task.setdefault(row["task_id"], {})[row["arm"]] = chars
    out: dict[str, dict] = {}
    for arm in ARMS:
        deltas = [
            sizes[arm] - sizes["B0"]
            for sizes in by_task.values()
            if arm in sizes and "B0" in sizes
        ]
        out[arm] = {
            "median_prompt_chars": median(
                [float(s[arm]) for s in by_task.values() if arm in s]
            ),
            "median_handoff_payload_chars": median([float(d) for d in deltas]),
            # ~4 chars per token is the conventional rough ratio; labelled as an
            # estimate because no tokenizer for this provider is available here.
            "median_handoff_payload_tokens_estimate": (
                None if not deltas else round(median([float(d) for d in deltas]) / 4)
            ),
        }
    return out


def main() -> int:
    if not RESULTS.is_file():
        print(f"no results at {RESULTS}")
        return 1
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    rows = payload["results"]
    manifest = payload["manifest"]

    for row in rows:
        valid, reasons = row_validity(row, manifest)
        row["_valid"] = valid
        row["_invalid_reasons"] = reasons
        row["_contamination"] = contamination(row)
        row["_rediscovery_operations"] = rediscovery_operations(row)
        if row["_contamination"]["flagged"]:
            row["_valid"] = False
            row["_invalid_reasons"].append(
                "repo_external_contamination: "
                + ",".join(row["_contamination"]["categories"])
            )

    valid_rows = [r for r in rows if r["_valid"]]
    tasks = sorted({r["task_id"] for r in rows})

    per_arm: dict[str, dict] = {}
    for arm in ARMS:
        arm_rows = [r for r in valid_rows if r.get("arm") == arm]
        n = len(arm_rows)
        successes = sum(1 for r in arm_rows if r.get("verified_success"))
        per_arm[arm] = {
            "valid_n": n,
            "total_n": sum(1 for r in rows if r.get("arm") == arm),
            "verified_completions": successes,
            "verified_completion_rate": round(successes / n, 3) if n else None,
            "verified_completion_wilson95": wilson(successes, n),
            "median_cached_input": median([float(r.get("cached_input_tokens") or 0) for r in arm_rows]),
            "median_uncached_input": median([float(r.get("uncached_input_tokens") or 0) for r in arm_rows]),
            "median_total_input": median([float(r.get("input_tokens") or 0) for r in arm_rows]),
            "median_output": median([float(r.get("output_tokens") or 0) for r in arm_rows]),
            # Explicitly NOT called reconstruction (see module docstring).
            "median_noncached_token_volume": median([
                float(r.get("uncached_input_tokens") or 0) + float(r.get("output_tokens") or 0)
                for r in arm_rows
            ]),
            "median_rediscovery_operations": median(
                [float(r["_rediscovery_operations"]) for r in arm_rows]
            ),
            "rediscovery_iqr": iqr([float(r["_rediscovery_operations"]) for r in arm_rows]),
            "total_duplicate_file_reads": sum(int(r.get("duplicate_file_reads") or 0) for r in arm_rows),
            "total_duplicate_commands": sum(int(r.get("duplicate_commands") or 0) for r in arm_rows),
            "total_duplicate_edits": sum(int(r.get("duplicate_edits") or 0) for r in arm_rows),
            "total_failed_commands": sum(int(r.get("failed_commands") or 0) for r in arm_rows),
            "median_command_count": median([float(r.get("command_count") or 0) for r in arm_rows]),
            "median_wall_seconds": median([float(r.get("wall_seconds") or 0) for r in arm_rows]),
        }

    # ── §25 paired task-level comparison ───────────────────────────────
    paired = []
    for task in tasks:
        entry: dict = {"task_id": task}
        for arm in ARMS:
            row = next((r for r in rows if r["task_id"] == task and r["arm"] == arm), None)
            entry[arm] = None if row is None else {
                "valid": row["_valid"],
                "verified": row.get("verified_success"),
                "uncached_in": row.get("uncached_input_tokens"),
                "cached_in": row.get("cached_input_tokens"),
                "out": row.get("output_tokens"),
                "cmds": row.get("command_count"),
                "rediscovery_ops": row["_rediscovery_operations"],
                "wall_s": row.get("wall_seconds"),
                "first_action": (row.get("first_action") or "")[:100],
            }
        b1, b3 = entry.get("B1"), entry.get("B3")
        if b1 and b3 and b1["valid"] and b3["valid"]:
            entry["delta_B3_minus_B1"] = {
                "uncached_in": b3["uncached_in"] - b1["uncached_in"],
                "out": b3["out"] - b1["out"],
                "rediscovery_ops": b3["rediscovery_ops"] - b1["rediscovery_ops"],
                "cmds": b3["cmds"] - b1["cmds"],
                "wall_s": round(b3["wall_s"] - b1["wall_s"], 1),
                "success_same": b3["verified"] == b1["verified"],
            }
        paired.append(entry)

    report = {
        "phase": "Pilot-0 (harness validation). NOT the commercial test.",
        "n_tasks": len(tasks),
        "statistical_note": (
            "At n=6 one failure moves a success rate by 16.7 points. No "
            "paid-wedge verdict is emitted from this phase."
        ),
        "provenance": {
            "suite_sha256": manifest["suite_sha256"],
            "harness_sha256": manifest["harness_sha256"],
            "entroly_source_commit": manifest["entroly_source_commit"],
            "native_loaded_sha256": manifest["native"]["loaded_sha256"],
            "native_match": manifest["native"]["match"],
            "codex_model": manifest["codex_model"],
            "codex_auth": manifest.get("codex_auth"),
            "sandbox": manifest.get("codex_sandbox"),
        },
        "validity": {
            "valid_rows": len(valid_rows),
            "total_rows": len(rows),
            "invalid": [
                {"task_id": r["task_id"], "arm": r["arm"],
                 "reasons": r["_invalid_reasons"]}
                for r in rows if not r["_valid"]
            ],
            "contamination_flagged_rows": sum(
                1 for r in rows if r["_contamination"]["flagged"]
            ),
            "contamination_detail": [
                {"task_id": r["task_id"], "arm": r["arm"],
                 **r["_contamination"]}
                for r in rows if r["_contamination"]["flagged"]
            ],
        },
        "per_arm": per_arm,
        "payload_sizes": payload_sizes(valid_rows),
        "handoff_generation_cost": {
            "B1": (
                "0 provider tokens BY CONSTRUCTION. The B1 summary is rendered "
                "deterministically from RecordedState by a template, not by a "
                "live Claude call, because no Anthropic key is present in this "
                "environment. This biases END-TO-END economics in B1's favour: "
                "a real agent-written handoff would cost output tokens. The "
                "median payload token estimate under payload_sizes is the "
                "closest available lower bound on what that generation would "
                "have produced."
            ),
            "B3": (
                "0 provider tokens. work_resume is local Rust+Python; its cost "
                "is latency, not provider usage. Measured separately."
            ),
        },
        "paired_tasks": paired,
        "direction_of_effect": None,
    }

    # ── §12 direction only, no verdict ─────────────────────────────────
    b1, b3 = per_arm["B1"], per_arm["B3"]
    if b1["valid_n"] and b3["valid_n"]:
        marks = []
        if b3["verified_completion_rate"] > b1["verified_completion_rate"]:
            marks.append("B3 higher verified completion")
        elif b3["verified_completion_rate"] < b1["verified_completion_rate"]:
            marks.append("B3 lower verified completion")
        else:
            marks.append("verified completion equal")
        if b3["median_rediscovery_operations"] is not None:
            if b3["median_rediscovery_operations"] < b1["median_rediscovery_operations"]:
                marks.append("B3 fewer rediscovery operations")
            elif b3["median_rediscovery_operations"] > b1["median_rediscovery_operations"]:
                marks.append("B3 more rediscovery operations")
            else:
                marks.append("rediscovery operations equal")
        if b3["median_noncached_token_volume"] and b1["median_noncached_token_volume"]:
            marks.append(
                "B3 noncached volume "
                f"{round(100 * (b3['median_noncached_token_volume'] / b1['median_noncached_token_volume'] - 1))}% vs B1"
            )
        report["direction_of_effect"] = marks

    out = PILOT_DIR / "analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k != "paired_tasks"}
    print(json.dumps(printable, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
