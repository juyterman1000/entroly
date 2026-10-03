"""Pilot-1 Attempt 2 runner: immutable rows, narrow circuit breaker, safe resume.

The scientific treatment is unchanged from Attempt 1 -- same ContinuityStress-24
tasks, same checkpoints, same B0/B1S/B3 content, same verifiers, same Codex model
and sandbox, same Entroly build, same rediscovery scorer. Only the *operational*
harness differs, so the harness hash necessarily differs too and that is recorded
rather than glossed over.

Attempt 1 failed operationally in three ways this file fixes:

  * it continued through 57 provider refusals after the quota was gone, because
    nothing inspected why a row produced no work;
  * it rewrote one monolithic results.json after every task, so an interruption
    left no way to distinguish a complete row from a truncated one;
  * its manifest named the usage quantity but not the aggregation, which turned
    out to matter: median showed -21.2% and sum showed -4.3% on the same five
    tasks.

Three distinctions are load-bearing throughout:

  TERMINAL PROVIDER FAILURE stops the attempt. Quota exhausted, global auth
  rejection, model globally unavailable. Further model calls cannot succeed, so
  continuing only destroys quota and fills the record with zero-work rows.

  ROW-LEVEL ENVIRONMENT FAILURE invalidates one row and continues. A missing
  interpreter or a task-specific sandbox problem says nothing about whether the
  provider can serve the next row.

  SCIENTIFIC FAILURE is evidence and must never stop anything. The agent's code
  failing pytest, a nonzero command, the verifier rejecting the solution, B3
  losing to B1S -- these are the measurements.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import harness as base  # noqa: E402
import rediscovery as rd  # noqa: E402

ATTEMPT_ID = "pilot1-attempt2"
OUT_DIR = ROOT / "research" / "ledger" / "pilot1_attempt2"
ROWS_DIR = OUT_DIR / "rows"
TRACE_DIR = OUT_DIR / "traces"
ARMS = ("B0", "B1S", "B3")

# ── row lifecycle ──────────────────────────────────────────────────────
NOT_STARTED = "NOT_STARTED"
RUNNING = "RUNNING"
COMPLETE = "COMPLETE"              # includes verified_success=False: evidence
ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
INTERRUPTED = "INTERRUPTED"

# ── provider failure classification (frozen) ───────────────────────────
# Attempt-level terminal: no later model call can succeed.
TERMINAL_PROVIDER = {
    "PROVIDER_QUOTA_EXHAUSTED": (
        r"hit your usage limit",
        r"purchase more credits",
        r"Upgrade to Pro",
        r"usage limit reached",
        r"credit.{0,12}exhaust",
    ),
    "PROVIDER_AUTH_FAILURE": (
        r"OAuth session expired",
        r"Failed to authenticate",
        r"invalid[_ ]api[_ ]key",
        r"401 Unauthorized",
    ),
    "MODEL_UNAVAILABLE_GLOBAL": (
        r"is not supported when using Codex",
        r"requires a newer version of Codex",
        r"model.{0,20}(?:not available|unavailable|not found)",
    ),
}
# Row-level only: one row is invalid, the attempt continues.
ROW_ENVIRONMENT = (
    r"not recognized as the name of a cmdlet",
    r"command not found",
    r"No such file or directory",
    r"Permission denied",
)


def classify_provider(stream: str) -> tuple[str | None, str | None]:
    """Return (terminal_class, matched_text) or (None, None).

    Scientific failures are never matched here: the patterns describe provider
    and auth refusals only, so pytest output and nonzero exits cannot trip it.
    """
    for label, patterns in TERMINAL_PROVIDER.items():
        for pattern in patterns:
            found = re.search(pattern, stream, re.I)
            if found:
                line = stream[max(0, found.start() - 80): found.end() + 80]
                return label, line.replace("\n", " ")[:240]
    return None, None


def classify_row_environment(stream: str) -> str | None:
    for pattern in ROW_ENVIRONMENT:
        if re.search(pattern, stream, re.I):
            return pattern
    return None


# ── atomic immutable persistence ───────────────────────────────────────

def row_path(task_id: str, arm: str) -> pathlib.Path:
    return ROWS_DIR / f"{task_id}.{arm}.json"


def row_digest(row: dict) -> str:
    """Digest over everything but the digest field, so integrity is checkable."""
    payload = {k: v for k, v in row.items() if k != "row_digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def write_row_atomic(row: dict, *, allow_overwrite: bool = False) -> pathlib.Path:
    """Temp file -> fsync -> atomic rename. Never exposes a partial row.

    Refuses to replace an existing COMPLETE row: a completed treatment execution
    is the scientific record and must not be silently superseded.
    """
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    target = row_path(row["task_id"], row["arm"])
    if target.exists() and not allow_overwrite:
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE:
            raise RuntimeError(
                f"refusing to overwrite COMPLETE row {target.name}"
            )
    row["row_digest"] = row_digest(row)
    handle, tmp_name = tempfile.mkstemp(dir=str(ROWS_DIR), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        pathlib.Path(tmp_name).unlink(missing_ok=True)
        raise
    return target


def load_row(task_id: str, arm: str) -> dict | None:
    path = row_path(task_id, arm)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def can_skip(task_id: str, arm: str, manifest: dict, task_hash: str,
             arm_hash: str) -> tuple[bool, str]:
    """Resume gate. File existence alone is never sufficient."""
    row = load_row(task_id, arm)
    if row is None:
        return False, "absent or unreadable"
    if row.get("status") != COMPLETE:
        return False, f"status={row.get('status')}"
    if row.get("attempt_id") != manifest["attempt_id"]:
        return False, "different attempt_id (no cross-attempt reuse)"
    for field, expected in (
        ("manifest_sha256", manifest["manifest_sha256"]),
        ("task_sha256", task_hash),
        ("arm_definition_sha256", arm_hash),
    ):
        if row.get(field) != expected:
            return False, f"{field} mismatch"
    if row.get("row_digest") != row_digest(row):
        return False, "row integrity digest mismatch"
    trace = row.get("trace_path")
    if trace:
        tp = ROOT / trace
        if not tp.is_file():
            return False, "trace missing"
        if row.get("trace_sha256") != base.sha256_file(tp):
            return False, "trace integrity mismatch"
    return True, "validated COMPLETE"


# ── hashes of the frozen treatment ─────────────────────────────────────

def task_hash(task) -> str:
    return base.sha256_text(json.dumps(asdict(task), sort_keys=True, default=str))


def arm_definition_hash(arm: str, task) -> str:
    """Hash the arm's actual prompt content, so a treatment change is visible."""
    if arm == "B0":
        body = ""
    elif arm == "B1S":
        body = base.b1_handoff(task)
    else:
        body = "B3:work_resume"       # payload is produced live; see prompt hash
    return base.sha256_text(f"{arm}\n{body}")


# ── one invocation ─────────────────────────────────────────────────────

def invoke(task, arm: str, manifest: dict, invocation: int,
           quota_boundary_phase: str) -> dict:
    work = pathlib.Path(os.environ.get("TEMP", "/tmp")) / (
        f"a2_{task.task_id}_{arm}_{int(time.time())}"
    )
    if work.exists():
        shutil.rmtree(work)
    repo = base.build_checkpoint(task, work)
    head = base.git(repo, "rev-parse", "HEAD")
    digest = base.worktree_digest(repo)

    parts = ["You are continuing work another agent started and could not finish.",
             "", f"## Task\n{task.statement}"]
    handoff = ""
    b3_latency = None
    if arm == "B1S":
        handoff = base.b1_handoff(task)
        parts += ["", handoff]
    elif arm == "B3":
        started = time.perf_counter()
        handoff, _payload = base.b3_payload(task, repo)
        b3_latency = round(time.perf_counter() - started, 3)
        parts += ["", handoff]
    parts += ["", "Finish the task. Run the test suite to confirm before you stop."]
    prompt = "\n".join(parts)

    env = dict(os.environ)
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "safe.directory"
    env["GIT_CONFIG_VALUE_0"] = "*"

    started_at = time.time()
    proc = subprocess.run(
        ["codex", "exec", "--json", "-C", str(repo), "-s", "workspace-write",
         "-m", base.CODEX_MODEL, "--skip-git-repo-check",
         "-c", "shell_environment_policy.inherit=all",
         "-c", 'sandbox_permissions=["disk-full-read-access"]',
         prompt],
        capture_output=True, text=True, timeout=base.TIMEOUT_S,
        encoding="utf-8", errors="replace", env=env,
    )
    ended_at = time.time()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    trace_file = TRACE_DIR / f"{task.task_id}.{arm}.inv{invocation:03d}.jsonl"
    trace_file.write_text(proc.stdout, encoding="utf-8")
    (TRACE_DIR / f"{task.task_id}.{arm}.inv{invocation:03d}.prompt.txt").write_text(
        prompt, encoding="utf-8"
    )

    terminal, matched = classify_provider(proc.stdout)
    row_env = None if terminal else classify_row_environment(proc.stdout)

    row: dict = {
        "attempt_id": manifest["attempt_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "task_id": task.task_id,
        "stratum": task.stratum,
        "arm": arm,
        "invocation": invocation,
        "task_sha256": task_hash(task),
        "arm_definition_sha256": arm_definition_hash(arm, task),
        "checkpoint_head": head,
        "worktree_digest": digest,
        "prompt_sha256": base.sha256_text(prompt),
        "prompt_chars": len(prompt),
        "handoff_chars": len(handoff),
        "handoff_tokens_estimate": round(len(handoff) / 4) if handoff else 0,
        # Both arms render their handoff locally, so neither spends provider
        # tokens generating it. Recorded explicitly because the primary usage
        # rule includes this term.
        "handoff_generation_provider_tokens": 0,
        "model": base.CODEX_MODEL,
        "codex_cli_version": manifest["codex_cli_version"],
        "sandbox": manifest["sandbox"],
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": round(ended_at - started_at, 2),
        "codex_exit": proc.returncode,
        "trace_path": str(trace_file.relative_to(ROOT)).replace("\\", "/"),
        "trace_sha256": base.sha256_file(trace_file),
        "quota_boundary_phase": quota_boundary_phase,
        "b3_local_latency_s": b3_latency,
    }

    if terminal:
        row["status"] = ENVIRONMENT_FAILURE
        row["environment_class"] = terminal
        row["environment_detail"] = matched
        row["verified_success"] = None
        shutil.rmtree(work, ignore_errors=True)
        return row

    # The verifier is run by the harness. Its outcome is collector evidence and
    # never appears in the agent trace.
    verified = subprocess.run(task.verifier, cwd=repo, capture_output=True,
                              text=True)
    metrics = base.parse_events(proc.stdout)
    recorded = {"rejected": list(task.recorded.rejected),
                "remaining_work": list(task.recorded.remaining_work)}
    scored = rd.score(proc.stdout, recorded)

    row.update({
        "verified_success": verified.returncode == 0,
        "verifier_returncode": verified.returncode,
        "verifier_tail": verified.stdout[-400:],
        "first_progress_action": rd.first_progress_action(proc.stdout, recorded),
        **metrics,
        **{k: v for k, v in scored.items()
           if k not in ("duplicate_read_detail", "duplicate_diagnostic_detail")},
    })
    if row_env and not metrics.get("command_count"):
        row["status"] = ENVIRONMENT_FAILURE
        row["environment_class"] = "ROW_ENVIRONMENT_FAILURE"
        row["environment_detail"] = row_env
    else:
        # COMPLETE regardless of verified_success. A legitimate task failure is
        # the experiment working, not the environment failing.
        row["status"] = COMPLETE
        row["environment_class"] = None
    shutil.rmtree(work, ignore_errors=True)
    return row


# ── manifest ───────────────────────────────────────────────────────────

def build_manifest(suite) -> dict:
    suite_text = json.dumps([asdict(t) for t in suite], sort_keys=True, default=str)
    cli = subprocess.run(["codex", "--version"], capture_output=True,
                         text=True).stdout.strip()
    run_order = {}
    for index, task in enumerate(suite):
        order = list(ARMS)
        random.Random(2000 + index).shuffle(order)
        run_order[task.task_id] = order
    manifest = {
        "attempt_id": ATTEMPT_ID,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "suite_type": "ContinuityStress-24",
        "suite_note": (
            "Every task carries a planted trap. Estimates BENEFIT when "
            "continuity matters; says nothing about how often it does. Not "
            "representative coding work."
        ),
        "suite_sha256": base.sha256_text(suite_text),
        "task_file_sha256": base.sha256_file(HERE / "tasks_pilot1.py"),
        "attempt2_harness_sha256": base.sha256_file(pathlib.Path(__file__)),
        "base_harness_sha256": base.sha256_file(HERE / "harness.py"),
        "scorer_sha256": base.sha256_file(HERE / "rediscovery.py"),
        "entroly_head": base.git(ROOT, "rev-parse", "HEAD"),
        "native": base.native_provenance(),
        "python_version": sys.version.split()[0],
        "codex_cli_version": cli,
        "codex_model": base.CODEX_MODEL,
        "codex_auth": "ChatGPT account (no token invoice; usage proxies only)",
        "sandbox": "workspace-write + disk-full-read-access + inherit env",
        "arms": {
            "B0": "repository checkpoint + task statement only",
            "B1S": "+ deterministic local text rendering of RecordedState "
                   "(0 provider tokens to generate)",
            "B3": "+ production work_resume payload only "
                  "(0 provider tokens to generate)",
            "B1N": "NOT FEASIBLE: Claude CLI OAuth expired; no ANTHROPIC_API_KEY",
        },
        "task_hashes": {t.task_id: task_hash(t) for t in suite},
        "arm_definition_hashes": {
            t.task_id: {a: arm_definition_hash(a, t) for a in ARMS} for t in suite
        },
        "run_order": run_order,
        "eligibility_taxonomy_version": "2",
        "contamination_audit_version": "1",
        # ── analysis contract, frozen before the first benchmark call ──
        "primary_comparison": "B3 vs B1S",
        "headline_name": (
            "Production continuation payload value (NOT representation value: "
            "B3 is materially larger than B1S, so structure cannot be isolated)"
        ),
        "matched_primary_rule": "tasks where BOTH B1S and B3 are scientifically eligible",
        "matched_three_arm_rule": "tasks where B0, B1S and B3 are all eligible",
        "success_metric": "verified_success from the independent verifier only",
        "success_denominator": "N = |MATCHED_PRIMARY_TASKS|; one discordant task = 100/N pp",
        "success_threshold_pp": 10.0,
        "paired_diagnostic": (
            "exact McNemar/binomial on discordant pairs; CONFIDENCE DIAGNOSTIC "
            "ONLY, does not replace the >=10pp threshold"
        ),
        "usage_metric": "uncached_input + output + handoff_generation_provider_tokens",
        "usage_aggregation": "RATIO OF SUMS over MATCHED_PRIMARY_TASKS",
        "usage_guard": "B3_total / B1S_total <= 1.10",
        "usage_sensitivity": (
            "per-task relative delta, median and IQR, reported as SENSITIVITY "
            "only. Attempt 1 showed median -21.2% vs sum -4.3% on the same 5 "
            "tasks, so aggregation materially changes apparent magnitude."
        ),
        "rediscovery_metric": "avoidable_rediscovery_operations (frozen scorer)",
        "rediscovery_rule": (
            ">=20% median reduction; if both medians are 0 report NOT RESOLVABLE "
            "and never manufacture a percentage from zero"
        ),
        "circuit_breaker": {
            "terminal_classes": sorted(TERMINAL_PROVIDER),
            "semantics": "stop further model calls; preserve all COMPLETE rows",
            "never_fires_on": "scientific failure (agent error, pytest failure, "
                              "nonzero exit, verifier rejection, any arm losing)",
        },
        "resume_semantics": (
            "skip a row only when status=COMPLETE and attempt_id, manifest, task, "
            "arm-definition, row digest and trace digest all validate. No "
            "cross-attempt reuse; Attempt-1 rows can never satisfy this."
        ),
        "quota_boundary_rule": (
            "rows tagged PRE_INTERRUPTION / POST_RESUME; final analysis must "
            "compare cached and uncached input across that boundary and must not "
            "attribute a difference to Entroly"
        ),
    }
    # Self-hash last, over everything else.
    manifest["manifest_sha256"] = base.sha256_text(
        json.dumps({k: v for k, v in manifest.items()}, sort_keys=True, default=str)
    )
    return manifest


def freeze_manifest(suite) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        print(f"manifest already frozen at {existing['frozen_at']}")
        return existing
    manifest = build_manifest(suite)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_frozen(manifest: dict) -> list[str]:
    """Refuse to run if the frozen treatment no longer matches disk."""
    problems: list[str] = []
    from tasks_pilot1 import SUITE

    suite_text = json.dumps([asdict(t) for t in SUITE], sort_keys=True, default=str)
    checks = {
        "suite_sha256": base.sha256_text(suite_text),
        "task_file_sha256": base.sha256_file(HERE / "tasks_pilot1.py"),
        "base_harness_sha256": base.sha256_file(HERE / "harness.py"),
        "scorer_sha256": base.sha256_file(HERE / "rediscovery.py"),
    }
    for field, actual in checks.items():
        if manifest.get(field) != actual:
            problems.append(f"{field}: manifest {manifest.get(field)} != disk {actual}")
    native = base.native_provenance()
    if native["loaded_sha256"] != manifest["native"]["loaded_sha256"]:
        problems.append("native binary hash changed since freeze")
    if native["match"] is False:
        problems.append("loaded native module != local build")
    return problems


def main(argv: list[str]) -> int:
    from tasks_pilot1 import SUITE

    limit = int(argv[1]) if len(argv) > 1 else len(SUITE)
    tasks = SUITE[:limit]

    manifest = freeze_manifest(SUITE)
    problems = verify_frozen(manifest)
    if problems:
        print("ABORT: frozen treatment does not match disk:")
        for p in problems:
            print("  -", p)
        return 2

    phase = os.environ.get("A2_PHASE", "PRE_INTERRUPTION")
    print(json.dumps({k: manifest[k] for k in (
        "attempt_id", "suite_sha256", "attempt2_harness_sha256",
        "entroly_head", "codex_model", "usage_aggregation")}, indent=2))
    print(f"phase={phase}\n")

    completed = 0
    for index, task in enumerate(tasks):
        order = manifest["run_order"][task.task_id]
        print(f"== [{index + 1}/{len(tasks)}] {task.task_id} [{task.stratum}] "
              f"order={order}")
        for arm in order:
            skip, why = can_skip(task.task_id, arm, manifest,
                                 manifest["task_hashes"][task.task_id],
                                 manifest["arm_definition_hashes"][task.task_id][arm])
            if skip:
                print(f"   {arm:4s} SKIP ({why})")
                completed += 1
                continue
            existing = load_row(task.task_id, arm)
            invocation = int((existing or {}).get("invocation") or 0) + 1
            if existing is not None:
                # Preserve the prior invocation artifact rather than losing it.
                prior = ROWS_DIR / (
                    f"{task.task_id}.{arm}.invocation-{invocation - 1:03d}.json"
                )
                if not prior.exists():
                    prior.write_text(json.dumps(existing, indent=2, default=str),
                                     encoding="utf-8")
            try:
                row = invoke(task, arm, manifest, invocation, phase)
            except Exception as exc:  # noqa: BLE001
                row = {"attempt_id": manifest["attempt_id"],
                       "manifest_sha256": manifest["manifest_sha256"],
                       "task_id": task.task_id, "arm": arm,
                       "invocation": invocation, "status": INTERRUPTED,
                       "harness_error": f"{type(exc).__name__}: {exc}"[:400],
                       "verified_success": None,
                       "quota_boundary_phase": phase}
            write_row_atomic(row, allow_overwrite=True)

            if row["status"] == ENVIRONMENT_FAILURE and \
                    row.get("environment_class") in TERMINAL_PROVIDER:
                marker = {
                    "attempt_id": manifest["attempt_id"],
                    "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "reason": row["environment_class"],
                    "detail": row.get("environment_detail"),
                    "reached_task": task.task_id,
                    "reached_arm": arm,
                    "completed_rows": completed,
                    "phase": phase,
                }
                (OUT_DIR / "INTERRUPTED.json").write_text(
                    json.dumps(marker, indent=2), encoding="utf-8")
                print(f"   {arm:4s} TERMINAL {row['environment_class']}")
                print("\nCIRCUIT BREAKER: stopping. "
                      f"{completed} scientific rows preserved. "
                      "Resume under the same manifest when the provider recovers.")
                return 3

            print(f"   {arm:4s} {row['status']:20s} "
                  f"ver={row.get('verified_success')} "
                  f"redisc={row.get('avoidable_rediscovery_operations')} "
                  f"unc={row.get('uncached_input_tokens')} "
                  f"out={row.get('output_tokens')} "
                  f"cmds={row.get('command_count')} {row.get('wall_seconds')}s")
            if row["status"] == COMPLETE:
                completed += 1

    (OUT_DIR / "FINISHED.json").write_text(json.dumps({
        "attempt_id": manifest["attempt_id"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "complete_rows": completed,
        "expected_rows": len(tasks) * len(ARMS),
    }, indent=2), encoding="utf-8")
    print(f"\nAttempt 2 finished: {completed} complete rows of "
          f"{len(tasks) * len(ARMS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
