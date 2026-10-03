"""Pilot-1 runner: ContinuityStress-24, arms B0 / B1S / B3.

A separate file from `harness.py` on purpose. Pilot-0's manifest records that
file's SHA-256, and editing it in place would leave a recorded hash that no
longer matches anything on disk. Shared machinery is imported rather than copied.

What differs from Pilot-0:

  * suite is ContinuityStress-24 (`tasks_pilot1`), not the 6-task validation set
  * the summary arm is named **B1S**, not B1. It is a deterministic rendering of
    RecordedState, so B1S and B3 draw on the SAME captured facts: this isolates
    the representation question. It is not an agent-written handoff, and the
    capture question (B1N) cannot be tested here -- `claude -p` exists and is
    wired for it, but returns "OAuth session expired and could not be refreshed"
    and there is no ANTHROPIC_API_KEY.
  * every event stream is copied into the ledger, so scoring does not depend on
    temp directories surviving
  * avoidable rediscovery and first-progress are scored per row at collection
    time, from the ordered timeline, using the frozen definitions in
    `rediscovery.py`
  * handoff payload size is recorded per arm, because B1S is ~700 characters and
    B3 ~4,400 for identical facts, so a B3 win could be a volume effect

Fairness properties inherited unchanged: byte-identical checkpoint per arm with
an asserted worktree digest, verifier run by the harness and never by an agent,
counterbalanced arm order, and B3 consuming only `work_resume`.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import harness as base  # noqa: E402  shared checkpoint/payload/parse machinery
import rediscovery as rd  # noqa: E402

OUT_DIR = ROOT / "research" / "ledger" / "pilot1"
TRACE_DIR = OUT_DIR / "traces"
ARMS = ("B0", "B1S", "B3")
SUITE_TYPE = "ContinuityStress-24"


def run_arm(task, arm: str, root: pathlib.Path) -> dict:
    arm_root = root / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    repo = base.build_checkpoint(task, arm_root)
    head = base.git(repo, "rev-parse", "HEAD")
    digest = base.worktree_digest(repo)

    parts = [
        "You are continuing work another agent started and could not finish.",
        "",
        f"## Task\n{task.statement}",
    ]
    extra: dict = {}
    handoff_text = ""
    if arm == "B1S":
        handoff_text = base.b1_handoff(task)
        parts += ["", handoff_text]
    elif arm == "B3":
        started = time.perf_counter()
        handoff_text, payload = base.b3_payload(task, repo)
        extra["b3_local_latency_s"] = round(time.perf_counter() - started, 3)
        extra["b3_resume"] = payload.get("resume")
        parts += ["", handoff_text]
    parts += ["", "Finish the task. Run the test suite to confirm before you stop."]
    prompt = "\n".join(parts)

    env = dict(os.environ)
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "safe.directory"
    env["GIT_CONFIG_VALUE_0"] = "*"

    started = time.time()
    proc = subprocess.run(
        ["codex", "exec", "--json", "-C", str(repo), "-s", "workspace-write",
         "-m", base.CODEX_MODEL, "--skip-git-repo-check",
         "-c", "shell_environment_policy.inherit=all",
         "-c", 'sandbox_permissions=["disk-full-read-access"]',
         prompt],
        capture_output=True, text=True, timeout=base.TIMEOUT_S,
        encoding="utf-8", errors="replace", env=env,
    )
    elapsed = time.time() - started

    verified = subprocess.run(task.verifier, cwd=repo, capture_output=True, text=True)
    metrics = base.parse_events(proc.stdout)

    recorded = {
        "rejected": list(task.recorded.rejected),
        "remaining_work": list(task.recorded.remaining_work),
    }
    scored = rd.score(proc.stdout, recorded)
    progress = rd.first_progress_action(proc.stdout, recorded)

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    (TRACE_DIR / f"{task.task_id}.{arm}.jsonl").write_text(
        proc.stdout, encoding="utf-8"
    )
    (TRACE_DIR / f"{task.task_id}.{arm}.prompt.txt").write_text(
        prompt, encoding="utf-8"
    )

    return {
        "task_id": task.task_id,
        "stratum": task.stratum,
        "arm": arm,
        "checkpoint_head": head,
        "worktree_digest": digest,
        "prompt_sha256": base.sha256_text(prompt),
        "prompt_chars": len(prompt),
        "handoff_chars": len(handoff_text),
        # ~4 chars/token; labelled an estimate because no provider tokenizer is
        # available locally.
        "handoff_tokens_estimate": round(len(handoff_text) / 4) if handoff_text else 0,
        "codex_exit": proc.returncode,
        "verified_success": verified.returncode == 0,
        "verifier_tail": verified.stdout[-400:],
        "wall_seconds": round(elapsed, 2),
        "first_progress_action": progress,
        **metrics,
        **{k: v for k, v in scored.items()
           if k not in ("duplicate_read_detail", "duplicate_diagnostic_detail")},
        "rediscovery_detail": {
            "duplicate_reads": scored["duplicate_read_detail"],
            "duplicate_diagnostics": scored["duplicate_diagnostic_detail"],
        },
        **extra,
    }


def main(argv: list[str]) -> int:
    from tasks_pilot1 import SUITE

    limit = int(argv[1]) if len(argv) > 1 else len(SUITE)
    tasks = SUITE[:limit]

    suite_text = json.dumps([asdict(t) for t in SUITE], sort_keys=True, default=str)
    manifest = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "suite_type": SUITE_TYPE,
        "suite_note": (
            "Every task carries a planted trap. This suite estimates the BENEFIT "
            "when continuity matters; it says nothing about how often that is. "
            "Do not describe it as representative coding work."
        ),
        "suite_sha256": base.sha256_text(suite_text),
        "suite_task_count": len(SUITE),
        "tasks_file_sha256": base.sha256_file(HERE / "tasks_pilot1.py"),
        "harness_sha256": base.sha256_file(pathlib.Path(__file__)),
        "base_harness_sha256": base.sha256_file(HERE / "harness.py"),
        "rediscovery_sha256": base.sha256_file(HERE / "rediscovery.py"),
        "entroly_source_commit": base.git(ROOT, "rev-parse", "HEAD"),
        "entroly_local_commits": base.git(ROOT, "log", "--format=%h", "-10").splitlines(),
        "native": base.native_provenance(),
        "python": sys.version.split()[0],
        "agent_b": subprocess.run(["codex", "--version"], capture_output=True,
                                  text=True).stdout.strip(),
        "codex_model": base.CODEX_MODEL,
        "codex_sandbox": "workspace-write + disk-full-read-access",
        "codex_auth": "ChatGPT account (no token invoice; usage proxies only)",
        "arms": {
            "B0": "repository checkpoint + task",
            "B1S": "… + deterministic text rendering of RecordedState "
                   "(0 provider tokens to generate, by construction)",
            "B3": "… + production work_resume payload only",
            "B1N": "NOT FEASIBLE: claude -p returns 'OAuth session expired and "
                   "could not be refreshed'; no ANTHROPIC_API_KEY",
        },
        "hypotheses": {
            "H1_representation": "B3 vs B1S — same captured facts, different form",
            "H2_capture": "B3 vs B1N — UNTESTABLE in this environment",
        },
        "primary_representation_comparison": "B3 vs B1S",
        "primary_commercial_comparison": "B3 vs B1N — not available",
        "metrics": {
            "primary_success": "verified_success from the task verifier only",
            "primary_reconstruction": (
                "avoidable_rediscovery_operations = duplicate reads with no "
                "intervening edit + duplicate diagnostics with no intervening "
                "file change. Failures excluded."
            ),
            "secondary": [
                "first_progress_action index", "failed_commands_total",
                "cached/uncached/output tokens", "handoff_tokens_estimate",
                "wall_seconds", "command_count", "b3_local_latency_s",
            ],
        },
        "thresholds": {
            "success": ">=10pp vs B1S; discrete at n=24 (1 task = 4.17pp, "
                       "3 tasks = 12.5pp)",
            "reconstruction": ">=20% median reduction in "
                              "avoidable_rediscovery_operations",
            "usage_guard": "noncached_usage_proxy (successor uncached input + "
                           "output) must not rise >10%",
            "caveat": "avoidable_rediscovery_operations was 0 on every Pilot-0 "
                      "row. If it is zero-variance here too, that half of the "
                      "criterion is undecidable and only success can decide.",
        },
        "contamination_rule": (
            "All agent file access in this CLI flows through shell commands or "
            "file_change events, so the command stream IS the strongest "
            "available trace source. A row reading the Entroly source tree, "
            "prior benchmark artifacts, another arm's directory, user documents, "
            "home config or agent session history is INVALID."
        ),
        "tasks_run": [t.task_id for t in tasks],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("suite_type", "suite_sha256", "harness_sha256",
                       "entroly_source_commit", "codex_model")}, indent=2))
    if manifest["native"]["match"] is False:
        print("ABORT: loaded native module != local build")
        return 1

    results: list[dict] = []
    for index, task in enumerate(tasks):
        root = pathlib.Path(os.environ.get("TEMP", "/tmp")) / (
            f"p1_{task.task_id}_{int(time.time())}"
        )
        if root.exists():
            shutil.rmtree(root)
        order = list(ARMS)
        random.Random(1000 + index).shuffle(order)
        print(f"\n== [{index + 1}/{len(tasks)}] {task.task_id} "
              f"[{task.stratum}] order={order}")
        digests = set()
        for arm in order:
            try:
                row = run_arm(task, arm, root)
            except Exception as exc:  # noqa: BLE001
                row = {"task_id": task.task_id, "arm": arm,
                       "harness_error": f"{type(exc).__name__}: {exc}"[:400],
                       "verified_success": False}
            row["arm_order"] = order
            digests.add(row.get("worktree_digest"))
            results.append(row)
            print(f"   {arm:4s} ver={row.get('verified_success')} "
                  f"ran={row.get('agent_ran_tests_successfully')} "
                  f"redisc={row.get('avoidable_rediscovery_operations')} "
                  f"prog={(row.get('first_progress_action') or {}).get('index')} "
                  f"unc={row.get('uncached_input_tokens')} "
                  f"out={row.get('output_tokens')} "
                  f"cmds={row.get('command_count')} {row.get('wall_seconds')}s"
                  + (f" ERR {row['harness_error']}" if "harness_error" in row else ""))
        assert len([d for d in digests if d]) <= 1, (
            f"arms did not start from identical worktrees: {digests}"
        )
        (OUT_DIR / "results.json").write_text(
            json.dumps({"manifest": manifest, "results": results},
                       indent=2, default=str), encoding="utf-8")
        # Abort early on an invalid environment rather than spending the suite.
        if index == 0 and not any(
            r.get("agent_ran_tests_successfully") for r in results
        ):
            print("\nABORT: no arm executed the test suite on task 1.")
            return 1
        try:
            shutil.rmtree(root)
        except OSError:
            pass

    print(f"\nwrote {(OUT_DIR / 'results.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
