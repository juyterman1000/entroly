"""Does the real B3 consumer now carry the state the graph already recorded?

Gate for the paid pilot. The pre-fix run of `b3_handoff_content.py` established
that `resume()`, `handoff()`, `context_scope()`, `continuation_proof()`,
`summary()` and `unfinished()` contained neither the failure verdict nor the
recorded remaining work, while `snapshot()`/`export_state()` did -- capture was
present, projection was not.

Two things this script does that the earlier probe did not:

1. It drives the **actual production B3 consumer**, `work_graph_mcp.work_resume`,
   not a hand-composed union of read surfaces. Section 8 of the plan forbids
   assembling `snapshot + export_state + resume` for the benchmark, because that
   would hand the Entroly arm an advantage no real second agent gets.
2. It asserts the required post-fix result as a pass/fail gate, so the pilot
   cannot start on an arm that silently still drops state.

The required result for the B3 consumer:

    failed verdict       YES
    failure detail       YES if recorded
    outstanding work     YES
    changed paths        YES
    decisions            YES
    commitment           YES

Run: python research/experiments/b3_projection_validation.py
Writes research/ledger/b3_projection_validation.json
Exit 0 only if every requirement is met.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "ledger" / "b3_projection_validation.json"

# The recorded facts. Each is something a real agent produces as an artefact; the
# assertions below look for these exact strings in the consumer payload, so a
# "present" result cannot be satisfied by a paraphrase or a status code.
REMAINING = "wire the new config field through the Rust engine"
DECISION = "budget must stay per-request; a global default broke cache alignment"


def make_repo() -> pathlib.Path:
    """A real git worktree, because work_resume takes a passive observation."""
    path = pathlib.Path(tempfile.mkdtemp(prefix="entroly_b3_repo_"))
    run = lambda *a: subprocess.run(  # noqa: E731
        a, cwd=path, capture_output=True, text=True, check=True
    )
    run("git", "init", "-q", "-b", "feat/configurable-budget")
    run("git", "config", "user.email", "pilot@example.invalid")
    run("git", "config", "user.name", "pilot")
    (path / "config.py").write_text("TOKEN_BUDGET = 8192\n", encoding="utf-8")
    (path / "engine.py").write_text("def optimize(budget):\n    return budget\n",
                                    encoding="utf-8")
    run("git", "add", "-A")
    run("git", "commit", "-q", "-m", "baseline")
    # Uncommitted edits: the interrupted state.
    (path / "config.py").write_text(
        "TOKEN_BUDGET = 8192\nPER_REQUEST_BUDGET = None\n", encoding="utf-8"
    )
    return path


def observation(repo_id: str, head_sha: str) -> dict:
    return {
        "repo_id": repo_id,
        "observed_at_ms": 1_000,
        "repository_label": "continuity pilot",
        "agent_id": "agent:claude",
        "session_id": "session-a",
        "task_hint": {
            "task_id": "task-budget",
            "title": "make the per-request token budget configurable",
            "trust": "observed",
            "explicit_status": "in_progress",
            "remaining_work": [REMAINING],
            "source_kind": "user_statement",
            "source_ref": "user:task",
        },
        "branch": {
            "name": "feat/configurable-budget",
            "head_sha": head_sha,
            "default_branch": "main",
            "ahead_by": 1,
        },
        "changes": [
            {"path": "config.py", "kind": "modified",
             "staged": False, "conflicted": False},
        ],
        "decisions": [
            {"decision_id": "decision-1", "text": DECISION,
             "source_ref": "checkpoint:1", "source_kind": "checkpoint",
             "trust": "observed"},
        ],
        # The failing test. This is the fact the pre-fix projection lost.
        "verifications": [
            {
                "verification_id": "verification:budget",
                "name": "budget tests",
                "state": "failed",
                "evidence_kind": "test_result",
                "source_ref": "pytest tests/test_engine_budget.py",
                "digest": "deadbeef",
                "observed_at_ms": 1_100,
            },
        ],
    }


def main() -> int:
    os.environ["ENTROLY_NO_SELF_HEAL"] = "1"
    repo = make_repo()
    # work_resume refuses a project outside ENTROLY_SOURCE ("project must stay
    # inside ENTROLY_SOURCE"). That path-containment guard is the production
    # behaviour, so the probe roots itself at the scratch repo rather than
    # bypassing the check.
    os.environ["ENTROLY_SOURCE"] = str(repo)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    from entroly import work_graph_mcp
    from entroly.work_graph_store import WorkGraphStore, discover_repository_identity

    repo_id = discover_repository_identity(repo)["repo_id"]
    store = WorkGraphStore(repo_id)
    store.submit_repository_observation(
        observation(repo_id, head), repository_path=repo
    )

    # ── The actual production B3 consumer ───────────────────────────────
    produced = work_graph_mcp.work_resume(project=str(repo), to_agent="agent:codex")
    payload_text = json.dumps(produced, default=str)

    resume = produced.get("work_resume", produced)
    if isinstance(resume, dict) and "resume" in resume:
        resume = resume["resume"]

    def verdicts() -> list[str]:
        out = []
        for item in (resume or {}).get("verification", []) or []:
            if isinstance(item, dict):
                out.append(str(item.get("verdict")))
            else:
                out.append(f"<string: {item}>")
        return out

    requirements = {
        # Substring checks against the serialized payload, because what matters
        # is what a second agent can read -- not what a Python object holds.
        "has_failed_verdict": "failed" in payload_text,
        "has_failure_detail_if_recorded": (
            "budget tests" in payload_text or "pytest" in payload_text
        ),
        "has_outstanding_work": REMAINING in payload_text,
        "has_changed_paths": "config.py" in payload_text,
        "has_decisions": DECISION in payload_text,
        "has_commitment": bool((resume or {}).get("graph_commitment")),
    }
    # Guard against the fail-open shape the fix was for: a verdict list of bare
    # strings would satisfy "failed" in payload_text only by accident.
    requirements["verdict_is_structured"] = all(
        isinstance(item, dict)
        for item in (resume or {}).get("verification", []) or []
    ) and bool((resume or {}).get("verification"))

    report = {
        "baseline_commit": "5cb3fe16 + projection fix",
        "repo": str(repo),
        "consumer": "entroly.work_graph_mcp.work_resume (MCP tool work_resume)",
        "note": (
            "single production-reachable continuation path; no snapshot or "
            "export_state fields are composed in"
        ),
        "requirements": requirements,
        "all_requirements_met": all(requirements.values()),
        "projected_verdicts": verdicts(),
        "outstanding_work": (resume or {}).get("outstanding_work"),
        "continuation_proof_outstanding_refs": (
            (produced.get("work_resume") or produced)
            .get("continuation_proof", {})
            .get("outstanding_work_refs")
            if isinstance(produced, dict) else None
        ),
        "resume_keys": sorted((resume or {}).keys()),
        "full_payload": produced,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "full_payload"},
                     indent=2, default=str)[:3000])
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    if not report["all_requirements_met"]:
        failed = [k for k, v in requirements.items() if not v]
        print(f"\nGATE FAILED -- do not start the paid pilot. Missing: {failed}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
