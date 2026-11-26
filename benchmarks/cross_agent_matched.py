"""Matched supplied-context trials across agent CLIs, with independent grading.

Prepare is local-only; run explicitly invokes the configured agent accounts.
Use ``python -m benchmarks.cross_agent_matched --help``. Plans and results may
contain private source and must be written outside the repository. This is a
single-response development experiment, not a native autonomous-agent score
or an agent-to-agent handoff evaluation. No installs, downloads, or retries.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from benchmarks.agentic_arms import build_compressed, build_raw
from benchmarks.cross_agent_cli import AgentSpec, generate, preflight
from benchmarks.real_commit_pilot import (
    _apply_patch, _digest, _fragments, _git, _load_tasks, _oracle, _prompt,
    _task_worktree, summarize as summarize_agent,
)

ARMS = ("raw", "selected", "raw_repeat", "null")
ROOT = Path(__file__).resolve().parent.parent
_BOUNDED_FAILURES = {"generation_timeout": "timeout", "response_limit": "unusable_patch"}
_SCORED = {"passed", "test_failure", "timeout", "unusable_patch", "selection_error", *_BOUNDED_FAILURES}


def _source_identity() -> dict:
    paths = (
        "benchmarks/cross_agent_matched.py", "benchmarks/cross_agent_cli.py",
        "benchmarks/agent_process_job.py",
        "benchmarks/agentic_arms.py", "benchmarks/real_commit_pilot.py",
    )
    return {
        "git_sha": _git(ROOT, "rev-parse", "HEAD").strip(),
        "files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
    }


def _positive(value: float, name: str) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _private_output(path: Path, repo: Path) -> None:
    for root in (repo.resolve(), ROOT):
        if path.resolve().is_relative_to(root):
            raise ValueError("write private plans and results outside the repository")
    if path.exists():
        raise FileExistsError(path)


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def prepare(
    repo: Path, tasks_file: Path, task_ids: list[str], agents: list[AgentSpec], *,
    selection_budget: int, timeout_s: float, test_timeout_s: float,
    max_response_bytes: int, oracle_image: str, optimize_fn=None,
) -> dict:
    """Freeze each prompt once, shared by every agent. Never invokes a model."""
    for name, value in (("selection_budget", selection_budget), ("timeout_s", timeout_s),
                        ("test_timeout_s", test_timeout_s), ("max_response_bytes", max_response_bytes)):
        _positive(value, name)
    if type(selection_budget) is not int or type(max_response_bytes) is not int:
        raise ValueError("token budget and response byte limit must be integers")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", oracle_image):
        raise ValueError("oracle requires an immutable Docker image ID")
    if not task_ids or len(set(task_ids)) != len(task_ids):
        raise ValueError("tasks must be nonempty and unique")
    if len(agents) < 2 or len({a.agent_id for a in agents}) != len(agents):
        raise ValueError("declare at least two unique agent configurations")
    tasks_bytes = tasks_file.read_bytes()
    with tempfile.TemporaryDirectory(prefix="entroly-matched-tasks-") as tmp:
        snapshot = Path(tmp) / "tasks.jsonl"
        snapshot.write_bytes(tasks_bytes)
        tasks = _load_tasks(snapshot, task_ids)
    for task in tasks:
        for path in [*task["source_files"], *task["test_files"]]:
            if path.startswith("-") or "\x00" in path or "\n" in path or "\r" in path:
                raise ValueError("task paths must be concrete file names, never command options")
    if optimize_fn is None:
        from entroly import optimize
        optimize_fn = optimize
    from benchmarks.agentic_tasks_run import _engine_provenance

    plans = []
    for task in tasks:
        fragments = _fragments(repo, task)
        raw = build_raw(fragments)
        selected = build_compressed(
            fragments, query=_prompt("", task), budget=selection_budget, optimize_fn=optimize_fn,
        )
        contexts = {"raw": raw, "selected": selected, "raw_repeat": raw}
        prompts = {arm: _prompt(contexts[arm].text if arm in contexts else "", task) for arm in ARMS}
        plans.append({
            "task": task, "corpus_sha256": _digest([asdict(f) for f in fragments]),
            "prompts": prompts,
            "prompt_sha256": {arm: _digest(prompt) for arm, prompt in prompts.items()},
            "context": {arm: built.to_dict() for arm, built in contexts.items()},
        })
    declared = []
    for agent in agents:
        try:
            runtime = preflight(agent)
            error = None
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            runtime, error = None, type(exc).__name__
        declared.append({"spec": asdict(agent), "runtime": runtime, "preflight_error": error})
    plan = {
        "schema": "entroly.cross-agent-matched-plan.v1", "source": _source_identity(),
        "engine": _engine_provenance(), "tasks_file_sha256": hashlib.sha256(tasks_bytes).hexdigest(),
        "agents": declared, "tasks": plans, "arms": list(ARMS),
        "limits": {"selection_budget_estimated_tokens": selection_budget,
                   "agent_timeout_s": timeout_s, "test_timeout_s": test_timeout_s,
                   "max_accepted_response_bytes": max_response_bytes},
        "oracle_image": oracle_image,
        "limitations": [
            "Fixed source/test corpus from development commits, not a repository holdout.",
            "One tool-free answer per cell; no native tool loop or cross-agent handoff measured.",
            "Equal wall-time limits and post-generation answer-size acceptance thresholds; no token/cost cap.",
            "Provider decoding and tokenizers may differ; CLI capture is separately bounded at four MiB.",
            "Usage fields retain provider semantics; no pooled token savings or dollar claims.",
            "Requested model labels and CLI receipts do not authenticate provider identity.",
            "Checksums detect edits, not fabricated receipts; no automatic quality claim.",
        ],
    }
    plan["plan_sha256"] = _digest(plan)
    return plan


def summarize(plan: dict, rows: list[dict], pending: dict | None = None) -> dict:
    """Use the same frozen denominator for every declared agent, including missing ones."""
    task_ids = [item["task"]["sha"] for item in plan["tasks"]]
    agent_ids = [item["spec"]["agent_id"] for item in plan["agents"]]
    seen = set()
    for row in rows:
        key = (row["agent_id"], row["task_id"], row["arm"])
        if (key in seen or key[0] not in agent_ids or key[1] not in task_ids or key[2] not in ARMS):
            raise ValueError("duplicate or unplanned cross-agent result")
        seen.add(key)
        task = next(t for t in plan["tasks"] if t["task"]["sha"] == key[1])
        if row.get("prompt_sha256") != task["prompt_sha256"][key[2]]:
            raise ValueError("result prompt does not match its frozen cell")
    by_agent = {}
    for agent_id in agent_ids:
        selected_rows = [row for row in rows if row["agent_id"] == agent_id]
        result = summarize_agent([
            {**row, "outcome": _BOUNDED_FAILURES.get(row["outcome"], row["outcome"])}
            for row in selected_rows
        ], task_ids, ARMS)
        result["outcomes_by_arm"] = {
            arm: dict(Counter(r["outcome"] for r in selected_rows if r["arm"] == arm)) for arm in ARMS
        }
        result["planned_cells"] = len(task_ids) * len(ARMS)
        result["recorded_cells"] = len(selected_rows)
        result["not_recorded_cells"] = result["planned_cells"] - len(selected_rows)
        result["observed_passes_by_arm"] = {
            arm: sum(r["outcome"] == "passed" for r in selected_rows if r["arm"] == arm) for arm in ARMS
        }
        by_agent[agent_id] = result
    common = []
    indexed = {(r["agent_id"], r["task_id"], r["arm"]): r["outcome"] for r in rows}
    for task_id in task_ids:
        cells = {agent: [indexed.get((agent, task_id, arm)) for arm in ("raw", "selected")]
                 for agent in agent_ids}
        if all(outcome in _SCORED for pair in cells.values() for outcome in pair):
            common.append({"task_id": task_id, "per_agent": {
                agent: {"raw": pair[0], "selected": pair[1],
                        "pass_delta": int(pair[1] == "passed") - int(pair[0] == "passed")}
                for agent, pair in cells.items()
            }})
    return {
        "per_agent": by_agent, "planned_cells": len(task_ids) * len(agent_ids) * len(ARMS),
        "recorded_cells": len(rows), "pending_attempt": pending,
        "complete_tasks_across_all_agents": common,
        "bounded_failure_comparison_mapping": _BOUNDED_FAILURES,
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "quality_claim_allowed": False, "handoff_quality_measured": False,
    }


def _grade(repo: Path, task: dict, response: str, *, timeout: float, image: str) -> dict:
    with _task_worktree(repo, task) as root:
        try:
            patch_hash, patch = _apply_patch(root, response, task["source_files"])
        except (ValueError, RuntimeError) as exc:
            return {"outcome": "unusable_patch", "detail": type(exc).__name__}
        outcome, detail = _oracle(root, task, timeout, image)
        return {"outcome": outcome, "oracle_tail": detail, "patch_sha256": patch_hash, "patch": patch}


def _validate_oracle(repo: Path, task: dict, *, timeout: float, image: str) -> None:
    # Re-establish behavioral fail-to-pass under this exact image before calls.
    with _task_worktree(repo, task) as root:
        broken, _ = _oracle(root, task, timeout, image)
    with _task_worktree(repo, task) as root:
        _git(root, "checkout", task["sha"], "--", *task["source_files"])
        fixed, _ = _oracle(root, task, timeout, image)
    if (broken, fixed) != ("test_failure", "passed"):
        raise RuntimeError(f"oracle fail-to-pass validation failed: {broken}/{fixed}")


def run(plan: dict, repo: Path, output: Path, *, max_calls: int, allow_account_calls: bool) -> dict:
    """Execute a fresh bounded attempt. Existing outputs are never overwritten or replayed."""
    if not allow_account_calls:
        raise ValueError("run requires explicit --allow-account-calls")
    if type(max_calls) is not int or max_calls <= 0:
        raise ValueError("max_calls must be a positive integer")
    if plan.get("schema") != "entroly.cross-agent-matched-plan.v1":
        raise ValueError("unsupported matched plan")
    if plan.get("plan_sha256") != _digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("matched plan checksum changed")
    if plan["source"] != _source_identity():
        raise ValueError("benchmark source changed since preparation")
    if os.environ.get("ENTROLY_REAL_COMMIT_SANDBOX") != "1":
        raise ValueError("grading requires ENTROLY_REAL_COMMIT_SANDBOX=1; host execution forbidden")
    _private_output(output, repo)
    artifact = {"schema": "entroly.cross-agent-matched-result.v1", "plan_sha256": plan["plan_sha256"],
                "plan": plan, "rows": [], "pending": None, "model_calls": 0, "max_calls": max_calls}
    artifact["summary"] = summarize(plan, [])
    artifact["artifact_sha256"] = _digest(artifact)
    _write_new(output, artifact)

    def checkpoint() -> None:
        artifact["summary"] = summarize(plan, artifact["rows"], artifact["pending"])
        artifact["artifact_sha256"] = _digest({k: v for k, v in artifact.items() if k != "artifact_sha256"})
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         suffix=".tmp", delete=False) as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)

    try:
        image = subprocess.run(["docker", "image", "inspect", plan["oracle_image"], "--format", "{{.Id}}"],
                               capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        if image != plan["oracle_image"]:
            raise RuntimeError("oracle image does not match the pinned image")
        for item in plan["tasks"]:
            task = item["task"]
            if _digest([asdict(f) for f in _fragments(repo, task)]) != item["corpus_sha256"]:
                raise ValueError("task corpus differs from frozen source")
            _validate_oracle(repo, task, timeout=plan["limits"]["test_timeout_s"], image=image)
        available = {}
        for declared in plan["agents"]:
            spec = AgentSpec(**declared["spec"])
            try:
                available[spec.agent_id] = declared["runtime"] is not None and preflight(spec) == declared["runtime"]
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
                available[spec.agent_id] = False
        for index, item in enumerate(plan["tasks"]):
            task = item["task"]
            # Pair first; rotate treatment and agent order by frozen task index.
            arms = ("raw", "selected", "raw_repeat", "null") if index % 2 == 0 else (
                "selected", "raw", "null", "raw_repeat")
            agents = plan["agents"][index % len(plan["agents"]):] + plan["agents"][:index % len(plan["agents"])]
            for pair in (arms[:2], arms[2:]):
                for declared in agents:
                    spec = AgentSpec(**declared["spec"])
                    for arm in pair:
                        row = {"agent_id": spec.agent_id, "task_id": task["sha"], "arm": arm,
                               "prompt_sha256": item["prompt_sha256"][arm]}
                        if not available[spec.agent_id]:
                            artifact["rows"].append({**row, "outcome": "agent_unavailable"})
                            checkpoint()
                            continue
                        if artifact["model_calls"] >= max_calls:
                            artifact["stop_reason"] = "call_budget"
                            checkpoint()
                            return artifact
                        artifact["pending"] = row
                        artifact["model_calls"] += 1
                        checkpoint()  # Persist possible consumption BEFORE invoking the account.
                        try:
                            generation = generate(spec, item["prompts"][arm], timeout_s=plan["limits"]["agent_timeout_s"])
                            row["generation"] = generation
                            row["response_sha256"] = hashlib.sha256(generation["text"].encode("utf-8")).hexdigest()
                            if generation["runtime"] != declared["runtime"]:
                                row["outcome"] = "runtime_changed"
                            elif len(generation["text"].encode("utf-8")) > plan["limits"]["max_accepted_response_bytes"]:
                                row["outcome"] = "response_limit"
                            else:
                                row.update(_grade(repo, task, generation["text"],
                                                  timeout=plan["limits"]["test_timeout_s"], image=image))
                        except TimeoutError:
                            row["outcome"] = "generation_timeout"
                        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                            row.update(outcome="execution_error", detail=type(exc).__name__)
                        artifact["rows"].append(row)
                        artifact["pending"] = None
                        checkpoint()
        artifact["stop_reason"] = "complete"
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        artifact.update(stop_reason="preflight_or_execution_error", error=type(exc).__name__)
    checkpoint()
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare", help="freeze prompts locally; never calls a model")
    freeze.add_argument("--tasks", type=Path, required=True)
    freeze.add_argument("--task-sha", action="append", required=True)
    freeze.add_argument("--agents", type=Path, required=True, help="JSON array of AgentSpec objects")
    freeze.add_argument("--selection-budget", type=int, default=1400)
    freeze.add_argument("--agent-timeout", type=float, default=120)
    freeze.add_argument("--test-timeout", type=float, default=40)
    freeze.add_argument("--max-accepted-response-bytes", type=int, default=64000,
                        help="post-generation acceptance threshold, not an inference token/cost cap")
    freeze.add_argument("--oracle-image", required=True, help="installed immutable sha256:... image ID")
    execute = sub.add_parser("run", help="invoke agent accounts; requires explicit opt-in")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--max-calls", type=int, required=True)
    execute.add_argument("--allow-account-calls", action="store_true")
    for cmd in (freeze, execute):
        cmd.add_argument("--repo", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        _private_output(args.output, args.repo)
        plan = prepare(args.repo, args.tasks, args.task_sha,
                       [AgentSpec(**a) for a in json.loads(args.agents.read_text(encoding="utf-8"))],
                       selection_budget=args.selection_budget, timeout_s=args.agent_timeout,
                       test_timeout_s=args.test_timeout, max_response_bytes=args.max_accepted_response_bytes,
                       oracle_image=args.oracle_image)
        _write_new(args.output, plan)
        print(json.dumps({"plan_sha256": plan["plan_sha256"], "planned_cells": len(plan["tasks"]) * len(plan["agents"]) * len(ARMS)}))
        return 0
    result = run(json.loads(args.plan.read_text(encoding="utf-8")), args.repo, args.output,
                 max_calls=args.max_calls, allow_account_calls=args.allow_account_calls)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if result["stop_reason"] in {"complete", "call_budget"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
