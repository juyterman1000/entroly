"""Local-model development pilot on mined, validated repository commits.

This is not the preregistered n>=400 cross-repository experiment. The fixed
protocol names every candidate, including candidates excluded for local-model
context size. Results are written outside the repository by default.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from benchmarks.agentic_arms import Fragment, build_compressed, build_raw
from benchmarks.agentic_tasks_run import call_model
from benchmarks.engine_isolation import assert_engine_isolated, isolated_engine_dir
from entroly.native_status import native_status


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr[-500:]}")
    return result.stdout


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _read_blob(repo: Path, revision: str, path: str) -> str:
    return _git(repo, "show", f"{revision}:{path}")


def _load_tasks(path: Path, expected_shas: list[str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_sha = {row["sha"]: row for row in rows}
    if len(by_sha) != len(rows) or set(by_sha) != set(expected_shas):
        raise ValueError("validated task rows must exactly match the frozen candidate list")
    ordered = [by_sha[sha] for sha in expected_shas]
    for task in ordered:
        if not re.fullmatch(r"[0-9a-f]{40}", task["sha"]):
            raise ValueError("candidate SHA must be a full Git object ID")
        if task.get("status") != "validated" or task.get("fail_outcome") != "test_failure":
            raise ValueError("candidate lacks a fail-to-pass oracle")
        if not task.get("source_files") or not task.get("test_files"):
            raise ValueError("candidate lacks source or test paths")
        for source in [*task["source_files"], *task["test_files"]]:
            if not isinstance(source, str) or not source:
                raise ValueError("candidate contains an unsafe repository path")
            parsed = PurePosixPath(source)
            if (
                parsed.is_absolute() or ".." in parsed.parts
                or ".git" in parsed.parts or "\\" in source or ":" in source
            ):
                raise ValueError("candidate contains an unsafe repository path")
    return ordered


def _fragments(repo: Path, task: dict[str, Any]) -> list[Fragment]:
    sha = task["sha"]
    paths = [*task["source_files"], *task["test_files"]]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate context paths")
    return [
        Fragment(path, _read_blob(repo, f"{sha}^" if path in task["source_files"] else sha, path))
        for path in paths
    ]


@contextmanager
def _task_worktree(repo: Path, task: dict[str, Any]) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="entroly-real-task-") as temp:
        root = Path(temp)
        _git(repo, "worktree", "add", "--detach", str(root), task["sha"])
        try:
            for source in [*task["source_files"], *task["test_files"]]:
                if (root / source).is_symlink():
                    raise ValueError("task symlink is unsupported")
            _git(root, "checkout", f"{task['sha']}^", "--", *task["source_files"])
            for source in [*task["source_files"], *task["test_files"]]:
                if (root / source).is_symlink():
                    raise ValueError("task symlink is unsupported")
            yield root
        finally:
            _git(repo, "worktree", "remove", "--force", str(root))


def _extract_patch(response: str) -> str:
    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)\n```", response, re.S | re.I)
    if fenced and (response[:fenced.start()].strip() or response[fenced.end():].strip()):
        raise ValueError("model output has content outside its diff fence")
    patch = fenced.group(1) if fenced else response
    start = patch.find("diff --git ")
    if start < 0:
        start = patch.find("--- a/")
    if start < 0:
        start = patch.find("--- ")
    if start < 0:
        raise ValueError("model output has no unified diff")
    return patch[start:].strip() + "\n"


def _check_patch_paths(patch: str, allowed: list[str]) -> None:
    allowed_set = set(allowed)
    old_paths = re.findall(r"^--- (.+)$", patch, re.M)
    new_paths = re.findall(r"^\+\+\+ (.+)$", patch, re.M)
    if not old_paths or len(old_paths) != len(new_paths):
        raise ValueError("patch lacks paired old/new file headers")
    for old, new in zip(old_paths, new_paths):
        old = old.split("\t", 1)[0].rstrip()
        new = new.split("\t", 1)[0].rstrip()
        if not old.startswith("a/") or not new.startswith("b/"):
            raise ValueError("patch may only modify existing source files")
        if old[2:] != new[2:] or old[2:] not in allowed_set:
            raise ValueError("patch changes a path outside the task source files")
    headers = re.findall(r"^diff --git a/(\S+) b/(\S+)$", patch, re.M)
    if any(a != b or a not in allowed_set for a, b in headers):
        raise ValueError("patch diff header changes an unapproved path")
    if any(marker in patch for marker in (
        "GIT binary patch", "rename from", "rename to", "new file mode",
        "deleted file mode", "old mode", "new mode", "copy from", "copy to",
        "Subproject commit",
    )):
        raise ValueError("binary, mode, symlink, copy and rename patches are unsupported")


def _apply_patch(root: Path, response: str, allowed: list[str]) -> tuple[str, str]:
    patch = _extract_patch(response)
    _check_patch_paths(patch, allowed)
    patch_hash = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    _git(root, "apply", "--check", "-", input_text=patch)
    _git(root, "apply", "-", input_text=patch)
    changed = set(_git(root, "diff", "--name-only").splitlines())
    if not changed <= set(allowed):
        raise ValueError("patch modified an unapproved file")
    return patch_hash, patch


def _oracle(
    root: Path, task: dict[str, Any], timeout: int, image_id: str | None = None,
) -> tuple[str, str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if image_id is not None and (
        env.get("ENTROLY_REAL_COMMIT_SANDBOX") != "1"
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id)
    ):
        raise ValueError("a pinned oracle requires a sandboxed immutable image ID")
    container = f"entroly-pilot-{uuid.uuid4().hex}" if image_id is not None else None
    if env.get("ENTROLY_REAL_COMMIT_SANDBOX") == "1":
        # Historical and model-generated Python execute ONLY in an isolated
        # container. The checkout is mounted read-only, no host credentials,
        # network, writeable bind volumes or extra Linux capabilities.
        root.chmod(0o755)
        for test_file in task["test_files"]:
            if Path(test_file).anchor or ".." in Path(test_file).parts:
                raise ValueError("invalid test path for sandbox")
        command = [
            "docker", "run", "--rm", "--pull=never", "--network=none",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--read-only", "--pids-limit=64", "--cpus=1", "--memory=768m",
            "--user=65534:65534", "--tmpfs=/tmp:rw,nosuid,nodev,size=64m",
            "--workdir=/task",
            f"--mount=type=bind,src={root},dst=/task,readonly",
            "--env=PYTHONPATH=/task",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            "--env=PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
            image_id or "entroly-dogfood-real-oracle:py312",
            "python", "-m", "pytest", *task["test_files"],
            "-x", "-q", "--tb=short", "-p", "no:cacheprovider",
        ]
        if container is not None:
            command[3:3] = ["--name", container]
    else:
        # Legacy trusted unit-test behavior only. Production model trials set
        # ENTROLY_REAL_COMMIT_SANDBOX=1 and fail if Docker is unavailable.
        command = [sys.executable, "-m", "pytest", *task["test_files"],
                   "-x", "-q", "--tb=short", "-p", "no:cacheprovider"]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "timeout", ""
    finally:
        if container is not None:
            # Killing a Docker client does not stop its container. Only remove
            # the unique container owned by this attempt, including on timeout.
            subprocess.run(
                ["docker", "rm", "--force", container], capture_output=True,
                timeout=10, check=False,
            )
    output = result.stdout + "\n" + result.stderr
    if env.get("ENTROLY_REAL_COMMIT_SANDBOX") == "1" and result.returncode in (125, 126, 127):
        return "infrastructure_error", output[-500:]
    if result.returncode == 0:
        return "passed", output[-500:]
    lowered = output.lower()
    if "error during collection" in lowered or "importerror" in lowered or "modulenotfounderror" in lowered:
        return "infrastructure_error", output[-500:]
    if "failed" in lowered and ("assert" in lowered or "short test summary info" in lowered):
        return "test_failure", output[-500:]
    return "unknown_failure", output[-500:]


def _prompt(context: str, task: dict[str, Any]) -> str:
    sources = ", ".join(task["source_files"])
    tests = ", ".join(task["test_files"])
    return (
        "Make the listed tests pass. Edit only the listed source files. "
        "Reply with one unified diff using a/ and b/ paths, without prose.\n"
        f"Source files: {sources}\nTests: {tests}\n\nContext:\n{context}\n"
    )


_ARMS_V1 = ("raw", "selected", "raw_repeat")
_ARMS_V2 = ("raw", "selected", "raw_repeat", "null")
# Exhausting the oracle's test budget is a failed attempt, not a transport error.
_SCORED_OUTCOMES = {"passed", "test_failure", "timeout", "unusable_patch", "selection_error"}
_PAIR_LABELS = {
    (True, True): "both_passed",
    (False, True): "selected_only_passed",
    (True, False): "baseline_only_passed",
    (False, False): "both_failed",
}


def _arm_order(version: int, index: int, seed: int) -> tuple[str, ...]:
    if version == 1:
        return _ARMS_V1
    offset = (index + seed) % len(_ARMS_V2)
    return _ARMS_V2[offset:] + _ARMS_V2[:offset]


def _artifact_digest(artifact: dict) -> str:
    # An integrity checksum, not authentication of a model answer or oracle.
    return _digest({key: value for key, value in artifact.items() if key != "artifact_sha256"})


def _next_arm(rows: list[dict], task_ids: list[str], seed: int) -> tuple[int, int]:
    task_index = arm_index = 0
    outcomes = _SCORED_OUTCOMES | {
        "execution_error", "infrastructure_error", "unknown_failure", "interrupted_unknown",
    }
    for row in rows:
        if task_index >= len(task_ids) or row.get("task_id") != task_ids[task_index]:
            raise ValueError("continuation rows are not a prefix of the frozen task order")
        arm = row.get("arm")
        if arm is None:
            if arm_index or row.get("outcome") not in {
                "ineligible_context_size", "ineligible_oracle_drift",
            }:
                raise ValueError("invalid continuation exclusion")
            task_index += 1
            continue
        if arm != _arm_order(2, task_index, seed)[arm_index] or row.get("outcome") not in outcomes:
            raise ValueError("invalid continuation arm order or outcome")
        for field in ("input_tokens", "output_tokens"):
            value = row.get(field)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("invalid continuation token usage")
        arm_index += 1
        if arm_index == len(_ARMS_V2):
            task_index += 1
            arm_index = 0
    return task_index, arm_index


def _read_continuation(path: Path, expected: dict, protocol: dict, runtime: dict) -> dict:
    data = path.read_bytes()
    parent = json.loads(data)
    if not isinstance(parent, dict) or parent.get("artifact_sha256") != _artifact_digest(parent):
        raise ValueError("continuation artifact integrity checksum differs")
    if parent.get("status") not in {"in_progress", "call_limit_reached"}:
        raise ValueError("continuation requires an incomplete artifact")
    if any(parent.get(key) != value for key, value in expected.items()):
        raise ValueError("continuation differs from the frozen protocol, source, tasks or model")
    contract = parent.get("continuation", {})
    if contract.get("schema") != "entroly.pilot-continuation.v1" or contract.get("runtime") != runtime:
        raise ValueError("continuation runtime identity differs or is unavailable")
    rows = parent.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("invalid continuation rows")
    task_index, arm_index = _next_arm(rows, protocol["candidate_shas"], protocol["seed"])
    if parent.get("summary") != summarize(rows, protocol["candidate_shas"], _ARMS_V2):
        raise ValueError("continuation summary differs from its observations")
    total = contract.get("total_arm_attempts")
    started = contract.get("new_arm_attempts")
    limit = contract.get("max_model_calls")
    if (type(total) is not int or type(started) is not int or not 0 <= started <= total
            or (limit is not None and (type(limit) is not int or not 0 < limit or started > limit))):
        raise ValueError("invalid continuation attempt accounting")
    plans = contract.get("task_plans")
    if not isinstance(plans, dict) or set(plans) - set(protocol["candidate_shas"]):
        raise ValueError("invalid continuation context plans")
    allowed = set(protocol["candidate_shas"][:task_index + 1])
    if set(plans) - allowed:
        raise ValueError("continuation contains context plans for future tasks")
    for plan in plans.values():
        if (not isinstance(plan, dict) or set(plan.get("arms", {})) != set(_ARMS_V2)
                or not isinstance(plan.get("corpus_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", plan["corpus_sha256"])):
            raise ValueError("invalid continuation context plan")
        for arm, arm_plan in plan["arms"].items():
            if (not isinstance(arm_plan, dict)
                    or (arm_plan == {"selection_error": True} and arm != "selected")
                    or (arm_plan != {"selection_error": True} and (
                        not isinstance(arm_plan.get("prompt_sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", arm_plan["prompt_sha256"])
                    ))):
                raise ValueError("invalid continuation arm plan")
    for row in rows:
        if row.get("arm") is None:
            continue
        plan = plans.get(row["task_id"], {}).get("arms", {}).get(row["arm"])
        if not isinstance(plan, dict) or row.get("plan_sha256") != _digest(plan):
            raise ValueError("continuation row differs from its context plan")
        if (row["outcome"] == "selection_error") != (plan == {"selection_error": True}):
            raise ValueError("continuation selection outcome differs from its context plan")
        if row.get("prompt_sha256") is not None and row["prompt_sha256"] != plan.get("prompt_sha256"):
            raise ValueError("continuation observed prompt differs from its plan")
        if row["outcome"] in {"passed", "test_failure", "timeout", "infrastructure_error", "unknown_failure"}:
            for field in ("prompt_sha256", "response_sha256", "patch_sha256"):
                if not isinstance(row.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                    raise ValueError("continuation oracle outcome lacks its generation and patch receipt")
    pending = contract.get("pending_arm")
    if pending is not None:
        if task_index >= len(protocol["candidate_shas"]) or pending != {
            "task_id": protocol["candidate_shas"][task_index],
            "arm": _arm_order(2, task_index, protocol["seed"])[arm_index],
        } or not started:
            raise ValueError("invalid continuation pending attempt")
        plan = plans.get(pending["task_id"], {}).get("arms", {}).get(pending["arm"])
        if not isinstance(plan, dict):
            raise ValueError("continuation pending attempt lacks its context plan")
        if plan == {"selection_error": True}:
            raise ValueError("continuation pending attempt has no model prompt")
    attempts = sum(row.get("arm") is not None and row["outcome"] != "selection_error" for row in rows)
    if total != attempts + int(pending is not None):
        raise ValueError("continuation attempt accounting differs from its observations")
    return {"artifact": parent, "sha256": hashlib.sha256(data).hexdigest()}


def summarize(rows: list[dict[str, Any]], task_ids: list[str], arms: tuple[str, ...]) -> dict:
    """Keep the frozen denominator and report incomplete comparisons explicitly."""
    grouped: dict[str, dict[str, dict]] = {task_id: {} for task_id in task_ids}
    excluded: dict[str, str] = {}
    for row in rows:
        task_id = row["task_id"]
        if task_id not in grouped:
            raise ValueError("result contains a task outside the frozen candidate list")
        arm = row.get("arm")
        if arm is None:
            if task_id in excluded or grouped[task_id]:
                raise ValueError("duplicate or mixed task exclusion")
            excluded[task_id] = row["outcome"]
        else:
            if arm not in arms or arm in grouped[task_id] or task_id in excluded:
                raise ValueError("duplicate or unexpected task arm")
            grouped[task_id][arm] = row
    comparisons = {}
    for baseline in ("raw", "raw_repeat", "null"):
        if baseline not in arms:
            continue
        counts = Counter()
        for task_id, values in grouped.items():
            if task_id in excluded:
                counts["excluded_tasks"] += 1
                continue
            pair = [values.get(name, {}).get("outcome") for name in (baseline, "selected")]
            if any(outcome not in _SCORED_OUTCOMES for outcome in pair):
                counts["incomplete_or_infrastructure_error"] += 1
                continue
            base_ok, selected_ok = [outcome == "passed" for outcome in pair]
            counts[_PAIR_LABELS[base_ok, selected_ok]] += 1
        comparisons[baseline] = {
            name: counts[name] for name in (
                "both_passed", "selected_only_passed", "baseline_only_passed",
                "both_failed", "excluded_tasks", "incomplete_or_infrastructure_error",
            )
        }
    return {
        "frozen_tasks": len(task_ids),
        "excluded_tasks": dict(sorted(excluded.items())),
        "outcomes_by_arm": {
            arm: dict(sorted(Counter(row["outcome"] for row in rows if row.get("arm") == arm).items()))
            for arm in arms
        },
        "comparisons_to_selected": comparisons,
        "quality_claim_allowed": False,
        "scope": "observed outcomes on frozen development tasks",
    }


def _local_model_digest(base_url: str, model: str) -> str:
    from benchmarks.pilot_runtime_identity import local_ollama_opener

    parsed = urllib.parse.urlparse(base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.query or parsed.fragment
            or parsed.port is None):
        raise ValueError("pilot requires a local Ollama HTTP endpoint")
    with local_ollama_opener(base_url).open(f"{base_url.rstrip('/')}/api/tags", timeout=10) as response:
        tags = json.loads(response.read().decode("utf-8"))
    matches = [item for item in tags.get("models", []) if item.get("name") == model]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise ValueError("requested local model and digest were not found")
    return matches[0]["digest"]


def _run_arm(
    repo: Path, task: dict[str, Any], *, arm: str, context: str,
    context_details: dict[str, Any], model: str, base_url: str, seed: int,
    model_timeout: float, test_timeout: int, output_tokens: int, context_window: int,
    preserve_execution_receipt: bool = False,
    oracle_image: str | None = None,
) -> dict[str, Any]:
    row = None
    try:
        with _task_worktree(repo, task) as root:
            prompt = _prompt(context, task)
            generation = call_model(
                base_url=base_url, model=model, prompt=prompt,
                seed=seed, timeout=model_timeout, max_output_tokens=output_tokens,
                context_window=context_window,
                local_only=True,
            )
            row = {
                "task_id": task["sha"], "arm": arm, "context": context_details,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "input_tokens": generation["input_tokens"],
                "output_tokens": generation["output_tokens"],
                "latency_s": generation["latency_s"],
                "response_sha256": hashlib.sha256(generation["text"].encode()).hexdigest(),
            }
            try:
                patch_hash, patch = _apply_patch(root, generation["text"], task["source_files"])
            except (ValueError, RuntimeError) as exc:
                row.update(
                    outcome="unusable_patch", patch_error=str(exc),
                    model_output=generation["text"],
                )
                return row
            row.update(patch_sha256=patch_hash, model_patch=patch)
            outcome, oracle_tail = (
                _oracle(root, task, test_timeout, oracle_image) if oracle_image is not None
                else _oracle(root, task, test_timeout)
            )
            row.update(outcome=outcome, oracle_tail=oracle_tail)
            return row
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        if not preserve_execution_receipt or row is None:
            raise
        # Cleanup can fail while returning: retain the receipt accumulated so far.
        row.update(outcome="execution_error", detail=str(exc))
        return row


def run(
    protocol_path: Path, tasks_path: Path, repo: Path, output: Path, base_url: str,
    *, max_model_calls: int | None = None, continue_from: Path | None = None,
) -> dict[str, Any]:
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError("development result artifact must be outside the repository")
    if output.exists() or output.is_symlink():
        raise ValueError("refusing to overwrite an existing pilot artifact")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    schemas = {"entroly.real-commit-pilot.v1": 1, "entroly.real-commit-pilot.v2": 2}
    if protocol["schema"] not in schemas:
        raise ValueError("unsupported pilot protocol")
    version = schemas[protocol["schema"]]
    bounded = max_model_calls is not None or continue_from is not None
    if bounded and version != 2:
        raise ValueError("bounded continuations require evaluation version 2")
    if max_model_calls is not None and (type(max_model_calls) is not int or max_model_calls <= 0):
        raise ValueError("max_model_calls must be a positive integer")
    arms = _ARMS_V1 if version == 1 else _ARMS_V2
    if version == 2 and type(protocol["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if not protocol["candidate_shas"] or len(set(protocol["candidate_shas"])) != len(protocol["candidate_shas"]):
        raise ValueError("frozen candidate list must be nonempty and unique")
    for field in ("max_corpus_bytes", "selection_budget", "model_timeout_s", "test_timeout_s", "max_output_tokens", "context_window"):
        if type(protocol[field]) not in (int, float) or protocol[field] <= 0:
            raise ValueError(f"{field} must be positive")
    tasks = _load_tasks(tasks_path, protocol["candidate_shas"])
    tasks_digest = hashlib.sha256(tasks_path.read_bytes()).hexdigest()
    if tasks_digest != protocol["tasks_sha256"]:
        raise ValueError("validated task file differs from frozen protocol")
    if _git(repo, "rev-parse", "HEAD").strip() != protocol["source_sha"]:
        raise ValueError("runner source checkout differs from frozen protocol")
    model_digest = _local_model_digest(base_url, protocol["model"])
    if model_digest != protocol["model_digest"]:
        raise ValueError("local model digest differs from frozen protocol")
    with isolated_engine_dir():
        assert_engine_isolated()
        from entroly import optimize

        engine = native_status()
        if not engine.ok:
            raise RuntimeError("pilot requires the current native engine")
        artifact = {
            "schema": f"entroly.real-commit-pilot-result.v{version}",
            "status": "in_progress",
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "tasks_sha256": tasks_digest,
            "source_sha": protocol["source_sha"],
            "engine": {"native_active": engine.ok, "version": engine.version},
            "model": protocol["model"], "model_digest": model_digest,
            "selection_query_policy": "legacy_generic" if version == 1 else "task_prompt",
            "arm_order_policy": "fixed" if version == 1 else "rotating_by_frozen_task_index",
            "rows": [],
            "limitations": (
                "Development pilot on one repository, no independent holdout, "
                "not the preregistered n>=400 study; full context is the declared "
                "source/test corpus, not the entire repository; repeated raw calls "
                "do not establish causal context attribution."
            ),
        }
        if bounded:
            from benchmarks.pilot_runtime_identity import runtime_identity

            artifact["limitations"] += (
                " Continuation fingerprints do not establish identical hardware load or model RNG state."
                " Stop the earlier process before continuing; an in-flight arm is recorded as unknown,"
                " never automatically retried. Artifact checksums detect edits but do not authenticate outcomes."
            )
            runtime = runtime_identity(repo, engine, base_url)
            expected = {key: artifact[key] for key in (
                "schema", "protocol_sha256", "tasks_sha256", "source_sha", "engine",
                "model", "model_digest", "selection_query_policy", "arm_order_policy",
            )}
            parent = _read_continuation(continue_from, expected, protocol, runtime) if continue_from else None
            inherited = parent["artifact"]["continuation"] if parent else {}
            artifact["rows"] = parent["artifact"]["rows"] if parent else []
            artifact["continuation"] = {
                "schema": "entroly.pilot-continuation.v1", "runtime": runtime,
                "parent_sha256": parent["sha256"] if parent else None,
                "new_arm_attempts": 0,
                "total_arm_attempts": inherited.get("total_arm_attempts", 0),
                "max_model_calls": max_model_calls,
                "task_plans": inherited.get("task_plans", {}), "pending_arm": None,
            }

        output_created = False

        def checkpoint() -> None:
            nonlocal output_created
            artifact["summary"] = summarize(artifact["rows"], protocol["candidate_shas"], arms)
            if bounded:
                artifact["artifact_sha256"] = _artifact_digest(artifact)
            output.parent.mkdir(parents=True, exist_ok=True)
            serialized = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if not output_created:
                # Claim the new output exclusively: two invocations passing the
                # initial exists check must not replace each other's artifacts.
                with output.open("x", encoding="utf-8") as handle:
                    handle.write(serialized)
                    if bounded:
                        handle.flush()
                        os.fsync(handle.fileno())
                output_created = True
                return
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent, prefix=output.name + ".",
                suffix=".tmp", delete=False,
            ) as handle:
                handle.write(serialized)
                if bounded:
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary = Path(handle.name)
            try:
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)

        def record(row: dict[str, Any]) -> None:
            if bounded and row.get("arm") is not None:
                plan = artifact["continuation"]["task_plans"][row["task_id"]]["arms"][row["arm"]]
                if row.get("prompt_sha256") is not None and row["prompt_sha256"] != plan.get("prompt_sha256"):
                    raise ValueError("observed prompt differs from the frozen arm plan")
                row["plan_sha256"] = _digest(plan)
            artifact["rows"].append(row)
            checkpoint()

        if bounded and parent and inherited.get("pending_arm") is not None:
            # A process may die after a request was sent but before its receipt
            # was saved. Retain that unknown attempt instead of replaying it.
            record({**inherited["pending_arm"], "outcome": "interrupted_unknown"})
        else:
            checkpoint()
        done = {
            task_id for task_id in protocol["candidate_shas"]
            if any(row["task_id"] == task_id and row.get("arm") is None for row in artifact["rows"])
            or {row.get("arm") for row in artifact["rows"] if row["task_id"] == task_id} == set(arms)
        }

        def at_limit() -> bool:
            return bounded and max_model_calls is not None and (
                artifact["continuation"]["new_arm_attempts"] >= max_model_calls
            )

        def stopped() -> dict:
            artifact["status"] = "call_limit_reached"
            checkpoint()
            return artifact

        for task_index, task in enumerate(tasks):
            if task["sha"] in done:
                continue
            if at_limit():
                return stopped()
            fragments = _fragments(repo, task)
            corpus_bytes = sum(len(fragment.content.encode("utf-8")) for fragment in fragments)
            if corpus_bytes > protocol["max_corpus_bytes"]:
                record({
                    "task_id": task["sha"], "outcome": "ineligible_context_size",
                    "corpus_bytes": corpus_bytes, "limit_bytes": protocol["max_corpus_bytes"],
                })
                continue
            plans = artifact["continuation"]["task_plans"] if bounded else {}
            if task["sha"] not in plans:
                with _task_worktree(repo, task) as baseline_root:
                    baseline_outcome, baseline_tail = (
                        _oracle(baseline_root, task, protocol["test_timeout_s"], runtime["oracle_image"])
                        if bounded else _oracle(baseline_root, task, protocol["test_timeout_s"])
                    )
                if baseline_outcome != "test_failure":
                    record({
                        "task_id": task["sha"], "outcome": "ineligible_oracle_drift",
                        "baseline_outcome": baseline_outcome,
                        "baseline_tail": baseline_tail,
                        "corpus_bytes": corpus_bytes,
                    })
                    continue
            raw = build_raw(fragments)
            try:
                selected = build_compressed(
                    fragments,
                    query="make the touched tests pass" if version == 1 else _prompt("", task),
                    budget=protocol["selection_budget"],
                    optimize_fn=optimize,
                )
                selection_error = None
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                selected = None
                selection_error = str(exc)
            contexts = {"raw": raw, "selected": selected, "raw_repeat": raw}
            if version == 2:
                contexts["null"] = build_raw([])
                contexts["null"].notes = ["no candidate source or test content"]
            if bounded:
                plan = {
                    "corpus_sha256": _digest([(f.source, f.digest) for f in fragments]),
                    "arms": {
                        arm: {"selection_error": True} if built is None else {
                            "prompt_sha256": hashlib.sha256(_prompt(built.text, task).encode("utf-8")).hexdigest(),
                            "context": {**built.to_dict(), "arm": arm},
                        }
                        for arm, built in contexts.items()
                    },
                }
                if task["sha"] in plans and plans[task["sha"]] != plan:
                    raise ValueError("continuation context selection differs from the frozen arm plan")
                plans[task["sha"]] = plan
                checkpoint()
            for arm in _arm_order(version, task_index, protocol["seed"]):
                if any(row["task_id"] == task["sha"] and row.get("arm") == arm for row in artifact["rows"]):
                    continue
                built = contexts[arm]
                if built is None:
                    record({
                        "task_id": task["sha"], "arm": arm,
                        "outcome": "selection_error", "detail": selection_error,
                        "corpus_bytes": corpus_bytes,
                    })
                    continue
                if at_limit():
                    return stopped()
                if bounded:
                    progress = artifact["continuation"]
                    progress["new_arm_attempts"] += 1
                    progress["total_arm_attempts"] += 1
                    progress["pending_arm"] = {"task_id": task["sha"], "arm": arm}
                    checkpoint()
                try:
                    row = _run_arm(
                        repo, task, arm=arm, context=built.text,
                        context_details=(
                            {**built.to_dict(), "arm": arm} if version == 2 else built.to_dict()
                        ),
                        model=protocol["model"],
                        base_url=base_url, seed=protocol["seed"],
                        model_timeout=protocol["model_timeout_s"],
                        test_timeout=protocol["test_timeout_s"],
                        output_tokens=protocol["max_output_tokens"],
                        context_window=protocol["context_window"],
                        preserve_execution_receipt=(version == 2),
                        **({"oracle_image": runtime["oracle_image"]} if bounded else {}),
                    )
                except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                    if version == 1:
                        raise
                    row = {"task_id": task["sha"], "arm": arm, "outcome": "execution_error",
                           "detail": str(exc)}
                if bounded:
                    artifact["continuation"]["pending_arm"] = None
                record({**row, "corpus_bytes": corpus_bytes})
                print(f"{task['sha'][:12]} {arm}: {row['outcome']}", flush=True)
        artifact["status"] = "complete"
        checkpoint()
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--max-model-calls", type=int, help="v2 cap on new model-generating arm attempts")
    parser.add_argument("--continue-from", type=Path, help="v2 incomplete artifact; write continuation to a new --out")
    args = parser.parse_args()
    run(args.protocol, args.tasks, args.repo.resolve(), args.out, args.base_url,
        max_model_calls=args.max_model_calls, continue_from=args.continue_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
