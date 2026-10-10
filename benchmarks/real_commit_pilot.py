"""Local-model development pilot on mined, validated repository commits.

This is not the preregistered n>=400 cross-repository experiment. The fixed
protocol names every candidate, including candidates excluded for local-model
context size. Results are written outside the repository by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
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
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
    patch = fenced.group(1) if fenced else response
    start = patch.find("diff --git ")
    if start < 0:
        start = patch.find("--- a/")
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
        if not old.startswith("a/") or not new.startswith("b/"):
            raise ValueError("patch may only modify existing source files")
        if old[2:] != new[2:] or old[2:] not in allowed_set:
            raise ValueError("patch changes a path outside the task source files")
    headers = re.findall(r"^diff --git a/(\S+) b/(\S+)$", patch, re.M)
    if any(a != b or a not in allowed_set for a, b in headers):
        raise ValueError("patch diff header changes an unapproved path")
    if any(marker in patch for marker in ("GIT binary patch", "rename from", "rename to")):
        raise ValueError("binary and rename patches are unsupported")


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


def _oracle(root: Path, task: dict[str, Any], timeout: int) -> tuple[str, str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *task["test_files"], "-x", "-q", "--tb=short", "-p", "no:cacheprovider"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "timeout", ""
    output = result.stdout + "\n" + result.stderr
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


def _local_model_digest(base_url: str, model: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("pilot requires a local Ollama HTTP endpoint")
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=10) as response:
        tags = json.loads(response.read().decode("utf-8"))
    matches = [item for item in tags.get("models", []) if item.get("name") == model]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise ValueError("requested local model and digest were not found")
    return matches[0]["digest"]


def _run_arm(
    repo: Path, task: dict[str, Any], *, arm: str, context: str,
    context_details: dict[str, Any], model: str, base_url: str, seed: int,
    model_timeout: float, test_timeout: int, output_tokens: int, context_window: int,
) -> dict[str, Any]:
    with _task_worktree(repo, task) as root:
        generation = call_model(
            base_url=base_url, model=model, prompt=_prompt(context, task),
            seed=seed, timeout=model_timeout, max_output_tokens=output_tokens,
            context_window=context_window,
        )
        row = {
            "task_id": task["sha"], "arm": arm, "context": context_details,
            "input_tokens": generation["input_tokens"],
            "output_tokens": generation["output_tokens"],
            "latency_s": generation["latency_s"],
            "response_sha256": hashlib.sha256(generation["text"].encode()).hexdigest(),
        }
        try:
            patch_hash, patch = _apply_patch(root, generation["text"], task["source_files"])
        except (ValueError, RuntimeError) as exc:
            return {**row, "outcome": "unusable_patch", "patch_error": str(exc)}
        outcome, oracle_tail = _oracle(root, task, test_timeout)
        return {
            **row, "outcome": outcome, "patch_sha256": patch_hash,
            "model_patch": patch, "oracle_tail": oracle_tail,
        }


def run(protocol_path: Path, tasks_path: Path, repo: Path, output: Path, base_url: str) -> dict[str, Any]:
    if output.resolve().is_relative_to(repo.resolve()):
        raise ValueError("development result artifact must be outside the repository")
    if output.exists():
        raise ValueError("refusing to overwrite an existing pilot artifact")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["schema"] != "entroly.real-commit-pilot.v1":
        raise ValueError("unsupported pilot protocol")
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
            "schema": "entroly.real-commit-pilot-result.v1",
            "status": "in_progress",
            "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "tasks_sha256": tasks_digest,
            "source_sha": protocol["source_sha"],
            "engine": {"native_active": engine.ok, "version": engine.version},
            "model": protocol["model"], "model_digest": model_digest,
            "rows": [],
            "limitations": (
                "Development pilot on one repository, no independent holdout, "
                "not the preregistered n>=400 study; full context is the declared "
                "source/test corpus, not the entire repository; repeated raw calls "
                "do not establish causal context attribution."
            ),
        }

        def checkpoint() -> None:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".tmp")
            temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(output)

        def record(row: dict[str, Any]) -> None:
            artifact["rows"].append(row)
            checkpoint()

        checkpoint()
        for task in tasks:
            fragments = _fragments(repo, task)
            corpus_bytes = sum(len(fragment.content.encode("utf-8")) for fragment in fragments)
            if corpus_bytes > protocol["max_corpus_bytes"]:
                record({
                    "task_id": task["sha"], "outcome": "ineligible_context_size",
                    "corpus_bytes": corpus_bytes, "limit_bytes": protocol["max_corpus_bytes"],
                })
                continue
            with _task_worktree(repo, task) as baseline_root:
                baseline_outcome, baseline_tail = _oracle(
                    baseline_root, task, protocol["test_timeout_s"]
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
                    fragments, query="make the touched tests pass", budget=protocol["selection_budget"],
                    optimize_fn=optimize,
                )
                selection_error = None
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                selected = None
                selection_error = str(exc)
            for arm, built in (("raw", raw), ("selected", selected), ("raw_repeat", raw)):
                if built is None:
                    record({
                        "task_id": task["sha"], "arm": arm,
                        "outcome": "selection_error", "detail": selection_error,
                        "corpus_bytes": corpus_bytes,
                    })
                    continue
                row = _run_arm(
                    repo, task, arm=arm, context=built.text,
                    context_details=built.to_dict(), model=protocol["model"],
                    base_url=base_url, seed=protocol["seed"],
                    model_timeout=protocol["model_timeout_s"],
                    test_timeout=protocol["test_timeout_s"],
                    output_tokens=protocol["max_output_tokens"],
                    context_window=protocol["context_window"],
                )
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
    args = parser.parse_args()
    run(args.protocol, args.tasks, args.repo.resolve(), args.out, args.base_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
