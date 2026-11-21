"""Exercise a real fail-to-pass task boundary without a model or network."""

import json
import hashlib
import subprocess
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from benchmarks.real_commit_pilot import (
    _apply_patch,
    _check_patch_paths,
    _extract_patch,
    _load_tasks,
    _local_model_digest,
    _oracle,
    _task_worktree,
    run,
    summarize,
    _arm_order,
)


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_patch_is_confined_and_oracle_executes_in_isolated_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "solver.py").write_text("def answer():\n    return 0\n", encoding="utf-8")
    (repo / "test_solver.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Pilot Test", "-c", "user.email=test@example.com", "commit", "-m", "before")
    (repo / "solver.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (repo / "test_solver.py").write_text("from solver import answer\n\ndef test_answer():\n    assert answer() == 42\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Pilot Test", "-c", "user.email=test@example.com", "commit", "-m", "fix")
    task = {
        "sha": _git(repo, "rev-parse", "HEAD"),
        "source_files": ["solver.py"],
        "test_files": ["test_solver.py"],
    }
    patch = (
        "diff --git a/solver.py b/solver.py\n"
        "--- a/solver.py\n+++ b/solver.py\n"
        "@@ -1,2 +1,2 @@\n def answer():\n-    return 0\n+    return 42\n"
    )
    with _task_worktree(repo, task) as root:
        assert _oracle(root, task, 20)[0] == "test_failure"
        patch_hash, applied = _apply_patch(root, f"```diff\n{patch}```", task["source_files"])
        assert patch_hash and applied == patch
        assert _oracle(root, task, 20)[0] == "passed"
    assert (repo / "solver.py").read_text(encoding="utf-8") == "def answer():\n    return 42\n"


@pytest.mark.parametrize(
    "path", ["tests/test_solver.py", "../outside.py", "solver2.py"]
)
def test_unapproved_patch_path_is_rejected(path):
    patch = f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n"
    with pytest.raises(ValueError, match="outside the task source files"):
        _check_patch_paths(patch, ["solver.py"])


def test_noncanonical_model_diff_is_reported_as_path_error():
    response = "```diff\n--- solver.py\n+++ test_solver.py\n@@ -1 +1 @@\n-old\n+new\n```"
    patch = _extract_patch(response)
    with pytest.raises(ValueError, match="only modify existing source files"):
        _check_patch_paths(patch, ["solver.py"])


def test_timestamped_unified_headers_keep_the_same_source_path():
    patch = (
        "--- a/solver.py\t2026-01-01\n"
        "+++ b/solver.py\t2026-01-02\n"
        "@@ -1 +1 @@\n-old\n+new\n"
    )
    _check_patch_paths(patch, ["solver.py"])  # must not raise
    assert patch.startswith("--- a/solver.py")


def test_extra_unapproved_diff_after_fence_is_not_ignored():
    response = (
        "```diff\n--- a/solver.py\n+++ b/solver.py\n@@ -1 +1 @@\n-old\n+new\n```\n"
        "--- a/tests/test_solver.py\n+++ b/tests/test_solver.py\n"
    )
    with pytest.raises(ValueError, match="outside its diff fence"):
        _extract_patch(response)


def test_task_file_must_match_frozen_candidate_set(tmp_path):
    path = tmp_path / "tasks.jsonl"
    sha = "a" * 40
    path.write_text(json.dumps({
        "sha": sha, "status": "validated", "fail_outcome": "test_failure",
        "source_files": ["source.py"], "test_files": ["test_source.py"],
    }) + "\n", encoding="utf-8")
    assert _load_tasks(path, [sha])[0]["sha"] == sha
    with pytest.raises(ValueError, match="exactly match"):
        _load_tasks(path, [sha, "b" * 40])
    path.write_text(json.dumps({
        "sha": sha, "status": "validated", "fail_outcome": "test_failure",
        "source_files": ["../outside.py"], "test_files": ["test_source.py"],
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe repository path"):
        _load_tasks(path, [sha])


def test_local_only_and_private_output_guards_run_before_model_calls(tmp_path):
    with pytest.raises(ValueError, match="local Ollama"):
        _local_model_digest("https://api.example.com", "model")
    protocol = tmp_path / "protocol.json"
    tasks = tmp_path / "tasks.jsonl"
    with pytest.raises(ValueError, match="outside the repository"):
        run(protocol, tasks, tmp_path, tmp_path / "result.json", "http://localhost:11434")


def test_v2_arm_order_balances_positions_and_v1_keeps_its_order():
    arms = ("raw", "selected", "raw_repeat", "null")
    orders = [_arm_order(2, index, 17) for index in range(4)]
    for position in range(4):
        assert {order[position] for order in orders} == set(arms)
    assert _arm_order(1, 3, 17) == ("raw", "selected", "raw_repeat")


def test_summary_keeps_losses_errors_exclusions_and_missing_arms():
    rows = [
        {"task_id": "loss", "arm": "raw", "outcome": "passed"},
        {"task_id": "loss", "arm": "selected", "outcome": "test_failure"},
        {"task_id": "error", "arm": "raw", "outcome": "passed"},
        {"task_id": "error", "arm": "selected", "outcome": "execution_error"},
        {"task_id": "excluded", "outcome": "ineligible_context_size"},
    ]
    result = summarize(rows, ["loss", "error", "excluded", "missing"], ("raw", "selected"))
    assert result["frozen_tasks"] == 4
    assert result["comparisons_to_selected"]["raw"] == {
        "both_passed": 0, "selected_only_passed": 0, "baseline_only_passed": 1,
        "both_failed": 0, "excluded_tasks": 1, "incomplete_or_infrastructure_error": 2,
    }
    assert result["quality_claim_allowed"] is False
    with pytest.raises(ValueError, match="duplicate"):
        summarize([*rows, rows[0]], ["loss", "error", "excluded", "missing"], ("raw", "selected"))


@pytest.mark.parametrize("selected_outcome", ["timeout", "execution_error"])
def test_summary_scores_oracle_timeout_but_not_execution_error(selected_outcome):
    result = summarize([
        {"task_id": "task", "arm": "raw", "outcome": "passed"},
        {"task_id": "task", "arm": "selected", "outcome": selected_outcome},
    ], ["task"], ("raw", "selected"))
    assert result["frozen_tasks"] == 1
    assert result["comparisons_to_selected"]["raw"] == {
        "both_passed": 0, "selected_only_passed": 0,
        "baseline_only_passed": int(selected_outcome == "timeout"),
        "both_failed": 0, "excluded_tasks": 0,
        "incomplete_or_infrastructure_error": int(selected_outcome == "execution_error"),
    }


@pytest.mark.parametrize("failure_stage", ["oracle", "cleanup"])
@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
@pytest.mark.parametrize("preserve_receipt", [False, True])
def test_run_arm_retains_receipt_on_post_generation_error(
    tmp_path, monkeypatch, failure_stage, error_type, preserve_receipt,
):
    import benchmarks.real_commit_pilot as pilot

    task = {"sha": "a" * 40, "source_files": ["solver.py"], "test_files": ["test_solver.py"]}
    context_details = {"arm": "selected", "included_sources": ["solver.py"]}
    context = "def solve(): return 0"
    response = "--- a/solver.py\n+++ b/solver.py\n@@ -1 +1 @@\n-old\n+new\n"
    patch_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
    error = error_type(f"{failure_stage} failed after generation")
    calls = []

    @contextmanager
    def worktree(*args):
        yield tmp_path
        if failure_stage == "cleanup":
            raise error

    def generate(**options):
        calls.append(options)
        return {"text": response, "input_tokens": 100, "output_tokens": 25, "latency_s": 1.25}

    def oracle(*args):
        if failure_stage == "oracle":
            raise error
        return "passed", "1 passed"

    monkeypatch.setattr(pilot, "_task_worktree", worktree)
    monkeypatch.setattr(pilot, "call_model", generate)
    monkeypatch.setattr(pilot, "_apply_patch", lambda *args: (patch_hash, response))
    monkeypatch.setattr(pilot, "_oracle", oracle)
    options = dict(
        arm="selected", context=context, context_details=context_details,
        model="local-model", base_url="http://localhost:11434", seed=17,
        model_timeout=5, test_timeout=5, output_tokens=32, context_window=1024,
    )
    if not preserve_receipt:
        # Omitting the option exercises the default v1 error-raising behavior.
        with pytest.raises(error_type) as raised:
            pilot._run_arm(tmp_path, task, **options)
        assert raised.value is error
        assert len(calls) == 1
        return

    row = pilot._run_arm(tmp_path, task, **options, preserve_execution_receipt=True)
    assert len(calls) == 1
    assert row == {
        "task_id": task["sha"], "arm": "selected", "context": context_details,
        "input_tokens": 100, "output_tokens": 25, "latency_s": 1.25,
        "prompt_sha256": hashlib.sha256(calls[0]["prompt"].encode("utf-8")).hexdigest(),
        "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
        "patch_sha256": patch_hash, "model_patch": response,
        "outcome": "execution_error", "detail": str(error),
        **({"oracle_tail": "1 passed"} if failure_stage == "cleanup" else {}),
    }


@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_run_arm_reraises_generation_errors_without_a_receipt(tmp_path, monkeypatch, error_type):
    import benchmarks.real_commit_pilot as pilot

    task = {"sha": "a" * 40, "source_files": ["solver.py"], "test_files": ["test_solver.py"]}
    error = error_type("generation failed before receipt")

    def generate(**options):
        raise error

    monkeypatch.setattr(pilot, "_task_worktree", lambda *args: nullcontext(tmp_path))
    monkeypatch.setattr(pilot, "call_model", generate)
    with pytest.raises(error_type) as raised:
        pilot._run_arm(
            tmp_path, task, arm="selected", context="", context_details={},
            model="local-model", base_url="http://localhost:11434", seed=17,
            model_timeout=5, test_timeout=5, output_tokens=32, context_window=1024,
            preserve_execution_receipt=True,
        )
    assert raised.value is error


@pytest.mark.parametrize("version", [1, 2])
def test_runner_protocol_preserves_v1_and_exercises_v2_controls(tmp_path, monkeypatch, version):
    import entroly
    import benchmarks.real_commit_pilot as pilot
    from benchmarks.agentic_arms import Fragment

    repo = tmp_path / "repo"
    repo.mkdir()
    tasks = [
        {"sha": str(index) * 40, "status": "validated", "fail_outcome": "test_failure",
         "source_files": [f"solver{index}.py"], "test_files": [f"test_solver{index}.py"]}
        for index in range(1, 5)
    ]
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
    protocol = {
        "schema": f"entroly.real-commit-pilot.v{version}",
        "candidate_shas": [task["sha"] for task in tasks],
        "source_sha": "f" * 40,
        "tasks_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
        "model": "local-model", "model_digest": "digest", "seed": 17,
        "max_corpus_bytes": 1000, "selection_budget": 40,
        "max_output_tokens": 32, "context_window": 1024,
        "model_timeout_s": 5, "test_timeout_s": 5,
    }
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    monkeypatch.setattr(pilot, "_git", lambda *args: protocol["source_sha"])
    monkeypatch.setattr(pilot, "_local_model_digest", lambda *args: "digest")
    monkeypatch.setattr(pilot, "native_status", lambda: SimpleNamespace(ok=True, version="1.0.91"))
    monkeypatch.setattr(pilot, "_fragments", lambda _repo, task: [
        Fragment(task["source_files"][0], "def solve(): return 0"),
        Fragment(task["test_files"][0], "def test_solve(): assert solve() == 42"),
    ])
    monkeypatch.setattr(pilot, "_task_worktree", lambda *args: nullcontext(repo))
    monkeypatch.setattr(pilot, "_oracle", lambda *args: ("test_failure", "assert 0 == 42"))
    queries = []

    def optimize(payload, *, budget, query):
        assert budget == 40
        queries.append(query)
        return {"selected": payload[:1]}

    monkeypatch.setattr(entroly, "optimize", optimize)
    calls = []

    def attempt(_repo, task, **options):
        calls.append((task["sha"], options))
        if version == 2 and options["arm"] == "selected" and task == tasks[0]:
            raise RuntimeError("model request timed out")
        return {
            "task_id": task["sha"], "arm": options["arm"], "outcome": "passed",
            "input_tokens": 12, "output_tokens": 8,
        }

    monkeypatch.setattr(pilot, "_run_arm", attempt)
    artifact = run(protocol_file, task_file, repo, tmp_path / "result.json", "http://localhost:11434")
    assert artifact["status"] == "complete"
    assert len(calls) == 4 * (3 if version == 1 else 4)
    for task in tasks:
        task_calls = [options for task_id, options in calls if task_id == task["sha"]]
        assert [options["arm"] for options in task_calls] == list(
            _arm_order(version, tasks.index(task), 17)
        )
        for options in task_calls:
            assert options["preserve_execution_receipt"] is (version == 2)
            assert (options["model"], options["seed"], options["output_tokens"],
                    options["context_window"], options["model_timeout"]) == (
                        "local-model", 17, 32, 1024, 5
                    )
            if options["arm"] == "null":
                assert options["context"] == ""
                assert options["context_details"]["arm"] == "null"
    if version == 1:
        assert queries == ["make the touched tests pass"] * 4
    else:
        assert all(task["source_files"][0] in query and task["test_files"][0] in query
                   for task, query in zip(tasks, queries))
        assert artifact["summary"]["outcomes_by_arm"]["selected"]["execution_error"] == 1
        assert artifact["summary"]["comparisons_to_selected"]["raw"]["incomplete_or_infrastructure_error"] == 1
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["summary"] == artifact["summary"]
