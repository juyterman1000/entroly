"""Exercise a real fail-to-pass task boundary without a model or network."""

import json
import subprocess

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
