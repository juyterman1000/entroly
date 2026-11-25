"""Real Git/patch boundaries with simulated generation, never provider-quality evidence."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import cross_agent_matched as matched
from benchmarks.cross_agent_cli import AgentSpec
from benchmarks.real_commit_pilot import _digest, _git

IMAGE = "sha256:" + "a" * 64
PATCH = "--- a/solver.py\n+++ b/solver.py\n@@ -1,2 +1,2 @@\n def answer():\n-    return 0\n+    return 42\n"


@pytest.fixture
def trial(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "solver.py").write_text("def answer():\n    return 0\n", encoding="utf-8")
    (repo / "test_solver.py").write_text("# prior test\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "before")
    (repo / "solver.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    (repo / "test_solver.py").write_text("from solver import answer\ndef test_answer():\n    assert answer() == 42\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fix")
    task = {"sha": _git(repo, "rev-parse", "HEAD").strip(), "status": "validated",
            "fail_outcome": "test_failure", "source_files": ["solver.py"], "test_files": ["test_solver.py"]}
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(json.dumps(task) + "\n", encoding="utf-8")
    agents = [AgentSpec("codex", "codex", "requested-codex", "codex.exe"),
              AgentSpec("claude", "claude", "requested-claude", "claude.exe")]
    runtimes = {a.agent_id: {"version": a.agent_id + "-version"} for a in agents}
    monkeypatch.setattr(matched, "preflight", lambda agent: runtimes[agent.agent_id])
    monkeypatch.setattr(matched, "_source_identity", lambda: {"source": "frozen"})
    plan = matched.prepare(repo, tasks_file, [task["sha"]], agents, selection_budget=100,
                           timeout_s=2, test_timeout_s=2, max_response_bytes=2000,
                           oracle_image=IMAGE,
                           optimize_fn=lambda fragments, **kw: {"selected": fragments[:1]})
    monkeypatch.setenv("ENTROLY_REAL_COMMIT_SANDBOX", "1")
    # Avoid replacing subprocess.run globally: _git shares that module.
    real_subprocess = matched.subprocess
    monkeypatch.setattr(matched, "subprocess", SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(stdout=IMAGE),
        SubprocessError=real_subprocess.SubprocessError,
    ))
    # Inspect known test-fixture bytes, never execute generated Python on host.
    oracle_calls = []

    def oracle(root, task, timeout, image):
        oracle_calls.append((Path(root), image))
        assert image == IMAGE
        return ("passed" if "return 42" in (root / "solver.py").read_text() else "test_failure"), "fixture oracle"

    monkeypatch.setattr(matched, "_oracle", oracle)
    calls = []

    def generate(agent, prompt, *, timeout_s):
        calls.append((agent.agent_id, prompt))
        return {"text": PATCH, "usage": {"provider_specific": 17}, "latency_s": 0.1,
                "runtime": runtimes[agent.agent_id]}

    monkeypatch.setattr(matched, "generate", generate)
    return SimpleNamespace(repo=repo, plan=plan, output=tmp_path / "result.json", calls=calls,
                           oracle_calls=oracle_calls, runtimes=runtimes, agents=agents)


def execute(trial, **kwargs):
    return matched.run(trial.plan, trial.repo, trial.output, max_calls=kwargs.pop("max_calls", 8),
                       allow_account_calls=kwargs.pop("allow_account_calls", True), **kwargs)


def test_matched_prompts_real_patch_grading_and_controls(trial):
    artifact = execute(trial)
    assert artifact["stop_reason"] == "complete"
    assert artifact["model_calls"] == 8
    assert artifact["artifact_sha256"] == _digest({k: v for k, v in artifact.items() if k != "artifact_sha256"})
    assert len(trial.oracle_calls) == 10  # Broken/fixed validation + eight independent grades.
    prompts = trial.plan["tasks"][0]["prompts"]
    assert prompts["raw"] == prompts["raw_repeat"]
    assert "return 0" in prompts["raw"]
    assert "return 42" not in prompts["raw"]
    assert "assert answer() == 42" in prompts["raw"]  # Declared test-visible task design.
    assert "def answer" not in prompts["null"]
    for agent in ("codex", "claude"):
        assert {prompt for aid, prompt in trial.calls if aid == agent} == set(prompts.values())
        counts = artifact["summary"]["per_agent"][agent]
        assert counts["comparisons_to_selected"]["raw"]["both_passed"] == 1
        assert counts["planned_cells"] == counts["recorded_cells"] == 4
    assert all(row["generation"]["usage"] == {"provider_specific": 17} for row in artifact["rows"])
    assert artifact["summary"]["quality_claim_allowed"] is False
    assert artifact["summary"]["handoff_quality_measured"] is False
    assert len(artifact["summary"]["complete_tasks_across_all_agents"]) == 1
    assert "return 42" in (trial.repo / "solver.py").read_text()  # Original checkout untouched.
    assert len(_git(trial.repo, "worktree", "list", "--porcelain").split("worktree ")) == 2


def test_call_limit_preserves_missing_pairs_and_full_denominator(trial):
    artifact = execute(trial, max_calls=3)
    assert len(trial.calls) == 3
    assert artifact["stop_reason"] == "call_budget"
    assert artifact["summary"]["planned_cells"] == 8
    counts = artifact["summary"]["per_agent"]["claude"]
    assert counts["not_recorded_cells"] == 3
    assert counts["comparisons_to_selected"]["raw"]["incomplete_or_infrastructure_error"] == 1


def test_missing_agent_is_visible_and_never_counted_as_quality_loss(trial):
    trial.plan["agents"][1]["runtime"] = None
    trial.plan["plan_sha256"] = _digest({k: v for k, v in trial.plan.items() if k != "plan_sha256"})
    artifact = execute(trial)
    assert len(trial.calls) == 4
    counts = artifact["summary"]["per_agent"]["claude"]
    assert counts["outcomes_by_arm"]["raw"] == {"agent_unavailable": 1}
    assert counts["comparisons_to_selected"]["raw"]["both_failed"] == 0
    assert counts["comparisons_to_selected"]["raw"]["incomplete_or_infrastructure_error"] == 1


@pytest.mark.parametrize("failure", ["checksum", "source", "sandbox", "opt_in", "existing_output"])
def test_fail_closed_before_model_calls(trial, monkeypatch, failure):
    if failure == "checksum":
        trial.plan["limits"]["agent_timeout_s"] = 900
    elif failure == "source":
        monkeypatch.setattr(matched, "_source_identity", lambda: {"source": "changed"})
    elif failure == "sandbox":
        monkeypatch.delenv("ENTROLY_REAL_COMMIT_SANDBOX")
    elif failure == "existing_output":
        trial.output.write_text("original", encoding="utf-8")
    with pytest.raises((ValueError, FileExistsError)):
        execute(trial, allow_account_calls=failure != "opt_in")
    assert trial.calls == []
    if failure == "existing_output":
        assert trial.output.read_text() == "original"


def test_oracle_failure_prevents_all_generation(trial, monkeypatch):
    monkeypatch.setattr(matched, "_oracle", lambda *a: ("infrastructure_error", ""))
    result = execute(trial)
    assert result["stop_reason"] == "preflight_or_execution_error"
    assert result["model_calls"] == 0
    assert trial.calls == []
    assert json.loads(trial.output.read_text())["summary"]["planned_cells"] == 8


def test_interrupted_attempt_is_durable_and_not_replayed(trial, monkeypatch):
    def interrupt(*a, **kw):
        on_disk = json.loads(trial.output.read_text())
        assert on_disk["model_calls"] == 1
        assert on_disk["pending"]["agent_id"] == "codex"
        raise KeyboardInterrupt

    monkeypatch.setattr(matched, "generate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        execute(trial)
    saved = json.loads(trial.output.read_text())
    assert saved["summary"]["pending_attempt"] is not None
    with pytest.raises(FileExistsError):
        execute(trial)


def test_generation_failure_keeps_unusable_accounting(trial, monkeypatch):
    monkeypatch.setattr(matched, "generate", lambda *a, **kw: {
        "text": "not a patch", "usage": {"cached_input_tokens": 19},
        "runtime": trial.runtimes["codex"], "latency_s": 1,
    })
    result = execute(trial, max_calls=1)
    assert result["rows"][0]["outcome"] == "unusable_patch"
    assert result["rows"][0]["generation"]["usage"]["cached_input_tokens"] == 19


def test_timeout_is_a_bounded_task_failure_not_silently_dropped(trial, monkeypatch):
    def timeout(*a, **kw):
        raise TimeoutError("deadline")

    monkeypatch.setattr(matched, "generate", timeout)
    result = execute(trial)
    assert result["model_calls"] == 8
    for agent in ("codex", "claude"):
        counts = result["summary"]["per_agent"][agent]
        assert counts["comparisons_to_selected"]["raw"]["both_failed"] == 1
        assert counts["outcomes_by_arm"]["raw"] == {"generation_timeout": 1}
    assert len(result["summary"]["complete_tasks_across_all_agents"]) == 1


def test_runtime_change_preserves_usage_and_blocks_grading(trial, monkeypatch):
    monkeypatch.setattr(matched, "generate", lambda *a, **kw: {
        "text": PATCH, "runtime": {"version": "changed"}, "usage": {"input": 4},
    })
    result = execute(trial, max_calls=1)
    assert result["rows"][0]["outcome"] == "runtime_changed"
    assert result["rows"][0]["generation"]["usage"] == {"input": 4}
    assert len(trial.oracle_calls) == 2
    assert result["summary"]["complete_tasks_across_all_agents"] == []


def test_summarize_rejects_duplicates_and_prompt_drift(trial):
    row = {"agent_id": "codex", "task_id": trial.plan["tasks"][0]["task"]["sha"],
           "arm": "raw", "outcome": "passed", "prompt_sha256": trial.plan["tasks"][0]["prompt_sha256"]["raw"]}
    with pytest.raises(ValueError, match="duplicate"):
        matched.summarize(trial.plan, [row, deepcopy(row)])
    row["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="prompt"):
        matched.summarize(trial.plan, [row])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True])
def test_invalid_limits_rejected(value):
    with pytest.raises(ValueError):
        matched._positive(value, "limit")


def test_private_artifacts_stay_out_of_source_tree(trial):
    with pytest.raises(ValueError, match="outside"):
        matched._private_output(trial.repo / "private-plan.json", trial.repo)


def test_option_like_fixture_paths_cannot_change_pytest_selection(trial, tmp_path):
    task = deepcopy(trial.plan["tasks"][0]["task"])
    task["test_files"] = ["-k"]
    tasks = tmp_path / "options.jsonl"
    tasks.write_text(json.dumps(task), encoding="utf-8")
    with pytest.raises(ValueError, match="command options"):
        matched.prepare(trial.repo, tasks, [task["sha"]], trial.agents,
                        selection_budget=100, timeout_s=1, test_timeout_s=1,
                        max_response_bytes=100, oracle_image=IMAGE,
                        optimize_fn=lambda *a, **kw: pytest.fail("must reject before selection"))


def test_prepare_hashes_the_same_snapshot_it_parses(trial, tmp_path):
    task = trial.plan["tasks"][0]["task"]
    tasks = tmp_path / "changing.jsonl"
    original = json.dumps(task).encode("utf-8")
    tasks.write_bytes(original)

    def mutate_after_parse(fragments, **kw):
        tasks.write_text("changed while preparing", encoding="utf-8")
        return {"selected": fragments[:1]}

    plan = matched.prepare(trial.repo, tasks, [task["sha"]], trial.agents,
                           selection_budget=100, timeout_s=1, test_timeout_s=1,
                           max_response_bytes=100, oracle_image=IMAGE,
                           optimize_fn=mutate_after_parse)
    assert plan["tasks_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert plan["tasks"][0]["task"] == task
