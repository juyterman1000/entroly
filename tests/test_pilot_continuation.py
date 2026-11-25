"""Bounded attempts and immutable continuation, without model or service calls."""

import copy
import hashlib
import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import benchmarks.real_commit_pilot as pilot
import benchmarks.pilot_runtime_identity as identity
from benchmarks.agentic_arms import Fragment


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    import entroly

    repo = tmp_path / "repo"
    repo.mkdir()
    tasks = [
        {"sha": char * 40, "status": "validated", "fail_outcome": "test_failure",
         "source_files": [f"solver_{char}.py"], "test_files": [f"test_{char}.py"]}
        for char in ("a", "b")
    ]
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
    protocol = {
        "schema": "entroly.real-commit-pilot.v2", "candidate_shas": [task["sha"] for task in tasks],
        "source_sha": "f" * 40, "tasks_sha256": hashlib.sha256(tasks_file.read_bytes()).hexdigest(),
        "model": "local-model", "model_digest": "model-digest", "seed": 17,
        "max_corpus_bytes": 1000, "selection_budget": 40, "max_output_tokens": 32,
        "context_window": 1024, "model_timeout_s": 5, "test_timeout_s": 5,
    }
    protocol_file = tmp_path / "protocol.json"
    protocol_file.write_text(json.dumps(protocol), encoding="utf-8")
    runtime = {"schema": "test-runtime", "oracle_image": "sha256:" + "1" * 64,
               "native": {"binary": "2" * 64}, "ollama_version": "test-version"}
    calls, baselines = [], []
    monkeypatch.setattr(pilot, "_git", lambda *args: "f" * 40)
    monkeypatch.setattr(pilot, "_local_model_digest", lambda *args: "model-digest")
    monkeypatch.setattr(pilot, "native_status", lambda: SimpleNamespace(ok=True, version="1.0.91"))
    monkeypatch.setattr(identity, "runtime_identity", lambda *args: copy.deepcopy(runtime))
    monkeypatch.setattr(pilot, "_task_worktree", lambda *args: nullcontext(repo))
    def fragments(_repo, task):
        return [
            Fragment(task["source_files"][0], "def solve(): return 0"),
            Fragment(task["test_files"][0], "def test_solve(): assert solve() == 42"),
        ]
    monkeypatch.setattr(pilot, "_fragments", fragments)
    monkeypatch.setattr(entroly, "optimize", lambda payload, **kwargs: {"selected": payload[:1]})

    def oracle(_root, task, _timeout, image):
        assert image == runtime["oracle_image"]
        baselines.append(task["sha"])
        return "test_failure", "assert 0 == 42"

    monkeypatch.setattr(pilot, "_oracle", oracle)

    def attempt(_repo, task, **options):
        calls.append((task["sha"], options["arm"]))
        assert options["oracle_image"] == runtime["oracle_image"]
        return {
            "task_id": task["sha"], "arm": options["arm"], "outcome": "passed",
            "prompt_sha256": hashlib.sha256(pilot._prompt(options["context"], task).encode()).hexdigest(),
            "response_sha256": "3" * 64, "patch_sha256": "4" * 64,
            "input_tokens": 12, "output_tokens": 8, "latency_s": 0.5,
            "oracle_tail": "1 passed", "model_patch": "test patch",
        }

    monkeypatch.setattr(pilot, "_run_arm", attempt)

    def run(name, **options):
        return pilot.run(protocol_file, tasks_file, repo, tmp_path / name,
                         "http://localhost:11434", **options)

    return SimpleNamespace(
        run=run, root=tmp_path, repo=repo, tasks=tasks, tasks_file=tasks_file,
        protocol=protocol, protocol_file=protocol_file, runtime=runtime,
        calls=calls, baselines=baselines, attempt=attempt, fragments=fragments,
    )


def test_bounded_chain_completes_without_replaying_or_overwriting(campaign):
    c = campaign
    first = c.run("first.json", max_model_calls=2)
    first_bytes = (c.root / "first.json").read_bytes()
    assert first["status"] == "call_limit_reached"
    assert first["continuation"]["new_arm_attempts"] == 2
    assert first["summary"]["frozen_tasks"] == 2
    assert len(c.calls) == 2
    second = c.run("second.json", max_model_calls=3, continue_from=c.root / "first.json")
    second_bytes = (c.root / "second.json").read_bytes()
    assert second["status"] == "call_limit_reached"
    assert len(c.calls) == 5
    final = c.run("final.json", max_model_calls=3, continue_from=c.root / "second.json")
    assert final["status"] == "complete"
    assert final["continuation"]["new_arm_attempts"] == 3
    assert final["continuation"]["total_arm_attempts"] == 8
    assert final["continuation"]["parent_sha256"] == hashlib.sha256(second_bytes).hexdigest()
    assert len(c.calls) == len(set(c.calls)) == 8
    assert c.baselines == [task["sha"] for task in c.tasks]
    assert (c.root / "first.json").read_bytes() == first_bytes
    assert (c.root / "second.json").read_bytes() == second_bytes
    assert sum(row["input_tokens"] for row in final["rows"]) == 96
    assert sum(row["output_tokens"] for row in final["rows"]) == 64
    assert final["summary"]["comparisons_to_selected"]["raw"]["both_passed"] == 2
    assert final["summary"]["quality_claim_allowed"] is False
    with pytest.raises(ValueError, match="incomplete artifact"):
        c.run("invalid.json", continue_from=c.root / "final.json")
    assert len(c.calls) == 8


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_invalid_caps_rejected_before_attempts(campaign, cap):
    with pytest.raises(ValueError, match="positive integer"):
        campaign.run("invalid.json", max_model_calls=cap)
    assert not campaign.calls


def test_v1_cannot_silently_switch_to_bounded_protocol(campaign):
    campaign.protocol["schema"] = "entroly.real-commit-pilot.v1"
    campaign.protocol_file.write_text(json.dumps(campaign.protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="version 2"):
        campaign.run("invalid.json", max_model_calls=1)
    assert not campaign.calls


def test_interrupted_request_is_unknown_and_not_replayed(campaign, monkeypatch):
    c = campaign

    def crash(_repo, task, **options):
        c.calls.append((task["sha"], options["arm"]))
        raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "_run_arm", crash)
    with pytest.raises(KeyboardInterrupt):
        c.run("interrupted.json", max_model_calls=1)
    original = (c.root / "interrupted.json").read_bytes()
    partial = json.loads(original)
    assert partial["status"] == "in_progress"
    assert partial["rows"] == []
    assert partial["continuation"]["pending_arm"] is not None
    monkeypatch.setattr(pilot, "_run_arm", c.attempt)
    result = c.run("continued.json", max_model_calls=7, continue_from=c.root / "interrupted.json")
    assert result["status"] == "complete"
    assert len(c.calls) == len(set(c.calls)) == 8
    assert result["rows"][0]["outcome"] == "interrupted_unknown"
    assert result["continuation"]["new_arm_attempts"] == 7
    assert result["continuation"]["total_arm_attempts"] == 8
    assert sum(row.get("input_tokens", 0) for row in result["rows"]) == 84
    assert result["summary"]["comparisons_to_selected"]["raw"]["incomplete_or_infrastructure_error"] == 1
    assert (c.root / "interrupted.json").read_bytes() == original


def test_selection_failures_do_not_spend_model_call_budget(campaign, monkeypatch):
    import entroly

    def broken_selector(*args, **kwargs):
        raise RuntimeError("selection failed")

    monkeypatch.setattr(entroly, "optimize", broken_selector)
    first = campaign.run("first.json", max_model_calls=1)
    assert first["continuation"]["new_arm_attempts"] == 1
    assert [row["outcome"] for row in first["rows"]] == ["selection_error", "passed"]
    final = campaign.run("final.json", max_model_calls=5, continue_from=campaign.root / "first.json")
    assert final["status"] == "complete"
    assert final["continuation"]["total_arm_attempts"] == 6
    assert len(campaign.calls) == len(set(campaign.calls)) == 6
    assert final["summary"]["comparisons_to_selected"]["raw"]["baseline_only_passed"] == 2


def test_crash_after_first_child_checkpoint_keeps_unknown_attempt_resumable(campaign, monkeypatch):
    c = campaign

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(pilot, "_run_arm", crash)
    with pytest.raises(KeyboardInterrupt):
        c.run("first.json", max_model_calls=1)
    first = (c.root / "first.json").read_bytes()
    fsync = pilot.os.fsync

    def stop_after_flush(fd):
        fsync(fd)
        raise KeyboardInterrupt

    monkeypatch.setattr(pilot.os, "fsync", stop_after_flush)
    with pytest.raises(KeyboardInterrupt):
        c.run("child.json", max_model_calls=1, continue_from=c.root / "first.json")
    child = json.loads((c.root / "child.json").read_bytes())
    assert child["rows"][0]["outcome"] == "interrupted_unknown"
    assert child["continuation"]["pending_arm"] is None
    assert child["continuation"]["total_arm_attempts"] == 1
    monkeypatch.setattr(pilot.os, "fsync", fsync)
    monkeypatch.setattr(pilot, "_run_arm", c.attempt)
    final = c.run("final.json", max_model_calls=7, continue_from=c.root / "child.json")
    assert final["status"] == "complete"
    assert final["continuation"]["total_arm_attempts"] == 8
    assert len(c.calls) == 7
    assert (c.root / "first.json").read_bytes() == first


def test_selection_failure_cannot_relabel_a_generated_arm_to_reduce_attempt_count(campaign):
    c = campaign
    data = c.run("first.json", max_model_calls=2)
    assert data["rows"][1]["arm"] == "raw_repeat"
    data["rows"][1]["outcome"] = "selection_error"
    data["continuation"]["new_arm_attempts"] -= 1
    data["continuation"]["total_arm_attempts"] -= 1
    data["summary"] = pilot.summarize(data["rows"], c.protocol["candidate_shas"], pilot._ARMS_V2)
    data["artifact_sha256"] = pilot._artifact_digest(data)
    (c.root / "first.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="selection outcome"):
        c.run("invalid.json", continue_from=c.root / "first.json")
    assert len(c.calls) == 2


@pytest.mark.parametrize("field", ["oracle_image", "native", "ollama_version"])
def test_runtime_changes_reject_continuation_before_attempts(campaign, field):
    campaign.run("first.json", max_model_calls=1)
    campaign.runtime[field] = "different"
    with pytest.raises(ValueError, match="runtime identity"):
        campaign.run("invalid.json", max_model_calls=1, continue_from=campaign.root / "first.json")
    assert len(campaign.calls) == 1
    assert not (campaign.root / "invalid.json").exists()


def test_changed_protocol_and_tasks_are_rejected(campaign):
    campaign.run("first.json", max_model_calls=1)
    campaign.protocol["max_output_tokens"] = 64
    campaign.protocol_file.write_text(json.dumps(campaign.protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen protocol"):
        campaign.run("invalid.json", continue_from=campaign.root / "first.json")
    campaign.tasks_file.write_text("changed\n", encoding="utf-8")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        campaign.run("invalid.json", continue_from=campaign.root / "first.json")
    assert len(campaign.calls) == 1


def test_changed_context_plan_rejected_without_new_attempt(campaign, monkeypatch):
    campaign.run("first.json", max_model_calls=1)
    original = (campaign.root / "first.json").read_bytes()
    monkeypatch.setattr(pilot, "_fragments", lambda _repo, task: [
        Fragment(task["source_files"][0], "different context"),
    ])
    with pytest.raises(ValueError, match="context selection"):
        campaign.run("invalid.json", continue_from=campaign.root / "first.json")
    assert len(campaign.calls) == 1
    assert (campaign.root / "first.json").read_bytes() == original


@pytest.mark.parametrize("mutation,reseal,match", [
    (lambda data: data["rows"][0].update(outcome="test_failure"), False, "integrity checksum"),
    (lambda data: data["rows"].reverse(), True, "arm order"),
    (lambda data: data["continuation"].update(total_arm_attempts=99), True, "attempt accounting"),
    (lambda data: data["rows"][0].pop("patch_sha256"), True, "receipt"),
])
def test_invalid_checkpoint_rejected_without_new_calls(campaign, mutation, reseal, match):
    campaign.run("first.json", max_model_calls=2)
    path = campaign.root / "first.json"
    data = json.loads(path.read_bytes())
    mutation(data)
    if reseal:
        data["artifact_sha256"] = pilot._artifact_digest(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        campaign.run("invalid.json", continue_from=path)
    assert len(campaign.calls) == 2


def test_parent_cannot_be_overwritten(campaign):
    campaign.run("first.json", max_model_calls=1)
    original = (campaign.root / "first.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        campaign.run("first.json", continue_from=campaign.root / "first.json")
    assert (campaign.root / "first.json").read_bytes() == original
    assert len(campaign.calls) == 1


def test_output_created_during_preflight_is_not_overwritten(campaign, monkeypatch):
    output = campaign.root / "race.json"

    def runtime(*args):
        output.write_bytes(b"another invocation owns this output")
        return copy.deepcopy(campaign.runtime)

    monkeypatch.setattr(identity, "runtime_identity", runtime)
    with pytest.raises(FileExistsError):
        campaign.run("race.json", max_model_calls=1)
    assert output.read_bytes() == b"another invocation owns this output"
    assert not campaign.calls


def test_post_generation_cleanup_timeout_retains_known_usage(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "_task_worktree", lambda *args: nullcontext(tmp_path))
    monkeypatch.setattr(pilot, "call_model", lambda **kwargs: {
        "text": "patch", "input_tokens": 12, "output_tokens": 8, "latency_s": 0.5,
    })
    monkeypatch.setattr(pilot, "_apply_patch", lambda *args: ("a" * 64, "patch"))

    def oracle(*args):
        raise subprocess.TimeoutExpired(["docker", "rm"], 10)

    monkeypatch.setattr(pilot, "_oracle", oracle)
    row = pilot._run_arm(
        tmp_path, {"sha": "a" * 40, "source_files": ["solver.py"], "test_files": ["test.py"]},
        arm="selected", context="context", context_details={}, model="local-model",
        base_url="http://localhost:11434", seed=17, model_timeout=5, test_timeout=5,
        output_tokens=32, context_window=1024, preserve_execution_receipt=True,
    )
    assert row["outcome"] == "execution_error"
    assert (row["input_tokens"], row["output_tokens"], row["patch_sha256"]) == (12, 8, "a" * 64)


def test_pinned_oracle_timeout_cleans_up_only_owned_container(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_REAL_COMMIT_SANDBOX", "1")
    commands = []

    def execute(command, **options):
        commands.append(command)
        if command[1] == "run":
            assert "sha256:" + "1" * 64 in command
            assert "entroly-dogfood-real-oracle:py312" not in command
            raise subprocess.TimeoutExpired(command, options["timeout"])
        assert options["timeout"] == 10
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pilot.subprocess, "run", execute)
    assert pilot._oracle(tmp_path, {"test_files": ["test_solver.py"]}, 5,
                         "sha256:" + "1" * 64) == ("timeout", "")
    name = commands[0][commands[0].index("--name") + 1]
    assert name.startswith("entroly-pilot-")
    assert commands[1] == ["docker", "rm", "--force", name]
