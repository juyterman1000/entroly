"""Frozen CLI checkpoints without model, Docker, or repository mining calls."""

import hashlib
import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from benchmarks import run_frozen_real_commit_pilot as cli


SOURCE_SHA = "a" * 40
_prepare = cli.prepare
_preflight = cli._preflight
OUTPUT_NAMES = (
    "effective-frozen-protocol.json", "frozen-validated-tasks.jsonl",
    "real-commit-model-results.json", "real-commit-summary.json",
)


def _forbidden(*args, **kwargs):
    pytest.fail("unexpected mining, preparation, model, or external command")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}


@pytest.fixture(autouse=True)
def no_external_calls(monkeypatch):
    monkeypatch.setattr(cli, "_preflight", lambda *args: None)
    for name in ("discover", "validate", "run"):
        monkeypatch.setattr(cli, name, _forbidden)
    monkeypatch.setattr(cli.subprocess, "run", _forbidden)
    monkeypatch.setattr(cli, "git", lambda *args: SOURCE_SHA)
    monkeypatch.setenv("ENTROLY_REAL_COMMIT_SANDBOX", "1")


@pytest.fixture
def campaign(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = {
        "schema": "entroly.real-commit-pilot-plan.v1",
        "source_baseline": "b" * 40,
        "candidate_shas": ["c" * 40],
        "model": "local-model", "model_digest": "local-digest", "seed": 17,
        "max_corpus_bytes": 5000, "selection_budget": 100,
        "max_output_tokens": 200, "context_window": 1000,
        "model_timeout_s": 20, "test_timeout_s": 10,
        "extra_manifest_field": {"description": "frozen scope"},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return SimpleNamespace(
        repo=repo, manifest=manifest, manifest_path=manifest_path,
        output=tmp_path / "new-output", parent_dir=tmp_path / "parent",
    )


def _fresh_prepare(monkeypatch, campaign, output, version):
    @dataclass
    class Task:
        sha: str

    with monkeypatch.context() as patch:
        patch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
        patch.setattr(cli, "discover", lambda *args, **kwargs: [Task("c" * 40)])
        patch.setattr(cli, "validate", lambda repo, task, **kwargs: task)
        return _prepare(campaign.manifest, campaign.repo, output, evaluation_version=version)


@pytest.fixture
def parent(monkeypatch, campaign):
    protocol, tasks = _fresh_prepare(monkeypatch, campaign, campaign.parent_dir, 2)
    # Preserve actual byte identity, including noncanonical JSON whitespace/CRLF.
    tasks.write_bytes(tasks.read_bytes().replace(b"\n", b"\r\n"))
    frozen = json.loads(protocol.read_bytes())
    frozen["tasks_sha256"] = _digest(tasks.read_bytes())
    protocol.write_bytes((json.dumps(frozen, indent=3) + "\r\n").encode("utf-8"))
    artifact = {
        "schema": "entroly.real-commit-pilot-result.v2",
        "source_sha": SOURCE_SHA, "status": "call_limit_reached",
        "protocol_sha256": _digest(protocol.read_bytes()),
        "tasks_sha256": _digest(tasks.read_bytes()),
        "rows": [], "summary": {},
    }
    path = campaign.parent_dir / OUTPUT_NAMES[2]
    path.write_text(json.dumps(artifact), encoding="utf-8")
    (campaign.parent_dir / OUTPUT_NAMES[3]).write_bytes(b"old summary\r\n")
    monkeypatch.setattr(cli, "prepare", _forbidden)
    return path


def _invoke(monkeypatch, campaign, *options):
    monkeypatch.setattr(sys, "argv", [
        "frozen-pilot", "--manifest", str(campaign.manifest_path),
        "--repo", str(campaign.repo), "--outdir", str(campaign.output), *map(str, options),
    ])
    return cli.main()


def _result(status="complete", rows=None, continuation=None):
    result = {
        "source_sha": SOURCE_SHA, "status": status,
        "rows": rows if rows is not None else [
            {"arm": "raw", "outcome": "passed", "input_tokens": 7, "output_tokens": 2},
        ],
        "summary": {"quality_claim_allowed": False},
    }
    if continuation is not None:
        result["continuation"] = continuation
    return result


@pytest.mark.parametrize("version", [1, 2])
def test_prepare_keeps_tuple_api_and_only_v2_binds_entire_manifest(monkeypatch, campaign, version):
    protocol_path, tasks_path = _fresh_prepare(monkeypatch, campaign, campaign.output, version)
    protocol = json.loads(protocol_path.read_bytes())
    expected = {key: campaign.manifest[key] for key in (
        "candidate_shas", "model", "model_digest", "seed", "max_corpus_bytes",
        "selection_budget", "max_output_tokens", "context_window", "model_timeout_s", "test_timeout_s",
    )}
    expected.update(
        schema=f"entroly.real-commit-pilot.v{version}", source_sha=SOURCE_SHA,
        tasks_sha256=_digest(tasks_path.read_bytes()), baseline=campaign.manifest["source_baseline"],
    )
    if version == 2:
        expected["manifest_sha256"] = _digest(json.dumps(
            campaign.manifest, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8"))
    assert protocol == expected


def test_continuation_copies_exact_bytes_without_preparing_or_mutating_parent(monkeypatch, campaign, parent):
    before = _snapshot(campaign.parent_dir)
    # Manifest formatting and key order are immaterial to its canonical digest.
    manifest = dict(reversed(list(campaign.manifest.items())))
    protocol, tasks = cli.prepare_continuation(
        manifest, campaign.repo, campaign.output, parent, evaluation_version=2,
    )
    assert protocol.read_bytes() == before[protocol.name]
    assert tasks.read_bytes() == before[tasks.name]
    assert _snapshot(campaign.parent_dir) == before
    assert set(_snapshot(campaign.output)) == set(OUTPUT_NAMES[:2])


@pytest.mark.parametrize("change", [
    "manifest_extra", "manifest_candidates", "manifest_budget", "missing_manifest_digest",
    "protocol_schema", "parent_schema", "source_sha", "baseline", "protocol_bytes", "tasks_bytes",
    "parent_protocol_hash", "parent_tasks_hash", "protocol_tasks_hash",
])
def test_continuation_rejects_changed_frozen_inputs_before_writes(campaign, parent, change):
    protocol_path = campaign.parent_dir / OUTPUT_NAMES[0]
    tasks_path = campaign.parent_dir / OUTPUT_NAMES[1]
    manifest = json.loads(json.dumps(campaign.manifest))
    protocol = json.loads(protocol_path.read_bytes())
    artifact = json.loads(parent.read_bytes())
    if change == "manifest_extra":
        manifest["extra_manifest_field"]["description"] = "changed scope"
    elif change == "manifest_candidates":
        manifest["candidate_shas"].append("d" * 40)
    elif change == "manifest_budget":
        manifest["selection_budget"] += 1
    elif change == "missing_manifest_digest":
        del protocol["manifest_sha256"]
    elif change == "protocol_schema":
        protocol["schema"] = "entroly.real-commit-pilot.v1"
    elif change == "parent_schema":
        artifact["schema"] = "entroly.real-commit-pilot-result.v1"
    elif change == "source_sha":
        protocol["source_sha"] = "d" * 40
    elif change == "baseline":
        protocol["baseline"] = "d" * 40
    elif change == "protocol_bytes":
        protocol_path.write_bytes(protocol_path.read_bytes() + b" ")
    elif change == "tasks_bytes":
        tasks_path.write_bytes(tasks_path.read_bytes() + b" ")
    elif change == "parent_protocol_hash":
        artifact["protocol_sha256"] = "0" * 64
    elif change == "parent_tasks_hash":
        artifact["tasks_sha256"] = "0" * 64
    elif change == "protocol_tasks_hash":
        protocol["tasks_sha256"] = "0" * 64
    if protocol != json.loads(protocol_path.read_bytes()):
        protocol_path.write_bytes(json.dumps(protocol).encode("utf-8"))
        # Keep the parent protocol binding valid to exercise individual checks.
        artifact["protocol_sha256"] = _digest(protocol_path.read_bytes())
    parent.write_bytes(json.dumps(artifact).encode("utf-8"))
    before = _snapshot(campaign.parent_dir)
    with pytest.raises(ValueError):
        cli.prepare_continuation(manifest, campaign.repo, campaign.output, parent, evaluation_version=2)
    assert not campaign.output.exists()
    assert _snapshot(campaign.parent_dir) == before


@pytest.mark.parametrize("name", OUTPUT_NAMES)
@pytest.mark.parametrize("mode", ["fresh_helper", "continue_helper", "fresh_cli", "continue_cli"])
def test_existing_outputs_rejected_before_preparation_or_writes(monkeypatch, campaign, parent, name, mode):
    campaign.output.mkdir()
    (campaign.output / name).write_bytes(b"preserve existing bytes\r\n")
    before = _snapshot(campaign.parent_dir)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        if mode == "fresh_helper":
            _prepare(campaign.manifest, campaign.repo, campaign.output, evaluation_version=2)
        elif mode == "continue_helper":
            cli.prepare_continuation(campaign.manifest, campaign.repo, campaign.output, parent, evaluation_version=2)
        else:
            options = ["--evaluation-version", "2"]
            if mode == "continue_cli":
                options.extend(["--continue-from", parent])
            _invoke(monkeypatch, campaign, *options)
    assert _snapshot(campaign.output) == {name: b"preserve existing bytes\r\n"}
    assert _snapshot(campaign.parent_dir) == before


def test_continuation_cannot_reuse_parent_directory(monkeypatch, campaign, parent):
    before = _snapshot(campaign.parent_dir)
    campaign.output = campaign.parent_dir
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _invoke(monkeypatch, campaign, "--evaluation-version", "2", "--continue-from", parent)
    assert _snapshot(campaign.parent_dir) == before


@pytest.mark.parametrize("options", [
    ["--max-model-calls", "1"], ["--continue-from", "parent.json"],
    ["--evaluation-version", "1", "--max-model-calls", "1"],
    ["--evaluation-version", "1", "--continue-from", "parent.json"],
])
def test_bounded_options_require_explicit_v2(monkeypatch, campaign, options):
    with pytest.raises(SystemExit) as error:
        _invoke(monkeypatch, campaign, *options)
    assert error.value.code == 2
    assert not campaign.output.exists()


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc"])
def test_call_limit_must_be_positive_integer(monkeypatch, campaign, value):
    with pytest.raises(SystemExit) as error:
        _invoke(monkeypatch, campaign, "--evaluation-version", "2", "--max-model-calls", value)
    assert error.value.code == 2
    assert not campaign.output.exists()


def test_continuation_helper_requires_explicit_v2(campaign, parent):
    with pytest.raises(ValueError, match="explicit evaluation version 2"):
        cli.prepare_continuation(campaign.manifest, campaign.repo, campaign.output, parent)
    assert not campaign.output.exists()


@pytest.mark.parametrize("bounded", [False, True])
def test_cli_continuation_kwargs_and_cumulative_stats(monkeypatch, campaign, parent, bounded):
    before = _snapshot(campaign.parent_dir)
    calls = []
    rows = [
        {"arm": "raw", "outcome": "passed", "input_tokens": 7, "output_tokens": 2},
        {"arm": "selected", "outcome": "execution_error"},
        {"arm": "null", "outcome": "test_failure", "input_tokens": 3, "output_tokens": 1},
    ]

    def run(protocol, tasks, repo, output, **kwargs):
        calls.append((protocol, tasks, repo, output, kwargs))
        assert protocol.read_bytes() == before[protocol.name]
        assert tasks.read_bytes() == before[tasks.name]
        return _result(rows=rows, continuation={"new_arm_attempts": 1, "total_arm_attempts": 3})

    monkeypatch.setattr(cli, "run", run)
    options = ["--evaluation-version", "2", "--continue-from", parent]
    if bounded:
        options.extend(["--max-model-calls", "1"])
    assert _invoke(monkeypatch, campaign, *options) == 0
    assert len(calls) == 1
    protocol, tasks, repo, output, kwargs = calls[0]
    assert (protocol, tasks, repo, output) == (
        campaign.output / OUTPUT_NAMES[0], campaign.output / OUTPUT_NAMES[1],
        campaign.repo, campaign.output / OUTPUT_NAMES[2],
    )
    expected = {"base_url": "http://127.0.0.1:11434", "continue_from": parent}
    if bounded:
        expected["max_model_calls"] = 1
    assert kwargs == expected
    stats = json.loads((campaign.output / OUTPUT_NAMES[3]).read_bytes())
    assert stats["status"] == "complete"
    assert stats["modeled_attempt_count"] == 2
    assert stats["model_reported_input_tokens"] == 10
    assert stats["model_reported_output_tokens"] == 3
    assert stats["new_arm_attempts"] == 1
    assert stats["total_arm_attempts"] == 3
    assert stats["outcome_counts"] == {"passed": 1, "execution_error": 1, "test_failure": 1}
    assert stats["quality_claim_allowed"] is False
    assert _snapshot(campaign.parent_dir) == before


@pytest.mark.parametrize("version,bounded", [(1, False), (2, False), (2, True)])
def test_fresh_cli_only_passes_requested_new_kwargs(monkeypatch, campaign, version, bounded):
    def prepare(manifest, repo, output, *, evaluation_version):
        assert evaluation_version == version
        return _fresh_prepare(monkeypatch, campaign, output, version)

    def run(protocol, tasks, repo, output, **kwargs):
        expected = {"base_url": "http://127.0.0.1:11434"}
        if bounded:
            expected["max_model_calls"] = 2
        assert kwargs == expected
        return _result()

    monkeypatch.setattr(cli, "prepare", prepare)
    monkeypatch.setattr(cli, "run", run)
    options = ["--evaluation-version", str(version)]
    if bounded:
        options.extend(["--max-model-calls", "2"])
    assert _invoke(monkeypatch, campaign, *options) == 0
    stats = json.loads((campaign.output / OUTPUT_NAMES[3]).read_bytes())
    assert stats["status"] == "complete"
    assert "new_arm_attempts" not in stats


@pytest.mark.parametrize("rows", [[], [{"arm": "raw", "outcome": "execution_error"}]])
def test_call_limit_without_observed_tokens_is_successful_checkpoint(monkeypatch, campaign, parent, capsys, rows):
    result = _result("call_limit_reached", rows, {"new_arm_attempts": 1, "total_arm_attempts": 1})
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: result)
    assert _invoke(monkeypatch, campaign, "--evaluation-version", "2",
                   "--continue-from", parent, "--max-model-calls", "1") == 0
    stats = json.loads((campaign.output / OUTPUT_NAMES[3]).read_bytes())
    assert json.loads(capsys.readouterr().out) == stats
    assert stats["status"] == "call_limit_reached"
    assert stats["modeled_attempt_count"] == 0
    assert stats["model_reported_input_tokens"] == 0
    assert stats["new_arm_attempts"] == stats["total_arm_attempts"] == 1
    assert stats["quality_claim_allowed"] is False


@pytest.mark.parametrize("status", ["complete", "in_progress"])
def test_no_modeled_completion_keeps_failure_behavior(monkeypatch, campaign, parent, status):
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: _result(status, []))
    with pytest.raises(RuntimeError, match="no real model result"):
        _invoke(monkeypatch, campaign, "--evaluation-version", "2", "--continue-from", parent)
    stats = json.loads((campaign.output / OUTPUT_NAMES[3]).read_bytes())
    assert stats["status"] == status
    assert stats["quality_claim_allowed"] is False


@pytest.mark.parametrize("problem", ["native", "model", "oracle", None])
def test_bounded_preflight_checks_prerequisites_before_mining(monkeypatch, campaign, problem):
    events = []

    def native():
        events.append("native")
        return SimpleNamespace(ok=problem != "native")

    def model(*args):
        events.append("model")
        return "wrong-digest" if problem == "model" else campaign.manifest["model_digest"]

    def runtime(*args):
        events.append("runtime")
        if problem == "oracle":
            raise RuntimeError("oracle image unavailable")
        return {}

    monkeypatch.setattr(cli, "native_status", native)
    monkeypatch.setattr(cli, "_local_model_digest", model)
    monkeypatch.setattr(cli, "runtime_identity", runtime)
    if problem is None:
        _preflight(campaign.manifest, campaign.repo)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            _preflight(campaign.manifest, campaign.repo)
    assert events == {
        "native": ["native"], "model": ["native", "model"],
    }.get(problem, ["native", "model", "runtime"])


def test_failed_preflight_does_not_mine_or_create_outputs(monkeypatch, campaign):
    def unavailable(*args):
        raise RuntimeError("oracle image unavailable")

    monkeypatch.setattr(cli, "_preflight", unavailable)
    with pytest.raises(RuntimeError, match="oracle image unavailable"):
        _invoke(monkeypatch, campaign, "--evaluation-version", "2", "--max-model-calls", "1")
    assert not campaign.output.exists()
