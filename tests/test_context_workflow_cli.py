from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["ENTROLY_DIR"] = str(tmp_path / "state")
    env["ENTROLY_DISABLE_UPDATE_CHECK"] = "1"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "entroly.cli", *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


def test_learn_history_json_is_content_blind_and_machine_readable(tmp_path: Path) -> None:
    history = tmp_path / "history"
    history.mkdir()
    (history / "one.jsonl").write_text(
        json.dumps({"role": "user", "content": "private prompt", "usage": {"input_tokens": 7}}),
        encoding="utf-8",
    )
    result = _run(tmp_path, "learn", "--history", "--history-root", str(history), "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["provider_reported"]["unknown_semantics_observed_sum"]["input_tokens"] == 7
    assert "private prompt" not in result.stdout


def test_shrink_preserves_exit_code_emits_recovery_and_writes_receipt(tmp_path: Path) -> None:
    script = (
        "import sys; "
        "[print(f'PASS test_item_{i % 5}') for i in range(500)]; "
        "print('ERROR failure-marker', file=sys.stderr); "
        "raise SystemExit(5)"
    )
    receipt = tmp_path / "command-receipt.json"
    result = _run(
        tmp_path, "shrink", "--budget", "80", "--receipt", str(receipt), "--",
        sys.executable, "-c", script,
    )
    assert result.returncode == 5
    assert "ERROR failure-marker" in result.stderr
    assert "Entroly command envelope" in result.stderr
    assert "exact recovery: entroly recover sha256:" in result.stderr
    assert json.loads(receipt.read_text(encoding="utf-8"))["exit_code"] == 5


def test_shrink_passes_non_utf8_bytes_through_without_false_recovery(tmp_path: Path) -> None:
    receipt = tmp_path / "binary-receipt.json"
    result = _run(
        tmp_path, "shrink", "--receipt", str(receipt), "--",
        sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([255, 0, 1]))",
    )
    assert result.returncode == 0
    row = json.loads(receipt.read_text(encoding="utf-8"))["streams"]["stdout"]
    assert row["mode"] == "passthrough-non-utf8"
    assert row["recovery_digest"] is None


def test_browser_snapshot_json_needs_no_browser_dependency(tmp_path: Path) -> None:
    snapshot = tmp_path / "page.aria.yml"
    snapshot.write_text(
        '- main:\n  - heading "Token receipt" [level=1]\n'
        + "\n".join(f"  - paragraph: noise {i}" for i in range(200)),
        encoding="utf-8",
    )
    result = _run(
        tmp_path, "browser", "--snapshot", str(snapshot), "--query", "token receipt",
        "--budget", "50", "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["receipt"]["mode"] == "compressed"
    assert payload["receipt"]["exact_recovery"] is True


def test_response_contract_round_trip_is_explicit_and_machine_readable(tmp_path: Path) -> None:
    result = _run(tmp_path, "response", "set", "concise", "--json")
    assert result.returncode == 0, result.stderr
    change = json.loads(result.stdout)
    assert change["name"] == "concise"
    assert change["reversible"] is True
    shown = _run(tmp_path, "response", "show", "--json")
    assert json.loads(shown.stdout)["name"] == "concise"


def test_trial_requires_explicit_experiment_arm_and_evaluation_is_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from entroly import cli
    from entroly import cli_context_workflows as workflows

    before = {
        "bypass_mode": False,
        "requests_total": 10,
        "tokens": {"original_total": 100, "optimized_total": 60},
        "usage_accounting": {
            "live": {"requests": 10, "uncached_input_tokens": 1000, "cache_read_tokens": 5,
                     "cache_write_tokens": 0, "output_tokens": 100},
            "ledger": {"cost_micro_usd": 1000},
        },
    }
    after = {
        "bypass_mode": False,
        "requests_total": 11,
        "tokens": {"original_total": 300, "optimized_total": 140},
        "usage_accounting": {
            "live": {"requests": 11, "uncached_input_tokens": 1120, "cache_read_tokens": 15,
                     "cache_write_tokens": 3, "output_tokens": 130},
            "ledger": {"cost_micro_usd": 2500},
        },
    }
    reports = iter((before, after))
    monkeypatch.setattr(workflows, "_stats", lambda _port: next(reports))
    bypass_values: list[bool] = []
    monkeypatch.setattr(workflows, "_set_bypass", lambda _port, enabled: bypass_values.append(enabled))
    monkeypatch.setattr(workflows, "_STATE_DIR_OVERRIDE", tmp_path)
    monkeypatch.setattr(workflows, "_START_PROXY", lambda _port: True)
    monkeypatch.setattr(
        workflows,
        "_RESOLVED_WRAP_ENV",
        lambda _spec, port: {"OPENAI_BASE_URL": f"http://localhost:{port}/v1"},
    )
    monkeypatch.setitem(workflows._WRAP_AGENTS, "fakeagent", {
        "kind": "cli", "env_key": "OPENAI_BASE_URL", "env_val": "http://localhost:{port}/v1"
    })
    monkeypatch.setattr(workflows.shutil, "which", lambda _name: sys.executable)
    monkeypatch.setattr(workflows.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0))
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({
        "task_success": True, "evidence_retained": True, "evaluator": "fixture-check"
    }), encoding="utf-8")

    rc = workflows.cmd_trial(SimpleNamespace(
        report=None, experiment="exp-1", arm="optimized", evaluation=str(evaluation),
        agent_command=["--", "fakeagent", "task"], port=9377, receipt=None, json_output=True,
    ))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality"]["task_success"] is True
    assert payload["quality"]["process_success"] is True
    assert payload["usage"]["provider_reported_active_input_tokens"] == 133
    assert bypass_values == [False, False]


def test_trial_report_refuses_single_unmatched_run_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from entroly import cli
    from entroly import cli_context_workflows as workflows

    monkeypatch.setattr(workflows, "_STATE_DIR_OVERRIDE", tmp_path)
    directory = tmp_path / "experiments" / "exp-2"
    directory.mkdir(parents=True)
    receipt = {
        "schema_version": "entroly.trial-run.v2",
        "arm": "optimized",
        "command_sha256": "sha256:same",
        "traffic": {"evidence_gate": "passed"},
        "usage": {"provider_reported_active_input_tokens": 10},
        "quality": {"task_success": True, "evidence_retained": True},
        "economics": {"cost_usd": 0.01},
    }
    (directory / "one.json").write_text(json.dumps(receipt), encoding="utf-8")
    report = workflows._trial_report("exp-2")
    assert report["comparison"]["status"] == "insufficient-evidence"
    assert report["comparison"]["provider_input_token_difference"] is None


def test_trial_report_ignores_corrupt_receipts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from entroly import cli
    from entroly import cli_context_workflows as workflows

    monkeypatch.setattr(workflows, "_STATE_DIR_OVERRIDE", tmp_path)
    directory = tmp_path / "experiments" / "exp-corrupt"
    directory.mkdir(parents=True)
    (directory / "bad.json").write_text(
        json.dumps({"schema_version": "entroly.trial-run.v2", "arm": "baseline"}),
        encoding="utf-8",
    )
    report = workflows._trial_report("exp-corrupt")
    assert report["receipts"] == {"accepted": 0, "ignored_invalid": 1}
    assert report["comparison"]["status"] == "insufficient-evidence"


@pytest.mark.parametrize(
    ("task_success", "cost_usd"),
    [
        (1, 0.01),
        (True, "NaN"),
        (True, "Infinity"),
    ],
)
def test_trial_report_quarantines_non_boolean_quality_and_nonfinite_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_success: object,
    cost_usd: object,
) -> None:
    from entroly import cli
    from entroly import cli_context_workflows as workflows

    monkeypatch.setattr(workflows, "_STATE_DIR_OVERRIDE", tmp_path)
    directory = tmp_path / "experiments" / "exp-invalid-evidence"
    directory.mkdir(parents=True)
    receipt = {
        "schema_version": "entroly.trial-run.v2",
        "arm": "baseline",
        "command_sha256": "sha256:" + "0" * 64,
        "traffic": {"evidence_gate": "passed"},
        "usage": {"provider_reported_active_input_tokens": 10},
        "quality": {"task_success": task_success, "evidence_retained": True},
        "economics": {"cost_usd": cost_usd},
    }
    (directory / "invalid.json").write_text(json.dumps(receipt), encoding="utf-8")

    report = workflows._trial_report("exp-invalid-evidence")

    assert report["receipts"] == {"accepted": 0, "ignored_invalid": 1}
    assert report["comparison"]["matched_command"] is False


def test_trial_report_requires_a_valid_command_digest_for_comparability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from entroly import cli
    from entroly import cli_context_workflows as workflows

    monkeypatch.setattr(workflows, "_STATE_DIR_OVERRIDE", tmp_path)
    directory = tmp_path / "experiments" / "exp-missing-digest"
    directory.mkdir(parents=True)
    for arm in ("baseline", "optimized"):
        receipt = {
            "schema_version": "entroly.trial-run.v2",
            "arm": arm,
            "traffic": {"evidence_gate": "passed"},
            "usage": {"provider_reported_active_input_tokens": 10},
            "quality": {"task_success": True, "evidence_retained": True},
            "economics": {"cost_usd": 0.01},
        }
        (directory / f"{arm}.json").write_text(json.dumps(receipt), encoding="utf-8")

    report = workflows._trial_report("exp-missing-digest")

    assert report["comparison"]["matched_command"] is False
    assert report["comparison"]["provider_input_token_difference"] is None
