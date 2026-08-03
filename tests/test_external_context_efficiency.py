from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from benchmarks.external_context_efficiency import (
    BenchmarkContractError,
    build_baseline_manifest,
    run_baseline,
    sanitize_baseline_environment,
    sha256_bytes,
    validate_jsonl,
    validate_record,
    validate_records,
)


def _digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


def _record(arm: str = "A_full") -> dict[str, object]:
    uses_entroly = arm.startswith(("C_", "D_", "E_"))
    return {
        "protocol_version": "external-context-efficiency-v1",
        "benchmark": "swe-bench-verified",
        "benchmark_version": "v1@abcdef0",
        "benchmark_task_id": "django__django-12345",
        "arm": arm,
        "run_id": f"run-{arm}",
        "pair_id": "swe:django__django-12345:seed-1",
        "model_id": "gpt-test-2026-08-01",
        "agent_id": "agent@abcdef0",
        "provider_id": "provider-test",
        "harness_commit": "abcdef0",
        "environment_digest": _digest("environment"),
        "task_input_digest": _digest("task-input"),
        "treatment_manifest_digest": _digest(f"manifest-{arm}"),
        "entroly_commit": "1234567" if uses_entroly else None,
        "entroly_config_digest": _digest("config") if uses_entroly else None,
        "seed": 1,
        "success": True,
        "benchmark_score": 1.0,
        "provider_input_tokens": 120,
        "provider_output_tokens": 20,
        "cached_input_tokens": 0,
        "context_tokens_before": 100,
        "context_tokens_after": 70 if uses_entroly else 100,
        "recovery_tokens": 0,
        "recovery_calls": 0,
        "wall_time_ms": 1000.0,
        "compression_time_ms": 5.0 if uses_entroly else 0.0,
        "peak_rss_bytes": 100_000,
        "context_overflow": False,
        "sufficiency_verdict": (
            "no_detected_gap_uncalibrated" if uses_entroly else None
        ),
        "calibration_policy_id": None,
        "required_evidence_present": True,
        "false_sufficient": False if uses_entroly else None,
        "error_class": None,
        "excluded": False,
        "exclusion_reason": None,
        "artifact_digests": {"stdout": _digest(f"stdout-{arm}")},
    }


def test_valid_a_full_record_passes_strict_contract() -> None:
    validate_record(_record())


def test_a_full_cannot_hide_context_changes() -> None:
    record = _record()
    record["context_tokens_after"] = 99
    with pytest.raises(BenchmarkContractError, match="context_tokens_after"):
        validate_record(record)


def test_a_full_cannot_carry_entroly_identity() -> None:
    record = _record()
    record["entroly_commit"] = "1234567"
    with pytest.raises(BenchmarkContractError, match="entroly_commit"):
        validate_record(record)


def test_false_sufficient_requires_calibrated_sufficiency() -> None:
    record = _record("C_entroly_conservative")
    record["required_evidence_present"] = False
    record["false_sufficient"] = True
    with pytest.raises(BenchmarkContractError, match="sufficient_calibrated"):
        validate_record(record)


def test_compared_treatment_requires_full_context_pair() -> None:
    with pytest.raises(BenchmarkContractError, match="require A_full"):
        validate_records([_record("C_entroly_conservative")])


def test_pair_identity_mismatch_fails_before_scoring() -> None:
    baseline = _record()
    treatment = _record("C_entroly_conservative")
    treatment["model_id"] = "different-model"
    with pytest.raises(BenchmarkContractError, match="model_id differs"):
        validate_records([baseline, treatment])


def test_duplicate_arm_is_rejected() -> None:
    first = _record()
    second = dict(first)
    second["run_id"] = "duplicate-run"
    with pytest.raises(BenchmarkContractError, match="duplicate result"):
        validate_records([first, second])


def test_jsonl_validator_reports_stable_summary(tmp_path: Path) -> None:
    records = [_record(), _record("C_entroly_conservative")]
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    summary = validate_jsonl(
        path,
        required_arms=("A_full", "C_entroly_conservative"),
    )
    assert summary["records"] == 2
    assert summary["pairs"] == 1
    assert summary["arms"] == {
        "A_full": 1,
        "C_entroly_conservative": 1,
    }
    assert str(summary["artifact_digest"]).startswith("sha256:")


def test_baseline_environment_removes_treatment_and_proxy_hooks() -> None:
    clean, removed = sanitize_baseline_environment(
        {
            "ENTROLY_AIR_GAP": "1",
            "ENTROLY_PROXY": "http://localhost:8000",
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "OPENAI_API_KEY": "secret-value",
            "PATH": "/usr/bin",
        }
    )
    assert set(removed) == {
        "ENTROLY_AIR_GAP",
        "ENTROLY_PROXY",
        "OPENAI_BASE_URL",
    }
    assert clean == {"OPENAI_API_KEY": "secret-value", "PATH": "/usr/bin"}


def test_manifest_records_secret_names_never_values(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]\n", encoding="utf-8")
    manifest, child_environment = build_baseline_manifest(
        benchmark="swe-bench-verified",
        benchmark_version="v1",
        run_id="baseline-smoke",
        model_id="gpt-test",
        agent_id="agent-test",
        provider_id="provider-test",
        harness_commit="abcdef0",
        task_set=tasks,
        command=[sys.executable, "-V"],
        environment={
            "ENTROLY_PROXY": "http://localhost:8000",
            "OPENAI_API_KEY": "never-write-this-value",
            "PATH": os.environ.get("PATH", ""),
        },
        module_names=(),
    )
    rendered = json.dumps(manifest, sort_keys=True)
    assert "never-write-this-value" not in rendered
    assert manifest["credential_environment_names"] == ["OPENAI_API_KEY"]
    assert manifest["removed_environment_keys"] == ["ENTROLY_PROXY"]
    assert child_environment["OPENAI_API_KEY"] == "never-write-this-value"
    assert manifest["arm"] == "A_full"
    assert manifest["treatment"] == "none"


def test_baseline_rejects_entroly_command(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    with pytest.raises(BenchmarkContractError, match="must not reference Entroly"):
        build_baseline_manifest(
            benchmark="swe-bench-verified",
            benchmark_version="v1",
            run_id="bad-command",
            model_id="gpt-test",
            agent_id="agent-test",
            provider_id=None,
            harness_commit=None,
            task_set=tasks,
            command=[sys.executable, "-m", "entroly"],
            module_names=(),
        )


def test_baseline_rejects_loaded_entroly_module(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    with pytest.raises(BenchmarkContractError, match="already loaded"):
        build_baseline_manifest(
            benchmark="swe-bench-verified",
            benchmark_version="v1",
            run_id="bad-process",
            model_id="gpt-test",
            agent_id="agent-test",
            provider_id=None,
            harness_commit=None,
            task_set=tasks,
            command=[sys.executable, "-V"],
            module_names=("entroly", "entroly.sdk"),
        )


def test_manifest_only_mode_does_not_execute_command(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    marker = tmp_path / "must-not-exist"
    script = tmp_path / "write_marker.py"
    script.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    manifest = run_baseline(
        output_directory=output,
        execute=False,
        timeout_seconds=10,
        benchmark="swe-bench-verified",
        benchmark_version="v1",
        run_id="dry-run",
        model_id="gpt-test",
        agent_id="agent-test",
        provider_id=None,
        harness_commit=None,
        task_set=tasks,
        command=[sys.executable, str(script)],
        environment={"PATH": os.environ.get("PATH", "")},
        module_names=(),
    )
    assert manifest["status"] == "planned"
    assert not marker.exists()
    persisted = json.loads(
        (output / "a_full_manifest.json").read_text(encoding="utf-8")
    )
    assert persisted["manifest_digest"] == manifest["manifest_digest"]


def test_executed_baseline_child_has_no_entroly_or_proxy_hooks(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.json"
    tasks.write_text("[]", encoding="utf-8")
    script = tmp_path / "inspect_environment.py"
    script.write_text(
        "import os, json\n"
        "print(json.dumps({\n"
        "  'entroly': sorted(k for k in os.environ if k.startswith('ENTROLY_')),\n"
        "  'base_url': os.environ.get('OPENAI_BASE_URL'),\n"
        "  'has_key': bool(os.environ.get('OPENAI_API_KEY')),\n"
        "}, sort_keys=True))\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    manifest = run_baseline(
        output_directory=output,
        execute=True,
        timeout_seconds=10,
        benchmark="swe-bench-verified",
        benchmark_version="v1",
        run_id="execute-smoke",
        model_id="gpt-test",
        agent_id="agent-test",
        provider_id="provider-test",
        harness_commit="abcdef0",
        task_set=tasks,
        command=[sys.executable, str(script)],
        environment={
            "ENTROLY_PROXY": "http://localhost:8000",
            "OPENAI_BASE_URL": "http://localhost:8000/v1",
            "OPENAI_API_KEY": "child-secret",
            "PATH": os.environ.get("PATH", ""),
        },
        module_names=(),
    )
    assert manifest["status"] == "completed"
    assert manifest["exit_code"] == 0
    child = json.loads((output / "a_full.stdout").read_text(encoding="utf-8"))
    assert child == {"base_url": None, "entroly": [], "has_key": True}
    rendered_manifest = (output / "a_full_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "child-secret" not in rendered_manifest
