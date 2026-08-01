from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from bench.neutral_gauntlet import (
    RunnerSpec,
    assert_comparable,
    pareto_dominates,
    resolve_identity,
    run_one,
)


def _echo_spec(**kwargs) -> RunnerSpec:
    return RunnerSpec(
        name="echo",
        version_command=(sys.executable, "--version"),
        run_command=(
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[2]).write_text(pathlib.Path(sys.argv[1]).read_text())",
            "{input}",
            "{output}",
        ),
        **kwargs,
    )


def test_runner_records_identical_input_version_and_output_identity() -> None:
    spec = _echo_spec()
    rows = [{"sample_id": "a", "text": "hello"}]
    result = run_one(
        spec,
        rows,
        experiment_contract={"model": "fixed", "seed": 7},
    )
    assert result.returncode == 0
    assert '"sample_id":"a"' in result.stdout
    assert result.input_sha256
    assert result.output_sha256 == hashlib.sha256(
        result.stdout.encode()
    ).hexdigest()
    assert result.experiment_sha256
    assert result.identity.executable_sha256
    assert not result.identity.pinned
    assert not result.timed_out


def test_pinned_identity_is_verified_before_execution() -> None:
    probe = run_one(_echo_spec(), [{"sample_id": "a"}])
    spec = _echo_spec(
        expected_version_pattern=r"Python 3\.",
        expected_executable_sha256=probe.identity.executable_sha256,
    )
    result = run_one(
        spec,
        [{"sample_id": "a"}],
        require_pinned_identity=True,
    )
    assert result.claim_ready
    assert result.identity.verified
    assert result.identity.pinned


def test_wrong_executable_hash_fails_without_running() -> None:
    spec = _echo_spec(expected_executable_sha256="0" * 64)
    result = run_one(
        spec,
        [{"sample_id": "a"}],
        require_pinned_identity=True,
    )
    assert result.returncode is None
    assert not result.claim_ready
    assert "executable SHA-256 does not match" in result.stderr
    assert result.latency_ms == 0.0


def test_explicit_artifact_tree_is_fingerprinted(tmp_path: Path) -> None:
    artifact = tmp_path / "package"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        '{"version":"1.2.3"}', encoding="utf-8"
    )
    base = _echo_spec(artifact_paths=(str(artifact),))
    identity = resolve_identity(
        base,
        {"PATH": str(Path(sys.executable).parent)},
    )
    assert len(identity.artifact_sha256) == 64
    pinned = replace(
        base,
        expected_artifact_sha256=identity.artifact_sha256,
    )
    result = run_one(
        pinned,
        [{"sample_id": "a"}],
        require_pinned_identity=True,
    )
    assert result.claim_ready


def test_comparability_rejects_mismatched_experiment_or_unpinned_runner() -> None:
    rows = [{"sample_id": "a"}]
    first = run_one(
        _echo_spec(), rows, experiment_contract={"seed": 1}
    )
    second = run_one(
        _echo_spec(), rows, experiment_contract={"seed": 2}
    )
    with pytest.raises(ValueError, match="experiment SHA-256"):
        assert_comparable(
            [first, second], require_claim_ready=False
        )
    with pytest.raises(ValueError, match="not claim-ready"):
        assert_comparable([first], require_claim_ready=True)


def test_pareto_requires_no_worse_and_one_strict_dimension() -> None:
    left = {"accuracy": 0.9, "tokens": 100.0, "latency": 5.0}
    right = {"accuracy": 0.9, "tokens": 110.0, "latency": 5.0}
    assert pareto_dominates(
        left,
        right,
        maximize=["accuracy"],
        minimize=["tokens", "latency"],
    )
    assert not pareto_dominates(
        right,
        left,
        maximize=["accuracy"],
        minimize=["tokens", "latency"],
    )
