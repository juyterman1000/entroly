from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly.process_safety import BoundedProcessResult
from entroly.ravs import ExecutorRegistry, TestRunnerExecutor
from entroly.ravs import executors as legacy_executors
from entroly.ravs.controller import SequentialController
from entroly.ravs.safe_executors import _sanitized_test_environment


def _completed(command, **_kwargs) -> BoundedProcessResult:
    return BoundedProcessResult(
        args=tuple(command),
        returncode=0,
        stdout="1 passed in 0.01s\n",
        stderr="",
    )


def test_public_and_legacy_imports_resolve_to_hardened_runner() -> None:
    assert legacy_executors.TestRunnerExecutor is TestRunnerExecutor
    assert legacy_executors.ExecutorRegistry is ExecutorRegistry
    assert isinstance(SequentialController()._executors, ExecutorRegistry)


def test_workspace_escape_is_rejected_before_process_launch(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside_test.py"
    outside.write_text("def test_bad(): assert True\n", encoding="utf-8")
    launched = False

    def fail_if_launched(*_args, **_kwargs):
        nonlocal launched
        launched = True
        raise AssertionError("escaped target reached process launch")

    monkeypatch.setattr(
        "entroly.ravs.safe_executors.run_bounded_process", fail_if_launched
    )
    result = TestRunnerExecutor(cwd=str(workspace)).execute(
        "run pytest ../outside_test.py"
    )

    assert not launched
    assert not result.succeeded
    assert "inside the selected workspace" in result.error


def test_option_like_target_is_rejected_before_process_launch(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "--capture_test.py").write_text(
        "def test_ok(): assert True\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "entroly.ravs.safe_executors.run_bounded_process",
        lambda *_args, **_kwargs: pytest.fail("unsafe option reached process launch"),
    )

    result = TestRunnerExecutor(cwd=str(tmp_path)).execute(
        "run pytest --capture_test.py"
    )

    assert not result.succeeded
    assert "safe path" in result.error


def test_valid_target_is_confined_and_separated_from_options(
    monkeypatch, tmp_path: Path
) -> None:
    target = tmp_path / "tests" / "test_ok.py"
    target.parent.mkdir()
    target.write_text("def test_ok(): assert True\n", encoding="utf-8")
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = list(command)
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        return _completed(command)

    monkeypatch.setattr(
        "entroly.ravs.safe_executors.run_bounded_process", fake_run
    )
    result = TestRunnerExecutor(cwd=str(tmp_path)).execute(
        "run pytest tests/test_ok.py"
    )

    assert result.succeeded
    assert captured["command"][-2:] == ["--", "tests/test_ok.py"]
    assert Path(captured["cwd"]) == tmp_path.resolve()
    payload = json.loads(result.result)
    assert payload["passed"] == 1


def test_credentials_are_removed_from_child_environment(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "do-not-leak-either")
    monkeypatch.setenv("SAFE_FEATURE_FLAG", "enabled")
    monkeypatch.delenv("ENTROLY_TEST_RUNNER_PASSTHROUGH_SECRETS", raising=False)

    child = _sanitized_test_environment()

    assert "OPENAI_API_KEY" not in child
    assert "AWS_SECRET_ACCESS_KEY" not in child
    assert child["SAFE_FEATURE_FLAG"] == "enabled"
    assert child["ENTROLY_TEST_RUNNER_ACTIVE"] == "1"


def test_secret_passthrough_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "trusted-workspace-key")
    monkeypatch.setenv("ENTROLY_TEST_RUNNER_PASSTHROUGH_SECRETS", "1")

    child = _sanitized_test_environment()

    assert child["OPENAI_API_KEY"] == "trusted-workspace-key"


def test_timeout_reports_full_tree_termination(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "test_slow.py"
    target.write_text("def test_slow(): assert True\n", encoding="utf-8")

    def fake_timeout(command, **_kwargs):
        return BoundedProcessResult(
            args=tuple(command),
            returncode=-9,
            stdout="",
            stderr="",
            timed_out=True,
        )

    monkeypatch.setattr(
        "entroly.ravs.safe_executors.run_bounded_process", fake_timeout
    )
    result = TestRunnerExecutor(timeout_s=0.1, cwd=str(tmp_path)).execute(
        "pytest test_slow.py"
    )

    assert not result.succeeded
    assert "process tree terminated" in result.error


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), "bad"])
def test_invalid_test_timeout_is_rejected(timeout) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        TestRunnerExecutor(timeout_s=timeout)
