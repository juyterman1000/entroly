"""Hardened RAVS executors for trust-sensitive local-code execution.

The legacy executor implementations remain available for compatibility, but
production registries use the classes in this module. Test execution is an
explicit local-code execution feature, not a sandbox; this layer still prevents
workspace escape, option injection, inherited secret exposure, unbounded output,
and orphaned descendants after timeout.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from entroly.path_safety import resolve_file_within
from entroly.process_safety import run_bounded_process

from .executors import (
    ASTExecutor,
    ExecutorRegistry as _LegacyExecutorRegistry,
    ExecutorResult,
    PythonExecutor,
    RetrievalExecutor,
    SymPyExecutor,
    TestRunnerExecutor as _LegacyTestRunnerExecutor,
)

_MAX_REQUEST_CHARS = 16_384
_MAX_CAPTURE_BYTES = 2 * 1024 * 1024
_SENSITIVE_ENV_EXACT = frozenset(
    {
        "AWS_SESSION_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "DOCKER_AUTH_CONFIG",
        "GITHUB_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "NPM_TOKEN",
        "PYPI_TOKEN",
    }
)
_SECRET_WORDS = frozenset(
    {"TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "CREDENTIALS", "COOKIE"}
)
_KEY_QUALIFIERS = frozenset({"API", "ACCESS", "PRIVATE", "SECRET", "SIGNING"})


def _is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    if upper in _SENSITIVE_ENV_EXACT:
        return True
    words = {word for word in re.split(r"[^A-Z0-9]+", upper) if word}
    if words & _SECRET_WORDS:
        return True
    if "KEY" in words and words & _KEY_QUALIFIERS:
        return True
    if "AUTH" in words and words & {"TOKEN", "CONFIG", "HEADER", "COOKIE"}:
        return True
    return False


def _sanitized_test_environment() -> dict[str, str]:
    """Return a child environment that omits credential-shaped variables.

    Operators may explicitly opt into legacy full inheritance for a trusted
    workspace with ``ENTROLY_TEST_RUNNER_PASSTHROUGH_SECRETS=1``.
    """
    passthrough_secrets = (
        os.environ.get("ENTROLY_TEST_RUNNER_PASSTHROUGH_SECRETS") == "1"
    )
    child: dict[str, str] = {}
    for key, value in os.environ.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if not passthrough_secrets and _is_sensitive_env_name(key):
            continue
        child[key] = value
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    child["ENTROLY_TEST_RUNNER_ACTIVE"] = "1"
    return child


class TestRunnerExecutor(_LegacyTestRunnerExecutor):
    """Execute allowlisted test frameworks within one resolved workspace."""

    def __init__(self, timeout_s: float = 30.0, cwd: str | None = None):
        try:
            normalized_timeout = float(timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout_s must be a finite positive number") from exc
        if not math.isfinite(normalized_timeout) or normalized_timeout <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        super().__init__(timeout_s=normalized_timeout, cwd=cwd)

    def _workspace(self) -> Path | None:
        try:
            workspace = Path(self._cwd or os.getcwd()).resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            return None
        return workspace if workspace.is_dir() else None

    def _target_from_input(self, input_text: str, workspace: Path) -> str | None:
        match = re.search(
            r"(?:tests?/\S+\.py|test_\S+\.py|\S+_test\.py|\S+\.spec\.\w+)",
            input_text,
        )
        if match is None:
            return None
        raw_target = match.group(0).strip("`'\"()[]{}<>,;:")
        if (
            not raw_target
            or raw_target.startswith("-")
            or "\x00" in raw_target
            or any(ord(char) < 32 for char in raw_target)
        ):
            raise ValueError("test target is not a safe path")
        resolved = resolve_file_within(workspace, raw_target)
        if resolved is None:
            raise ValueError("test target does not exist inside the selected workspace")
        return str(resolved.relative_to(workspace))

    def execute(self, input_text: str) -> ExecutorResult:
        started = time.perf_counter()
        if not isinstance(input_text, str):
            return self._blocked_result(started, "test request must be text")
        if len(input_text) > _MAX_REQUEST_CHARS:
            return self._blocked_result(started, "test request is too long")

        workspace = self._workspace()
        if workspace is None:
            return self._blocked_result(started, "test workspace is unavailable")

        command: list[str] | None = None
        for pattern, base_command in self._FRAMEWORKS:
            if pattern.search(input_text):
                command = list(base_command)
                break
        if command is None:
            command = ["python", "-m", "pytest", "--tb=short", "-q"]

        try:
            target = self._target_from_input(input_text, workspace)
        except ValueError as exc:
            return self._blocked_result(started, str(exc))

        if target:
            if command[:3] == ["python", "-m", "pytest"]:
                command.append("--")
            command.append(target)

        if os.environ.get("ENTROLY_TEST_RUNNER_ACTIVE") == "1":
            return self._blocked_result(started, "nested test execution blocked")
        if "PYTEST_CURRENT_TEST" in os.environ and target is None:
            return self._blocked_result(
                started,
                "ambiguous full-suite test execution blocked from inside pytest",
            )

        process = run_bounded_process(
            command,
            timeout=self._timeout,
            cwd=workspace,
            env=_sanitized_test_environment(),
            max_output_bytes=_MAX_CAPTURE_BYTES,
            preserve_stdout_tail=True,
            preserve_stderr_tail=True,
        )
        elapsed = (time.perf_counter() - started) * 1000

        if process.timed_out:
            error = f"timeout after {self._timeout}s; process tree terminated"
            return ExecutorResult(
                result=json.dumps(
                    {
                        "exit_code": -1,
                        "error": error,
                        "passed": 0,
                        "failed": 0,
                        "output_truncated": (
                            process.stdout_truncated or process.stderr_truncated
                        ),
                    }
                ),
                succeeded=False,
                error=error,
                execution_time_ms=round(elapsed, 2),
                executor_name="test_runner",
            )

        if process.execution_error:
            return ExecutorResult(
                result=json.dumps(
                    {
                        "exit_code": -1,
                        "error": process.execution_error,
                        "passed": 0,
                        "failed": 0,
                    }
                ),
                succeeded=False,
                error=f"command launch failed: {process.execution_error}"[:200],
                execution_time_ms=round(elapsed, 2),
                executor_name="test_runner",
            )

        returncode = process.returncode if process.returncode is not None else -1
        output = process.stdout + "\n" + process.stderr
        parsed = self._parse_results(output, returncode)
        parsed["output_truncated"] = (
            process.stdout_truncated or process.stderr_truncated
        )
        return ExecutorResult(
            result=json.dumps(parsed),
            succeeded=returncode == 0,
            error="" if returncode == 0 else f"exit_code={returncode}",
            execution_time_ms=round(elapsed, 2),
            executor_name="test_runner",
        )


class ExecutorRegistry(_LegacyExecutorRegistry):
    """Production registry with the hardened test runner installed."""

    def __init__(self, test_cwd: str | None = None):
        self._sympy = SymPyExecutor()
        self._python = PythonExecutor()
        self._ast = ASTExecutor()
        self._test = TestRunnerExecutor(cwd=test_cwd)
        self._retrieval = RetrievalExecutor()

    def get(self, executor_type: str) -> Any:
        return super().get(executor_type)
