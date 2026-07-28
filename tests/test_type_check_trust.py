from __future__ import annotations

import json

import pytest

from entroly.process_safety import BoundedProcessResult
from entroly.verifiers.service import ExtendedResult
from entroly.verifiers.type_check import (
    _typecheck_environment,
    check_snippet,
    check_snippet_result,
)


def _completed(
    command=("pyright",),
    *,
    returncode: int | None = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
    execution_error: str = "",
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
) -> BoundedProcessResult:
    return BoundedProcessResult(
        args=tuple(command),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        execution_error=execution_error,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )


def _install_pyright(monkeypatch, responses) -> None:
    monkeypatch.setattr(
        "entroly.verifiers.type_check.shutil.which",
        lambda _name: "/usr/bin/pyright",
    )
    iterator = iter(responses)
    monkeypatch.setattr(
        "entroly.verifiers.type_check.run_bounded_process",
        lambda *args, **kwargs: next(iterator),
    )


def test_missing_pyright_is_not_reported_as_clean_pass(monkeypatch) -> None:
    monkeypatch.setattr("entroly.verifiers.type_check.shutil.which", lambda _name: None)

    result = check_snippet_result("value: int = 'wrong'")

    assert result.status == "unavailable"
    assert result.diagnostics == []
    assert not result.completed
    assert not result.trusted_clean
    assert check_snippet("value: int = 'wrong'") == []


def test_pyright_timeout_is_explicit_and_reports_tree_cleanup(monkeypatch) -> None:
    _install_pyright(
        monkeypatch,
        [
            _completed(stdout="pyright 1.1"),
            _completed(returncode=-9, timed_out=True),
        ],
    )

    result = check_snippet_result("x = 1", timeout_s=0.01)

    assert result.status == "timed_out"
    assert not result.completed
    assert "process tree terminated" in result.detail


def test_malformed_pyright_output_is_explicit(monkeypatch) -> None:
    _install_pyright(
        monkeypatch,
        [
            _completed(stdout="pyright 1.1"),
            _completed(stdout="not-json"),
        ],
    )

    result = check_snippet_result("x = 1")

    assert result.status == "malformed_output"
    assert not result.completed
    assert result.diagnostics == []


def test_oversized_pyright_output_is_not_parsed_as_a_pass(monkeypatch) -> None:
    _install_pyright(
        monkeypatch,
        [
            _completed(stdout="pyright 1.1"),
            _completed(stdout="{}", stdout_truncated=True),
        ],
    )

    result = check_snippet_result("x = 1")

    assert result.status == "output_too_large"
    assert not result.completed
    assert "safety limit" in result.detail


def test_clean_pyright_run_is_the_only_clean_status(monkeypatch) -> None:
    _install_pyright(
        monkeypatch,
        [
            _completed(stdout="pyright 1.1"),
            _completed(stdout=json.dumps({"generalDiagnostics": []})),
        ],
    )

    result = check_snippet_result("x: int = 1")

    assert result.status == "passed"
    assert result.completed
    assert result.trusted_clean
    assert result.diagnostics == []


def _run_issue_case(monkeypatch, diagnostic: dict):
    _install_pyright(
        monkeypatch,
        [
            _completed(stdout="pyright 1.1"),
            _completed(
                returncode=1,
                stdout=json.dumps({"generalDiagnostics": [diagnostic]}),
            ),
        ],
    )
    return check_snippet_result("x: int = 'bad'")


def test_pyright_issues_are_preserved_and_reject_verdict(monkeypatch) -> None:
    diagnostic = {
        "file": "/tmp/entroly_typecheck_123/_snippet.py",
        "severity": "error",
        "message": 'Type "str" is not assignable to "int"',
        "rule": "reportAssignmentType",
        "range": {"start": {"line": 0, "character": 4}},
    }

    result = _run_issue_case(monkeypatch, diagnostic)

    assert result.status == "issues_found"
    assert result.completed
    assert not result.trusted_clean
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].line == 1
    assert result.diagnostics[0].column == 5


def test_windows_diagnostic_path_is_not_silently_discarded(monkeypatch) -> None:
    diagnostic = {
        "file": r"C:\Users\agent\AppData\Local\Temp\entroly_typecheck\_snippet.py",
        "severity": "error",
        "message": 'Type "str" is not assignable to "int"',
        "rule": "reportAssignmentType",
        "range": {"start": {"line": 2, "character": 1}},
    }

    result = _run_issue_case(monkeypatch, diagnostic)

    assert result.status == "issues_found"
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].line == 3
    assert result.diagnostics[0].column == 2


@pytest.mark.parametrize(
    ("source", "python_version", "timeout_s", "expected_detail"),
    [
        (None, "3.11", 1.0, "source must be a string"),
        ("x = 1", "3.11", 0.0, "finite positive"),
        ("x = 1", "3.11", float("nan"), "finite positive"),
        ("x = 1", "3.11", float("inf"), "finite positive"),
        ("x = 1", "3.11", "bad", "finite positive"),
        ("x = 1", "3.11;--verifytypes", 1.0, "major.minor"),
    ],
)
def test_invalid_type_check_inputs_fail_auditably(
    source, python_version: str, timeout_s, expected_detail: str
) -> None:
    result = check_snippet_result(
        source,
        python_version=python_version,
        timeout_s=timeout_s,
    )

    assert result.status == "execution_failed"
    assert expected_detail in result.detail


def test_unbounded_extra_paths_are_rejected_before_launch(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.verifiers.type_check.run_bounded_process",
        lambda *_args, **_kwargs: pytest.fail("invalid paths reached process launch"),
    )

    result = check_snippet_result("x = 1", extra_paths=["ok"] * 65)

    assert result.status == "execution_failed"
    assert "extra_paths" in result.detail


def test_oversized_source_is_rejected_before_launch(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.verifiers.type_check.run_bounded_process",
        lambda *_args, **_kwargs: pytest.fail("oversized source reached process launch"),
    )

    result = check_snippet_result("x" * 2_000_001)

    assert result.status == "output_too_large"
    assert "source exceeds" in result.detail


def test_typecheck_environment_scrubs_credentials_but_preserves_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-leak")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Entroly Maintainer")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.delenv("ENTROLY_TYPECHECK_PASSTHROUGH_SECRETS", raising=False)

    child = _typecheck_environment()

    assert "OPENAI_API_KEY" not in child
    assert child["GIT_AUTHOR_NAME"] == "Entroly Maintainer"
    assert child["TOKENIZERS_PARALLELISM"] == "false"


def _extended_result(status: str, detail: str = "") -> ExtendedResult:
    return ExtendedResult(
        code="x = 1",
        archetype="code/implement",
        lambda_used=0.5,
        judgments=[],
        h_strict=0.0,
        h_lenient=0.0,
        n_grounded=0,
        n_unreachable=0,
        n_hallucinated=0,
        manifest_size=1,
        type_errors=[],
        type_check_status=status,
        type_check_detail=detail,
    )


@pytest.mark.parametrize(
    "status",
    [
        "issues_found",
        "unavailable",
        "timed_out",
        "execution_failed",
        "malformed_output",
        "output_too_large",
    ],
)
def test_requested_type_check_must_complete_cleanly_to_pass(status: str) -> None:
    result = _extended_result(status, "adversarial failure")

    assert not result.passed_strict()
    assert not result.passed_lenient()
    explanation = result.explain()
    assert "REJECTED" in explanation
    assert "type-check:" in explanation


def test_unrequested_type_check_preserves_optional_behavior() -> None:
    result = _extended_result("not_requested")

    assert result.passed_strict()
    assert result.passed_lenient()


def test_completed_clean_type_check_allows_symbol_verdict_to_pass() -> None:
    result = _extended_result("passed")

    assert result.passed_strict()
    assert result.passed_lenient()
    assert "type-check: PASS" in result.explain()
