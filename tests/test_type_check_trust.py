from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from entroly.verifiers.service import ExtendedResult
from entroly.verifiers.type_check import check_snippet, check_snippet_result


def _completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_missing_pyright_is_not_reported_as_clean_pass(monkeypatch) -> None:
    monkeypatch.setattr("entroly.verifiers.type_check.shutil.which", lambda _name: None)

    result = check_snippet_result("value: int = 'wrong'")

    assert result.status == "unavailable"
    assert result.diagnostics == []
    assert not result.completed
    assert not result.trusted_clean
    # Compatibility callers still receive a list, but trust-sensitive callers
    # can no longer confuse this with a completed clean run.
    assert check_snippet("value: int = 'wrong'") == []


def test_pyright_timeout_is_explicit(monkeypatch) -> None:
    calls = 0

    monkeypatch.setattr(
        "entroly.verifiers.type_check.shutil.which",
        lambda _name: "/usr/bin/pyright",
    )

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _completed(stdout="pyright 1.1")
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    monkeypatch.setattr("entroly.verifiers.type_check.subprocess.run", fake_run)

    result = check_snippet_result("x = 1", timeout_s=0.01)

    assert result.status == "timed_out"
    assert not result.completed
    assert "timed out" in result.detail


def test_malformed_pyright_output_is_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.verifiers.type_check.shutil.which",
        lambda _name: "/usr/bin/pyright",
    )
    responses = iter(
        [
            _completed(stdout="pyright 1.1"),
            _completed(stdout="not-json"),
        ]
    )
    monkeypatch.setattr(
        "entroly.verifiers.type_check.subprocess.run",
        lambda *args, **kwargs: next(responses),
    )

    result = check_snippet_result("x = 1")

    assert result.status == "malformed_output"
    assert not result.completed
    assert result.diagnostics == []


def test_clean_pyright_run_is_the_only_clean_status(monkeypatch) -> None:
    monkeypatch.setattr(
        "entroly.verifiers.type_check.shutil.which",
        lambda _name: "/usr/bin/pyright",
    )
    responses = iter(
        [
            _completed(stdout="pyright 1.1"),
            _completed(stdout=json.dumps({"generalDiagnostics": []})),
        ]
    )
    monkeypatch.setattr(
        "entroly.verifiers.type_check.subprocess.run",
        lambda *args, **kwargs: next(responses),
    )

    result = check_snippet_result("x: int = 1")

    assert result.status == "passed"
    assert result.completed
    assert result.trusted_clean
    assert result.diagnostics == []


def _run_issue_case(monkeypatch, diagnostic: dict) -> object:
    monkeypatch.setattr(
        "entroly.verifiers.type_check.shutil.which",
        lambda _name: "/usr/bin/pyright",
    )
    responses = iter(
        [
            _completed(stdout="pyright 1.1"),
            _completed(
                returncode=1,
                stdout=json.dumps({"generalDiagnostics": [diagnostic]}),
            ),
        ]
    )
    monkeypatch.setattr(
        "entroly.verifiers.type_check.subprocess.run",
        lambda *args, **kwargs: next(responses),
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
        ("x = 1", "3.11", 0.0, "timeout must be positive"),
        ("x = 1", "3.11;--verifytypes", 1.0, "major.minor"),
    ],
)
def test_invalid_type_check_inputs_fail_auditably(
    source, python_version: str, timeout_s: float, expected_detail: str
) -> None:
    result = check_snippet_result(
        source,
        python_version=python_version,
        timeout_s=timeout_s,
    )

    assert result.status == "execution_failed"
    assert expected_detail in result.detail


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
