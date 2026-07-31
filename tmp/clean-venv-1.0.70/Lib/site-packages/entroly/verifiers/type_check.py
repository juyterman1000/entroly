"""
Pyright-Backed Type Compatibility Verifier
==========================================

Sometimes a symbol exists AND is in scope, but is being used wrong:

    requests.get(url, parans={"q": x})    # 'parans' is wrong, real is 'params'

The symbol `requests.get` is fine. The keyword arg `parans` doesn't
exist on the function. The Bayesian symbol verifier won't catch this
because `parans` isn't even a symbol reference per AST — it's a kwarg name.

Pyright catches it. We invoke pyright on the generated snippet in a
sandbox tempdir and parse `--outputjson` for diagnostics that map to
verifier-style judgments.

Trust constraints
-----------------
- Never represent an unavailable or failed verifier as a clean pass.
- Filter to errors *originating in our snippet*, not in dependency types.
- Preserve a compatibility helper for callers that only consume diagnostics.
- Bound source, process lifetime, descendants, output, and diagnostic detail.
- Do not expose host credentials to an external verifier by default.

For ad-hoc snippets, we run with `--outputjson` and no project mode so the
check remains file-scoped.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from entroly.process_safety import run_bounded_process, sanitized_environment

logger = logging.getLogger("entroly.verifiers.type_check")

TYPE_CHECK_STATUSES = frozenset(
    {
        "passed",
        "issues_found",
        "unavailable",
        "timed_out",
        "execution_failed",
        "malformed_output",
        "output_too_large",
    }
)
_PYTHON_VERSION_RE = re.compile(r"^[2-9]\.[0-9]{1,2}$")
_MAX_SOURCE_CHARS = 2_000_000
_MAX_EXTRA_PATHS = 64
_MAX_EXTRA_PATH_CHARS = 4096
_MAX_PYRIGHT_OUTPUT_BYTES = 4 * 1024 * 1024
_MAX_DIAGNOSTICS = 10_000


@dataclass
class TypeError_:
    """A type-compat diagnostic mapped from Pyright output."""

    line: int
    column: int
    severity: str
    message: str
    rule: str
    likely_symbol: str | None = None


@dataclass
class TypeCheckResult:
    """Auditable outcome of one Pyright invocation.

    An empty diagnostics list is meaningful only when ``status == "passed"``.
    Every degraded state remains distinguishable from a successful clean run.
    """

    status: str
    diagnostics: list[TypeError_] = field(default_factory=list)
    detail: str = ""

    def __post_init__(self) -> None:
        if self.status not in TYPE_CHECK_STATUSES:
            raise ValueError(f"unknown type-check status: {self.status!r}")
        self.detail = self.detail[:500]

    @property
    def completed(self) -> bool:
        return self.status in {"passed", "issues_found"}

    @property
    def trusted_clean(self) -> bool:
        return self.status == "passed"


def _bounded_detail(value: object) -> str:
    return str(value or "").strip()[-500:]


def _typecheck_environment() -> dict[str, str]:
    allow_secrets = os.environ.get("ENTROLY_TYPECHECK_PASSTHROUGH_SECRETS") == "1"
    return sanitized_environment(
        allow_secrets=allow_secrets,
        overrides={"PYTHONUTF8": "1", "NO_COLOR": "1"},
    )


def _normalized_timeout(timeout_s: object) -> float | None:
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError):
        return None
    return timeout if math.isfinite(timeout) and timeout > 0 else None


def _validated_extra_paths(extra_paths: object) -> list[str] | None:
    if extra_paths is None:
        return []
    if not isinstance(extra_paths, (list, tuple)):
        return None
    if len(extra_paths) > _MAX_EXTRA_PATHS:
        return None
    validated: list[str] = []
    for path in extra_paths:
        if (
            not isinstance(path, str)
            or not path
            or len(path) > _MAX_EXTRA_PATH_CHARS
            or "\x00" in path
            or any(ord(char) < 32 for char in path)
        ):
            return None
        validated.append(path)
    return validated


def pyright_available() -> bool:
    """Return whether Pyright is installed and its bounded probe succeeds."""
    executable = shutil.which("pyright")
    if executable is None:
        return False
    result = run_bounded_process(
        [executable, "--version"],
        timeout=3,
        env=_typecheck_environment(),
        max_output_bytes=64 * 1024,
    )
    return result.succeeded and not result.stdout_truncated and not result.stderr_truncated


def check_snippet_result(
    source: str,
    extra_paths: list[str] | None = None,
    python_version: str = "3.11",
    timeout_s: float = 8.0,
) -> TypeCheckResult:
    """Run Pyright and return a structured, fail-auditable result."""
    if not isinstance(source, str):
        return TypeCheckResult(
            status="execution_failed",
            detail="type-check source must be a string",
        )
    if len(source) > _MAX_SOURCE_CHARS:
        return TypeCheckResult(
            status="output_too_large",
            detail=f"type-check source exceeds {_MAX_SOURCE_CHARS} characters",
        )

    timeout = _normalized_timeout(timeout_s)
    if timeout is None:
        return TypeCheckResult(
            status="execution_failed",
            detail="type-check timeout must be a finite positive number",
        )
    if not isinstance(python_version, str) or not _PYTHON_VERSION_RE.fullmatch(
        python_version
    ):
        return TypeCheckResult(
            status="execution_failed",
            detail="python_version must use a supported major.minor form",
        )

    validated_paths = _validated_extra_paths(extra_paths)
    if validated_paths is None:
        return TypeCheckResult(
            status="execution_failed",
            detail="extra_paths must be a bounded list of safe strings",
        )

    executable = shutil.which("pyright")
    if executable is None:
        return TypeCheckResult(
            status="unavailable",
            detail="pyright executable was not found",
        )

    environment = _typecheck_environment()
    version_probe = run_bounded_process(
        [executable, "--version"],
        timeout=min(3.0, timeout),
        env=environment,
        max_output_bytes=64 * 1024,
    )
    if version_probe.timed_out:
        return TypeCheckResult(
            status="timed_out",
            detail="pyright version probe timed out; process tree terminated",
        )
    if version_probe.execution_error:
        return TypeCheckResult(
            status="execution_failed",
            detail=f"pyright version probe failed: {_bounded_detail(version_probe.execution_error)}",
        )
    if version_probe.stdout_truncated or version_probe.stderr_truncated:
        return TypeCheckResult(
            status="output_too_large",
            detail="pyright version probe output exceeded the safety limit",
        )
    if version_probe.returncode != 0:
        detail = _bounded_detail(version_probe.stderr or version_probe.stdout)
        return TypeCheckResult(
            status="execution_failed",
            detail=f"pyright version probe exited {version_probe.returncode}: {detail}",
        )

    with tempfile.TemporaryDirectory(prefix="entroly_typecheck_") as td:
        snippet_path = Path(td) / "_snippet.py"
        try:
            snippet_path.write_text(source, encoding="utf-8")
        except OSError as exc:
            return TypeCheckResult(
                status="execution_failed",
                detail=f"could not stage snippet: {_bounded_detail(exc)}",
            )

        command = [
            executable,
            "--outputjson",
            "--pythonversion",
            python_version,
            str(snippet_path),
        ]
        if validated_paths:
            command.extend(["--extrapaths", os.pathsep.join(validated_paths)])

        process = run_bounded_process(
            command,
            timeout=timeout,
            env=environment,
            max_output_bytes=_MAX_PYRIGHT_OUTPUT_BYTES,
            preserve_stdout_tail=False,
            preserve_stderr_tail=True,
        )
        if process.timed_out:
            logger.debug("pyright timed out after %ss", timeout)
            return TypeCheckResult(
                status="timed_out",
                detail=f"pyright timed out after {timeout}s; process tree terminated",
            )
        if process.execution_error:
            logger.debug("pyright execution failed: %s", process.execution_error)
            return TypeCheckResult(
                status="execution_failed",
                detail=f"pyright execution failed: {_bounded_detail(process.execution_error)}",
            )
        if process.stdout_truncated or process.stderr_truncated:
            return TypeCheckResult(
                status="output_too_large",
                detail="pyright output exceeded the safety limit",
            )
        if not process.stdout:
            detail = _bounded_detail(process.stderr)
            return TypeCheckResult(
                status="malformed_output",
                detail=f"pyright produced no JSON output: {detail}",
            )

        try:
            data = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            logger.debug("pyright stdout parse failed: %s", exc)
            return TypeCheckResult(
                status="malformed_output",
                detail=f"pyright returned invalid JSON: {_bounded_detail(exc)}",
            )

        if not isinstance(data, dict):
            return TypeCheckResult(
                status="malformed_output",
                detail="pyright JSON root was not an object",
            )
        diagnostics = data.get("generalDiagnostics")
        if (
            not isinstance(diagnostics, list)
            or len(diagnostics) > _MAX_DIAGNOSTICS
            or not all(isinstance(item, dict) for item in diagnostics)
        ):
            return TypeCheckResult(
                status="malformed_output",
                detail="pyright generalDiagnostics was not a bounded list of objects",
            )

        errors: list[TypeError_] = []
        try:
            for diagnostic in diagnostics:
                diagnostic_file = str(diagnostic.get("file", ""))
                if Path(diagnostic_file.replace("\\", "/")).name != "_snippet.py":
                    continue
                severity = str(diagnostic.get("severity", "information"))
                if severity not in ("error", "warning"):
                    continue
                range_data = diagnostic.get("range", {})
                if not isinstance(range_data, dict):
                    raise TypeError("diagnostic range was not an object")
                start = range_data.get("start", {})
                if not isinstance(start, dict):
                    raise TypeError("diagnostic range.start was not an object")
                line = int(start.get("line", 0))
                column = int(start.get("character", 0))
                if line < 0 or column < 0:
                    raise ValueError("diagnostic positions must be non-negative")
                message = str(diagnostic.get("message", ""))[:4000]
                rule = str(diagnostic.get("rule", ""))[:500]
                errors.append(
                    TypeError_(
                        line=line + 1,
                        column=column + 1,
                        severity=severity,
                        message=message,
                        rule=rule,
                        likely_symbol=_extract_symbol_from_message(message),
                    )
                )
        except (TypeError, ValueError) as exc:
            return TypeCheckResult(
                status="malformed_output",
                detail=f"pyright diagnostic schema was invalid: {_bounded_detail(exc)}",
            )

        if errors:
            return TypeCheckResult(status="issues_found", diagnostics=errors)

        if process.returncode != 0:
            detail = _bounded_detail(process.stderr or process.stdout)
            return TypeCheckResult(
                status="execution_failed",
                detail=f"pyright exited {process.returncode} without snippet diagnostics: {detail}",
            )

        return TypeCheckResult(status="passed")


def check_snippet(
    source: str,
    extra_paths: list[str] | None = None,
    python_version: str = "3.11",
    timeout_s: float = 8.0,
) -> list[TypeError_]:
    """Compatibility wrapper returning diagnostics only.

    New trust-sensitive callers must use :func:`check_snippet_result` so a
    degraded verifier cannot be mistaken for a clean pass.
    """
    return check_snippet_result(
        source,
        extra_paths=extra_paths,
        python_version=python_version,
        timeout_s=timeout_s,
    ).diagnostics


_SYMBOL_RE = None


def _extract_symbol_from_message(msg: str) -> str | None:
    global _SYMBOL_RE
    if _SYMBOL_RE is None:
        _SYMBOL_RE = re.compile(r'"([a-zA-Z_][a-zA-Z0-9_]*)"')
    match = _SYMBOL_RE.search(msg)
    return match.group(1) if match else None
