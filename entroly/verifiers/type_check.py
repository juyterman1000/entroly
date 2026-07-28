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
- Bound diagnostic detail so failed tooling cannot flood receipts or logs.

For ad-hoc snippets, we run with `--outputjson` and no project mode so the
check remains file-scoped.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("entroly.verifiers.type_check")

TYPE_CHECK_STATUSES = frozenset(
    {
        "passed",
        "issues_found",
        "unavailable",
        "timed_out",
        "execution_failed",
        "malformed_output",
    }
)


@dataclass
class TypeError_:
    """A type-compat diagnostic mapped from Pyright output."""

    line: int
    column: int
    severity: str  # "error", "warning", "information"
    message: str
    rule: str  # e.g. "reportGeneralTypeIssues"
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


def pyright_available() -> bool:
    """Return whether Pyright is installed and its version probe succeeds."""
    executable = shutil.which("pyright")
    if executable is None:
        return False
    try:
        proc = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def check_snippet_result(
    source: str,
    extra_paths: list[str] | None = None,
    python_version: str = "3.11",
    timeout_s: float = 8.0,
) -> TypeCheckResult:
    """Run Pyright and return a structured, fail-auditable result."""
    executable = shutil.which("pyright")
    if executable is None:
        return TypeCheckResult(
            status="unavailable",
            detail="pyright executable was not found",
        )

    try:
        version_probe = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=min(3.0, max(timeout_s, 0.01)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return TypeCheckResult(
            status="timed_out",
            detail="pyright version probe timed out",
        )
    except OSError as exc:
        return TypeCheckResult(
            status="execution_failed",
            detail=f"pyright version probe failed: {_bounded_detail(exc)}",
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

        cmd = [
            executable,
            "--outputjson",
            "--pythonversion",
            python_version,
            str(snippet_path),
        ]
        if extra_paths:
            cmd.extend(["--extrapaths", os.pathsep.join(extra_paths)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.debug("pyright timed out after %ss", timeout_s)
            return TypeCheckResult(
                status="timed_out",
                detail=f"pyright timed out after {timeout_s}s",
            )
        except OSError as exc:
            logger.debug("pyright execution failed: %s", exc)
            return TypeCheckResult(
                status="execution_failed",
                detail=f"pyright execution failed: {_bounded_detail(exc)}",
            )

        if not proc.stdout:
            detail = _bounded_detail(proc.stderr)
            return TypeCheckResult(
                status="malformed_output",
                detail=f"pyright produced no JSON output: {detail}",
            )

        try:
            data = json.loads(proc.stdout)
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
        if not isinstance(diagnostics, list) or not all(
            isinstance(item, dict) for item in diagnostics
        ):
            return TypeCheckResult(
                status="malformed_output",
                detail="pyright generalDiagnostics was not a list of objects",
            )

        errors: list[TypeError_] = []
        try:
            for diagnostic in diagnostics:
                d_file = str(diagnostic.get("file", ""))
                if Path(d_file).name != "_snippet.py":
                    continue
                severity = str(diagnostic.get("severity", "information"))
                if severity not in ("error", "warning"):
                    continue
                start = diagnostic.get("range", {}).get("start", {})
                if not isinstance(start, dict):
                    raise TypeError("diagnostic range.start was not an object")
                message = str(diagnostic.get("message", ""))
                errors.append(
                    TypeError_(
                        line=int(start.get("line", 0)) + 1,
                        column=int(start.get("character", 0)) + 1,
                        severity=severity,
                        message=message,
                        rule=str(diagnostic.get("rule", "")),
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

        if proc.returncode != 0:
            detail = _bounded_detail(proc.stderr or proc.stdout)
            return TypeCheckResult(
                status="execution_failed",
                detail=f"pyright exited {proc.returncode} without snippet diagnostics: {detail}",
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
    import re

    global _SYMBOL_RE
    if _SYMBOL_RE is None:
        _SYMBOL_RE = re.compile(r'"([a-zA-Z_][a-zA-Z0-9_]*)"')
    match = _SYMBOL_RE.search(msg)
    return match.group(1) if match else None
