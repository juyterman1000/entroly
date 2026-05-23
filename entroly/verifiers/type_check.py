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

Design constraints
------------------
- Fail OPEN if pyright is not installed (don't break the user's flow)
- Fail OPEN on timeout (>5s = abandon)
- Filter to errors *originating in our snippet*, not in dep types
- Convert to (symbol_name, kind, line, message) tuples that compose with
  the rest of the verifier results

For projects with a venv, we pyright in --pythonversion 3.13 (current
process) and --venvpath if available, so the type checks see installed
packages. For ad-hoc snippets, we run with --outputjson + no project
mode (just file-scoped).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("entroly.verifiers.type_check")


@dataclass
class TypeError_:
    """A type-compat diagnostic mapped from pyright output."""
    line: int
    column: int
    severity: str   # "error", "warning", "information"
    message: str
    rule: str       # e.g. "reportGeneralTypeIssues"
    # Best-effort symbol name extracted from the message
    likely_symbol: str | None = None


def pyright_available() -> bool:
    """Is pyright installed and runnable? Returns False on any error."""
    if shutil.which("pyright") is None:
        return False
    try:
        proc = subprocess.run(
            ["pyright", "--version"],
            capture_output=True, text=True, timeout=3,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def check_snippet(
    source: str,
    extra_paths: list[str] | None = None,
    python_version: str = "3.11",
    timeout_s: float = 8.0,
) -> list[TypeError_]:
    """Run pyright on a code snippet and return parsed errors.

    Fails open: any error returns [] rather than raising.
    """
    if not pyright_available():
        return []

    with tempfile.TemporaryDirectory(prefix="entroly_typecheck_") as td:
        td_path = Path(td)
        snippet_path = td_path / "_snippet.py"
        snippet_path.write_text(source, encoding="utf-8")

        cmd = [
            "pyright",
            "--outputjson",
            "--pythonversion", python_version,
            str(snippet_path),
        ]
        if extra_paths:
            cmd.extend(["--extrapaths", os.pathsep.join(extra_paths)])

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout_s, check=False,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug("pyright failed (%s) — failing open", e)
            return []

        # Pyright returns non-zero when there are errors — that's expected.
        # We care about stdout content.
        try:
            data = json.loads(proc.stdout) if proc.stdout else {}
        except json.JSONDecodeError as e:
            logger.debug("pyright stdout parse failed (%s)", e)
            return []

        diagnostics = data.get("generalDiagnostics", [])
        errors: list[TypeError_] = []
        for d in diagnostics:
            # Only keep our snippet's diagnostics; ignore deps
            d_file = d.get("file", "")
            if "_snippet.py" not in d_file:
                continue
            sev = d.get("severity", "information")
            if sev not in ("error", "warning"):
                continue
            rng = d.get("range", {}).get("start", {})
            msg = d.get("message", "")
            errors.append(TypeError_(
                line=int(rng.get("line", 0)) + 1,
                column=int(rng.get("character", 0)) + 1,
                severity=sev,
                message=msg,
                rule=d.get("rule", ""),
                likely_symbol=_extract_symbol_from_message(msg),
            ))
        return errors


# Pyright error messages have surprisingly stable shapes; this is a
# best-effort extraction of the offending symbol name from common
# templates. Used only for explainability.
_SYMBOL_RE = None


def _extract_symbol_from_message(msg: str) -> str | None:
    import re
    global _SYMBOL_RE
    if _SYMBOL_RE is None:
        # Common pyright messages:
        #   "No overloads for "get" match the provided arguments"
        #   "Argument missing for parameter "..."
        #   "Cannot access member "fooBar" for type "Module""
        #   "X is not a known member of "..."
        _SYMBOL_RE = re.compile(r'"([a-zA-Z_][a-zA-Z0-9_]*)"')
    m = _SYMBOL_RE.search(msg)
    return m.group(1) if m else None
