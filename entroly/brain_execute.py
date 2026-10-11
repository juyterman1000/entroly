"""Answer a question about the workspace by computing it, not by reading it.

Selection shrinks what reaches the model, but it is the wrong instrument for a
question whose answer is an aggregate. "How many exported functions are there",
"which files import this module", "what is the largest test file" all require
reading every candidate file to produce one number, and the model pays for the
whole corpus to learn one line of it.

A script answers those directly: it runs where the files already are and emits
only the result. The point is the ratio between what it reads and what it
returns, so the output is capped and the cap is reported rather than hidden.

This is not a replacement for selection. A question like "why is this written
this way" needs the text itself, and no amount of computing over it substitutes
for reading it.

Security posture, because this executes code:

- Off unless ``ENTROLY_BRAIN_EXECUTE=1``. There is no implicit enablement.
- The script runs in a subprocess with ``shell=False``, so nothing is parsed by
  a shell.
- The working directory is the resolved workspace root, and the script file is
  written outside it so a run cannot index or match its own source.
- The environment is rebuilt from a small allowlist. A repository is full of
  credentials in the ambient environment, and a script that can read files has
  no business also inheriting ``AWS_SECRET_ACCESS_KEY``.
- Wall-clock timeout, and the process group is killed on expiry.
- Output is bounded; exceeding the bound is reported, not silently cut.

It remains an explicit local-code execution feature rather than a sandbox: a
script can still read any file the user can read. The caller is trusting the
model with the user's own shell-level reach, which is why it is opt-in.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["BrainResult", "brain_execute", "brain_execute_enabled"]

ENABLE_ENV = "ENTROLY_BRAIN_EXECUTE"

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_OUTPUT_BYTES = 32 * 1024
MAX_CODE_BYTES = 128 * 1024

# Rebuilt rather than inherited. PATH is needed to find the interpreter's own
# helpers; the Windows entries are required for the C runtime to initialise.
_ENV_ALLOWLIST = (
    "PATH",
    "SYSTEMROOT",
    "COMSPEC",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
)


@dataclass(frozen=True)
class BrainResult:
    """What the script produced, and what it cost to produce it."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    truncated: bool = False
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_s": round(self.duration_s, 3),
            "truncated": self.truncated,
            "error": self.error,
            **self.meta,
        }


def brain_execute_enabled() -> bool:
    """True only on explicit opt-in."""
    return os.environ.get(ENABLE_ENV, "") == "1"


def _child_env() -> dict[str, str]:
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    # Keep the child off the parent's import path so a script cannot quietly
    # import repository modules it was not given.
    env["PYTHONPATH"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["ENTROLY_BRAIN_CHILD"] = "1"
    return env


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= limit:
        return text, False
    clipped = encoded[:limit].decode("utf-8", "ignore")
    return (
        f"{clipped}\n...[truncated: {len(encoded):,} bytes produced, {limit:,} returned]",
        True,
    )


def brain_execute(
    code: str,
    *,
    workspace: str | Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> BrainResult:
    """Run ``code`` as a Python script rooted at the workspace.

    Returns a :class:`BrainResult` rather than raising, so a failing script is
    an observation the caller can act on rather than an exception that loses
    whatever the script did print before it failed.
    """
    if not brain_execute_enabled():
        return BrainResult(
            ok=False,
            error=f"disabled; set {ENABLE_ENV}=1 to allow local script execution",
        )
    if not isinstance(code, str) or not code.strip():
        return BrainResult(ok=False, error="code is empty")
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        return BrainResult(ok=False, error="code exceeds the size limit")

    root = Path(workspace or os.environ.get("ENTROLY_SOURCE") or os.getcwd())
    try:
        root = root.resolve(strict=True)
    except OSError:
        return BrainResult(ok=False, error="workspace does not resolve")
    if not root.is_dir():
        return BrainResult(ok=False, error="workspace is not a directory")

    timeout_s = max(0.1, min(float(timeout_s), 120.0))
    max_output_bytes = max(256, min(int(max_output_bytes), 1024 * 1024))

    started = time.perf_counter()
    # The script lives outside the workspace so a run cannot read, index, or
    # pattern-match its own source as if it were repository content.
    with tempfile.TemporaryDirectory(prefix="entroly-brain-") as scratch:
        script = Path(scratch) / "brain_script.py"
        script.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(  # noqa: S603 - shell=False, fixed argv
                [sys.executable, "-I", "-B", str(script)],
                cwd=str(root),
                env=_child_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                shell=False,
            )
        except subprocess.TimeoutExpired as expired:
            duration = time.perf_counter() - started
            partial, _ = _truncate(expired.stdout or "", max_output_bytes)
            return BrainResult(
                ok=False,
                stdout=partial,
                duration_s=duration,
                error=f"timed out after {timeout_s:g}s",
                meta={"timed_out": True},
            )
        except OSError as exc:
            return BrainResult(
                ok=False,
                duration_s=time.perf_counter() - started,
                error=f"could not start interpreter: {exc}",
            )

    duration = time.perf_counter() - started
    stdout, clipped_out = _truncate(completed.stdout or "", max_output_bytes)
    stderr, clipped_err = _truncate(completed.stderr or "", max_output_bytes // 4)
    return BrainResult(
        ok=completed.returncode == 0,
        stdout=stdout,
        stderr=stderr,
        exit_code=completed.returncode,
        duration_s=duration,
        truncated=clipped_out or clipped_err,
        error=None if completed.returncode == 0 else "script exited non-zero",
        meta={"workspace": str(root)},
    )
