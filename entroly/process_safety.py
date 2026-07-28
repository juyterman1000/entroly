"""Cross-platform bounded subprocess execution for trust-sensitive paths.

The standard ``subprocess.run(..., capture_output=True, timeout=...)`` API
terminates only the direct child. Descendants can survive and can keep inherited
pipe handles open, which creates leaked processes, blocked readers, and false
confidence that a timed-out command was fully stopped.

This module starts commands in an isolated process group, captures output in
anonymous temporary files, kills the complete process tree on timeout, and
returns a structured result without raising for ordinary process failures.
"""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_DEFAULT_MAX_OUTPUT_BYTES = 2 * 1024 * 1024
_TRUNCATION_MARKER = b"\n...[output truncated]...\n"
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


@dataclass(frozen=True)
class BoundedProcessResult:
    """Auditable outcome from :func:`run_bounded_process`."""

    args: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    execution_error: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def succeeded(self) -> bool:
        return (
            not self.timed_out
            and not self.execution_error
            and self.returncode == 0
        )


def is_sensitive_environment_name(name: str) -> bool:
    """Return whether an environment name is credential-shaped.

    Tokenized matching avoids false positives such as ``GIT_AUTHOR_NAME`` and
    ``TOKENIZERS_PARALLELISM`` while still catching common provider credentials.
    """
    if not isinstance(name, str):
        return False
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


def sanitized_environment(
    base: Mapping[str, str] | None = None,
    *,
    allow_secrets: bool = False,
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy an environment while removing credential-shaped values by default."""
    source = os.environ if base is None else base
    child: dict[str, str] = {}
    for key, value in source.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if not allow_secrets and is_sensitive_environment_name(key):
            continue
        child[key] = value
    if overrides:
        for key, value in overrides.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("environment overrides must contain only strings")
            if not key or "\x00" in key or "=" in key:
                raise ValueError("environment override names are invalid")
            if "\x00" in value:
                raise ValueError("environment override values must not contain NUL bytes")
            child[key] = value
    return child


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("command must be a non-empty list or tuple")
    normalized: list[str] = []
    for value in command:
        if not isinstance(value, str) or not value:
            raise ValueError("command arguments must be non-empty strings")
        if "\x00" in value:
            raise ValueError("command arguments must not contain NUL bytes")
        normalized.append(value)
    return tuple(normalized)


def _validate_timeout(timeout: float) -> float:
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be a finite positive number")
    return value


def _validate_max_output_bytes(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError("max_output_bytes must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_output_bytes must be a positive integer") from exc
    if normalized <= 0:
        raise ValueError("max_output_bytes must be a positive integer")
    return normalized


def terminate_process_tree(
    proc: subprocess.Popen[bytes], *, timeout: float = 1.0
) -> None:
    """Best-effort terminate a process and all descendants, then reap it."""
    try:
        wait_timeout = max(0.1, float(timeout))
    except (TypeError, ValueError):
        wait_timeout = 1.0
    if not math.isfinite(wait_timeout):
        wait_timeout = 1.0

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=wait_timeout,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=wait_timeout)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass


def _read_bounded(
    handle, *, max_bytes: int, preserve_tail: bool
) -> tuple[str, bool]:
    handle.flush()
    handle.seek(0, os.SEEK_END)
    size = handle.tell()
    truncated = size > max_bytes
    if not truncated:
        handle.seek(0)
        payload = handle.read()
    elif preserve_tail:
        handle.seek(max(0, size - max_bytes))
        payload = _TRUNCATION_MARKER + handle.read(max_bytes)
    else:
        head_size = max_bytes // 2
        tail_size = max_bytes - head_size
        handle.seek(0)
        head = handle.read(head_size)
        handle.seek(max(0, size - tail_size))
        tail = handle.read(tail_size)
        payload = head + _TRUNCATION_MARKER + tail
    return payload.decode("utf-8", errors="replace"), truncated


def run_bounded_process(
    command: Sequence[str],
    *,
    timeout: float,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    preserve_stdout_tail: bool = False,
    preserve_stderr_tail: bool = True,
) -> BoundedProcessResult:
    """Execute a command with bounded output and full process-tree cleanup.

    No shell is involved. Ordinary launch failures and timeouts are represented
    in the returned result. Invalid API inputs raise ``ValueError`` before a
    process is started.
    """
    args = _validate_command(command)
    timeout_value = _validate_timeout(timeout)
    output_limit = _validate_max_output_bytes(max_output_bytes)
    resolved_cwd = None if cwd is None else str(Path(cwd))
    proc: subprocess.Popen[bytes] | None = None

    with tempfile.TemporaryFile(mode="w+b") as stdout_capture, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_capture:
        try:
            proc = subprocess.Popen(
                list(args),
                cwd=resolved_cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout_capture,
                stderr=stderr_capture,
                env=None if env is None else dict(env),
                start_new_session=os.name != "nt",
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            return BoundedProcessResult(
                args=args,
                returncode=None,
                stdout="",
                stderr="",
                execution_error=str(exc)[:500],
            )

        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout_value)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(proc)
            returncode = proc.returncode
        finally:
            if proc.poll() is None:
                terminate_process_tree(proc)

        stdout, stdout_truncated = _read_bounded(
            stdout_capture,
            max_bytes=output_limit,
            preserve_tail=preserve_stdout_tail,
        )
        stderr, stderr_truncated = _read_bounded(
            stderr_capture,
            max_bytes=output_limit,
            preserve_tail=preserve_stderr_tail,
        )
        return BoundedProcessResult(
            args=args,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
