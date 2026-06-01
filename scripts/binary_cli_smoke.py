#!/usr/bin/env python3
"""Smoke-test the shipped entroly-rs CLI surface."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(binary: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *args],
        input=input_text,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: binary_cli_smoke.py PATH_TO_ENTROLY_RS")

    binary = Path(sys.argv[1]).resolve()
    require(binary.is_file(), f"binary does not exist: {binary}")

    version = run(binary, "--version")
    require(version.returncode == 0, f"--version failed: {version.stderr}")
    require(
        re.fullmatch(r"entroly-rs \d+\.\d+\.\d+\s*", version.stdout) is not None,
        f"unexpected --version output: {version.stdout!r}",
    )

    help_result = run(binary, "--help")
    require(help_result.returncode == 0, f"--help failed: {help_result.stderr}")
    require("entroly-rs compress" in help_result.stdout, "help omits compress command")
    require("entroly-rs proxy" in help_result.stdout, "help omits proxy command")

    sample = "\n".join(
        f"Section {index}: Entroly compresses repeated context while preserving useful details."
        for index in range(120)
    )
    stdin_result = run(binary, "compress", "--budget", "80", input_text=sample)
    require(stdin_result.returncode == 0, f"stdin compression failed: {stdin_result.stderr}")
    require(stdin_result.stdout.strip(), "stdin compression returned empty output")
    require("budget 80" in stdin_result.stderr, "stdin compression omitted summary")

    with tempfile.TemporaryDirectory() as temp_dir:
        first = Path(temp_dir, "first.txt")
        second = Path(temp_dir, "second.txt")
        first.write_text(sample, encoding="utf-8")
        second.write_text("second input", encoding="utf-8")

        file_result = run(binary, "compress", "--budget", "80", str(first))
        require(file_result.returncode == 0, f"file compression failed: {file_result.stderr}")
        require(file_result.stdout.strip(), "file compression returned empty output")

        duplicate = run(binary, "compress", str(first), str(second))
        require(duplicate.returncode == 64, "multiple input files must be rejected")

    bad_budget = run(binary, "compress", "--budget", "0", input_text=sample)
    require(bad_budget.returncode == 64, "zero budget must be rejected")

    unknown = run(binary, "not-a-command")
    require(unknown.returncode == 64, "unknown command must be rejected")

    print("PASS: entroly-rs CLI smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
