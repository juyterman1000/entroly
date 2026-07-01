"""Pytest wrappers for the standalone native validation programs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("script_name", ["test_integration.py", "test_brutal.py"])
def test_native_validation_script(script_name: str) -> None:
    script = Path(__file__).with_name(script_name)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
