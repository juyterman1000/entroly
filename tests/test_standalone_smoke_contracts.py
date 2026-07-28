"""Turn legacy standalone smoke scripts into fail-closed pytest contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(path: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "ENTROLY_DIR": str(tmp_path / "entroly-state"),
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
        "PYTHONUTF8": "1",
    }
    return subprocess.run(
        [sys.executable, path],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=120,
        check=False,
    )


def test_cogops_smoke_output_cannot_report_failure_with_exit_zero(tmp_path: Path) -> None:
    result = _run("tests/test_cogops_smoke.py", tmp_path)
    combined = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, combined
    assert "ALL COGOPS DATA PLANE TESTS PASSED" in combined, combined
    assert "WARNING:" not in combined, combined
    assert "FAIL:" not in combined, combined


def test_federation_crossagent_smoke_fails_closed(tmp_path: Path) -> None:
    result = _run("tests/test_federation_crossagent.py", tmp_path)
    combined = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, combined
    assert "ALL TESTS PASSED" in combined, combined
    assert "[FAIL]" not in combined, combined
    assert "need attention" not in combined, combined
