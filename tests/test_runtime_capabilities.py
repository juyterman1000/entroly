from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from entroly.runtime_capabilities import runtime_capabilities


def _walk(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_report_is_stable_conservative_and_privacy_safe() -> None:
    report = runtime_capabilities()

    assert report["schema_version"] == "entroly.runtime-capabilities.v1"
    assert report["engine"]["active"] in {"native", "pure-python"}
    assert report["claims"] == {
        "provider_connectivity_verified": False,
        "benchmark_leadership_implied": False,
        "production_readiness_implied": False,
    }
    assert report["assurance"]["secure_recovery_store"] is True
    assert report["operations"]["unwrap"] is True

    forbidden = {"path", "error", "exception", "traceback"}
    assert not any(str(key).casefold() in forbidden for key, _ in _walk(report))


def test_json_cli_is_offline_machine_readable_and_does_not_leak_paths(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-m", "entroly.cli", "capabilities", "--json"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["schema_version"] == "entroly.runtime-capabilities.v1"
    assert report["claims"]["provider_connectivity_verified"] is False
    assert str(tmp_path) not in result.stdout
    assert "Traceback" not in result.stderr
