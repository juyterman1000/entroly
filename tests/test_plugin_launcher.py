from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "entroly-plugin-launch.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="the launcher is a Node script; node is absent in this environment",
)


def _run(env_path: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = env_path
    env.update(extra_env or {})
    return subprocess.run(
        ["node", str(LAUNCHER)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def test_reports_one_actionable_line_when_no_runner_exists(tmp_path: Path) -> None:
    # Create a PATH with only node, excluding npm/npx which may ship with node.
    # This ensures we actually test the "no runner found" path.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    node_exe = shutil.which("node")
    node_copy = fake_bin / Path(node_exe).name
    shutil.copy2(node_exe, node_copy)

    result = _run(str(fake_bin))

    assert result.returncode == 1
    # The failure mode this guards is a plugin that starts nothing and says
    # nothing, which is indistinguishable from a broken install.
    assert "entroly" in result.stderr.lower()
    assert "uv" in result.stderr.lower()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake PATH executables need .cmd shims on Windows",
)
def test_prefers_uvx_and_passes_the_expected_arguments(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "invoked.txt"
    uvx = fake_bin / "uvx"
    uvx.write_text(
        '#!/bin/sh\nprintf "%s" "$*" > "{marker}"\nexit 0\n'.format(marker=marker),
        encoding="utf-8",
    )
    uvx.chmod(0o755)

    node_dir = str(Path(shutil.which("node")).parent)
    result = _run(f"{fake_bin}{os.pathsep}{node_dir}")

    assert result.returncode == 0
    assert marker.read_text(encoding="utf-8") == "--from entroly entroly"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake PATH executables need .cmd shims on Windows",
)
def test_falls_through_to_npx_when_uvx_is_absent(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "invoked.txt"
    npx = fake_bin / "npx"
    npx.write_text(
        '#!/bin/sh\nprintf "npx %s" "$*" > "{marker}"\nexit 0\n'.format(marker=marker),
        encoding="utf-8",
    )
    npx.chmod(0o755)

    node_dir = str(Path(shutil.which("node")).parent)
    result = _run(f"{fake_bin}{os.pathsep}{node_dir}")

    assert result.returncode == 0
    assert marker.read_text(encoding="utf-8") == "npx -y entroly@latest"


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="exercises the Windows .cmd invocation path",
)
def test_invokes_a_cmd_shim_on_windows(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "invoked.txt"
    (fake_bin / "uvx.cmd").write_text(
        f'@echo off\r\n> "{marker}" echo %*\r\n', encoding="utf-8"
    )

    node_dir = str(Path(shutil.which("node")).parent)
    result = _run(f"{fake_bin}{os.pathsep}{node_dir}")

    assert result.returncode == 0
    assert "entroly" in marker.read_text(encoding="utf-8")
