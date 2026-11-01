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


def _run(
    env_path: str,
    extra_env: dict | None = None,
    args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PATH"] = env_path
    env.update(extra_env or {})
    return subprocess.run(
        ["node", str(LAUNCHER), *(args or [])],
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
    assert marker.read_text(encoding="utf-8") == "--from entroly entroly serve"


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
    assert marker.read_text(encoding="utf-8") == "npx -y entroly@latest serve"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake PATH executables need .cmd shims on Windows",
)
def test_hook_prefers_installed_entroly_and_forwards_arguments(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "invoked.txt"
    executable = fake_bin / "entroly"
    executable.write_text(
        '#!/bin/sh\nprintf "%s" "$*" > "{marker}"\nexit 0\n'.format(marker=marker),
        encoding="utf-8",
    )
    executable.chmod(0o755)

    node_dir = str(Path(shutil.which("node")).parent)
    args = ["activation", "hook", "--host", "claude-code"]
    result = _run(f"{fake_bin}{os.pathsep}{node_dir}", args=args)

    assert result.returncode == 0
    assert marker.read_text(encoding="utf-8") == " ".join(args)


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


def _bin_with_only_node(tmp_path: Path) -> Path:
    """A PATH directory holding node and nothing else: no npx, no uvx."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    node_exe = Path(shutil.which("node"))
    try:
        (fake_bin / node_exe.name).symlink_to(node_exe)
    except OSError:
        shutil.copy2(node_exe, fake_bin / node_exe.name)
    return fake_bin


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake PATH executables need .cmd shims on Windows",
)
def test_a_broken_install_reports_its_own_error_not_a_missing_runner(
    tmp_path: Path,
) -> None:
    """An installed Entroly that fails must not be reported as absent.

    Claude Code shows only the first line of a failing hook's stderr. The
    launcher printed "no runner found. Install uv ... or Node, or run `pip
    install -U entroly`" whenever every candidate exited non-zero, so a user
    whose `entroly` was installed but broken was told to install it, and the
    real error was discarded.
    """
    fake_bin = _bin_with_only_node(tmp_path)
    broken = fake_bin / "entroly"
    broken.write_text(
        "#!/bin/sh\n"
        "echo 'Traceback (most recent call last):' >&2\n"
        "echo 'ImportError: entroly_core 1.0.87 is older than the required 1.0.88' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    broken.chmod(0o755)

    env = dict(os.environ, PATH=str(fake_bin))
    result = subprocess.run(
        ["node", str(LAUNCHER), "activation", "hook", "--host", "claude-code"],
        input="{}",
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert result.returncode == 1
    first_line = result.stderr.splitlines()[0]
    assert "ImportError: entroly_core 1.0.87 is older than the required 1.0.88" in first_line
    assert "no runner found" not in result.stderr.lower()
    assert "exit 1" in result.stderr


def test_a_missing_install_does_not_ask_for_the_node_that_is_running_it(
    tmp_path: Path,
) -> None:
    result = _run(str(_bin_with_only_node(tmp_path)))

    assert result.returncode == 1
    first_line = result.stderr.splitlines()[0].lower()
    assert "pip install -u entroly" in first_line
    assert "node" not in first_line


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="fake PATH executables need .cmd shims on Windows",
)
def test_a_server_that_exits_non_zero_is_reported_with_its_status(tmp_path: Path) -> None:
    fake_bin = _bin_with_only_node(tmp_path)
    failing = fake_bin / "entroly"
    failing.write_text("#!/bin/sh\nexit 3\n", encoding="utf-8")
    failing.chmod(0o755)

    result = _run(str(fake_bin))

    assert result.returncode == 1
    assert "exit 3" in result.stderr
    assert "no runner found" not in result.stderr.lower()
