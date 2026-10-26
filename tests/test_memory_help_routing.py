"""`entroly memory` must be discoverable and its help must work.

Reproduced on a fresh HOME: `entroly memory --help` and
`entroly memory remember --help` exited 2 with "argument command: invalid
choice: 'memory'", and `entroly --help` did not list `memory`, although
`entroly memory remember ...` works. The launcher sent any argv containing
--help to the main CLI before its `memory` route
(entroly/_docker_launcher.py launch), and the main parser has no `memory`.
"""

from __future__ import annotations

import subprocess
import sys

_LAUNCH = (
    "import sys; sys.argv = ['entroly', *sys.argv[1:]]; "
    "from entroly.docker_launcher_safe import launch; launch()"
)


def _entroly(*args: str, home) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _LAUNCH, *args],
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "HOME": str(home), "NO_COLOR": "1"},
        check=False,
    )


def test_memory_help_reaches_the_memory_cli(tmp_path):
    result = _entroly("memory", "--help", home=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "invalid choice" not in result.stderr
    assert "usage: entroly memory" in result.stdout


def test_memory_subcommand_help_reaches_the_memory_cli(tmp_path):
    result = _entroly("memory", "remember", "--help", home=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "usage: entroly memory remember" in result.stdout


def test_top_level_help_lists_memory(tmp_path):
    result = _entroly("--help", home=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "memory" in result.stdout


def test_main_cli_dispatches_memory(tmp_path):
    # `entroly.cli.main` is also an entry (python -m entroly.cli); it must not
    # parse `memory` and then print the top-level help instead of running it.
    code = (
        "import sys; sys.argv = ['entroly', 'memory', 'recall', '--help']; "
        "from entroly.cli import main; main()"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**__import__("os").environ, "HOME": str(tmp_path), "NO_COLOR": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: entroly memory recall" in result.stdout
