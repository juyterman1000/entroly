"""`entroly doctor` must not call a normal install "a different installation".

Reproduced with a regular (non-editable) `pip install` into a fresh venv:
doctor printed "! `entroly` on PATH is a different installation than the
imported package" and told the user to run `pip install -e .`, although the
PATH launcher and the imported package were the same install. The old test
compared directory ancestry (site-packages is never an ancestor of the
venv), so it fired on every venv, `--user`, system and editable-outside-venv
install. README sends users to `entroly doctor` when ranking looks wrong.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

import entroly.cli as cli

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX console-script shebangs"
)

_REAL_WHICH = shutil.which


def _write_exec(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _console_script(path: Path, interpreter: str) -> Path:
    # The exact shape pip writes for `[project.scripts] entroly = ...`.
    return _write_exec(
        path,
        f"#!{interpreter}\n"
        "import sys\n"
        "from entroly.docker_launcher_safe import launch\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(launch())\n",
    )


def _doctor_output(monkeypatch, capsys, launcher: Path) -> str:
    def which(name, *args, **kwargs):
        if name == "entroly":
            return str(launcher)
        return _REAL_WHICH(name, *args, **kwargs)

    monkeypatch.setattr(cli.shutil, "which", which)
    cli.cmd_doctor(argparse.Namespace(json_output=False, port=21099))
    return capsys.readouterr().out


def test_doctor_accepts_a_launcher_that_imports_the_running_package(
    tmp_path, monkeypatch, capsys
):
    # A venv laid out the standard way: bin/entroly beside lib/.../site-packages.
    launcher = _console_script(tmp_path / "venv" / "bin" / "entroly", sys.executable)

    out = _doctor_output(monkeypatch, capsys, launcher)

    assert "different installation" not in out, out


def test_doctor_still_flags_a_launcher_that_imports_another_copy(
    tmp_path, monkeypatch, capsys
):
    other = tmp_path / "other-install"
    (other / "entroly").mkdir(parents=True)
    (other / "entroly" / "__init__.py").write_text("", encoding="utf-8")
    interpreter = _write_exec(
        tmp_path / "other-python",
        f'#!/bin/sh\nPYTHONPATH="{other}" exec "{sys.executable}" "$@"\n',
    )
    launcher = _console_script(tmp_path / "venv" / "bin" / "entroly", str(interpreter))

    out = _doctor_output(monkeypatch, capsys, launcher)

    assert "different installation" in out, out
    assert str(launcher) in out


def test_pip_trampoline_shebang_is_understood(tmp_path, monkeypatch, capsys):
    # pip writes a /bin/sh trampoline when the interpreter path has spaces.
    # Give the whole environment a spaced path, the way a user's venv has one.
    # Symlinking only the interpreter would strand it outside its venv: with no
    # pyvenv.cfg beside it, Python resolves the link to the base interpreter,
    # which cannot import an editable install (CI's `.venv`), so the check
    # would be judging a different Python than the one the launcher names.
    executable = Path(sys.executable)
    spaced = tmp_path / "with space"
    os.symlink(executable.parent.parent, spaced, target_is_directory=True)
    python = spaced / executable.parent.name / executable.name
    launcher = _write_exec(
        tmp_path / "venv" / "bin" / "entroly",
        "#!/bin/sh\n"
        f"'''exec' \"{python}\" \"$0\" \"$@\"\n"
        "' '''\n"
        "import sys\n"
        "from entroly.docker_launcher_safe import launch\n"
        "sys.exit(launch())\n",
    )

    out = _doctor_output(monkeypatch, capsys, launcher)

    assert "different installation" not in out, out
