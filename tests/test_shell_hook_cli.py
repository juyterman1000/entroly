"""`entroly hook` must not leave a user's shell worse off.

Run in a clean HOME with published 1.0.88:

* `entroly hook install` appended a block to ~/.bashrc whose functions only
  set and unset an environment variable nothing reads, so it compressed
  nothing (the README says it "adds transparent compression to your shell").
  The bash block also ran `trap ... DEBUG`, replacing any DEBUG trap the user
  already had, which is how tools such as bash-preexec hook commands.
* It reported "Hook installed for ['/home/u/.bashrc']" and told the user to
  run "source ~/.['/home/u/.bashrc']rc".
* `entroly hook uninstall` failed with "uninstall_hook() takes 0 positional
  arguments but 1 was given", so the block could not be removed.
* `entroly hook status` printed the keys of its result dictionary:
  "installed installed / available not installed / any_installed installed".
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from entroly import shell_hook

ORIGINAL = 'export PATH=$HOME/bin:$PATH\ntrap "echo my-debug-trap" DEBUG\n'


def _entroly(home: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HOME": str(home),
        "NO_COLOR": "1",
        "ENTROLY_NO_DOCKER": "1",
        "ENTROLY_DISABLE_UPDATE_CHECK": "1",
    }
    return subprocess.run(
        [sys.executable, "-m", "entroly", "hook", *argv],
        env=env, capture_output=True, text=True, timeout=120,
    )


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    (tmp_path / ".config" / "fish").mkdir(parents=True)
    return tmp_path


def _old_install(config: Path, hook: str) -> None:
    """What `entroly hook install` used to append."""
    config.write_text(ORIGINAL + "\n" + hook, encoding="utf-8")


def test_uninstall_restores_each_file_byte_for_byte(home: Path) -> None:
    files = {
        home / ".bashrc": shell_hook.BASH_HOOK,
        home / ".zshrc": shell_hook.ZSH_HOOK,
        home / ".config" / "fish" / "config.fish": shell_hook.FISH_HOOK,
    }
    for path, hook in files.items():
        _old_install(path, hook)

    done = _entroly(home, "uninstall")

    assert done.returncode == 0, done.stdout + done.stderr
    for path in files:
        assert path.read_text(encoding="utf-8") == ORIGINAL, path.name
    assert "takes 0 positional arguments" not in done.stdout + done.stderr


def test_uninstall_honours_the_shell_filter(home: Path) -> None:
    _old_install(home / ".bashrc", shell_hook.BASH_HOOK)
    _old_install(home / ".zshrc", shell_hook.ZSH_HOOK)

    assert _entroly(home, "uninstall", "--shell", "zsh").returncode == 0

    assert (home / ".zshrc").read_text(encoding="utf-8") == ORIGINAL
    assert shell_hook.HOOK_MARKER_START in (home / ".bashrc").read_text(encoding="utf-8")


def test_install_changes_nothing_and_says_what_works(home: Path) -> None:
    rc = home / ".bashrc"
    rc.write_text(ORIGINAL, encoding="utf-8")

    done = _entroly(home, "install")

    assert rc.read_text(encoding="utf-8") == ORIGINAL, "the user's DEBUG trap must survive"
    assert done.returncode == 1
    assert "entroly shrink --" in done.stdout
    assert "['" not in done.stdout


def test_status_names_files_not_dictionary_keys(home: Path) -> None:
    (home / ".zshrc").write_text(ORIGINAL, encoding="utf-8")
    _old_install(home / ".bashrc", shell_hook.BASH_HOOK)

    done = _entroly(home, "status")

    assert done.returncode == 0
    assert ".bashrc" in done.stdout and ".zshrc" in done.stdout
    for key in ("available", "any_installed"):
        assert key not in done.stdout
