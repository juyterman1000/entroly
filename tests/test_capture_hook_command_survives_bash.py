"""The RAVS capture hook command must survive the shell Claude Code runs it in.

`entroly ravs hook install --claude-code` writes this installation's absolute
`entroly` path into Claude Code's settings, unquoted. Claude Code runs hook
commands with bash -- Git Bash on Windows -- so:

* a path containing a space split into words. A venv under "my venv", or a
  Windows user folder such as C:\\Users\\Jane Doe, failed with "No such file or
  directory" on every Bash tool call;
* bash removed every unquoted backslash, so on Windows
  C:\\Python312\\Scripts\\entroly.exe ran as C:Python312Scriptsentroly.exe,
  a command that does not exist, for every user.

Claude Code's own issue tracker records both failures for its plugin hooks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from entroly.ravs import hooks

BASH = shutil.which("bash")


def _words(command: str) -> list[str]:
    """How bash splits a command line into words."""
    done = subprocess.run(
        [BASH, "-c", f"for w in {command}; do printf '%s\\n' \"$w\"; done"],
        capture_output=True, text=True, check=True,
    )
    return done.stdout.splitlines()


@pytest.mark.skipif(BASH is None or os.name == "nt", reason="needs a POSIX bash")
def test_the_hook_runs_from_a_path_with_a_space(tmp_path, monkeypatch) -> None:
    bindir = tmp_path / "my venv" / "bin"
    bindir.mkdir(parents=True)
    fake = bindir / "entroly"
    fake.write_text('#!/bin/sh\nprintf "ran %s\\n" "$*"\n', encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake) if name == "entroly" else None)

    command = hooks._capture_command()
    done = subprocess.run([BASH, "-c", command], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert done.stdout == "ran ravs capture --stdin --host claude-code --quiet\n"


@pytest.mark.skipif(BASH is None or os.name == "nt", reason="needs a POSIX bash")
def test_the_python_fallback_survives_a_space_too(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "my venv" / "bin" / "python3"))

    words = _words(hooks._capture_command())

    assert words[:3] == [str(tmp_path / "my venv" / "bin" / "python3"), "-m", "entroly.cli"]


@pytest.mark.skipif(BASH is None, reason="needs bash to check how Git Bash splits it")
@pytest.mark.parametrize(
    "path",
    [
        r"C:\Python312\Scripts\entroly.exe",
        r"C:\Users\Jane Doe\AppData\Local\Programs\Python\Python312\Scripts\entroly.exe",
    ],
)
def test_a_windows_path_reaches_git_bash_as_one_intact_word(path) -> None:
    word = hooks._hook_word(path, windows=True)

    assert _words(word) == [path.replace("\\", "/")]


def test_a_plain_posix_path_is_written_unchanged() -> None:
    """Registrations already on disk keep matching byte for byte."""
    assert hooks._hook_word("/home/u/.venv/bin/entroly", windows=False) == "/home/u/.venv/bin/entroly"


def test_a_quoted_registration_is_still_recognised(tmp_path, monkeypatch) -> None:
    bindir = tmp_path / "my venv" / "bin"
    bindir.mkdir(parents=True)
    fake = bindir / "entroly"
    fake.write_text("#!/bin/sh\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake) if name == "entroly" else None)
    monkeypatch.setattr(hooks, "_claude_code_settings_path", lambda: tmp_path / "settings.json")
    monkeypatch.chdir(tmp_path)  # no project-level registration in scope

    first = hooks.install_claude_code()
    again = hooks.install_claude_code()

    assert first.get("action") == "installed", first
    assert again.get("action") == "already_present", again
    assert hooks.uninstall_claude_code().get("action") == "uninstalled"


@pytest.mark.skipif(BASH is None or os.name == "nt", reason="needs a POSIX bash")
def test_reinstalling_repairs_a_registration_written_unquoted(tmp_path, monkeypatch) -> None:
    """The changelog tells affected users to re-run install; that must fix it."""
    import json

    bindir = tmp_path / "my venv" / "bin"
    bindir.mkdir(parents=True)
    fake = bindir / "entroly"
    fake.write_text('#!/bin/sh\nprintf "ran %s\\n" "$*"\n', encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(shutil, "which", lambda name: str(fake) if name == "entroly" else None)
    monkeypatch.chdir(tmp_path)
    settings = tmp_path / "settings.json"
    monkeypatch.setattr(hooks, "_claude_code_settings_path", lambda: settings)
    old = f"{fake} ravs capture --stdin --host claude-code --quiet"  # as 1.0.88 wrote it
    settings.write_text(json.dumps({"hooks": {event: [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": old}]}
    ] for event in ("PostToolUse", "PostToolUseFailure")}}), encoding="utf-8")

    outcome = hooks.install_claude_code()

    assert outcome.get("action") == "updated", outcome
    written = json.loads(settings.read_text(encoding="utf-8"))
    for event in ("PostToolUse", "PostToolUseFailure"):
        [entry] = written["hooks"][event]
        command = entry["hooks"][0]["command"]
        done = subprocess.run([BASH, "-c", command], capture_output=True, text=True)
        assert done.returncode == 0, (event, command, done.stderr)
