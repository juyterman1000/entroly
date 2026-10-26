"""The first-run local checks must say what they leave in the user's repo.

Reproduced on a fresh HOME in a clean click checkout: after
`entroly verify-claims` and `entroly simulate`, `git status` showed
`?? .entroly/` and `?? .entroly_verification.json` (simulate also wrote 11
belief files under .entroly/vault/beliefs/). Neither command said so; only
"Report saved: .entroly_verification.json" was printed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess

import pytest

import entroly.cli as cli
import entroly.value_tracker as value_tracker
from entroly import verify_claims
from entroly.value_tracker import ValueTracker

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "app.py").write_text(
        "def main_module_structure():\n    return 'explain the main module'\n",
        encoding="utf-8",
    )
    if shutil.which("git"):
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ENTROLY_DIR", raising=False)
    monkeypatch.chdir(root)
    monkeypatch.setattr(value_tracker, "_tracker", ValueTracker(tmp_path / "state"))
    return root


def test_verify_claims_names_what_it_left_in_the_project(project, capsys):
    verify_claims.run()
    out = capsys.readouterr().out

    assert (project / ".entroly_verification.json").exists()
    line = next((x for x in out.splitlines() if "Entroly writes local state" in x), "")
    assert ".entroly_verification.json" in line, out
    assert ".entroly/" in line  # the autoseed vault, even if still being written
    assert ".gitignore" in line


@needs_git
def test_no_notice_once_the_user_ignores_it(project, capsys):
    (project / ".gitignore").write_text(".entroly/\n.entroly_verification.json\n", encoding="utf-8")

    verify_claims.run()

    assert "Entroly writes local state" not in capsys.readouterr().out


def test_simulate_names_the_project_vault(project, capsys):
    (project / ".entroly").mkdir(exist_ok=True)  # created by any earlier run

    cli.cmd_simulate(
        argparse.Namespace(
            query=["explain the main module"], budget=None, baseline=None,
            max_files=100, json=False,
        )
    )
    out = capsys.readouterr().out

    assert "Entroly writes local state into this project: .entroly/" in out, out
