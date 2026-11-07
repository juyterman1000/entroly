"""Entroly's per-project state must not end up in the user's commits.

`entroly go` (and the MCP server and proxy it starts) fills
`<project>/.entroly/vault/` with one generated summary per source file: 172
files and 1.1 MB on pallets/click. Nothing told git to ignore them, so the
first `git add -A` after onboarding committed summaries of the user's code.

The directory now ignores itself the way `.pytest_cache` and `.ruff_cache`
do, with a `.gitignore` containing `*`. The user's own `.gitignore` is never
edited, and a user who wants the vault in git deletes that one file.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from entroly.project_footprint import mark_local_state_dir, project_footprint_notice
from entroly.vault import VaultConfig, VaultManager

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def _ignored(repo: Path, rel: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-q", "--", rel], check=False
    )
    return result.returncode == 0


def test_vault_under_project_entroly_ignores_itself(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    VaultManager(VaultConfig(base_path=str(repo / ".entroly" / "vault"))).ensure_structure()
    belief = repo / ".entroly" / "vault" / "beliefs" / "src_app.md"
    belief.write_text("generated summary", encoding="utf-8")

    assert _ignored(repo, ".entroly/vault/beliefs/src_app.md")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert ".entroly" not in status
    assert project_footprint_notice(repo) is None  # nothing left to warn about


def test_an_existing_gitignore_inside_entroly_is_left_alone(tmp_path: Path) -> None:
    state = tmp_path / ".entroly"
    state.mkdir()
    (state / ".gitignore").write_text("vault/media/\n", encoding="utf-8")

    mark_local_state_dir(state)

    assert (state / ".gitignore").read_text(encoding="utf-8") == "vault/media/\n"


def test_only_a_directory_named_entroly_is_marked(tmp_path: Path) -> None:
    elsewhere = tmp_path / "custom-vault-home"
    VaultManager(VaultConfig(base_path=str(elsewhere / "vault"))).ensure_structure()

    assert not (elsewhere / ".gitignore").exists()


def test_marking_never_raises_on_a_read_only_directory(tmp_path: Path) -> None:
    state = tmp_path / ".entroly"
    state.mkdir()
    state.chmod(0o500)
    try:
        mark_local_state_dir(state)  # best effort: must not raise
    finally:
        state.chmod(0o700)


def test_autoseed_marker_write_marks_the_directory(tmp_path: Path) -> None:
    from entroly import belief_autoseed

    repo = _repo(tmp_path)
    vault = repo / ".entroly" / "vault"
    belief_autoseed._write_marker(vault, repo, {"signature": "x"})

    assert _ignored(repo, ".entroly/vault/" + belief_autoseed._MARKER_NAME)
