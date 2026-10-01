"""`entroly init` must count the files the indexer will actually index.

`init` is the first command a new user runs, and it ends by telling them how
much of their project Entroly will pick up. It counted with `_git_ls_files`
alone:

    from entroly.auto_index import _git_ls_files, _should_index
    files = _git_ls_files(os.getcwd())
    indexable = [f for f in files if _should_index(f)]
    print(f"Entroly will auto-index {len(indexable)} files on first run")

`auto_index` does not. It falls back to a filesystem walk when git discovery
returns nothing, in both of the places it discovers files. So in a directory
that is not a git repository -- a new user trying the tool on a folder before
`git init`, which is a normal thing to do -- the two disagree completely.

Measured on a five-file project with no `.git`:

    _git_ls_files   ->  0 files,  0 indexable   <- what init reported
    _walk_fallback  ->  5 files,  5 indexable   <- what auto_index would use

"Entroly will auto-index 0 files on first run" is both false and discouraging,
at the exact moment a user decides whether the tool works on their project.

The cause is duplication: the git-then-walk decision existed three times, and
the copy in `cli.py` was missing its second half. These tests pin the shared
helper and the agreement, so a fourth copy cannot drift the same way.
"""
from __future__ import annotations

import subprocess

import pytest

from entroly.auto_index import (
    _git_ls_files,
    _should_index,
    _walk_fallback,
    discover_project_files,
)


@pytest.fixture()
def project(tmp_path):
    """A small project with no git repository."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        "def login(user, password):\n    return verify(user, password)\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "billing.py").write_text(
        "def charge(account, amount):\n    return gateway.submit(account, amount)\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# demo\n", encoding="utf-8")
    return tmp_path


def _indexable(paths):
    return [p for p in paths if _should_index(p)]


# ── The defect ───────────────────────────────────────────────────────


def test_a_non_git_project_discovers_its_files(project):
    """The case a new user hits before running `git init`."""
    files, discovery = discover_project_files(str(project))

    assert discovery == "walk", f"expected the walk fallback, got {discovery!r}"
    assert _indexable(files), (
        "a project with source files discovered nothing, so `init` would tell "
        "the user Entroly indexes 0 files"
    )


def test_the_helper_agrees_with_what_init_would_report(project):
    """`init`'s count and the indexer's discovery must be the same number.

    This is the assertion that would have failed before the fix: `init` used
    `_git_ls_files` alone, which returns nothing here.
    """
    helper_files, _ = discover_project_files(str(project))
    git_only = _git_ls_files(str(project))

    assert not git_only, "fixture is no longer a non-git project"
    assert len(_indexable(helper_files)) > len(_indexable(git_only)), (
        "the helper found no more than git-only discovery, so the fallback is "
        "not engaged and the original defect would still be present"
    )


def test_the_helper_matches_the_walk_fallback_when_git_is_empty(project):
    helper_files, _ = discover_project_files(str(project))

    assert sorted(helper_files) == sorted(_walk_fallback(str(project)))


# ── Guard: git repositories must keep using git ──────────────────────


def test_a_git_project_still_uses_git_discovery(project):
    """Without this the fix could always walk, losing .gitignore handling.

    `_git_ls_files` respects `.gitignore`; the walk fallback is the less precise
    path and must stay the fallback.
    """
    try:
        subprocess.run(
            ["git", "init", "-q"], cwd=project, check=True, capture_output=True,
            timeout=60,
        )
        subprocess.run(
            ["git", "add", "-A"], cwd=project, check=True, capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")

    if not _git_ls_files(str(project)):  # pragma: no cover - git present but odd
        pytest.skip("git produced no tracked files in the fixture")

    files, discovery = discover_project_files(str(project))

    assert discovery == "git"
    assert _indexable(files)


def test_an_empty_directory_reports_nothing_rather_than_failing(tmp_path):
    """Zero is the honest answer here, and must not raise."""
    files, discovery = discover_project_files(str(tmp_path))

    assert files == []
    assert discovery == "walk"
