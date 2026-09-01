"""The receipt store belongs to a project, not to a current directory.

`DEFAULT_STORE` was the relative path `.entroly/receipts`, resolved against
whatever cwd the process happened to have. Measured by running the documented
flow: `entroly ingest .` at a project root, then `entroly select -q ...` from a
subdirectory of that same project, printed

    No Context Receipt index found at .entroly\\receipts\\index.json.
    Run `entroly ingest ./docs` first or pass `--docs ./docs`.

The index existed one directory up. The advice was also wrong -- re-running
ingest from the subdirectory would not have found it, it would have written a
second, unrelated index, leaving the project with two disagreeing sets of
evidence about what context was selected.

Every project tool a user already has -- git, cargo, npm, ruff -- searches
upward for its project root. The store now does the same, bounded by the
repository so it can never read some unrelated parent directory's evidence.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from entroly.context_receipts import store


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    (root / "src" / "deep").mkdir(parents=True)
    (root / ".git").mkdir()
    monkeypatch.chdir(root)
    return root


class TestUpwardDiscovery:
    def test_an_existing_store_is_found_from_a_subdirectory(self, project):
        (project / ".entroly" / "receipts").mkdir(parents=True)
        os.chdir(project / "src" / "deep")

        assert store.resolve_store() == project / ".entroly" / "receipts"
        assert store.default_index_path() == (
            project / ".entroly" / "receipts" / "index.json"
        )

    def test_the_store_is_the_same_from_every_directory_in_the_project(self, project):
        (project / ".entroly" / "receipts").mkdir(parents=True)

        seen = set()
        for directory in (project, project / "src", project / "src" / "deep"):
            os.chdir(directory)
            seen.add(store.resolve_store())
        assert len(seen) == 1, (
            "one project must have one store; a per-directory store is how the "
            "same project ends up with two disagreeing indexes"
        )

    def test_a_new_store_is_created_at_the_repository_root(self, project):
        """Not in whichever subdirectory the command happened to run from."""
        os.chdir(project / "src" / "deep")
        assert store.resolve_store() == project / ".entroly" / "receipts"


class TestTheWalkStaysInsideTheRepository:
    def test_an_unrelated_parent_store_is_never_adopted(self, tmp_path, monkeypatch):
        """Reading another project's evidence is worse than finding none."""
        outer = tmp_path / "outer"
        (outer / ".entroly" / "receipts").mkdir(parents=True)
        inner = outer / "inner"
        (inner / ".git").mkdir(parents=True)
        monkeypatch.chdir(inner)

        assert store.resolve_store() == inner / ".entroly" / "receipts"

    def test_a_store_at_the_repository_root_still_wins_over_the_boundary(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "repo"
        (root / ".git").mkdir(parents=True)
        (root / ".entroly" / "receipts").mkdir(parents=True)
        (root / "pkg").mkdir()
        monkeypatch.chdir(root / "pkg")

        assert store.resolve_store() == root / ".entroly" / "receipts"


class TestPathsAreNotFrozenAtImport:
    def test_resolution_follows_the_current_project(self, tmp_path, monkeypatch):
        """A module-level relative Path silently froze this to the import cwd."""
        first = tmp_path / "a"
        second = tmp_path / "b"
        for p in (first, second):
            (p / ".git").mkdir(parents=True)

        monkeypatch.chdir(first)
        a = store.resolve_store()
        monkeypatch.chdir(second)
        b = store.resolve_store()

        assert a != b
        assert a.is_absolute() and b.is_absolute()

    def test_the_latest_pointer_lives_in_the_store_it_describes(self, project):
        os.chdir(project / "src")
        assert store.latest_pointer_path().parent == store.resolve_store()


class TestTheLatestPointerResolvesFromAnywhere:
    def test_a_relative_pointer_is_anchored_to_its_own_store(self, project):
        """Pointers written by earlier versions recorded a cwd-relative path."""
        receipts = project / ".entroly" / "receipts"
        receipts.mkdir(parents=True)
        (receipts / "cr_abc.json").write_text("{}", encoding="utf-8")
        store.latest_pointer_path().write_text(
            str(Path(".entroly") / "receipts" / "cr_abc.json"), encoding="utf-8"
        )

        os.chdir(project / "src" / "deep")
        resolved = store.latest_receipt_path()
        assert resolved is not None and resolved.exists(), (
            "a pointer written at the project root must still resolve from a "
            "subdirectory, or `entroly receipt` cannot find the receipt it just "
            "wrote"
        )

    def test_an_absolute_pointer_is_returned_unchanged(self, project):
        receipts = project / ".entroly" / "receipts"
        receipts.mkdir(parents=True)
        target = receipts / "cr_xyz.json"
        target.write_text("{}", encoding="utf-8")
        store.latest_pointer_path().write_text(str(target), encoding="utf-8")

        assert store.latest_receipt_path() == target

    def test_an_empty_pointer_reports_nothing_rather_than_the_store_root(
        self, project
    ):
        (project / ".entroly" / "receipts").mkdir(parents=True)
        store.latest_pointer_path().write_text("   ", encoding="utf-8")

        assert store.latest_receipt_path() is None
