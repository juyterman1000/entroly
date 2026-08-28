"""Beliefs must appear because the repository was indexed, not because the user
was told to compile them.

The dashboard shipped an empty panel reading "No beliefs yet -- run
compile_beliefs to seed the vault". Indexing had already walked the tree, so
the product knew the next step and asked someone else to take it.

The risks of doing it automatically are delay, repetition and failure, so each
is pinned here: it must not block indexing, must skip an unchanged tree, and
must never turn a vault problem into an indexing problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entroly import belief_autoseed


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("ENTROLY_VAULT", raising=False)
    belief_autoseed.reset_for_tests()
    yield
    belief_autoseed.reset_for_tests()


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj" / "src"
    root.mkdir(parents=True)
    (root / "auth.py").write_text(
        "import hashlib\n\n\n"
        "class SessionStore:\n"
        "    '''Holds sessions.'''\n"
        "    def __init__(self):\n"
        "        self.tokens = {}\n\n\n"
        "def hash_password(pw, salt):\n"
        "    '''Hash a password with a salt.'''\n"
        "    return hashlib.sha256((salt + pw).encode()).hexdigest()\n",
        encoding="utf-8",
    )
    return tmp_path / "proj"


class TestSeeding:
    def test_beliefs_are_written_without_a_user_command(self, tmp_path):
        result = belief_autoseed.compile_now(_project(tmp_path))

        assert result["status"] == "compiled"
        assert result["beliefs_written"] > 0
        vault = tmp_path / "state" / "vault" / "beliefs"
        assert list(vault.glob("*.md")), "the vault panel would still be empty"

    def test_an_unchanged_tree_is_not_recompiled(self, tmp_path):
        project = _project(tmp_path)
        belief_autoseed.compile_now(project)
        second = belief_autoseed.compile_now(project)

        assert second["status"] == "skipped", (
            "opening a repository repeatedly must compile once, not once per open"
        )

    def test_a_changed_tree_is_recompiled(self, tmp_path):
        project = _project(tmp_path)
        belief_autoseed.compile_now(project)
        (project / "src" / "billing.py").write_text(
            "def charge(cents):\n    '''Charge in cents.'''\n    return cents\n",
            encoding="utf-8")

        assert belief_autoseed.compile_now(project)["status"] == "compiled", (
            "skipping must be based on the tree, not on having ever run"
        )

    def test_the_marker_records_what_was_compiled(self, tmp_path):
        belief_autoseed.compile_now(_project(tmp_path))
        marker = json.loads(
            (tmp_path / "state" / "vault" / "autoseed.json").read_text(encoding="utf-8"))

        assert marker["signature"]
        assert marker["files_processed"] >= 1


class TestFailsOpen:
    def test_a_broken_vault_returns_an_error_rather_than_raising(
        self, tmp_path, monkeypatch
    ):
        project = _project(tmp_path)

        def explode(*_a, **_k):
            raise OSError("vault is read-only")

        monkeypatch.setattr(
            "entroly.vault.VaultManager.ensure_structure", explode, raising=False)

        result = belief_autoseed.compile_now(project)
        assert result["status"] == "error", (
            "a vault problem must be reported, not raised into the caller"
        )

    def test_start_autoseed_is_idempotent_per_directory(self, tmp_path):
        project = _project(tmp_path)
        assert belief_autoseed.start_autoseed(project) is True
        assert belief_autoseed.start_autoseed(project) is False, (
            "repeated indexing in one session must not stack compilations"
        )

    def test_it_can_be_turned_off(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ENTROLY_BELIEF_AUTOSEED", "0")
        assert belief_autoseed.autoseed_enabled() is False
        assert belief_autoseed.start_autoseed(_project(tmp_path)) is False


class TestIndexingIntegration:
    def test_indexing_starts_the_seeder(self, tmp_path, monkeypatch):
        """The wiring, not the compiler: indexing must trigger seeding."""
        project = _project(tmp_path)
        calls: list[str] = []
        monkeypatch.setattr(
            belief_autoseed, "start_autoseed",
            lambda directory=None, max_files=400: calls.append(str(directory)) or True)

        from entroly import auto_index as module

        monkeypatch.setattr(module, "_auto_index",
                            lambda *_a, **_k: {"status": "indexed", "files_indexed": 1})
        module.auto_index(engine=object(), project_dir=str(project))

        assert calls, "indexing completed without seeding beliefs"

    def test_a_seeder_failure_does_not_break_indexing(self, tmp_path, monkeypatch):
        from entroly import auto_index as module

        monkeypatch.setattr(
            belief_autoseed, "start_autoseed",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(module, "_auto_index",
                            lambda *_a, **_k: {"status": "indexed", "files_indexed": 3})

        result = module.auto_index(engine=object(), project_dir=str(tmp_path))
        assert result["status"] == "indexed", (
            "a belief failure must cost a panel, not the index"
        )
