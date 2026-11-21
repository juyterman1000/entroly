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
import os
import shutil
import threading
import time
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
        project = _project(tmp_path)
        belief_autoseed.compile_now(project)
        markers = json.loads(
            (tmp_path / "state" / "vault" / "autoseed.json").read_text(encoding="utf-8"))

        entry = markers[str(project.resolve())]
        assert entry["signature"]
        assert entry["files_processed"] >= 1

    def test_two_projects_do_not_overwrite_each_other(self, tmp_path):
        """A single unkeyed signature made every alternation recompile both."""
        first = _project(tmp_path)
        second = tmp_path / "other" / "src"
        second.mkdir(parents=True)
        (second / "billing.py").write_text(
            "def charge(cents):\n    'Charge.'\n    return cents\n", encoding="utf-8")

        assert belief_autoseed.compile_now(first)["status"] == "compiled"
        assert belief_autoseed.compile_now(tmp_path / "other")["status"] == "compiled"
        assert belief_autoseed.compile_now(first)["status"] == "skipped", (
            "returning to a project already compiled must not recompile it"
        )

    def test_a_file_beyond_the_compile_cap_still_invalidates(self, tmp_path):
        """The signature covers the whole tree; only compilation is capped.

        Breaking the signature loop at max_files made every file sorting after
        the cap invisible, so editing one never triggered recompilation again
        and its beliefs stayed stale permanently.
        """
        project = _project(tmp_path)
        source = project / "src"
        for index in range(12):
            (source / f"mod_{index:03d}.py").write_text(
                f"def f{index}():\n    'Doc.'\n    return {index}\n", encoding="utf-8")

        before = belief_autoseed._tree_signature(project.resolve(), max_files=2)
        (source / "mod_011.py").write_text(
            "def f11():\n    'Changed.'\n    return 999\n", encoding="utf-8")
        after = belief_autoseed._tree_signature(project.resolve(), max_files=2)

        assert before != after, (
            "a file past the compile cap must still invalidate the signature"
        )


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


#: Every one of these holds a file with a source suffix, so only directory
#: pruning can keep it out of the fingerprint -- the suffix filter cannot.
_VENDORED_FIXTURE = {
    "node_modules": "index.js",
    ".git": "hook.py",
    ".venv": "site.py",
    "target": "build.rs",
    "__pycache__": "cached.py",
    ".entroly": "state.py",
}

#: Bypasses the Windows MAX_PATH limit, so a tree can be *created* that a
#: plain-path enumeration of the same tree cannot open.
_EXTENDED_PREFIX = "\\\\?\\"


def _tree_with_vendored_dirs(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "main.py").write_text(
        "def main():\n    '''Entry point.'''\n    return 0\n", encoding="utf-8")
    (root / "src" / "app.py").write_text(
        "def app():\n    '''Nested source.'''\n    return 1\n", encoding="utf-8")
    for directory, filename in _VENDORED_FIXTURE.items():
        nested = root / directory / "pkg"
        nested.mkdir(parents=True)
        (nested / filename).write_text("x = 1\n", encoding="utf-8")
    return root


def _remove_tree(root: Path) -> None:
    """Delete a tree that may hold paths Windows cannot open by plain name."""
    target = _EXTENDED_PREFIX + str(root) if os.name == "nt" else str(root)
    shutil.rmtree(target, ignore_errors=True)


def _unopenable_subtree(parent: Path) -> Path:
    """Build a directory that exists but cannot be enumerated by plain path.

    This is the vendored pnpm store that broke autoseed on a real checkout: the
    directories are created through the extended-length prefix, so they are
    really there, but every ordinary ``scandir`` of them fails with
    ``FileNotFoundError [WinError 3]``.
    """
    parent.mkdir(parents=True, exist_ok=True)
    deep = parent
    while len(str(deep)) < 300:
        deep = deep / ("d" * 40)
        os.mkdir(_EXTENDED_PREFIX + str(deep))
    with open(_EXTENDED_PREFIX + str(deep / "vendored.js"), "w",
              encoding="utf-8") as handle:
        handle.write("export const vendored = 1;\n")
    try:
        os.listdir(str(deep))
    except OSError:
        return deep
    pytest.skip("long paths are enabled here, so MAX_PATH cannot be reproduced")


def _wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestVendoredTreesArePruned:
    """The exclusion list has to be applied before descending, not after.

    It used to read ``sorted(directory.rglob("*"))`` with the exclusion applied
    to each yielded path. ``rglob`` must enumerate a directory to yield from it
    and ``sorted`` drains the generator before any filter runs, so the
    exclusion excluded nothing from the walk -- only from the digest.
    """

    def test_a_vendored_file_is_never_a_candidate(self, tmp_path):
        root = _tree_with_vendored_dirs(tmp_path)

        found, unreadable = belief_autoseed._candidate_files(root)
        names = {path.name for path in found}

        assert "main.py" in names, (
            "a file directly under the root must still be found"
        )
        assert "app.py" in names, (
            "pruning must not stop the walk descending into real source trees, "
            "or absence below would prove nothing"
        )
        for directory, filename in _VENDORED_FIXTURE.items():
            assert filename not in names, (
                f"{directory}/pkg/{filename} reached the fingerprint; the "
                "exclusion ran after enumeration instead of pruning the walk"
            )
        assert unreadable == 0, "nothing in this tree is unreadable"

    def test_no_candidate_lies_under_a_pruned_directory(self, tmp_path):
        root = _tree_with_vendored_dirs(tmp_path)

        found, _ = belief_autoseed._candidate_files(root)
        offenders = [
            path for path in found
            if set(path.parts) & set(belief_autoseed._PRUNED_DIR_NAMES)
        ]

        assert found, "the walk returned nothing at all"
        assert not offenders, (
            f"the walk descended into a pruned tree and yielded {offenders}"
        )


@pytest.mark.skipif(
    os.name != "nt",
    reason="the MAX_PATH enumeration failure is specific to Windows")
class TestAnUnopenableDirectoryIsNotFatal:
    """A directory that cannot be enumerated must cost a stale belief, not the
    whole feature.

    The ``FileNotFoundError`` came out of ``rglob`` itself -- past the
    ``except OSError`` that only guards ``path.stat()`` -- and killed the
    autoseed thread.
    """

    def test_a_vendored_tree_too_deep_to_open_is_never_descended(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        source = root / "main.py"
        source.write_text("def main():\n    '''Entry.'''\n    return 0\n",
                          encoding="utf-8")
        try:
            _unopenable_subtree(root / "node_modules" / "pkg")

            before = belief_autoseed._tree_signature(root, max_files=400)
            source.write_text(
                "def main():\n    '''Changed.'''\n    return 1\n", encoding="utf-8")
            after = belief_autoseed._tree_signature(root, max_files=400)

            assert before != after, (
                "the fingerprint must keep tracking real source next to a "
                "vendored tree that cannot be opened"
            )
            _, unreadable = belief_autoseed._candidate_files(root)
            assert unreadable == 0, (
                "a pruned tree must never be opened, so it cannot fail to open"
            )
        finally:
            _remove_tree(root)

    def test_an_unopenable_source_directory_is_counted_not_raised(self, tmp_path):
        root = tmp_path / "repo"
        root.mkdir()
        (root / "main.py").write_text(
            "def main():\n    '''Entry.'''\n    return 0\n", encoding="utf-8")
        try:
            _unopenable_subtree(root / "src" / "generated")

            signature = belief_autoseed._tree_signature(root, max_files=400)
            assert signature, (
                "an enumeration failure must not be raised out of the "
                "fingerprint; it is reported and the rest of the tree covered"
            )

            found, unreadable = belief_autoseed._candidate_files(root)
            assert unreadable >= 1, (
                "an enumeration failure must be counted, not silently dropped"
            )
            assert any(path.name == "main.py" for path in found), (
                "one unreadable directory must not cost the rest of the tree"
            )
        finally:
            _remove_tree(root)


class TestABackgroundFailureDoesNotLatch:
    """``_started.add(target)`` happens before the thread launches.

    So a thread that died without releasing it left autoseed disabled for the
    life of the process -- the failure latched, and the only symptom was one
    traceback on the MCP server's stderr at startup.
    """

    @staticmethod
    def _fail_in_the_thread(monkeypatch):
        """Make the background compile raise, and capture thread deaths."""
        escaped: list[object] = []
        monkeypatch.setattr(threading, "excepthook", escaped.append)
        entered: list[threading.Thread] = []

        def explode(*_a, **_k):
            entered.append(threading.current_thread())
            raise RuntimeError("the tree could not be enumerated")

        monkeypatch.setattr(belief_autoseed, "compile_now", explode)
        return entered, escaped

    @staticmethod
    def _await_attempt(entered, index: int) -> None:
        assert _wait_for(lambda: len(entered) > index), (
            "the autoseed thread never reached the compile step"
        )
        entered[index].join(timeout=15)
        assert not entered[index].is_alive(), "the autoseed thread never finished"

    def test_a_failure_does_not_take_the_thread_down_with_a_traceback(
        self, tmp_path, monkeypatch
    ):
        project = _project(tmp_path)
        entered, escaped = self._fail_in_the_thread(monkeypatch)

        assert belief_autoseed.start_autoseed(project) is True
        self._await_attempt(entered, 0)

        assert not escaped, (
            "the autoseed thread died of an unhandled "
            f"{getattr(escaped[0], 'exc_type', None)} and printed a raw "
            "traceback on the server's stderr"
        )

    def test_a_failure_releases_the_directory_so_it_is_retried(
        self, tmp_path, monkeypatch
    ):
        project = _project(tmp_path)
        entered, _escaped = self._fail_in_the_thread(monkeypatch)

        assert belief_autoseed.start_autoseed(project) is True
        self._await_attempt(entered, 0)

        assert belief_autoseed.start_autoseed(project) is True, (
            "the failed directory stayed latched in _started, so autoseed "
            "never ran again for the life of the process"
        )
        self._await_attempt(entered, 1)

    def test_the_latch_still_holds_while_the_thread_is_working(
        self, tmp_path, monkeypatch
    ):
        """Releasing on failure must not make start_autoseed non-idempotent."""
        project = _project(tmp_path)
        release = threading.Event()
        started = threading.Event()

        def block(*_a, **_k):
            started.set()
            release.wait(15)
            return {"status": "compiled", "beliefs_written": 0, "files_processed": 0}

        monkeypatch.setattr(belief_autoseed, "compile_now", block)
        try:
            assert belief_autoseed.start_autoseed(project) is True
            assert started.wait(15), "the autoseed thread never started"
            assert belief_autoseed.start_autoseed(project) is False, (
                "a compile already in flight must not be stacked"
            )
        finally:
            release.set()
