"""Takeover happens because an agent opened the repository, not because it asked.

`work_resume` reconstructed work state correctly, but something had to call it.
An agent that never learned the tool existed got no takeover, which made the
feature semi-automatic in the only sense that matters to a user who did not read
the docs.

Also covers the modification timeline. Detection previously happened only when
something asked -- a resume, an observation, a verification. A point-in-time
refresh shows what a file looks like now; it cannot show that the file changed
at 14:02 and again at 14:07.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from entroly import work_graph_session as session
from entroly.work_graph_watcher import WorkspaceModificationWatcher


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.test")
    _git(root, "config", "user.name", "t")
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture(autouse=True)
def _clean_session_state():
    session.reset_for_tests()
    yield
    session.stop_session_watchers()
    session.reset_for_tests()


class TestWatcherTimeline:
    def test_records_transitions_not_just_current_state(self):
        samples = [
            {"a.py": "d1"},
            {"a.py": "d2"},
            {"a.py": "d2", "b.py": "d3"},
            {"b.py": "d3"},
        ]
        watcher = WorkspaceModificationWatcher(lambda: samples.pop(0), interval_seconds=60)
        watcher._sample(seed=True)   # consumes the first
        watcher.poll_once()
        watcher.poll_once()
        watcher.poll_once()

        observed = [(m["path"], m["change"]) for m in watcher.modifications()]
        assert observed == [
            ("a.py", "modified"),
            ("b.py", "appeared"),
            ("a.py", "vanished"),
        ]

    def test_seeding_does_not_announce_existing_state_as_new(self):
        watcher = WorkspaceModificationWatcher(lambda: {"a.py": "d1"}, interval_seconds=60)
        watcher._sample(seed=True)
        assert watcher.modifications() == [], (
            "a file that was already modified when watching began is not a new "
            "modification"
        )

    def test_a_sampler_failure_is_counted_not_swallowed(self):
        def broken():
            raise OSError("disk went away")

        watcher = WorkspaceModificationWatcher(broken, interval_seconds=60)
        watcher.poll_once()
        # A gap in the timeline must be visible; a watcher that silently stops
        # recording looks identical to a repository where nothing changed.
        assert watcher.status()["sampler_errors"] == 1

    def test_drain_clears_so_the_timeline_is_consumed_once(self):
        samples = [{"a.py": "d1"}, {"a.py": "d2"}]
        watcher = WorkspaceModificationWatcher(lambda: samples.pop(0), interval_seconds=60)
        watcher._sample(seed=True)
        watcher.poll_once()

        assert len(watcher.drain()) == 1
        assert watcher.drain() == []

    def test_capacity_exhaustion_is_reported(self):
        digests = iter(range(100))
        watcher = WorkspaceModificationWatcher(
            lambda: {"a.py": str(next(digests))}, interval_seconds=60, max_records=3
        )
        watcher._sample(seed=True)
        for _ in range(10):
            watcher.poll_once()

        status = watcher.status()
        assert status["at_capacity"] is True, (
            "a dropped record is a hole in the timeline and must be visible"
        )


class TestAutomaticTakeover:
    def test_clean_repository_is_a_null_control_not_an_error(self, repo, monkeypatch):
        """A fresh clone must not greet its agent with a refusal.

        `resume` raises "no unfinished workstream" for a clean checkout, which
        is correct fail-closed behaviour. Recording that as an error made it
        look broken and returned early, so the watcher never started on exactly
        the repository where a timeline is most useful.
        """
        monkeypatch.setenv("ENTROLY_DIR", str(repo.parent / "state"))
        monkeypatch.setenv("ENTROLY_SOURCE", str(repo))

        summary = session.start_session(force=True)

        assert summary["recovered"] is False
        assert summary["gate_armed"] is False
        assert summary.get("error") is None
        assert summary.get("null_control")

    def test_unfinished_work_arms_the_gate_without_the_agent_asking(
        self, repo, monkeypatch
    ):
        monkeypatch.setenv("ENTROLY_DIR", str(repo.parent / "state"))
        monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
        from entroly.work_graph_mcp import work_claim

        work_claim(
            project=str(repo), agent_id="a", task_title="harden",
            task_id="t1", scope_paths=["app.py"],
        )
        (repo / "app.py").write_text("x = 2  # interrupted\n", encoding="utf-8")

        # No work_resume call anywhere in this test. That is the point.
        summary = session.start_session(force=True)

        assert summary["recovered"] is True
        assert summary["gate_armed"] is True
        assert summary["acknowledgement"]["token"].startswith("recovery:")

        blocked = work_claim(
            project=str(repo), agent_id="b", task_title="t2",
            task_id="t2", scope_paths=["app.py"],
        )
        assert blocked["status"] == "error"

    def test_takeover_is_idempotent_per_repository(self, repo, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(repo.parent / "state"))
        monkeypatch.setenv("ENTROLY_SOURCE", str(repo))

        session.start_session()
        again = session.start_session()
        assert again["reused"] is True, (
            "a second agent action must not demand acknowledgement of the same "
            "facts twice"
        )

    def test_autostart_can_be_disabled(self, repo, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(repo.parent / "state"))
        monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
        monkeypatch.setenv("ENTROLY_WORK_GRAPH_AUTOSTART", "0")

        summary = session.start_session(force=True)
        assert summary["attempted"] is False
        assert summary["gate_armed"] is False

    def test_watcher_start_failure_is_reported_not_silent(self, repo, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(repo.parent / "state"))
        monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
        monkeypatch.setenv("ENTROLY_WORK_GRAPH_WATCH", "0")

        summary = session.start_session(force=True)
        assert summary["watcher_started"] is False
        # "disabled" and "tried and broke" must not look identical.
        assert "disabled" in summary.get("watcher_note", "")

    @pytest.mark.timeout(120)
    def test_watcher_runs_on_a_clean_repository(self, repo, monkeypatch):
        monkeypatch.setenv("ENTROLY_DIR", str(repo.parent / "state"))
        monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
        monkeypatch.setenv("ENTROLY_WORK_GRAPH_WATCH", "1")

        summary = session.start_session(force=True)
        assert summary["watcher_started"] is True

        watcher = session.session_watcher("")
        assert watcher is not None
        (repo / "later.py").write_text("y = 1\n", encoding="utf-8")
        watcher.poll_once()

        assert any(
            m["path"] == "later.py" and m["change"] == "appeared"
            for m in watcher.modifications()
        )


class TestTakeoverCost:
    """Takeover may be slow once. It may not be slow repeatedly.

    Measured on this repository (~4.7k files): the cold call costs ~3.6s, of
    which ~380ms is Git observation and content digests and the rest is the
    Rust store. That cost is deliberate -- the trust gate has to be armed
    before the first mutating call, so deferring takeover to a background
    thread would risk `work_claim` arriving while nothing was armed yet, which
    is the exact fail-open the gate exists to prevent.

    What must not regress is the per-call cost. An absolute bound on the cold
    path would be machine-dependent and flaky; the contract worth pinning is
    that takeover is paid once.
    """

    def test_repeated_takeover_is_effectively_free(self, repo, monkeypatch):
        import time

        monkeypatch.setenv("ENTROLY_DIR", str(repo.parent / "state"))
        monkeypatch.setenv("ENTROLY_SOURCE", str(repo))

        session.start_session()          # pay the cold cost once

        started = time.perf_counter()
        for _ in range(200):
            session.start_session()
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert elapsed_ms < 50, (
            f"200 cached takeovers took {elapsed_ms:.1f}ms; takeover is being "
            "re-run per tool call rather than memoised"
        )

    def test_disabled_takeover_costs_nothing(self, repo, monkeypatch):
        import time

        monkeypatch.setenv("ENTROLY_DIR", str(repo.parent / "state"))
        monkeypatch.setenv("ENTROLY_SOURCE", str(repo))
        monkeypatch.setenv("ENTROLY_WORK_GRAPH_AUTOSTART", "0")

        started = time.perf_counter()
        summary = session.start_session(force=True)
        elapsed_ms = (time.perf_counter() - started) * 1000

        assert summary["attempted"] is False
        assert elapsed_ms < 100, (
            f"opting out still cost {elapsed_ms:.1f}ms; the opt-out must skip "
            "observation entirely, not merely discard its result"
        )
