"""Regression tests for daemon learning-worker lifecycle telemetry."""

import sys
import time
from types import SimpleNamespace

from entroly.daemon import EntrolyDaemon


def _wait_for(predicate, timeout_s: float = 1.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def test_learning_worker_lifecycle_is_exposed_in_stats(monkeypatch):
    class FeedbackJournal:
        def __init__(self, _checkpoint_dir):
            pass

        def count(self):
            return 0

        def prune(self, max_age):
            return None

        def stats(self):
            return {"episodes": 0}

    class TaskProfileOptimizer:
        def __init__(self, _journal):
            self._profiles = {}

        def optimize_all(self):
            return {}

    class DreamingLoop:
        def __init__(self, **_kwargs):
            pass

        def stats(self):
            return {"status": "idle"}

    monkeypatch.setitem(
        sys.modules,
        "entroly.autotune",
        SimpleNamespace(
            DreamingLoop=DreamingLoop,
            FeedbackJournal=FeedbackJournal,
            TaskProfileOptimizer=TaskProfileOptimizer,
        ),
    )

    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    initial = daemon.get_learning_stats()["learning_loop"]["worker"]
    assert initial == {
        "running": False,
        "started_at": None,
        "stopped_at": None,
        "exit_reason": "not_started",
    }

    daemon._start_learning_loop()
    worker = daemon._workers["learning"]
    _wait_for(lambda: daemon._learning_worker_running)

    active = daemon.get_learning_stats()["learning_loop"]["worker"]
    assert active["running"] is True
    assert active["started_at"] is not None
    assert active["stopped_at"] is None
    assert active["exit_reason"] is None

    daemon._shutdown.set()
    daemon._learning_wake.set()
    worker.join(timeout=1)

    stopped = daemon.get_learning_stats()["learning_loop"]["worker"]
    assert not worker.is_alive()
    assert stopped["running"] is False
    assert stopped["stopped_at"] is not None
    assert stopped["exit_reason"] == "shutdown"
