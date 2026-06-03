"""Regression test for learning-loop startup failure telemetry."""

import sys
from types import SimpleNamespace

from entroly.daemon import EntrolyDaemon


def test_learning_startup_failure_is_exposed_in_stats(monkeypatch):
    class FeedbackJournal:
        def __init__(self, _checkpoint_dir):
            raise RuntimeError("journal unavailable")

    monkeypatch.setitem(
        sys.modules,
        "entroly.autotune",
        SimpleNamespace(
            DreamingLoop=object,
            FeedbackJournal=FeedbackJournal,
            TaskProfileOptimizer=object,
        ),
    )

    daemon = EntrolyDaemon(enable_proxy=False, enable_mcp=False)
    daemon._start_learning_loop()

    stats = daemon.get_learning_stats()
    loop = stats["learning_loop"]
    assert loop["status"] == "error"
    assert loop["last_error"] == "journal unavailable"
    assert loop["worker"] == {
        "running": False,
        "started_at": None,
        "stopped_at": None,
        "exit_reason": "start_error",
    }
