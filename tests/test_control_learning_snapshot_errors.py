"""Regression tests for degraded control API learning snapshots."""

from entroly.dashboard import _control_learning_snapshot


class _State:
    learning_enabled = False
    autotune_enabled = True


def test_learning_snapshot_reports_aggregated_stats_failure():
    class Daemon:
        state = _State()

        def get_learning_stats(self):
            raise RuntimeError("stats unavailable")

        def get_learning_weights(self):
            return {"source": "defaults"}

    snap = _control_learning_snapshot(Daemon())

    assert snap["local_enabled"] is False
    assert snap["weights"] == {"source": "defaults"}
    assert snap["status"] == "degraded"
    assert snap["errors"] == [{
        "section": "learning_stats",
        "type": "RuntimeError",
        "message": "stats unavailable",
    }]


def test_learning_snapshot_isolates_legacy_weights_failure():
    class Daemon:
        state = _State()

        def get_learning_weights(self):
            raise RuntimeError("weights unavailable")

    snap = _control_learning_snapshot(Daemon())

    assert snap["local_enabled"] is False
    assert snap["weights"] == {
        "status": "error",
        "error": "weights unavailable",
    }
    assert snap["status"] == "degraded"
    assert snap["errors"] == [{
        "section": "learning_weights",
        "type": "RuntimeError",
        "message": "weights unavailable",
    }]
