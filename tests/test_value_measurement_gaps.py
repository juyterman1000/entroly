"""A receipt must say what it failed to measure.

Telemetry is dropped rather than allowed to block a user's request when the
value-tracker lock is contended. That trade is correct and stays. What was
wrong is that the drop was invisible: measured on this machine with eight
concurrent writers, 10% of events were lost (360 of 400 recorded) and the
receipt still presented its totals as though they were exact.

CLAUDE.md requires that "omitted evidence" be inspectable. An event the tracker
failed to record is omitted evidence, and a total that silently excludes it is
a floor being presented as a count.

Drops are counted in memory -- persisting them at the moment of failure would
need the very lock that just timed out -- and folded into the persisted totals
by the next successful mutation.
"""

from __future__ import annotations

import pytest

from entroly.value_tracker import ValueTracker


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    return ValueTracker(tmp_path)


class TestDropsAreCounted:
    def test_a_dropped_event_is_recorded_as_a_gap(self, tracker, monkeypatch):
        def refuse(*_a, **_k):
            raise TimeoutError("timed out acquiring value-tracker lock")

        monkeypatch.setattr(tracker, "_record_with_lock", refuse)
        tracker.record(tokens_saved=250, source="mcp")

        assert tracker._pending_dropped == 1
        assert tracker._pending_dropped_tokens == 250

    def test_a_drop_never_raises_into_the_caller(self, tracker, monkeypatch):
        """Telemetry must not become the reason a request fails."""
        monkeypatch.setattr(
            tracker, "_record_with_lock",
            lambda *_a, **_k: (_ for _ in ()).throw(TimeoutError("busy")))
        tracker.record(tokens_saved=100, source="mcp")  # must not raise

    def test_pending_drops_reach_the_persisted_total(self, tracker, monkeypatch):
        calls = {"n": 0}
        real = tracker._record_with_lock

        def fail_twice(*a, **k):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise TimeoutError("busy")
            return real(*a, **k)

        monkeypatch.setattr(tracker, "_record_with_lock", fail_twice)
        tracker.record(tokens_saved=100, source="mcp")   # dropped
        tracker.record(tokens_saved=100, source="mcp")   # dropped
        tracker.record(tokens_saved=100, source="mcp")   # succeeds, folds them in

        lifetime = tracker.get_lifetime()
        assert lifetime["events_dropped"] == 2
        assert lifetime["tokens_dropped"] == 200
        assert tracker._pending_dropped == 0, "folded drops must not double-count"

    def test_every_attempt_is_either_recorded_or_counted(self, tracker, monkeypatch):
        """The invariant: recorded + dropped == attempted. No silent loss."""
        calls = {"n": 0}
        real = tracker._record_with_lock

        def flaky(*a, **k):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise TimeoutError("busy")
            return real(*a, **k)

        monkeypatch.setattr(tracker, "_record_with_lock", flaky)
        attempts = 30
        for _ in range(attempts):
            tracker.record(tokens_saved=10, source="mcp")
        tracker.record(tokens_saved=0, source="mcp")     # flush trailing drops

        lifetime = tracker.get_lifetime()
        accounted = lifetime["local_operations"] + lifetime["events_dropped"]
        assert accounted == attempts + 1, (
            f"{attempts + 1} attempted, {accounted} accounted for -- "
            "an unaccounted event is an unmeasured saving"
        )


class TestTheReceiptDisclosesIt:
    def test_a_clean_receipt_says_its_totals_are_complete(self, tracker):
        tracker.record(tokens_saved=100, source="mcp")
        gaps = tracker.get_value_receipt()["measurement_gaps"]

        assert gaps["events_dropped"] == 0
        assert gaps["totals_are"] == "complete"

    def test_a_lossy_receipt_says_its_totals_are_a_floor(self, tracker, monkeypatch):
        calls = {"n": 0}
        real = tracker._record_with_lock

        def fail_first(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("busy")
            return real(*a, **k)

        monkeypatch.setattr(tracker, "_record_with_lock", fail_first)
        tracker.record(tokens_saved=500, source="mcp")
        tracker.record(tokens_saved=100, source="mcp")

        gaps = tracker.get_value_receipt()["measurement_gaps"]
        assert gaps["events_dropped"] == 1
        assert gaps["tokens_dropped"] == 500
        assert gaps["totals_are"] == "a floor", (
            "presenting an incomplete total as exact is the defect this pins"
        )
        assert "contention" in gaps["reason"]
