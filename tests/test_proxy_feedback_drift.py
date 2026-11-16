from math import prod
from random import Random

import pytest

from entroly.proxy_feedback import ImplicitFeedbackTracker, _SequentialConfusionMonitor


def send(tracker, confusion, count):
    for i in range(count):
        tracker.record_assessment(-0.5 if i % 100 < confusion else 0.3)


def test_drift_has_frozen_reference_and_requires_two_complete_windows():
    tracker = ImplicitFeedbackTracker()
    send(tracker, 20, 2048)
    baseline = tracker.stats()["drift_detector"]["baseline_confusion_rate"]
    assert baseline is not None
    send(tracker, 20, 8192)
    assert tracker.quality_trend() == "stable"
    send(tracker, 36, 1024)
    assert tracker.quality_trend() == "stable"
    send(tracker, 36, 1024)
    assert tracker.quality_trend() == "declining"
    assert tracker.stats()["drift_detector"]["baseline_confusion_rate"] == baseline
    send(tracker, 20, 2048)
    assert tracker.quality_trend() == "improving"
    send(tracker, 20, 1024)
    assert tracker.quality_trend() == "stable"


def test_drift_warmup_does_not_raise_a_decline_for_a_steady_nonzero_baseline():
    tracker = ImplicitFeedbackTracker()
    send(tracker, 35, 1024)
    assert tracker.stats()["drift_detector"]["baseline_confusion_rate"] is None
    send(tracker, 35, 3072)
    assert tracker.stats()["drift_detector"]["baseline_ready"] is True
    assert tracker.quality_trend() == "stable"


def test_sequential_drift_matches_explicit_restart_mixture_with_ties():
    monitor = _SequentialConfusionMonitor(seed=4)
    rng = Random(4)
    scores = []
    bets = [[], [], []]
    for confused in (False, False, True, True, False, True):
        monitor.update(confused)
        score = int(not confused)
        scores.append(score)
        rank = (sum(s < score for s in scores) + rng.random() * scores.count(score)) / len(scores)
        for factors, tail in zip(bets, (0.05, 0.20, 0.50)):
            factors.append(0.8 + (0.2 / tail if rank < tail else 0.0))
        expected = 2048 / (2048 + len(scores)) + sum(
            2048 / ((2048 + k) * (2049 + k)) * prod(factors[k:])
            for factors in bets
            for k in range(len(scores))
        ) / len(bets)
        assert monitor.to_dict()["e_value"] == pytest.approx(expected)


def test_sequential_drift_is_opt_in_and_alarm_leaves_frozen_monitor_in_control():
    assert "sequential_drift_shadow" not in ImplicitFeedbackTracker().stats()
    tracker = ImplicitFeedbackTracker(sequential_drift=True)
    tracker._sequential_drift = _SequentialConfusionMonitor(seed=0)
    source = Random(1000)
    for _ in range(4096):
        tracker.record_assessment(-0.5 if source.random() < 0.20 else 0.3)
    assert tracker.stats()["sequential_drift_shadow"]["first_alarm_at"] is None

    for _ in range(1024):
        tracker.record_assessment(-0.5 if source.random() < 0.38 else 0.3)
    shadow = tracker.stats()["sequential_drift_shadow"]
    assert 4096 < shadow["first_alarm_at"] <= 5120
    assert shadow["e_value"] >= 20
    assert shadow["production_authority"] is False
    assert tracker.quality_trend() == "stable"

    for _ in range(1024):
        tracker.record_assessment(-0.5 if source.random() < 0.38 else 0.3)
    assert tracker.stats()["sequential_drift_shadow"] == shadow
    assert tracker.quality_trend() == "declining"
