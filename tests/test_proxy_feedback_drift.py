from entroly.proxy_feedback import ImplicitFeedbackTracker


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
