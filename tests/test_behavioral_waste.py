from __future__ import annotations

from entroly.behavioral_waste import BehavioralWasteDetector


def test_identical_tool_retry_and_repeated_error_are_reported_once() -> None:
    detector = BehavioralWasteDetector(repeat_threshold=3)
    findings = ()
    for index in range(3):
        findings = detector.observe(
            "conversation",
            tool_name="read_file",
            arguments={"path": "missing.py"},
            result_text=f"ERROR file not found attempt {index + 10}",
            input_tokens=100,
            observed_at=float(index),
        )
    kinds = {finding.kind for finding in findings}
    assert kinds == {"identical_tool_retry", "repeated_error"}
    assert all(finding.tier == "opportunity" for finding in findings)
    assert detector.observe(
        "conversation",
        tool_name="other",
        arguments={},
        result_text="ok",
        observed_at=4.0,
    ) == ()


def test_alternating_tool_loop_is_detected() -> None:
    detector = BehavioralWasteDetector(repeat_threshold=3)
    findings = ()
    for index, tool in enumerate(("search", "read", "search", "read")):
        findings = detector.observe(
            "loop", tool_name=tool, arguments={"q": tool}, observed_at=float(index)
        )
    assert any(finding.kind == "alternating_tool_loop" for finding in findings)


def test_model_switch_churn_is_advisory() -> None:
    detector = BehavioralWasteDetector(repeat_threshold=3)
    findings = ()
    for index, model in enumerate(("a", "b", "a", "b")):
        findings = detector.observe(
            "routing", model=model, input_tokens=50, observed_at=float(index)
        )
    churn = next(finding for finding in findings if finding.kind == "model_switch_churn")
    assert churn.occurrences == 3
    assert churn.estimated_wasted_tokens == 150
