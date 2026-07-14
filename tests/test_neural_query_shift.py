from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.neural_query_shift import _sentence_ranges, verify_report

ROOT = Path(__file__).resolve().parents[1]


def test_sentence_ranges_preserve_exact_source_offsets() -> None:
    context = "First fact. Second fact? Final fact!"

    ranges = _sentence_ranges(context)

    assert [text for _, _, text in ranges] == [
        "First fact.",
        "Second fact?",
        "Final fact!",
    ]
    assert all(context[start:end] == text for start, end, text in ranges)


@pytest.mark.parametrize(
    "filename",
    (
        "neural_query_shift_2x.json",
        "neural_query_shift.json",
        "neural_query_shift_8x.json",
    ),
)
def test_committed_query_shift_artifacts_verify(filename: str) -> None:
    path = ROOT / "benchmarks" / "results" / filename
    report = json.loads(path.read_text(encoding="utf-8"))

    verify_report(report)


def test_query_shift_verifier_rejects_tampered_trial() -> None:
    path = ROOT / "benchmarks" / "results" / "neural_query_shift.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(report)
    tampered["trials"][0]["q2_retained_after_rehydration"] = not tampered["trials"][0][
        "q2_retained_after_rehydration"
    ]

    with pytest.raises(ValueError, match="does not match trial rows"):
        verify_report(tampered)


def test_query_shift_verifier_rejects_promoted_or_forged_report() -> None:
    path = ROOT / "benchmarks" / "results" / "neural_query_shift.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    promoted = copy.deepcopy(report)
    promoted["headline_eligible"] = True
    with pytest.raises(ValueError, match="headline_eligible=false"):
        verify_report(promoted)

    forged = copy.deepcopy(report)
    forged["metrics"]["selection_reasons"] = {"neural_override": len(report["trials"])}
    with pytest.raises(ValueError, match="selection reason metrics"):
        verify_report(forged)
