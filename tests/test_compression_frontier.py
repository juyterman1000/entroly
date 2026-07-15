from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("tiktoken", reason="compression frontier requires benchmark extras")

from benchmarks.compression_frontier import (
    Trial,
    _canonical_sha256,
    _sha256,
    _superiority_gate,
    analyze,
    render_markdown,
    render_svg,
    verify_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _adapter(system: str, trials: list[Trial], output: str) -> dict[str, object]:
    return {
        "system": system,
        "package": system,
        "version": "test",
        "algorithm": "fixture",
        "config": {},
        "runtime": {"python": "test"},
        "results": [
            {
                "trial_id": trial.trial_id,
                "target_ratio": 0.5,
                "output_text": output,
                "output_sha256": _sha256(output),
                "deterministic": True,
                "latency_ms": {"samples": [1.0], "p50": 1.0, "p95": 1.0},
                "native_metrics": {},
            }
            for trial in trials
        ],
    }


def test_frontier_report_verifier_detects_output_tampering() -> None:
    content = " ".join(["irrelevant context"] * 120) + " gold-answer"
    trials = [
        Trial(
            trial_id=f"trial-{index}",
            example_id=f"example-{index}",
            question="What is the gold answer?",
            answers=("gold-answer",),
            content=content,
            gold_document=0,
            document_ids=(f"example-{index}",),
        )
        for index in range(2)
    ]
    report = analyze(
        trials=trials,
        adapters=(
            _adapter("entroly", trials, "gold-answer"),
            _adapter("headroom", trials, "unrelated"),
        ),
        ratios=(0.5,),
        protocol={"seed": 1, "dataset_fingerprint": "test"},
        downstream_options=None,
    )

    verify_report(report)
    assert report["aggregates"]["entroly"]["0.5"]["answer_retention"] == 1.0
    assert report["aggregates"]["headroom"]["0.5"]["answer_retention"] == 0.0
    assert report["superiority_gate"]["passed"] is False

    tampered = copy.deepcopy(report)
    tampered["rows"][0]["output_text"] = "changed"
    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        verify_report(tampered)


def test_superiority_gate_requires_every_statistical_and_downstream_guard() -> None:
    ratios = (0.5, 0.25, 0.125)
    aggregates: dict[str, dict[str, object]] = {"entroly": {}, "headroom": {}}
    paired: dict[str, object] = {}
    for ratio in ratios:
        key = f"{ratio:g}"
        aggregates["entroly"][key] = {
            "passed": True,
            "target_attainment": 1.0,
            "answer_retention": 1.0,
        }
        aggregates["headroom"][key] = {
            "passed": True,
            "target_attainment": 1.0,
            "answer_retention": 0.0,
        }
        paired[key] = {"mcnemar_exact_two_sided_p": 0.0001}
    downstream = {
        "aggregates": {
            "raw": {"exact_match": 0.75, "mean_token_f1": 0.8, "errors": 0},
            "entroly": {"exact_match": 0.75, "mean_token_f1": 0.8, "errors": 0},
            "headroom": {"exact_match": 0.25, "mean_token_f1": 0.3, "errors": 0},
        }
    }

    gate = _superiority_gate(
        aggregates=aggregates,
        paired_statistics=paired,
        ratios=ratios,
        downstream=downstream,
    )

    assert gate["passed"] is True
    assert not gate["reasons"]


def test_committed_frontier_artifacts_and_readme_are_in_sync() -> None:
    result_path = ROOT / "benchmarks" / "results" / "compression_frontier.json"
    report = json.loads(result_path.read_text(encoding="utf-8"))

    verify_report(report)
    assert report["superiority_gate"]["passed"] is True
    assert (ROOT / "benchmarks" / "results" / "compression_frontier.md").read_text(
        encoding="utf-8"
    ) == render_markdown(report)
    assert (ROOT / "docs" / "assets" / "compression_frontier.svg").read_text(
        encoding="utf-8"
    ) == render_svg(report)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for expected in (
        "**95.0%**",
        "**93.3%**",
        "**88.3%**",
        "The published Headroom 0.31.0 baseline retained **1.7%**",
        "Headroom 0.31.0 baseline at 12.5%",
        "Using Headroom today?",
        "1.0.59 source candidate",
    ):
        assert expected in readme

    tampered = copy.deepcopy(report)
    tampered["downstream"]["rows"][0]["exact_match"] = not tampered["downstream"][
        "rows"
    ][0]["exact_match"]
    tampered["payload_sha256"] = _canonical_sha256(
        {key: value for key, value in tampered.items() if key != "payload_sha256"}
    )
    with pytest.raises(ValueError, match="downstream exact_match mismatch"):
        verify_report(tampered)
