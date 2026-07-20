from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.neural_evidence_frontier import (
    SCHEMA_VERSION,
    _calibrate_override,
    _gated_metrics,
    _guarded_metrics,
    _mcnemar_exact,
    _metrics,
    _wilson_upper,
    render_svg,
    verify_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _row(
    index: int,
    *,
    lexical_top: int,
    neural_top: int,
    gold: int,
    margin: float,
    partition: str = "calibration",
) -> dict:
    lexical_order = [lexical_top, 1 - lexical_top]
    neural_order = [neural_top, 1 - neural_top]
    return {
        "trial_index": index,
        "gold_index": gold,
        "lexical_order": lexical_order,
        "neural_order": neural_order,
        "lexical_top": lexical_top,
        "neural_top": neural_top,
        "lexical_correct": lexical_top == gold,
        "neural_correct": neural_top == gold,
        "neural_margin": margin,
        "partition": partition,
        "gated_top": lexical_top,
        "gated_correct": lexical_top == gold,
    }


def test_wilson_bound_is_conservative_and_shrinks_with_evidence() -> None:
    assert _wilson_upper(0, 10) > 0.20
    assert _wilson_upper(0, 100) < 0.04
    assert _wilson_upper(10, 100) > 0.10


def test_calibration_requires_enough_low_risk_noninferior_overrides() -> None:
    clean = [
        _row(index, lexical_top=0, neural_top=1, gold=1, margin=0.2)
        for index in range(50)
    ]

    calibration = _calibrate_override(
        clean, max_error_upper=0.10, minimum_overrides=40
    )

    assert calibration["eligible"] is True
    assert calibration["threshold"] == pytest.approx(0.2)
    assert calibration["override_errors"] == 0

    risky = [
        _row(
            index,
            lexical_top=0,
            neural_top=1,
            gold=0 if index < 10 else 1,
            margin=0.2,
        )
        for index in range(50)
    ]
    rejected = _calibrate_override(
        risky, max_error_upper=0.10, minimum_overrides=40
    )
    assert rejected["eligible"] is False


def test_mcnemar_exact_is_paired_and_symmetric() -> None:
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(12, 1) == _mcnemar_exact(1, 12)
    assert _mcnemar_exact(12, 1) < 0.05


def test_verifier_recomputes_orders_metrics_and_headline_gate() -> None:
    rows = [
        _row(
            0,
            lexical_top=0,
            neural_top=1,
            gold=1,
            margin=0.4,
            partition="test",
        ),
        _row(
            1,
            lexical_top=1,
            neural_top=1,
            gold=1,
            margin=0.3,
            partition="test",
        ),
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {"candidates_per_trial": 2},
        "test_metrics": {
            "lexical_bm25": _metrics(rows, "lexical"),
            "local_transformer": _metrics(rows, "neural"),
            "risk_gated": _gated_metrics(rows),
            "dual_channel_guard": _guarded_metrics(
                rows, candidates_per_trial=2
            ),
            "paired": {
                "transformer_only_correct": 1,
                "lexical_only_correct": 0,
                "mcnemar_exact_p": 1.0,
            },
        },
        "headline_eligible": False,
        "model": {
            "id": "snapshot-id",
            "fingerprint_sha256": "f" * 64,
            "path_disclosed": False,
        },
        "trials": rows,
    }

    verify_report(report)

    tampered = copy.deepcopy(report)
    tampered["test_metrics"]["local_transformer"]["top1_correct"] = 0
    with pytest.raises(ValueError, match="metrics"):
        verify_report(tampered)


def test_share_card_is_bound_to_verified_trial_evidence() -> None:
    report_path = ROOT / "benchmarks" / "results" / "neural_evidence_frontier.json"
    card_path = ROOT / "docs" / "assets" / "neural_evidence_frontier.svg"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    rendered = render_svg(report)

    assert "Keep your agent. Give it a Context OS." in rendered
    assert "Read. Remember. Trust. Recover. Spend wisely. Learn from outcomes." in rendered
    assert "16" in rendered
    assert "1.02" in rendered
    assert "298 / 300" in rendered
    assert "ENTROLY EVIDENCE LAB" not in rendered
    assert "NO BREAKTHROUGH CLAIM" not in rendered
    assert "Generated answers and production savings were not measured." in rendered
    assert card_path.read_text(encoding="utf-8") == rendered


def test_public_story_keeps_the_negative_result_and_provenance_attached() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    evidence_page = (
        ROOT / "docs" / "benchmarks" / "neural-evidence-frontier.md"
    ).read_text(encoding="utf-8")

    for claim in (
        "Keep your agent. Give it a Context OS.",
        "One measured job of the Context OS",
        "Frontier models reason. OpenClaw and Hermes run agents.",
        "**297 of 300**",
        "**293 of 300**",
        "**298 of 300**",
        "**1.02 of 16 passages**",
        "`p=0.21875`",
        "this experiment measures retrieval",
    ):
        assert claim in readme

    for provenance in (
        "sentence-transformers/all-MiniLM-L6-v2",
        "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "3efc859f2086261cc877fff243cddd2e4532e55e4157747861f705cd36b05a13",
        "0dc83f1f7759d7ace58cfc2d7ae19380473452f1",
    ):
        assert provenance in evidence_page

    assert "It does not measure generated-answer quality" in evidence_page
    assert "This audits the committed experiment; it does not rerun" in evidence_page
