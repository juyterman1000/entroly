from __future__ import annotations

import pytest

from bench.benchmark_evidence import (
    ProviderUsage,
    SampleEvidence,
    exact_mcnemar,
    summarize,
    wilson_interval,
)


def record(sample_id: str, baseline: bool, treatment: bool) -> SampleEvidence:
    return SampleEvidence(
        sample_id=sample_id,
        dataset="squad",
        split="holdout",
        model="model-a",
        seed=0,
        raw_context_tokens=200,
        selected_tokens_pre_trim=100,
        emitted_context_tokens=90,
        prompt_input_tokens=130,
        output_tokens=10,
        provider_total_tokens=140,
        baseline_correct=baseline,
        treatment_correct=treatment,
    )


def test_provider_totals_are_not_compressor_tokens() -> None:
    usage = ProviderUsage(input_tokens=130, output_tokens=10, total_tokens=140)
    assert usage.total_tokens == 140
    with pytest.raises(ValueError):
        ProviderUsage(input_tokens=130, output_tokens=10, total_tokens=90)


def test_mcnemar_four_regressions_zero_gains_is_one_eighth_two_sided() -> None:
    records = [record(str(i), True, False) for i in range(4)]
    assert exact_mcnemar(records) == pytest.approx(0.125)


def test_summary_reports_emitted_savings_and_paired_outcomes() -> None:
    result = summarize([record("a", True, True), record("b", True, False)])
    assert result["mean_emitted_savings"] == pytest.approx(0.55)
    assert result["regressions"] == 1
    assert result["gains"] == 0


def test_wilson_empty_is_fail_closed() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
