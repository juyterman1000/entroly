from __future__ import annotations

import json

import pytest

from benchmarks.context_efficiency_frontier import (
    SCHEMA_VERSION,
    Trial,
    analyze_frontier,
    load_trials,
)
from benchmarks.context_efficiency_report import render_markdown


def _record(task: int, condition: str, **overrides):
    payload = {
        "schema_version": SCHEMA_VERSION,
        "workload": "fixture-agent-tasks",
        "workload_version": "1",
        "task_id": f"task-{task}",
        "provider": "fixture",
        "model": "fixture-model",
        "provider_request_id": f"fixture-request-{task}-{condition}",
        "usage_source": "deterministic_fixture",
        "cost_source": "zero_cost_fixture",
        "replicate": 0,
        "condition": condition,
        "scorer": "exact-match-v1",
        "task_score": 1.0,
        "evidence_recall": 1.0,
        "unsupported_claim_rate": 0.0,
        "context_tokens": 1_000 if condition == "raw" else 400,
        "reasoning_tokens": 100,
        "output_tokens": 50,
        "billed_cost_usd": 0.01 if condition == "raw" else 0.005,
        "latency_ms": 1_000 if condition == "raw" else 800,
        "context_commit_id": "ctx_fixture" if condition in {"entroly", "combined"} else None,
    }
    payload.update(overrides)
    return payload


def test_frontier_reports_paired_quality_preserving_win():
    trials = [
        Trial.from_dict(_record(task, condition))
        for task in range(8)
        for condition in ("raw", "entroly")
    ]

    report = analyze_frontier(trials, bootstrap_samples=200, seed=7)
    comparison = report["comparisons_to_raw"]["entroly"]

    assert report["pair_count"] == 8
    assert comparison["mean_quality_delta"] == 0.0
    assert comparison["mean_context_reduction"] == 0.6
    assert comparison["mean_billed_cost_reduction"] == 0.5
    assert comparison["quality_preserving_context_win"] is True
    assert report["pareto_frontier"] == ["entroly"]
    assert report["provenance"]["usage_sources"] == ["deterministic_fixture"]


def test_frontier_refuses_to_call_context_savings_a_quality_win():
    trials = [
        Trial.from_dict(_record(task, "raw"))
        for task in range(6)
    ] + [
        Trial.from_dict(
            _record(task, "entroly", task_score=0.8, evidence_recall=0.8)
        )
        for task in range(6)
    ]

    report = analyze_frontier(trials, bootstrap_samples=100)

    assert report["comparisons_to_raw"]["entroly"]["quality_preserving_context_win"] is False
    assert set(report["pareto_frontier"]) == {"raw", "entroly"}


def test_trial_requires_context_commit_for_entroly_conditions():
    payload = _record(1, "entroly")
    payload["context_commit_id"] = None

    with pytest.raises(ValueError, match="require context_commit_id"):
        Trial.from_dict(payload)


def test_trial_rejects_fractional_token_counts():
    payload = _record(1, "raw")
    payload["context_tokens"] = 999.5

    with pytest.raises(ValueError, match="context_tokens must be an integer"):
        Trial.from_dict(payload)


def test_trial_rejects_unverifiable_usage_provenance():
    payload = _record(1, "raw")
    payload["usage_source"] = "local_estimate"

    with pytest.raises(ValueError, match="usage_source must be one of"):
        Trial.from_dict(payload)


def test_analyzer_rejects_unpaired_or_duplicate_trials():
    entroly_only = Trial.from_dict(_record(1, "entroly"))
    with pytest.raises(ValueError, match="missing the raw baseline"):
        analyze_frontier([entroly_only], bootstrap_samples=10)

    raw = Trial.from_dict(_record(1, "raw"))
    with pytest.raises(ValueError, match="duplicate 'raw'"):
        analyze_frontier([raw, raw, entroly_only], bootstrap_samples=10)


def test_load_trials_reports_the_invalid_line(tmp_path):
    path = tmp_path / "trials.jsonl"
    path.write_text(
        json.dumps(_record(1, "raw")) + "\n" + "{not-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"trials\.jsonl:2"):
        load_trials(path)


def test_public_report_exposes_pass_rule_and_caveats():
    trials = [
        Trial.from_dict(_record(task, condition))
        for task in range(4)
        for condition in ("raw", "entroly")
    ]
    report = analyze_frontier(trials, bootstrap_samples=50)

    markdown = render_markdown(report)

    assert "| entroly | 4 |" in markdown
    assert "**PASS**" in markdown
    assert "95% paired-bootstrap bounds" in markdown
    assert "Usage sources: deterministic_fixture" in markdown
    assert "Context Commit IDs prove artifact integrity" in markdown


def test_public_report_rejects_unknown_schema():
    with pytest.raises(ValueError, match="schema_version"):
        render_markdown({"schema_version": "future"})
