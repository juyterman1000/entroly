from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.context_efficiency_frontier import analyze_frontier, load_trials
from benchmarks.context_efficiency_openai import (
    DEFAULT_MODEL,
    LONG_BENCH_METRICS_SHA256,
    SCORER,
    ProviderConfig,
    WorkloadItem,
    call_openai,
    normalize_answer,
    qa_f1_score,
    run_trials,
    select_longbench_rows,
)


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _response(answer: str, request_id: str, prompt_tokens: int):
    return SimpleNamespace(
        id=request_id,
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=4,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        ),
    )


def _client(responses):
    completions = _FakeCompletions(responses)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        fake_completions=completions,
    )


def _item() -> WorkloadItem:
    context = "Distractor details. " * 600 + "The launch code is ORCHID-17."
    return WorkloadItem(
        task_id="hotpotqa-fixture",
        context=context,
        question="What is the launch code?",
        answers=("ORCHID-17",),
    )


def test_call_openai_captures_provider_usage_and_request_id():
    client = _client([_response("ORCHID-17", "chatcmpl_fixture", 123)])

    observation = call_openai(
        client,
        model=DEFAULT_MODEL,
        context="The launch code is ORCHID-17.",
        question="What is the launch code?",
    )

    assert observation.request_id == "chatcmpl_fixture"
    assert observation.prompt_tokens == 123
    assert observation.completion_tokens == 4
    assert client.fake_completions.requests[0]["temperature"] == 0


def test_call_openai_forwards_explicit_reasoning_configuration():
    client = _client([_response("ORCHID-17", "chatcmpl_reasoning", 123)])

    call_openai(
        client,
        model="gemini-2.5-flash",
        context="The launch code is ORCHID-17.",
        question="What is the launch code?",
        max_output_tokens=80,
        reasoning_effort="none",
    )

    request = client.fake_completions.requests[0]
    assert request["max_tokens"] == 80
    assert request["reasoning_effort"] == "none"


def test_shortest_context_selection_is_deterministic_and_explicitly_biased():
    rows = [
        {"context": "long context", "question": "q3"},
        {"context": "x", "question": "q1"},
        {"context": "medium", "question": "q2"},
    ]

    selected = select_longbench_rows(rows, 2, "shortest-context")

    assert [row["context"] for row in selected] == ["x", "medium"]
    assert select_longbench_rows(rows, 2, "random") == rows[:2]


def test_longbench_hotpotqa_scorer_matches_official_normalization_and_f1():
    assert normalize_answer("The, ORCHID-17!") == "orchid17"
    assert qa_f1_score("Alpha Beta", "the alpha beta gamma") == 0.8
    assert "longbench-v1-hotpotqa-official-qa-f1" in SCORER
    assert len(LONG_BENCH_METRICS_SHA256) == 64


def test_run_trials_writes_complete_auditable_matrix(tmp_path):
    output = tmp_path / "trials.jsonl"
    client = _client(
        [
            _response("ORCHID-17", "chatcmpl_raw", 3_100),
            _response("ORCHID-17", "chatcmpl_entroly", 520),
        ]
    )

    trials = run_trials(
        items=[_item()],
        client=client,
        output=output,
        token_budget=120,
        seed=0,
    )
    loaded = load_trials(output)

    assert len(trials) == len(loaded) == 2
    assert {trial.condition for trial in loaded} == {"raw", "entroly"}
    entroly = next(trial for trial in loaded if trial.condition == "entroly")
    assert entroly.context_commit_id and entroly.context_commit_id.startswith("ctx_")
    assert entroly.usage_source == "provider_response"
    assert entroly.usage_observed is True
    assert entroly.cost_observed is True
    assert entroly.task_success is True
    assert entroly.scorer == SCORER
    assert '"selection_policy":"preselected"' in entroly.experiment_config
    assert "2026-07-11" in entroly.cost_source_reference
    assert "input-0.15" in entroly.cost_source_reference
    assert list(output.parent.glob("trials_context_commits/*.json"))
    report = analyze_frontier(loaded, bootstrap_samples=20)
    assert report["comparisons_to_raw"]["entroly"]["mean_context_reduction"] > 0


def test_run_trials_keeps_provider_errors_as_zero_score_outcomes(tmp_path):
    output = tmp_path / "trials.jsonl"
    client = _client(
        [
            TimeoutError("fixture timeout"),
            _response("ORCHID-17", "chatcmpl_success", 800),
        ]
    )

    run_trials(
        items=[_item()],
        client=client,
        output=output,
        token_budget=120,
        seed=0,
    )
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert len(records) == 2
    error = next(record for record in records if record["outcome"] == "error")
    assert error["task_score"] == 0.0
    assert error["context_tokens"] == 0
    assert error["error_type"] == "TimeoutError"
    assert error["usage_observed"] is False
    assert error["cost_observed"] is False
    assert len(error["error_fingerprint"]) == 64


def test_run_trials_refuses_tasks_without_answer_evidence_before_calling_provider(
    tmp_path,
):
    output = tmp_path / "trials.jsonl"
    client = _client([])
    item = WorkloadItem(
        task_id="missing-evidence",
        context="Only distractors are present.",
        question="What is the launch code?",
        answers=("ORCHID-17",),
    )

    with pytest.raises(ValueError, match="raw context does not contain"):
        run_trials(items=[item], client=client, output=output)

    assert not output.exists()
    assert client.fake_completions.requests == []


def test_resume_does_not_rebill_completed_conditions(tmp_path):
    output = tmp_path / "trials.jsonl"
    first = _client(
        [
            _response("ORCHID-17", "chatcmpl_one", 900),
            _response("ORCHID-17", "chatcmpl_two", 500),
        ]
    )
    run_trials(items=[_item()], client=first, output=output, token_budget=120)
    second = _client([])

    resumed = run_trials(
        items=[_item()],
        client=second,
        output=output,
        token_budget=120,
        resume=True,
    )

    assert len(resumed) == 2
    assert second.fake_completions.requests == []


def test_self_hosted_provider_records_zero_api_fee_without_claiming_zero_compute(tmp_path):
    output = tmp_path / "trials.jsonl"
    provider = ProviderConfig(
        name="ollama",
        cost_source="self_hosted_no_api_fee",
        cost_source_reference="ollama:model=digest;hardware-cost=unmeasured",
        input_usd_per_million=0.0,
        cached_input_usd_per_million=0.0,
        output_usd_per_million=0.0,
    )
    client = _client(
        [
            _response("ORCHID-17", "local-one", 900),
            _response("ORCHID-17", "local-two", 500),
        ]
    )

    trials = run_trials(
        items=[_item()],
        client=client,
        output=output,
        model="local-model",
        token_budget=120,
        provider=provider,
    )

    assert {trial.provider for trial in trials} == {"ollama"}
    assert {trial.cost_source for trial in trials} == {"self_hosted_no_api_fee"}
    assert all(trial.billed_cost_usd == 0.0 for trial in trials)
