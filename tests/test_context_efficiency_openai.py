from __future__ import annotations

import json
from types import SimpleNamespace

from benchmarks.context_efficiency_frontier import analyze_frontier, load_trials
from benchmarks.context_efficiency_openai import (
    DEFAULT_MODEL,
    ProviderConfig,
    WorkloadItem,
    call_openai,
    run_trials,
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
