from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from benchmarks.context_efficiency_baseline import build_head_tail_manifest
from benchmarks.context_efficiency_frontier import analyze_frontier, load_trials
from benchmarks.context_efficiency_openai import (
    DEFAULT_MODEL,
    LONG_BENCH_METRICS_SHA256,
    SCORER,
    ProviderConfig,
    WorkloadItem,
    call_openai,
    load_frozen_baseline,
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
    experiment = json.loads(entroly.experiment_config)
    assert experiment["selection_policy"] == "preselected"
    runtime = experiment["entroly_runtime"]
    assert runtime["entroly_package_version"]
    assert len(runtime["entroly_python_source_sha256"]) == 64
    assert len(runtime["benchmark_runner_sha256"]) == 64
    assert runtime["entroly_git_revision"]
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


def test_frozen_algorithmic_baseline_joins_the_same_paired_matrix(tmp_path):
    tiktoken = pytest.importorskip("tiktoken")

    item = _item()
    manifest = build_head_tail_manifest(
        items=[item],
        token_budget=120,
        encoding=tiktoken.get_encoding("o200k_base"),
        implementation_sha256="a" * 64,
    )
    baseline_schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "benchmarks/context_efficiency_baseline.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(baseline_schema)
    Draft202012Validator(baseline_schema).validate(manifest)
    assert (
        len(
            tiktoken.get_encoding("o200k_base").encode(
                manifest["tasks"][0]["selected_context"]
            )
        )
        <= 120
    )
    manifest_path = tmp_path / "baseline.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    baseline = load_frozen_baseline(manifest_path, [item])
    output = tmp_path / "trials.jsonl"
    client = _client(
        [
            _response("ORCHID-17", "request-one", 900),
            _response("ORCHID-17", "request-two", 500),
            _response("ORCHID-17", "request-three", 150),
        ]
    )

    trials = run_trials(
        items=[item],
        client=client,
        output=output,
        token_budget=120,
        frozen_baseline=baseline,
    )
    report = analyze_frontier(trials, bootstrap_samples=20)

    assert {trial.condition for trial in trials} == {
        "raw",
        "entroly",
        "algorithmic_baseline",
    }
    assert set(report["comparisons_to_raw"]) == {
        "algorithmic_baseline",
        "entroly",
    }
    assert baseline.manifest_sha256 in trials[0].experiment_config


def test_frozen_baseline_rejects_source_or_output_tampering(tmp_path):
    tiktoken = pytest.importorskip("tiktoken")

    item = _item()
    manifest = build_head_tail_manifest(
        items=[item],
        token_budget=120,
        encoding=tiktoken.get_encoding("o200k_base"),
        implementation_sha256="a" * 64,
    )
    manifest["tasks"][0]["selected_context"] += " tampered"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="selected digest mismatch"):
        load_frozen_baseline(path, [item])
