from __future__ import annotations

import json
from copy import deepcopy

import pytest
import entroly.decisions_trial as decisions_trial
from entroly.context_receipts.models import stable_hash

from entroly.decisions_trial import (
    MANIFEST_SCHEMA,
    REPLAY_SCHEMA,
    _parse_answer,
    evaluate,
    main,
)


def manifest():
    return {
        "schema": MANIFEST_SCHEMA,
        "task_family": "fixture-routing",
        "context_policy": "fixture-selected-v1",
        "tokenizer": "fixture-counts-v1",
        "budget": 100,
        "trials": [
            {
                "trial_id": "trial-1",
                "task_id": "task-1",
                "question": "Route the issue.",
                "full_context": "private full context",
                "selected_context": "private selected context",
                "full_context_tokens": 20,
                "selected_tokens": 10,
                "decision": {
                    "name": "route",
                    "instructions": "Choose the responsible team.",
                    "choices": [
                        {"value": "a", "description": "Team A"},
                        {"value": "b", "description": "Team B"},
                    ],
                },
                "truth_choice": "a",
            }
        ],
    }


def response(choice, first):
    return {
        "model": "gpt-6-luna",
        "answers": [
            {
                "type": "choice",
                "name": "route",
                "choice": choice,
                "confidence": 0.7,
                "probabilities": [
                    {"value": "a", "probability": first},
                    {"value": "b", "probability": 1 - first},
                ],
            }
        ],
        "usage": {"input_tokens": 17},
    }


def replay():
    return {
        "schema": REPLAY_SCHEMA,
        "trials": [
            {
                "trial_id": "trial-1",
                "full": response("a", 0.8),
                "selected": response("b", 0.2),
                "full_repeat": response("a", 0.7),
            }
        ],
    }


def evaluate_fixture(manifest_value=None, replay_value=None):
    return evaluate(
        manifest_value or manifest(),
        workload_sha256="a" * 64,
        entroly_sha="b" * 40,
        replay=replay_value or replay(),
    )


def test_replay_records_choice_flip_and_independent_probability_control():
    report = evaluate_fixture()
    assert report["source"] == "offline_replay"
    assert report["production_authority"] is False
    assert report["context_regret"]["decision_divergence_regret"] == 1
    assert report["context_regret"]["model_self_divergence"] == 0
    assert report["context_regret"]["task_context_regret"] == 1
    assert report["probability_shifts"][0]["full_selected_js_bits"] > 0
    assert report["probability_shifts"][0]["full_repeat_js_bits"] > 0
    assert (
        report["observations"][0]["provider_observed_usage"]["input_tokens"]["full"]
        == 17
    )
    assert report["report_id"] == stable_hash(
        {key: value for key, value in report.items() if key != "report_id"}
    )
    assert (
        report["probability_shifts"][0]["request_commitments"]["full"]
        == report["probability_shifts"][0]["request_commitments"]["full_repeat"]
    )
    assert "private full context" not in str(report)
    assert '"choice": "a"' not in str(report)


def test_missing_truth_remains_unavailable():
    item = manifest()
    del item["trials"][0]["truth_choice"]
    report = evaluate_fixture(manifest_value=item)
    assert report["context_regret"]["task_context_regret"] is None


def test_replay_change_gets_a_distinct_scope():
    first = evaluate_fixture()
    changed = replay()
    changed["trials"][0]["full_repeat"]["answers"][0]["confidence"] = 0.6
    second = evaluate_fixture(replay_value=changed)
    assert first["scope_id"] != second["scope_id"]


def test_live_collector_sends_exact_paired_inputs_without_truth(monkeypatch):
    sent = []
    answers = [response("a", 0.8), response("b", 0.2), response("a", 0.7)]

    def fake_call(payload, api_key):
        assert api_key == "test-key"
        sent.append(payload)
        return answers[len(sent) - 1], 10.0

    monkeypatch.setattr(decisions_trial, "_call_openai", fake_call)
    task = manifest()
    task["trials"][0]["decision"]["type"] = "predicate"
    report = evaluate(
        task,
        workload_sha256="a" * 64,
        entroly_sha="b" * 40,
        api_key="test-key",
    )
    assert len(sent) == 3
    assert sent[0] == sent[2]
    assert sent[0]["input"] != sent[1]["input"]
    assert sent[0]["questions"][0]["type"] == "choice"
    assert all("truth_choice" not in str(payload) for payload in sent)
    assert report["latency_source"] == "client_wall_clock"
    assert report["observations"][0]["latency_ms"] == 30.0
    assert report["source"] == "openai_decisions_api"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["answers"][0].update(type="refusal"),
        lambda value: value["answers"][0].update(name="other"),
        lambda value: value["answers"][0].update(choice="unknown"),
        lambda value: value["answers"][0]["probabilities"].pop(),
        lambda value: value["answers"][0]["probabilities"][0].update(
            probability=float("nan")
        ),
        lambda value: value["answers"][0]["probabilities"][0].update(probability=0.4),
        lambda value: value.update(model="other-model"),
    ],
)
def test_invalid_provider_answers_fail_closed(mutation):
    value = response("a", 0.8)
    mutation(value)
    with pytest.raises(ValueError):
        _parse_answer(value, manifest()["trials"][0])


def test_incomplete_triplet_and_duplicate_trials_fail_closed():
    fixture = replay()
    del fixture["trials"][0]["full_repeat"]
    with pytest.raises(ValueError, match="incomplete"):
        evaluate_fixture(replay_value=fixture)
    task = manifest()
    task["trials"].append(deepcopy(task["trials"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_fixture(manifest_value=task)


def test_cli_requires_explicit_egress_and_writes_only_complete_offline_report(
    tmp_path, monkeypatch
):
    input_path = tmp_path / "manifest.json"
    replay_path = tmp_path / "replay.json"
    output_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(manifest()), encoding="utf-8")
    replay_path.write_text(json.dumps(replay()), encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        main(
            [
                "--manifest",
                str(input_path),
                "--output",
                str(output_path),
                "--send-content",
                "--entroly-sha",
                "b" * 40,
            ]
        )
        == 1
    )
    assert not output_path.exists()
    assert (
        main(
            [
                "--manifest",
                str(input_path),
                "--output",
                str(output_path),
                "--responses",
                str(replay_path),
                "--entroly-sha",
                "b" * 40,
            ]
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["source"] == "offline_replay"
    assert report["responses_sha256"]
    assert (
        main(
            [
                "--manifest",
                str(input_path),
                "--output",
                str(output_path),
                "--responses",
                str(replay_path),
                "--entroly-sha",
                "b" * 40,
            ]
        )
        == 1
    )
