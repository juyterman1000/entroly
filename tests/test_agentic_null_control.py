"""Tests for the mandatory no-context construct-validity arm."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agentic_null_control import (  # noqa: E402
    NULL_ARM,
    build_null_prompt,
    run_null_arm,
)
from benchmarks.agentic_tasks_run import build_tasks  # noqa: E402


def test_null_prompt_contains_no_repository_source():
    task = build_tasks(distractor_count=6)[0]
    prompt = build_null_prompt(task)

    assert task.query in prompt
    assert task.test_source in prompt
    assert task.broken_source not in prompt
    for body in task.distractors.values():
        assert body not in prompt


def test_null_arm_records_every_source_as_omitted(monkeypatch):
    task = build_tasks(distractor_count=2)[0]
    seen = {}

    def fake_call_model(**kwargs):
        seen["prompt"] = kwargs["prompt"]
        return {
            "text": task.broken_source,
            "input_tokens": 10,
            "output_tokens": 5,
            "latency_s": 0.01,
        }

    monkeypatch.setattr(
        "benchmarks.agentic_null_control.call_model", fake_call_model
    )
    monkeypatch.setattr(
        "benchmarks.agentic_null_control.run_oracle",
        lambda _task, _source: (False, "1 failed"),
    )

    row = run_null_arm(
        task,
        model="test-model",
        base_url="http://invalid",
        seed=7,
        timeout=1,
    )

    assert row["arm"] == NULL_ARM
    assert row["context"]["estimated_context_tokens"] == 0
    assert row["context"]["included_sources"] == []
    assert row["context"]["omitted_sources"] == [
        fragment.source for fragment in task.fragments()
    ]
    assert task.broken_source not in seen["prompt"]
