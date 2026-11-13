"""Ensure real agent pilot has strict no-oracle recovery and measured billing."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from benchmarks.agentic_arms import Fragment
from benchmarks.agentic_tasks_run import build_dependency_tasks
from benchmarks.intensive_live_agent_pilot import import_recovery, model_identity, summarize


def test_imports_recover_only_visible_source() -> None:
    for task in build_dependency_tasks(6):
        candidates = [Fragment(task.dependency_file, task.dependency_source),
                      Fragment("secrets/private.py", "SECRET='unrelated'")]
        found = import_recovery(target_path=task.target_file,
                                target_source=task.broken_source, omitted=candidates)
        assert [f.source for f in found] == [task.dependency_file]


def test_parse_fail_closed() -> None:
    assert import_recovery(target_path="core/x.py", target_source="<not-python>",
                           omitted=[Fragment("config/secret.py", "SECRET=1")]) == []


def test_refuses_remote_model_endpoint() -> None:
    with pytest.raises(ValueError, match="only a local"):
        model_identity("https://example.com", "qwen2.5-coder:0.5b")


def test_model_must_be_pinned_by_digest() -> None:
    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def read(self):
            return json.dumps({"models": [{"name": "other:latest", "digest": "abc"}]}).encode()
    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        with pytest.raises(RuntimeError, match="not locally available"):
            model_identity("http://127.0.0.1:11434", "qwen2.5-coder:0.5b")


def test_closed_loop_must_charge_both_model_requests() -> None:
    trial = {
        "raw": {"passed": True, "input_tokens": 1000, "output_tokens": 10, "latency_s": 5},
        "compress": {"passed": False, "input_tokens": 200, "output_tokens": 20, "latency_s": 3},
        "closed_loop": {"passed": True, "retry_performed": True,
                        "input_tokens": 700, "output_tokens": 25, "latency_s": 7},
    }
    summary = summarize([trial])
    assert summary["raw"]["passed"] == 1
    assert summary["compress"]["passed"] == 0
    assert summary["closed_loop"]["reported_input_tokens"] == 700
    assert summary["input_token_reduction_compress_vs_raw_pct"] == 80.0
    assert summary["input_token_reduction_closed_loop_vs_raw_pct"] == 30.0
    assert summary["closed_loop_real_retry_count"] == 1
