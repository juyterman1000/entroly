"""Older file reads must reach task-conditioned selection before any age cut.

These are exact-evidence regressions, not generated-answer quality benchmarks.
"""

from __future__ import annotations

import copy

import pytest

from entroly import compress_messages
from entroly.tokens import count_tokens


@pytest.fixture(autouse=True)
def lexical_only(monkeypatch):
    # This regression exercises local evidence retention, not an encoder.
    monkeypatch.setattr(
        "entroly.context_receipts.embedding_scorer.default_scorer", lambda: None
    )


def _file_read(call_id: str, path: str, text: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "' + path + '"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": text},
    ]


def _conversation(later_reads: int) -> list[dict]:
    noise = "".join(f"unused_setting_{i} = 'archive'\n" for i in range(120))
    messages = [
        {"role": "system", "content": "Answer using the code that was read."},
        {"role": "user", "content": "Inspect the retry implementation."},
        *_file_read(
            "policy",
            "retry_policy.py",
            f"# retry_policy.py\n{noise}\nRETRY_LIMIT = 7\n\n{noise}",
        ),
        *_file_read(
            "backoff",
            "backoff.py",
            f"# backoff.py\n{noise}\nBACKOFF_BASE = 13\n\n{noise}",
        ),
    ]
    for i in range(later_reads):
        messages.extend(
            _file_read(
                f"later_{i}",
                f"unrelated_{i}.py",
                f"# unrelated_{i}.py\n" + "other_setting = 'routine'\n" * 14,
            )
        )
    messages.append(
        {
            "role": "user",
            "content": (
                "What are RETRY_LIMIT in retry_policy.py and BACKOFF_BASE "
                "in backoff.py? Cite the exact assignments."
            ),
        }
    )
    return messages


@pytest.mark.parametrize("profile", ["safe", "balanced"])
@pytest.mark.parametrize("later_reads", [4, 20, 60])
@pytest.mark.parametrize("preserve_last_n", [1, 4])
def test_aged_cross_file_evidence_survives_growing_history(
    profile, later_reads, preserve_last_n
):
    messages = _conversation(later_reads)
    original = copy.deepcopy(messages)
    budget = 1_600
    assert sum(
        count_tokens(m["content"])
        for m in messages
        if isinstance(m.get("content"), str)
    ) > budget

    out = compress_messages(
        messages,
        budget=budget,
        preserve_last_n=preserve_last_n,
        distill=False,
        profile=profile,
    )

    assert "RETRY_LIMIT = 7" in out[3]["content"]
    assert "BACKOFF_BASE = 13" in out[5]["content"]
    assert out[0] == original[0]
    assert out[-1] == original[-1]
    assert messages == original, "compression must not mutate the caller's history"
    assert [m.get("tool_call_id") for m in out] == [
        m.get("tool_call_id") for m in original
    ]
    assert sum(
        count_tokens(m["content"])
        for m in out
        if isinstance(m.get("content"), str)
    ) <= budget


@pytest.mark.parametrize(
    ("profile", "question"),
    [
        ("max", "What is RETRY_LIMIT in retry_policy.py?"),
        ("balanced", "Summarize"),
        ("safe", "Summarize"),
    ],
)
def test_structural_and_queryless_paths_still_use_age_prepass(
    monkeypatch, profile, question
):
    from entroly import hardening

    calls = []
    original_prune = hardening.prune_aged_tool_outputs

    def record_prune(messages, tail_window):
        calls.append(tail_window)
        return original_prune(messages, tail_window=tail_window)

    monkeypatch.setattr(hardening, "prune_aged_tool_outputs", record_prune)
    messages = _conversation(4)
    messages[-1]["content"] = question

    compress_messages(
        messages,
        budget=1_600,
        preserve_last_n=1,
        distill=False,
        profile=profile,
    )

    assert calls == [1]


def test_request_adapter_retains_aged_code_without_changing_provider_controls():
    from entroly.integrations.request_adapter import compress_request_payload

    payload = {
        "temperature": 0,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
        "messages": _conversation(20),
    }
    original = copy.deepcopy(payload)

    result = compress_request_payload(payload, budget=1_600)

    assert result.changed
    assert "RETRY_LIMIT = 7" in result.payload["messages"][3]["content"]
    assert "BACKOFF_BASE = 13" in result.payload["messages"][5]["content"]
    assert result.payload["tools"] == original["tools"]
    assert result.payload["temperature"] == original["temperature"]
    assert payload == original
    assert sum(
        count_tokens(m["content"])
        for m in result.payload["messages"]
        if isinstance(m.get("content"), str)
    ) <= 1_600
