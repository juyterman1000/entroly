from __future__ import annotations

import copy

import pytest

from entroly.semantic_assurance import (
    SemanticWireError,
    assure_provider_request,
    project_retrieval_intent,
    validate_anthropic_cache_topology,
)


def test_intent_projection_removes_whole_harness_block_without_mutating_body():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "fix the auth retry"},
                    {
                        "type": "text",
                        "text": "<system-reminder>internal harness state</system-reminder>",
                    },
                ],
            }
        ]
    }
    original = copy.deepcopy(body)

    projected = project_retrieval_intent(body, "anthropic")

    assert projected.retrieval_text == "fix the auth retry"
    assert projected.removed_blocks == 1
    assert body == original


def test_intent_projection_removes_newline_suffix_but_preserves_inline_literal():
    body = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "debug the parser\n"
                    "<system-reminder>daemon metadata</system-reminder>"
                ),
            }
        ]
    }
    projected = project_retrieval_intent(body, "anthropic")
    assert projected.retrieval_text == "debug the parser"

    inline = {
        "messages": [
            {
                "role": "user",
                "content": "Explain why the literal <system-reminder> tag is parsed",
            }
        ]
    }
    assert project_retrieval_intent(inline, "anthropic").retrieval_text == inline[
        "messages"
    ][0]["content"]


def test_stale_tool_edges_become_plain_historical_evidence():
    body = {
        "model": "claude-test",
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "toolu_old",
                        "name": "RemovedTool",
                        "input": {"path": "a.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_old",
                        "content": "old result",
                    }
                ],
            },
        ],
    }
    original = copy.deepcopy(body)

    assured, report = assure_provider_request(body, "anthropic")

    serialized = repr(assured["messages"])
    assert "'type': 'tool_use'" not in serialized
    assert "'type': 'tool_result'" not in serialized
    assert "old result" in serialized
    assert {repair.code for repair in report.repairs} >= {
        "tool_use_retired",
        "tool_result_retired",
    }
    assert body == original


def test_current_tool_edge_is_preserved_and_results_are_frontloaded():
    body = {
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "result follows"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                ],
            },
        ],
    }

    assured, report = assure_provider_request(body, "anthropic")

    assert assured["messages"][0]["content"][0]["type"] == "tool_use"
    assert assured["messages"][1]["content"][0]["type"] == "tool_result"
    assert "tool_results_frontloaded" in {repair.code for repair in report.repairs}


def test_orphan_tool_result_is_retired_not_dropped():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "missing", "content": "evidence"}
                ],
            }
        ]
    }
    assured, _ = assure_provider_request(body, "anthropic")
    assert assured["messages"][0]["content"][0]["type"] == "text"
    assert "evidence" in assured["messages"][0]["content"][0]["text"]


def test_system_role_is_relocated_and_cache_control_survives():
    body = {
        "system": [{"type": "text", "text": "existing"}],
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "harness",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {"role": "user", "content": "hi"},
        ],
    }

    assured, report = assure_provider_request(body, "anthropic")

    assert [message["role"] for message in assured["messages"]] == ["user"]
    assert assured["system"][1]["text"] == "harness"
    assert assured["system"][1]["cache_control"]["ttl"] == "1h"
    assert "anthropic_system_role_relocated" in {
        repair.code for repair in report.repairs
    }


def test_cache_topology_accepts_longer_before_shorter():
    body = {
        "tools": [
            {
                "name": "Read",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "system": [
            {
                "type": "text",
                "text": "stable",
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "recent",
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    }
                ],
            }
        ],
    }
    points = validate_anthropic_cache_topology(body)
    assert [point.ttl for point in points] == ["1h", "1h", "5m"]


def test_cache_topology_blocks_shorter_before_longer():
    body = {
        "system": [
            {
                "type": "text",
                "text": "first",
                "cache_control": {"type": "ephemeral", "ttl": "5m"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "later",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        ],
    }
    with pytest.raises(SemanticWireError, match="longer-lived") as caught:
        validate_anthropic_cache_topology(body)
    assert caught.value.code == "cache_ttl_non_monotonic"


def test_automatic_cache_conflict_is_blocked_not_silently_repriced():
    body = {
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "x",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        ],
    }
    with pytest.raises(SemanticWireError) as caught:
        validate_anthropic_cache_topology(body)
    assert caught.value.code == "automatic_cache_ttl_conflict"


def test_more_than_four_explicit_cache_breakpoints_is_blocked():
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": str(index),
                        "cache_control": {"type": "ephemeral", "ttl": "5m"},
                    }
                    for index in range(5)
                ],
            }
        ]
    }
    with pytest.raises(SemanticWireError) as caught:
        validate_anthropic_cache_topology(body)
    assert caught.value.code == "cache_breakpoint_limit_exceeded"
