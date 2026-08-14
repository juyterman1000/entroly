from __future__ import annotations

import pytest

from entroly.anthropic_state_contract import validate_active_tool_state
from entroly.semantic_assurance import SemanticWireError


def test_active_tool_use_requires_immediate_user_result_message():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "later"}]},
        ]
    }

    with pytest.raises(SemanticWireError) as caught:
        validate_active_tool_state(body)
    assert caught.value.code == "tool_result_missing"


def test_active_tool_use_requires_every_matching_result():
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
                    {"type": "tool_use", "id": "toolu_2", "name": "Edit", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                ],
            },
        ]
    }

    with pytest.raises(SemanticWireError) as caught:
        validate_active_tool_state(body)
    assert caught.value.code == "tool_result_missing"


def test_tool_results_must_lead_user_content():
    body = {
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
                    {"type": "text", "text": "thanks"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                ],
            },
        ]
    }

    with pytest.raises(SemanticWireError) as caught:
        validate_active_tool_state(body)
    assert caught.value.code == "tool_result_order_invalid"


def test_complete_active_tool_state_passes():
    body = {
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
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                    {"type": "text", "text": "continue"},
                ],
            },
        ]
    }

    validate_active_tool_state(body)
