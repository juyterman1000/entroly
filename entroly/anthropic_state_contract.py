"""State-machine proof for active Anthropic client tool calls.

Historical capability edges may be retired by semantic_assurance, but an active
client tool call is still protocol state. Before transport, every active
``tool_use`` must be satisfied by the immediately following user message and its
``tool_result`` blocks must lead that content. Ambiguous/incomplete active state
is rejected locally rather than guessed into a provider request.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from . import semantic_assurance as _semantic


def validate_active_tool_state(body: Mapping[str, Any]) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return

    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
            continue

        active_ids = [
            str(block.get("id") or "")
            for block in content
            if isinstance(block, Mapping) and block.get("type") == "tool_use"
        ]
        active_ids = [value for value in active_ids if value]
        if not active_ids:
            continue

        next_index = index + 1
        if next_index >= len(messages):
            raise _semantic.SemanticWireError(
                "tool_result_missing",
                f"messages[{index}]",
                "active Anthropic tool_use has no immediately following user tool_result message",
            )
        following = messages[next_index]
        if not isinstance(following, Mapping) or following.get("role") != "user":
            raise _semantic.SemanticWireError(
                "tool_result_missing",
                f"messages[{next_index}]",
                "active Anthropic tool_use must be followed immediately by a user tool_result message",
            )
        result_content = following.get("content")
        if not isinstance(result_content, list):
            raise _semantic.SemanticWireError(
                "tool_result_missing",
                f"messages[{next_index}].content",
                "active Anthropic tool_use requires list-shaped tool_result content",
            )

        result_ids: list[str] = []
        ordinary_seen = False
        for block_index, block in enumerate(result_content):
            is_result = isinstance(block, Mapping) and block.get("type") == "tool_result"
            if is_result:
                if ordinary_seen:
                    raise _semantic.SemanticWireError(
                        "tool_result_order_invalid",
                        f"messages[{next_index}].content[{block_index}]",
                        "Anthropic tool_result blocks must precede ordinary user content",
                    )
                result_ids.append(str(block.get("tool_use_id") or ""))
            else:
                ordinary_seen = True

        active_set = set(active_ids)
        result_set = {value for value in result_ids if value}
        missing = sorted(active_set - result_set)
        if missing:
            raise _semantic.SemanticWireError(
                "tool_result_missing",
                f"messages[{next_index}].content",
                "one or more active Anthropic tool_use IDs have no matching immediate tool_result",
            )


def install_active_tool_state_proof() -> None:
    current = _semantic.assure_provider_request
    if hasattr(current, "__entroly_active_tool_state_original__"):
        return

    def assured(body: Mapping[str, Any], provider: str):
        candidate, report = current(body, provider)
        if provider == "anthropic":
            validate_active_tool_state(candidate)
        return candidate, report

    assured.__entroly_active_tool_state_original__ = current
    _semantic.assure_provider_request = assured


__all__ = ["install_active_tool_state_proof", "validate_active_tool_state"]
