"""Unit tests for pre-flight context guard and Merkle compaction stubs."""

from __future__ import annotations

import json
import pytest

from entroly.tokens import count_messages_tokens, trim_messages


def test_preflight_trim_with_stubs():
    """Verify that messages exceeding budget are compacted into Merkle stubs."""
    messages = [
        {"role": "system", "content": "System prompt describing tool policy and guidelines."},
        {"role": "user", "content": "Here is file 1 content: " + ("def calculate_total():\n    return 42\n" * 50)},
        {"role": "assistant", "content": "Received file 1. Here is my analysis: " + ("analysis line\n" * 50)},
        {"role": "user", "content": "Now fix the calculation bug in file 1."},
    ]

    total_tokens = count_messages_tokens(messages)
    assert total_tokens > 200

    # Compact to a tight budget (e.g., 100 tokens)
    compacted = trim_messages(
        messages,
        max_tokens=100,
        strategy="last",
        include_system=True,
        create_stubs=True,
        store_recovery=True,
    )

    # 1. System prompt is preserved
    assert compacted[0]["role"] == "system"
    assert "System prompt" in compacted[0]["content"]

    # 2. Stub is created in the middle
    stubs = [m for m in compacted if "ENTROLY CONTEXT COMPACTION" in str(m.get("content", ""))]
    assert len(stubs) == 1
    stub_content = stubs[0]["content"]
    assert "historical turn(s)" in stub_content
    assert "sha256:" in stub_content
    assert "Recoverable via:" in stub_content

    # 3. Latest active user intent is preserved
    assert compacted[-1]["role"] == "user"
    assert "Now fix the calculation bug" in compacted[-1]["content"]

    # 4. Total tokens of compacted message list is within bounds
    compacted_tokens = count_messages_tokens(compacted)
    assert compacted_tokens <= 150  # comfortably fits well below the original >200 tokens
