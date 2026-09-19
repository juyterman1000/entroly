"""Unit tests for bounded, recoverable message compaction."""

from __future__ import annotations

import json
from entroly.cli_recover import default_recovery_store_path
from entroly.codec import RecoveryStore
from entroly.tokens import count_messages_tokens, trim_messages


def test_preflight_trim_with_stubs(tmp_path, monkeypatch):
    """The exact omitted messages are recoverable through the CLI store."""
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path))
    messages = [
        {"role": "system", "content": "System prompt describing tool policy and guidelines."},
        {"role": "user", "content": "Here is file 1 content: " + ("def calculate_total():\n    return 42\n" * 50)},
        {"role": "assistant", "content": "Received file 1. Here is my analysis: " + ("analysis line\n" * 50)},
        {"role": "user", "content": "Now fix the calculation bug in file 1."},
    ]

    total_tokens = count_messages_tokens(messages)
    assert total_tokens > 200

    # Compact to a tight budget that includes the complete recovery digest.
    compacted = trim_messages(
        messages,
        max_tokens=110,
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
    assert "omitted message(s)" in stub_content
    assert "sha256:" in stub_content
    assert "Recoverable via:" in stub_content
    digest = stub_content.split("entroly recover ", 1)[1].split("`", 1)[0]
    store = RecoveryStore(default_recovery_store_path())
    ref = store.reference_for(digest)
    assert ref is not None
    assert json.loads(store.recover(ref)) == messages[1:3]

    # 3. Latest active user intent is preserved
    assert compacted[-1]["role"] == "user"
    assert "Now fix the calculation bug" in compacted[-1]["content"]

    # 4. Total tokens of compacted message list is within bounds
    compacted_tokens = count_messages_tokens(compacted)
    assert compacted_tokens <= 110
