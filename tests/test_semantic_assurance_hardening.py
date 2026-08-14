from __future__ import annotations

from entroly import semantic_assurance as semantic
from entroly.semantic_assurance_hardening import (
    conservative_purify_block,
    install_semantic_assurance_hardening,
)


def test_mixed_legitimate_text_between_reminders_is_not_swallowed():
    text = (
        "question\n"
        "<system-reminder>first</system-reminder>\n"
        "keep this legitimate text\n"
        "<system-reminder>trailing</system-reminder>"
    )

    purified, removed = conservative_purify_block(text)

    assert "keep this legitimate text" in purified
    assert "<system-reminder>first</system-reminder>" in purified
    assert "trailing" not in purified
    assert removed == 1


def test_adjacent_harness_only_regions_are_fully_removed():
    text = (
        "<system-reminder>one</system-reminder>\n"
        "<system-reminder>two</system-reminder>"
    )
    assert conservative_purify_block(text) == ("", 2)


def test_inline_closed_literal_is_preserved():
    text = "Explain <system-reminder>literal</system-reminder> as parser syntax"
    assert conservative_purify_block(text) == (text, 0)


def test_inline_trailing_closed_literal_is_preserved():
    text = "Explain this literal <system-reminder>example</system-reminder>"
    assert conservative_purify_block(text) == (text, 0)


def test_mapping_shaped_historical_result_is_preserved_as_evidence():
    install_semantic_assurance_hardening()
    block = {
        "type": "tool_result",
        "tool_use_id": "old",
        "content": {"type": "text", "text": "structured evidence"},
    }

    retired = semantic._historical_tool_result(block)

    assert retired[0]["type"] == "text"
    assert any(
        item.get("text") == "structured evidence"
        for item in retired[1:]
        if isinstance(item, dict)
    )
