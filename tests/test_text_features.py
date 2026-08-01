from __future__ import annotations

from entroly.text_features import (
    is_instruction_path,
    protected_input_reason,
    query_terms,
    text_terms,
)


def test_unicode_terms_include_cjk_bigrams_and_identifier_parts() -> None:
    terms = query_terms("resolvePayment 認証失敗の原因")
    assert {"resolve", "payment", "認証", "失敗", "原因"} <= terms
    assert {"resolve", "payment"} <= text_terms("resolvePayment")


def test_instruction_paths_cover_agent_rule_surfaces() -> None:
    assert is_instruction_path("AGENTS.md")
    assert is_instruction_path("project/.claude/rules/security.md")
    assert is_instruction_path("skills/release/SKILL.md")
    assert is_instruction_path(".github/instructions/python.instructions.md")
    assert not is_instruction_path("docs/architecture.md")


def test_short_input_guard_is_bounded_and_structured_data_exempt() -> None:
    assert protected_input_reason(
        "one concise but important result",
        budget_tokens=2,
        content_type="text",
    ) == "short_input_full_fidelity"
    assert protected_input_reason(
        '{"payload":"' + "x" * 2_000 + '"}',
        budget_tokens=10,
        content_type="json",
    ) is None
    assert protected_input_reason(
        "x" * 4_000,
        budget_tokens=10,
        content_type="text",
    ) is None
