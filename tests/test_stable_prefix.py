from __future__ import annotations

import pytest

from entroly.stable_prefix import (
    CanonicalPrefixBuilder,
    conversation_anchor,
)


def test_prefix_is_stable_across_mapping_and_tool_order() -> None:
    first = (
        CanonicalPrefixBuilder()
        .add("policy", {"b": 2, "a": 1}, priority=10)
        .add_tools([{"name": "write", "z": 1}, {"name": "read", "a": 2}])
        .build(dynamic_tail="turn one")
    )
    second = (
        CanonicalPrefixBuilder()
        .add_tools([{"a": 2, "name": "read"}, {"z": 1, "name": "write"}])
        .add("policy", {"a": 1, "b": 2}, priority=10)
        .build(dynamic_tail="turn two")
    )

    assert first.stable_prefix == second.stable_prefix
    assert first.prefix_hash == second.prefix_hash
    assert first.dynamic_tail != second.dynamic_tail


def test_conversation_anchor_survives_appended_turns() -> None:
    initial = [
        {"role": "system", "content": "coding policy"},
        {"role": "user", "content": "fix auth"},
    ]
    later = initial + [
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": "add tests"},
    ]

    assert conversation_anchor(initial) == conversation_anchor(later)


def test_conflicting_duplicate_section_is_rejected() -> None:
    builder = CanonicalPrefixBuilder().add("policy", "one")
    with pytest.raises(ValueError, match="already has different content"):
        builder.add("policy", "two")
