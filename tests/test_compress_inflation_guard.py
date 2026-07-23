"""The compressor must never inflate, never crash, never empty a non-empty input.

`compress()` is query-agnostic structural compaction. Its one unbreakable
contract is that it is not net-negative: for any input, with no explicit
budget, it must return no more characters than it received. Structural passes
(the code symbol-summary prepend, format markers) can otherwise inflate small
or already-dense inputs — a "compressor" that silently costs tokens.

This is the fail-closed / falsification-first invariant applied to compression:
we throw an adversarial grid at it and assert the contract holds on every case.
"""

from __future__ import annotations

import pytest

from entroly import compress
from entroly.sdk import _guard_against_inflation

# An adversarial grid: tiny code (where a prepended symbol summary would
# inflate), already-dense text, pathological whitespace/markers, unicode/CJK,
# degenerate structure. None may crash; none may inflate.
ADVERSARIAL_INPUTS = [
    "def f(): return 1",                       # tiny code — prepend would inflate
    "class A:\n    pass",                       # tiny class
    "fn main() {}",                             # tiny rust
    "x",                                        # single char
    "{}",                                       # empty json object
    "[]",                                       # empty json array
    "a,b,c\n1,2,3",                             # tiny csv
    "```\ncode\n```",                           # already-fenced
    "\n\n\n\n\n",                               # pure whitespace
    "语言 处理 测试 " * 3,                        # CJK
    "🔥" * 50,                                   # emoji
    "SELECT * FROM t WHERE id=1;",              # tiny sql
    "import os\nimport sys",                    # tiny imports
    "//" * 200,                                 # degenerate comment run
    '{"k":"v"}',                                # minimal json
]


@pytest.mark.parametrize("content", ADVERSARIAL_INPUTS)
def test_compress_never_inflates(content: str) -> None:
    out = compress(content)  # no budget → target_ratio path (the gap)
    assert len(out) <= len(content), (
        f"compress inflated {len(content)}→{len(out)} chars for {content!r}"
    )


@pytest.mark.parametrize("content", ADVERSARIAL_INPUTS)
def test_compress_never_crashes_and_preserves_non_empty(content: str) -> None:
    out = compress(content, content_type="code")
    assert isinstance(out, str)
    if content.strip():
        assert out.strip(), "compress emptied a non-empty input"


def test_inflation_guard_is_a_pure_clamp() -> None:
    # Larger structural output falls back to the original …
    assert _guard_against_inflation("A" * 100, "orig") == "orig"
    # … and a genuinely smaller output is passed through untouched.
    assert _guard_against_inflation("small", "A" * 100) == "small"
    # Equal length is not inflation.
    assert _guard_against_inflation("abcd", "wxyz") == "abcd"


def test_budget_path_stays_inflation_safe() -> None:
    big = "def function_with_a_long_name():\n    return 42\n" * 40
    out = compress(big, budget=20)
    assert len(out) // 4 <= 20 * 3  # bounded near the estimated ceiling
    assert len(out) <= len(big)
