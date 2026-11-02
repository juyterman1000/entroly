"""Response distillation must not leave broken text behind.

Distillation is on by default (``ENTROLY_DISTILL``) and rewrites the model's
answer on every buffered response through the proxy. Run on ordinary answers
it produced text the model never wrote:

* "...```\\n\\nI hope this helps!" became "...```\\n\\nI": the closing
  pattern matched "hope this helps!" and left the "I" behind.
* "I think the bug is probably in the retry loop" became "the bug is probably
  in the retry loop": the hedge was removed and the sentence lost its capital.
"""

from __future__ import annotations

import pytest

from entroly.proxy_transform import distill_response

FIX = "```python\ndef f(x):\n    return x or 0\n```"


@pytest.mark.parametrize(
    "closing",
    ["I hope this helps!", "I hope that helps.", "Hope it helps!", "Hope this helps!"],
)
def test_a_closing_pleasantry_is_removed_whole(closing: str) -> None:
    text = f"Sure! Here is the fix:\n\n{FIX}\n\n{closing}"

    result, _, _ = distill_response(text, mode="full")

    assert result == FIX, repr(result)


def test_removing_a_leading_hedge_keeps_the_sentence_capitalised() -> None:
    text = "I think the bug is probably in the retry loop, but I'm not certain. It might also be the cache."

    result, _, _ = distill_response(text, mode="full")

    assert result.startswith("The bug is probably in the retry loop"), repr(result)


def test_removing_a_hedge_after_a_full_stop_keeps_the_capital() -> None:
    text = "The cache is fine and the config loads. I believe the retry loop is wrong, so check it first."

    result, _, _ = distill_response(text, mode="full")

    assert "loads. The retry loop is wrong" in result, repr(result)


def test_a_hedge_inside_a_sentence_leaves_the_next_word_alone() -> None:
    text = "The config loads correctly, and It seems like the retry loop never runs at all here."

    result, _, _ = distill_response(text, mode="full")

    assert "and the retry loop never runs" in result, repr(result)
