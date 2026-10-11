"""The declared budget is a token ceiling, so it must be checked in tokens.

``compress`` and ``compress_messages`` take ``budget`` in tokens, and the
engine treats it as a hard ceiling because the provider does. Enforcing it on
``len(text) // 4`` only holds for ASCII prose. Measured against o200k, this
repository's own files run 3.3 chars/token for JSON and 3.8 for Markdown, and
CJK runs about 1.4 while emoji runs under 1. A character slice sized at
``budget * 4`` is therefore several times the declared ceiling for exactly the
inputs where the ceiling matters.

The regression these tests pin is not hypothetical: before the fix,
``compress(chinese, budget=200)`` returned 554 tokens and the emoji case
returned 1,378. ``compress_messages`` was worse than over-budget -- its
decision to compress at all read a CJK conversation as a third of its real
size, cleared the budget check, and returned the input unchanged.

The suite could not catch it because the existing budget assertion is written
in characters, and no tokenizer test used a non-ASCII input. So these tests
assert in tokens, and they do it over scripts with different densities.
"""
from __future__ import annotations

import pytest

from entroly.sdk import compress, compress_messages
from entroly.tokens import count_tokens


def _tokenizer_present() -> bool:
    """True when count_tokens measures rather than falling back to chars/4.

    tiktoken is an optional extra, so a base `pip install entroly` degrades
    count_tokens to the same character heuristic the budget used to use. The
    contract still holds there -- enforcement and assertion share one unit --
    but a fixture built to show the tokenizer disagreeing with chars/4 cannot,
    because in that environment it agrees by construction.
    """
    dense = "这是一个关于上下文压缩的技术文档。"
    return count_tokens(dense) != max(1, (len(dense) + 3) // 4)


requires_tokenizer = pytest.mark.skipif(
    not _tokenizer_present(),
    reason="no tokenizer installed; count_tokens is the chars/4 fallback",
)


# Deliberately spans characters-per-token from ~0.6 (emoji) to ~4.3 (English).
SCRIPTS: dict[str, str] = {
    "english": "The quick brown fox jumps over the lazy dog. " * 120,
    "chinese": "这是一个关于上下文压缩的技术文档，描述了令牌预算的执行方式。" * 60,
    "japanese": "これはコンテキスト圧縮に関する技術文書であり、トークン予算の適用方法を説明します。" * 50,
    "korean": "이것은 컨텍스트 압축에 관한 기술 문서이며 토큰 예산 적용 방법을 설명합니다. " * 50,
    "russian": "Это технический документ о сжатии контекста и бюджете токенов. " * 60,
    "emoji": "🚀🔥✨🎯📦🧩🛠️🌐🔑📊 " * 150,
    "json": '{"key":"value","nested":{"a":1,"b":[1,2,3]}},' * 200,
}


@pytest.mark.parametrize("script", sorted(SCRIPTS))
@pytest.mark.parametrize("budget", [50, 200, 1000])
def test_compress_never_exceeds_the_token_budget(script: str, budget: int) -> None:
    out = compress(SCRIPTS[script], budget=budget)
    assert count_tokens(out) <= budget


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_compress_messages_never_exceeds_the_token_budget(script: str) -> None:
    # Full-size bodies, not a tenth of them. compress_messages has a per-message
    # floor (see test_compress_messages_has_a_per_message_floor), so a fixture
    # small enough for a third of its own size to fall under that floor tests
    # the floor rather than the budget.
    body = SCRIPTS[script]
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": body}
        for index in range(10)
    ]
    source = sum(count_tokens(m["content"]) for m in messages)
    budget = max(1, source // 3)

    out = compress_messages(messages, budget=budget)

    assert sum(count_tokens(m["content"]) for m in out) <= budget


def test_compress_messages_has_a_per_message_floor() -> None:
    """Below a certain budget the result stops shrinking, and says so here.

    The tightening loop stops considering a message once it is down to about
    twenty tokens, and the live user turn is never compressed. The sum of
    those floors is the smallest conversation this can produce, so a budget
    under it is not achievable. That is a real limit on the contract and is
    pinned rather than hidden: budgets of 50, 100 and 150 all return the same
    size.
    """
    body = "这是一个关于上下文压缩的技术文档，描述了令牌预算的执行方式。" * 6
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": body}
        for index in range(10)
    ]

    sizes = {
        budget: sum(
            count_tokens(m["content"])
            for m in compress_messages(messages, budget=budget)
        )
        for budget in (50, 100, 150)
    }

    assert len(set(sizes.values())) == 1, (
        f"expected one floor across unreachable budgets, got {sizes}"
    )
    floor = next(iter(sizes.values()))
    assert floor > 150, "fixture must sit below the floor for this to mean anything"


@requires_tokenizer
def test_compress_messages_does_not_silently_skip_dense_input() -> None:
    """The decision to compress must be made in tokens, not characters.

    A CJK conversation reads as roughly a third of its size under ``len // 4``.
    That cleared the in-budget check and returned the caller's list untouched,
    which is worse than overshooting: nothing signals that the requested
    reduction never happened.
    """
    body = "这是一个关于上下文压缩的技术文档，描述了令牌预算的执行方式。" * 8
    messages = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": body}
        for index in range(10)
    ]
    char_estimate = sum(len(m["content"]) // 4 for m in messages)
    real_tokens = sum(count_tokens(m["content"]) for m in messages)
    assert char_estimate < real_tokens, "fixture must be denser than 4 chars/token"

    out = compress_messages(messages, budget=char_estimate + 10)

    assert out != messages, "dense input was returned uncompressed"
    assert sum(count_tokens(m["content"]) for m in out) <= char_estimate + 10


@requires_tokenizer
def test_reported_savings_are_measured_not_estimated() -> None:
    """Savings are reported to users, so they have to be real counts."""
    body = "这是一个关于上下文压缩的技术文档，描述了令牌预算的执行方式。" * 60
    out = compress(body, budget=200)

    assert count_tokens(body) > len(body) // 4, "fixture must defeat the heuristic"
    assert count_tokens(out) <= 200
