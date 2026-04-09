"""
Entroly SDK — 3-Line Integration
==================================

Drop-in context optimization for ANY AI application.
Works with LangChain, CrewAI, Agno, or raw API calls.

Usage::

    from entroly import compress
    compressed = compress(text, budget=2000)

    # Or with content type hint:
    compressed = compress(json_blob, budget=500, content_type="json")

    # Or compress a full message list (LLM conversation):
    from entroly import compress_messages
    messages = compress_messages(messages, budget=50000)

    # LangChain integration:
    from entroly.integrations.langchain import EntrolyCompressor
    chain = EntrolyCompressor(budget=30000) | llm
"""

from __future__ import annotations

from typing import Any

from .universal_compress import (
    detect_content_type,
    tfidf_extractive_summarize,
    universal_compress,
)


def compress(
    content: str,
    budget: int | None = None,
    content_type: str | None = None,
    target_ratio: float = 0.3,
) -> str:
    """Compress any content to fit within a token budget.

    This is the simplest possible API — one function, one import.

    Args:
        content: Any text content (code, prose, JSON, logs, emails, etc.)
        budget: Target token count. If set, overrides target_ratio.
        content_type: Optional hint ("json", "code", "prose", "log", etc.)
        target_ratio: Compression ratio if budget not specified (0.3 = keep 30%)

    Returns:
        Compressed text that preserves the most important information.

    Example::

        from entroly import compress

        # Compress a large API response
        compressed = compress(api_response, budget=1000)

        # Compress code with type hint
        compressed = compress(source_code, budget=2000, content_type="code")
    """
    if not content:
        return content

    # Estimate current token count (4 chars ≈ 1 token)
    current_tokens = len(content) // 4

    # If budget specified, compute target ratio from it
    if budget is not None and current_tokens > budget:
        target_ratio = max(0.05, budget / max(current_tokens, 1))
    elif budget is not None and current_tokens <= budget:
        return content  # Already within budget

    # Handle code content with the Rust engine if available
    if content_type == "code" or (content_type is None and _looks_like_code(content)):
        try:
            return _compress_code(content, target_ratio)
        except Exception:
            pass  # Fall through to universal compressor

    compressed, _, _ = universal_compress(content, target_ratio, content_type)
    return compressed


def compress_messages(
    messages: list[dict[str, Any]],
    budget: int = 50_000,
    preserve_last_n: int = 4,
) -> list[dict[str, Any]]:
    """Compress a conversation message list to fit within a token budget.

    Preserves the most recent messages verbatim and progressively
    compresses older messages using content-aware compression.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        budget: Target total token count for all messages
        preserve_last_n: Number of most recent messages to keep verbatim

    Returns:
        Compressed message list.

    Example::

        from entroly import compress_messages

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": very_long_response},
            {"role": "user", "content": "Fix the bug"},
        ]
        compressed = compress_messages(messages, budget=30000)
    """
    if not messages:
        return messages

    # Estimate total tokens
    total_tokens = sum(
        len(m.get("content", "")) // 4
        for m in messages if isinstance(m.get("content"), str)
    )
    if total_tokens <= budget:
        return messages  # Already within budget

    # Adaptive preserve_last_n: shrink if there aren't enough
    # older messages to compress
    effective_preserve = min(preserve_last_n, max(1, len(messages) - 1))

    # Split: recent (verbatim) vs older (compressible)
    recent = messages[-effective_preserve:]
    older = messages[:-effective_preserve]

    recent_tokens = sum(
        len(m.get("content", "")) // 4
        for m in recent if isinstance(m.get("content"), str)
    )
    remaining_budget = max(budget - recent_tokens, 500)

    # If recent alone busts budget, compress even recent messages
    # (except the very last user message)
    if recent_tokens > budget:
        return _compress_all_messages(messages, budget)

    if not older:
        # All messages are "recent" — compress all except the last
        return _compress_all_messages(messages, budget)

    # Compute per-message compression ratio
    older_tokens = sum(
        len(m.get("content", "")) // 4
        for m in older if isinstance(m.get("content"), str)
    )
    ratio = max(0.05, remaining_budget / max(older_tokens, 1))

    result = []
    for msg in older:
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < 200:
            result.append(msg)
            continue

        role = msg.get("role", "")
        # Tool results get more aggressive compression
        msg_ratio = ratio * 0.5 if role in ("tool", "function") else ratio

        compressed_content = compress(content, target_ratio=msg_ratio)
        new_msg = dict(msg)
        new_msg["content"] = compressed_content
        result.append(new_msg)

    result.extend(recent)
    return result


def _compress_all_messages(
    messages: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Compress all messages proportionally, preserving the last user message.

    Used when even the 'recent' window busts the token budget.
    Sorts messages by size and compresses the largest first.
    """
    # Always preserve the last user message verbatim
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") in ("user", "human"):
            last_user_idx = i
            break

    total_tokens = sum(
        len(m.get("content", "")) // 4
        for m in messages if isinstance(m.get("content"), str)
    )
    ratio = max(0.05, budget / max(total_tokens, 1))

    result = []
    for i, msg in enumerate(messages):
        if i == last_user_idx:
            result.append(msg)  # Preserve last user message
            continue

        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < 200:
            result.append(msg)
            continue

        role = msg.get("role", "")
        msg_ratio = ratio * 0.3 if role in ("tool", "function") else ratio

        compressed_content = compress(content, target_ratio=msg_ratio)
        new_msg = dict(msg)
        new_msg["content"] = compressed_content
        result.append(new_msg)

    return result


def _looks_like_code(text: str) -> bool:
    """Heuristic: does this text look like source code?"""
    code_indicators = 0
    lines = text[:2000].split("\n")
    for line in lines[:30]:
        stripped = line.strip()
        if any(kw in stripped for kw in [
            "def ", "class ", "import ", "from ", "fn ", "pub ",
            "function ", "const ", "let ", "var ", "return ",
            "if (", "for (", "while (", "struct ", "#include",
        ]):
            code_indicators += 1
        if stripped.endswith(("{", "}", ";", ":")):
            code_indicators += 1
    return code_indicators >= 3


def _compress_code(content: str, target_ratio: float) -> str:
    """Compress code using the Rust engine if available."""
    try:
        from entroly_core import py_compress_block
        return py_compress_block("assistant", content, len(content) // 4, "skeleton", None)
    except ImportError:
        # Rust engine not available — use universal compressor
        compressed, _, _ = universal_compress(content, target_ratio, "prose")
        return compressed
