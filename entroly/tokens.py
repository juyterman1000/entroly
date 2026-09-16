"""Canonical token counting for the entire Entroly package.

Every internal caller that needs token counts should import from here.
Tiktoken ``o200k_base`` is the primary encoder (GPT-4o / Claude family).
When tiktoken is not installed the fallback is deliberately conservative
(``ceil(len(text) / 4)``), so budget math over-reserves rather than
overflows.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@lru_cache(maxsize=1)
def _encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("o200k_base")
    except (ImportError, ValueError):
        return None


def count_tokens(text: str) -> int:
    """Return the exact o200k token count, or a conservative estimate."""
    if not text:
        return 0
    enc = _encoding()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, (len(text) + 3) // 4)


def estimate_tokens(text: str) -> int:
    """Alias kept for callers that do not need exact counts."""
    return count_tokens(text)


def count_messages_tokens(
    messages: Sequence[dict],
    *,
    per_message_overhead: int = 4,
    reply_priming: int = 3,
) -> int:
    """Approximate total tokens for a chat-style message list.

    Overhead constants follow the OpenAI token-counting cookbook:
    4 tokens per message (role, separators) + 3 for reply priming.
    """
    total = reply_priming
    for msg in messages:
        total += per_message_overhead
        for value in msg.values():
            if isinstance(value, str):
                total += count_tokens(value)
            elif isinstance(value, list):
                for part in value:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or ""
                        if isinstance(text, str):
                            total += count_tokens(text)
    return total


def trim_messages(
    messages: list[dict],
    *,
    max_tokens: int,
    strategy: str = "last",
    include_system: bool = True,
    token_counter: Callable[[str], int] | None = None,
    create_stubs: bool = False,
    store_recovery: bool = False,
) -> list[dict]:
    """Trim a chat message list to fit within *max_tokens*.

    Strategies:
      - ``"last"``:  keep the most recent messages (default).
      - ``"first"``: keep the oldest messages.

    When *include_system* is True (default), any leading message whose
    ``role`` is ``"system"`` is always retained.  Its token cost is
    deducted from *max_tokens* before the remaining messages are
    selected.

    When *create_stubs* is True, omitted messages are represented by a
    content digest.  A recovery command is advertised only after exact local
    storage has been verified.  If the final stub cannot fit, the original
    messages are returned unchanged rather than silently losing the task.

    *token_counter* overrides the built-in ``count_tokens`` if the
    caller needs a model-specific tokenizer.
    """
    if not messages:
        return []
    if strategy not in ("last", "first"):
        raise ValueError(f"Unknown trim strategy: {strategy!r}")

    import json

    counter = token_counter or count_tokens
    per_msg = 4
    reply_priming = 3

    system: list[dict] = []
    rest: list[dict] = []
    budget = max_tokens - reply_priming

    for msg in messages:
        if include_system and msg.get("role") == "system" and not rest:
            system.append(msg)
            cost = per_msg
            for v in msg.values():
                if isinstance(v, str):
                    cost += counter(v)
            budget -= cost
        else:
            rest.append(msg)

    if budget <= 0:
        return messages if create_stubs else system

    def _msg_tokens(msg: dict) -> int:
        t = per_msg
        for v in msg.values():
            if isinstance(v, str):
                t += counter(v)
            elif isinstance(v, list):
                for part in v:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or ""
                        if isinstance(text, str):
                            t += counter(text)
        return t

    dropped: list[dict] = []
    if strategy == "last":
        selected: list[dict] = []
        remaining = budget
        for idx, msg in enumerate(reversed(rest)):
            cost = _msg_tokens(msg)
            if remaining - cost < 0:
                # All preceding messages in rest are dropped
                cutoff = len(rest) - idx
                dropped = rest[:cutoff]
                break
            selected.append(msg)
            remaining -= cost
        selected.reverse()
    else:
        selected = []
        remaining = budget
        for idx, msg in enumerate(rest):
            cost = _msg_tokens(msg)
            if remaining - cost < 0:
                dropped = rest[idx:]
                break
            selected.append(msg)
            remaining -= cost

    if create_stubs and dropped:
        from .codec import content_digest

        # Recompute the complete representation after each omission.  A digest
        # and decimal count change stub tokenization, so a fixed reserve can
        # still overflow a tight budget.
        while True:
            if strategy == "last" and (not selected or selected[-1] is not rest[-1]):
                return messages
            canonical = json.dumps(dropped, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
            digest = content_digest(canonical)
            detail = (
                f"Recoverable via: `entroly recover {digest}`"
                if store_recovery else f"Digest: {digest}"
            )
            stub = {
                "role": "system",
                "content": (
                    f"[ENTROLY CONTEXT COMPACTION: {len(dropped)} omitted message(s). "
                    f"{detail}]"
                ),
            }
            compacted = (
                system + [stub] + selected if strategy == "last"
                else system + selected + [stub]
            )
            if reply_priming + sum(_msg_tokens(msg) for msg in compacted) <= max_tokens:
                break
            if not selected:
                return messages
            if strategy == "last":
                dropped.append(selected.pop(0))
            else:
                dropped.insert(0, selected.pop())

        if store_recovery:
            from .cli_recover import default_recovery_store_path
            from .codec import RecoveryStore

            try:
                path = default_recovery_store_path()
                store = RecoveryStore(path)
                reference = store.put(canonical, item_count=len(dropped), item_label="message(s)")
                reopened = RecoveryStore(path)
                recovered_ref = reopened.reference_for(reference.digest)
                if recovered_ref is None or reopened.recover(recovered_ref) != canonical:
                    return messages
            except Exception:
                return messages
        return compacted

    return system + selected

