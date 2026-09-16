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

    When *create_stubs* is True, any omitted historical turns are replaced
    by a structured, deterministic Merkle compaction stub containing the
    turn count, estimated token savings, and SHA-256 digest, preventing
    silent amnesia and preserving context receipt honesty.

    *token_counter* overrides the built-in ``count_tokens`` if the
    caller needs a model-specific tokenizer.
    """
    if not messages:
        return []
    if strategy not in ("last", "first"):
        raise ValueError(f"Unknown trim strategy: {strategy!r}")

    import hashlib
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
        return system

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
        dropped_tokens = sum(_msg_tokens(m) for m in dropped)
        canonical = json.dumps(dropped, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        if store_recovery:
            try:
                from .context_receipts.store import ensure_store, write_json
                store_dir = ensure_store()
                target_path = store_dir / f"compacted_{digest[:16]}.json"
                write_json(target_path, {
                    "digest": digest,
                    "count": len(dropped),
                    "tokens": dropped_tokens,
                    "messages": dropped,
                })
            except Exception:
                pass

        stub = {
            "role": "system",
            "content": (
                f"[ENTROLY CONTEXT COMPACTION: {len(dropped)} historical turn(s) "
                f"({dropped_tokens} tokens) compacted into Merkle stub. "
                f"Digest: sha256:{digest[:16]}... "
                f"Recoverable via: `entroly recover {digest[:16]}`]"
            ),
        }
        if strategy == "last":
            return system + [stub] + selected
        else:
            return system + selected + [stub]

    return system + selected

