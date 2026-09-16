"""Recoverable, request-anchored context boundaries for provider payloads.

Only complete earlier user turns may be omitted. The current turn, provider
schema, and any non-text content are preserved. A boundary is sent only after
the final payload fits the local estimate and CLI recovery of omitted items
has been verified. The estimate is not a provider tokenizer guarantee.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tokens import count_tokens

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextBoundary:
    body: dict[str, Any]
    omitted_messages: int
    estimated_tokens: int
    recovery_digest: str

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The selected provider-native sequence (for SDK callers)."""
        return self.body.get("messages", self.body.get("contents", self.body.get("input", [])))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def estimate_request_tokens(body: dict[str, Any]) -> int:
    """Local serialization-based estimate, including tools and system text."""
    return count_tokens(_json(body)) + 3


def _plain_text_content(content: Any) -> bool:
    if isinstance(content, str):
        return True
    return (
        isinstance(content, list)
        and all(
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and set(block) <= {"type", "text", "cache_control"}
            for block in content
        )
    )


def _sequence(
    body: dict[str, Any], provider: str,
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Return variant, sequence key, immutable prefix, and conversation."""
    if provider == "gemini" and isinstance(body.get("contents"), list):
        variant, key, items = "gemini", "contents", body["contents"]
        roles = {"user", "model"}
        def valid(item: dict[str, Any]) -> bool:
            return (
                set(item) == {"role", "parts"}
                and isinstance(item.get("parts"), list)
                and all(
                    isinstance(part, dict) and set(part) == {"text"}
                    and isinstance(part["text"], str)
                    for part in item["parts"]
                )
            )
    elif isinstance(body.get("messages"), list):
        variant = "anthropic" if provider == "anthropic" else "chat"
        key, items = "messages", body["messages"]
        roles = {"user", "assistant"}
        def valid(item: dict[str, Any]) -> bool:
            return set(item) == {"role", "content"} and _plain_text_content(item.get("content"))
    elif isinstance(body.get("input"), list):
        variant, key, items = "responses", "input", body["input"]
        roles = {"user", "assistant"}
        def valid(item: dict[str, Any]) -> bool:
            return set(item) == {"role", "content"} and _plain_text_content(item.get("content"))
    else:
        return None
    if not items or any(not isinstance(item, dict) or not valid(item) for item in items):
        return None

    prefix: list[dict[str, Any]] = []
    index = 0
    if variant in {"chat", "responses"}:
        while index < len(items) and items[index]["role"] in {"system", "developer"}:
            prefix.append(items[index])
            index += 1
    conversation = items[index:]
    if not conversation or conversation[0]["role"] != "user":
        return None
    if any(item["role"] not in roles for item in conversation):
        return None
    return variant, key, prefix, conversation


def _with_stub(
    body: dict[str, Any], variant: str, key: str, prefix: list[dict[str, Any]],
    remaining: list[dict[str, Any]], stub_text: str,
) -> dict[str, Any] | None:
    candidate = dict(body)
    if variant == "chat":
        candidate[key] = prefix + [{"role": "system", "content": stub_text}] + remaining
    elif variant == "responses":
        candidate[key] = prefix + remaining
        instructions = body.get("instructions", "")
        if not isinstance(instructions, str):
            return None
        candidate["instructions"] = (
            f"{instructions}\n\n{stub_text}" if instructions else stub_text
        )
    elif variant == "anthropic":
        candidate[key] = remaining
        system = body.get("system", "")
        if isinstance(system, str):
            candidate["system"] = f"{system}\n\n{stub_text}" if system else stub_text
        elif isinstance(system, list) and all(
            isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str) for block in system
        ):
            candidate["system"] = [*system, {"type": "text", "text": stub_text}]
        else:
            return None
    else:  # Gemini
        candidate[key] = remaining
        instruction = body.get("systemInstruction")
        if instruction is None:
            parts: list[dict[str, str]] = []
        elif isinstance(instruction, str):
            parts = [{"text": instruction}]
        elif (
            isinstance(instruction, dict)
            and set(instruction) == {"parts"}
            and isinstance(instruction["parts"], list)
            and all(
                isinstance(part, dict) and set(part) == {"text"}
                and isinstance(part["text"], str)
                for part in instruction["parts"]
            )
        ):
            parts = list(instruction["parts"])
        else:
            return None
        candidate["systemInstruction"] = {"parts": [*parts, {"text": stub_text}]}
    return candidate


def compact_request_context(
    body: dict[str, Any], *, provider: str, max_tokens: int,
    store_path: str | Path | None = None,
) -> ContextBoundary | None:
    """Select the largest complete historical suffix that fits and recovers."""
    if max_tokens <= 0:
        return None
    parsed = _sequence(body, provider)
    if parsed is None:
        return None
    variant, key, prefix, conversation = parsed
    turns: list[list[dict[str, Any]]] = []
    for item in conversation:
        if item["role"] == "user":
            turns.append([])
        turns[-1].append(item)
    if len(turns) < 2:
        return None

    history, active = turns[:-1], turns[-1]
    for kept_turns in range(len(history) - 1, -1, -1):
        omitted = [item for turn in history[: len(history) - kept_turns] for item in turn]
        remaining = [item for turn in history[len(history) - kept_turns :] for item in turn] + active
        payload = _json(omitted)
        from .codec import content_digest

        digest = content_digest(payload)
        stub = (
            f"[Entroly omitted {len(omitted)} earlier messages. "
            f"Exact local recovery: entroly recover {digest}]"
        )
        candidate = _with_stub(body, variant, key, prefix, remaining, stub)
        if candidate is None:
            return None
        try:
            estimated = estimate_request_tokens(candidate)
        except (TypeError, ValueError, OverflowError):
            return None
        if estimated > max_tokens:
            continue

        from .cli_recover import default_recovery_store_path
        from .codec import RecoveryStore

        path = store_path if store_path is not None else default_recovery_store_path()
        try:
            store = RecoveryStore(path)
            reference = store.put(
                payload, item_count=len(omitted), item_label="message(s)",
                note="request-anchored context boundary",
            )
            reopened = RecoveryStore(path)
            recovered_ref = reopened.reference_for(reference.digest)
            if recovered_ref is None or reopened.recover(recovered_ref) != payload:
                return None
        except Exception as exc:
            logger.warning("Context boundary recovery unavailable: %s", type(exc).__name__)
            return None
        return ContextBoundary(candidate, len(omitted), estimated, digest)
    return None


def compact_chat_history(
    messages: list[dict[str, Any]], *, max_tokens: int,
    store_path: str | Path | None = None,
) -> ContextBoundary | None:
    """Convenience facade for plain chat SDK callers."""
    return compact_request_context(
        {"messages": messages}, provider="openai", max_tokens=max_tokens,
        store_path=store_path,
    )
