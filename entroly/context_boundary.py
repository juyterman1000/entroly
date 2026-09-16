"""Recoverable, request-anchored context boundaries for provider payloads.

Only complete earlier user turns may be omitted. The current turn and provider
schema are preserved; requests with media pass through. A boundary is sent only after
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


def _contains_media(value: Any) -> bool:
    """Avoid budgeting provider media from its small URL or opaque metadata."""
    if isinstance(value, list):
        return any(_contains_media(item) for item in value)
    if not isinstance(value, dict):
        return False
    if any(key in value for key in (
        "image_url", "image", "inlineData", "inline_data", "fileData",
        "file_data", "audio", "video", "input_audio", "input_image",
    )):
        return True
    if value.get("type") in {
        "image", "input_image", "audio", "input_audio", "video",
        "document", "file", "input_file",
    }:
        return True
    if isinstance(value.get("source"), dict) and value["source"].get("type") in {
        "base64", "url", "file",
    }:
        return True
    return any(_contains_media(item) for item in value.values())


def _sequence(
    body: dict[str, Any], provider: str,
) -> tuple[str, str, list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Select a wire contract from the payload, using provider as a hint."""
    sequence_keys = [
        key for key in ("messages", "contents", "input")
        if isinstance(body.get(key), list)
    ]
    if len(sequence_keys) != 1:
        return None
    if sequence_keys[0] == "contents":
        variant, key, items = "gemini", "contents", body["contents"]
        roles = {"user", "model"}
        def valid(item: dict[str, Any]) -> bool:
            return (
                item.get("role") in roles
                and isinstance(item.get("parts"), list)
                and all(isinstance(part, dict) for part in item["parts"])
                and not _contains_media(item)
            )
    elif sequence_keys[0] == "messages":
        native_system = "system" in body and not any(
            isinstance(item, dict) and item.get("role") in {"system", "developer"}
            for item in body["messages"]
        )
        variant = "anthropic" if provider == "anthropic" or native_system else "chat"
        key, items = "messages", body["messages"]
        roles = {"user", "assistant"}
        def valid(item: dict[str, Any]) -> bool:
            return (
                item.get("role") in {*roles, "system", "developer", "tool", "function"}
                and not _contains_media(item)
            )
    elif sequence_keys[0] == "input":
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
    permitted_roles = roles | ({"tool", "function"} if variant == "chat" else set())
    if any(item["role"] not in permitted_roles for item in conversation):
        return None
    return variant, key, prefix, conversation


def _starts_user_turn(item: dict[str, Any], variant: str) -> bool:
    if item["role"] != "user":
        return False
    if variant == "anthropic" and isinstance(item.get("content"), list):
        if any(
            isinstance(block, dict) and block.get("type") == "tool_result"
            for block in item["content"]
        ):
            return False
    if variant == "gemini" and isinstance(item.get("parts"), list):
        if any(
            isinstance(part, dict) and "functionResponse" in part
            for part in item["parts"]
        ):
            return False
    return True


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
    """Select a recent complete historical suffix that fits and recovers."""
    if max_tokens <= 0:
        return None
    parsed = _sequence(body, provider)
    if parsed is None:
        return None
    variant, key, prefix, conversation = parsed
    turns: list[list[dict[str, Any]]] = []
    for item in conversation:
        if _starts_user_turn(item, variant):
            turns.append([])
        if not turns:
            return None
        turns[-1].append(item)
    if len(turns) < 2:
        return None

    history, active = turns[:-1], turns[-1]

    def proposal(omitted_turns: int):
        omitted = [item for turn in history[:omitted_turns] for item in turn]
        remaining = [item for turn in history[omitted_turns:] for item in turn] + active
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
        return candidate, estimated, digest, payload, len(omitted)

    # Exponential bracketing avoids serializing a long request once for every
    # historical turn. Binary refinement then finds a recent fitting suffix.
    # Stub digests make exact token counts slightly non-monotonic; every
    # returned candidate is still checked against the complete final payload.
    failed = 0
    omitted_turns = 1
    selected = None
    while True:
        candidate = proposal(omitted_turns)
        if candidate is None:
            return None
        if candidate[1] <= max_tokens:
            selected = candidate
            break
        failed = omitted_turns
        if omitted_turns == len(history):
            return None
        omitted_turns = min(len(history), omitted_turns * 2)
    fitting = omitted_turns
    while fitting - failed > 1:
        middle = (fitting + failed) // 2
        candidate = proposal(middle)
        if candidate is None:
            return None
        if candidate[1] <= max_tokens:
            fitting = middle
            selected = candidate
        else:
            failed = middle

    if selected is None:
        return None
    candidate_body, estimated, digest, payload, omitted_count = selected

    from .cli_recover import default_recovery_store_path
    from .codec import RecoveryStore

    path = store_path if store_path is not None else default_recovery_store_path()
    try:
        store = RecoveryStore(path)
        reference = store.put(
            payload, item_count=omitted_count, item_label="message(s)",
            note="request-anchored context boundary",
        )
        reopened = RecoveryStore(path)
        recovered_ref = reopened.reference_for(reference.digest)
        if recovered_ref is None or reopened.recover(recovered_ref) != payload:
            return None
    except Exception as exc:
        logger.warning("Context boundary recovery unavailable: %s", type(exc).__name__)
        return None
    return ContextBoundary(candidate_body, omitted_count, estimated, digest)


def compact_chat_history(
    messages: list[dict[str, Any]], *, max_tokens: int,
    store_path: str | Path | None = None,
) -> ContextBoundary | None:
    """Convenience facade for plain chat SDK callers."""
    return compact_request_context(
        {"messages": messages}, provider="openai", max_tokens=max_tokens,
        store_path=store_path,
    )
