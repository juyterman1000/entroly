"""Local JSONL bridge for the Entroly OpenClaw context-engine plugin."""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from .context_receipts.retrieval import tokenize
from .sdk import compress

BRIDGE_SCHEMA = "entroly.openclaw.bridge.v2"
RECEIPT_SCHEMA = "entroly.openclaw.receipt.v2"
DEFAULT_PRESERVE_LAST_N = 4
PROVIDER_MODE = "openclaw_managed"
_BUDGET_SOURCES = {
    "openclaw_token_budget",
    "openclaw_runtime_settings",
    "entroly_model_registry",
    "operator_fallback",
}
DEFAULT_RECEIPT_MAX_FILES = 512
DEFAULT_RECEIPT_MAX_BYTES = 64 * 1024 * 1024
_EMPTY_ACCEPTANCE_SIGNATURE = "0" * 64


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _signing_key_id(key: bytes) -> str:
    """Return a non-secret identifier that detects receipt-key rotation."""
    return hashlib.sha256(b"entroly.openclaw.receipt-key.v1\0" + key).hexdigest()


def _hmac_sha256(key: bytes, label: str, value: str) -> str:
    payload = f"entroly.openclaw.{label}.v1:{value}".encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _estimate_value_tokens(value: Any) -> int:
    if isinstance(value, str):
        return max(1, (len(value) + 3) // 4) if value else 0
    if value is None or isinstance(value, (bool, int, float)):
        return 1
    if isinstance(value, list):
        return sum(_estimate_value_tokens(item) for item in value)
    if isinstance(value, dict):
        return sum(
            _estimate_value_tokens(key) + _estimate_value_tokens(item)
            for key, item in value.items()
        )
    return _estimate_value_tokens(str(value))


def _compressible_text_locations(message: dict[str, Any]) -> list[tuple[int | None, str]]:
    """Return unsigned normalized text fields that Entroly may safely rewrite."""
    content = message.get("content")
    if isinstance(content, str):
        return [(None, content)]
    if not isinstance(content, list):
        return []
    locations: list[tuple[int | None, str]] = []
    for index, block in enumerate(content):
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and set(block) <= {"type", "text"}
        ):
            locations.append((index, block["text"]))
    return locations


# A tool_result block carries protocol identity (`tool_use_id`) and an error
# flag the agent branches on. Only its textual payload may be rewritten, and
# only when the block is otherwise shaped exactly as the protocol defines it.
# Anything carrying a key outside this set is treated as opaque and passed
# through untouched, so an unrecognized extension can never be silently eaten.
_TOOL_RESULT_KEYS = frozenset({"type", "tool_use_id", "content", "is_error"})


def _is_plain_text_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and set(block) <= {"type", "text"}
    )


def _tool_result_text_locations(
    message: dict[str, Any],
) -> list[tuple[int, int | None, str]]:
    """Return ``(block_index, inner_index, text)`` for rewritable tool output.

    ``inner_index`` is ``None`` when the block's ``content`` is a bare string
    and an index when it is a list of plain text blocks. Tool results are the
    bulk of a long agent session -- a single pytest or compiler run dwarfs the
    prose around it -- but they are excluded from
    ``_compressible_text_locations`` because that helper is deliberately
    restricted to unsigned free text.
    """

    content = message.get("content")
    if not isinstance(content, list):
        return []

    locations: list[tuple[int, int | None, str]] = []
    for block_index, block in enumerate(content):
        if (
            not isinstance(block, dict)
            or block.get("type") != "tool_result"
            or not set(block) <= _TOOL_RESULT_KEYS
        ):
            continue
        inner = block.get("content")
        if isinstance(inner, str):
            locations.append((block_index, None, inner))
        elif isinstance(inner, list):
            for inner_index, inner_block in enumerate(inner):
                if _is_plain_text_block(inner_block):
                    locations.append((block_index, inner_index, inner_block["text"]))
    return locations


def _compress_tool_results(
    message: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Compress a message's tool_result payloads. Returns (message, tokens_saved).

    Delegates to the proxy's ``compress_tool_output``, which already applies the
    specialized codecs, then the pattern rules, then ESC. Reusing it keeps a
    single definition of what a compressed tool result looks like across the
    proxy and the OpenClaw bridge rather than growing a second one here.
    """

    locations = _tool_result_text_locations(message)
    if not locations:
        return message, 0

    try:
        from .proxy_transform import compress_tool_output
    except Exception:
        return message, 0

    updated = copy.deepcopy(message)
    saved = 0
    for block_index, inner_index, raw_text in locations:
        try:
            compressed, kind, _ratio = compress_tool_output(raw_text)
        except Exception:
            # Fail open: an unexpected payload keeps its original bytes.
            continue
        if kind == "none" or compressed == raw_text:
            continue
        delta = _estimate_value_tokens(raw_text) - _estimate_value_tokens(compressed)
        if delta <= 0:
            continue
        saved += delta
        if inner_index is None:
            updated["content"][block_index]["content"] = compressed
        else:
            updated["content"][block_index]["content"][inner_index]["text"] = compressed
    if saved == 0:
        return message, 0
    return updated, saved


def _message_text(message: dict[str, Any], *, compressible_only: bool) -> str:
    if compressible_only:
        return "\n".join(text for _, text in _compressible_text_locations(message))
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(block["text"])
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def _provider_visible_message_tokens(message: dict[str, Any]) -> int:
    """Estimate normalized prompt content without routing/accounting metadata."""
    role = message.get("role")
    if role not in {"user", "assistant", "toolResult", "system", "developer", "human"}:
        return _estimate_value_tokens(message) + 3

    content = message.get("content")
    content_tokens = _estimate_value_tokens(content)
    envelope_tokens = 4
    if role == "toolResult":
        envelope_tokens += _estimate_value_tokens(message.get("toolCallId"))
        envelope_tokens += _estimate_value_tokens(message.get("toolName"))
        envelope_tokens += 1  # isError
    return content_tokens + envelope_tokens


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate provider-visible normalized context, independent of routing metadata."""
    return sum(_provider_visible_message_tokens(message) for message in messages)


def _compressible_text_tokens(message: dict[str, Any]) -> int:
    return sum(_estimate_value_tokens(text) for _, text in _compressible_text_locations(message))


def _tool_result_text_tokens(message: dict[str, Any]) -> int:
    return sum(
        _estimate_value_tokens(text) for _, _, text in _tool_result_text_locations(message)
    )


def _fixed_message_tokens(message: dict[str, Any]) -> int:
    """Tokens the budget cannot reclaim from this message.

    Tool output is subtracted alongside free text. Counting it as fixed made
    the arithmetic self-defeating: a single large log inflated the fixed total
    past the whole budget, `content_budget` went non-positive, and the
    assembler returned the original context -- so the one message responsible
    for the overflow was the one guaranteed to survive it whole.
    """

    reclaimable = _compressible_text_tokens(message) + _tool_result_text_tokens(message)
    return max(0, _provider_visible_message_tokens(message) - reclaimable)


def _protected_message(message: dict[str, Any]) -> bool:
    """Return whether a message must remain byte-for-byte equivalent."""
    if message.get("role") in {"system", "developer"}:
        return True
    # "No rewritable free text" used to imply "structured, therefore
    # untouchable". A tool_result-only message satisfies that test, which made
    # every large tool log permanently exempt from compression.
    return not (
        _compressible_text_locations(message) or _tool_result_text_locations(message)
    )


def _message_relevance(
    messages: list[dict[str, Any]], query: str
) -> list[dict[str, Any]]:
    query_terms = sorted(set(tokenize(query)))
    if not query_terms:
        return [
            {"score": 0.0, "matched_terms": [], "token_count": 0}
            for _ in messages
        ]

    documents = [
        tokenize(_message_text(message, compressible_only=True))
        for message in messages
    ]
    document_frequency: dict[str, int] = defaultdict(int)
    for terms in documents:
        for term in set(terms):
            document_frequency[term] += 1
    document_count = max(1, len(documents))
    average_length = max(
        1.0, sum(len(terms) for terms in documents) / document_count
    )

    ranked: list[dict[str, Any]] = []
    for index, terms in enumerate(documents):
        frequencies = Counter(terms)
        document_length = max(1, len(terms))
        matched = sorted(term for term in query_terms if frequencies.get(term, 0))
        score = 0.0
        for term in matched:
            frequency = frequencies[term]
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(
                1.0
                + (document_count - frequency_docs + 0.5)
                / (frequency_docs + 0.5)
            )
            normalization = 1.0 - 0.75 + 0.75 * document_length / average_length
            score += inverse_frequency * (frequency * 2.2) / (
                frequency + 1.2 * normalization
            )
        coverage = len(matched) / max(1, len(query_terms))
        pin_eligible = True
        security_flags: list[str] = []
        pin_blocked_reason: str | None = None
        if matched:
            try:
                from .context_firewall import scan

                scan_result = scan(
                    _message_text(messages[index], compressible_only=False),
                    source=f"openclaw_message_{index}",
                    check_repetition=False,
                )
                security_flags = sorted(
                    {
                        f"{threat.severity}:{threat.threat_type}"
                        for threat in scan_result.threats
                    }
                )
                pin_eligible = scan_result.is_safe
                if not pin_eligible:
                    pin_blocked_reason = "context_firewall"
            except Exception:
                pin_eligible = False
                security_flags = ["critical:scanner_error"]
                pin_blocked_reason = "context_firewall_error"
        ranked.append(
            {
                "score": round(score * (1.0 + coverage), 6),
                "matched_terms": matched,
                "token_count": len(terms),
                "pin_eligible": pin_eligible,
                "security_flags": security_flags,
                "pin_blocked_reason": pin_blocked_reason,
            }
        )
    return ranked


def _query_for_request(
    request: dict[str, Any], messages: list[dict[str, Any]]
) -> str:
    prompt = request.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    for message in reversed(messages):
        if message.get("role") not in {"user", "human"}:
            continue
        content = _message_text(message, compressible_only=False)
        if content.strip():
            return content.strip()
    return ""


def _compress_message_bodies(
    messages: list[dict[str, Any]],
    *,
    content_budget: int,
    distill: bool,
    query: str,
    compress_tool_results: bool = True,
) -> tuple[list[dict[str, Any]], set[int], list[dict[str, Any]], int, int]:
    # Tool output is compressed before the text budget is allocated. A single
    # test or compiler run can outweigh every prose message around it, so
    # spending the budget on prose first would squeeze the explanation while
    # the log that caused the overflow passes through whole.
    tool_tokens_saved = 0
    if compress_tool_results:
        reduced: list[dict[str, Any]] = []
        for message in messages:
            message, saved = _compress_tool_results(message)
            tool_tokens_saved += saved
            reduced.append(message)
        messages = reduced

    text_tokens = [
        max(1, _compressible_text_tokens(message)) for message in messages
    ]
    relevance = _message_relevance(messages, query)
    reserve_for_compression = min(int(content_budget * 0.35), len(messages) * 8)
    pin_budget = min(
        int(content_budget * 0.65), max(0, content_budget - reserve_for_compression)
    )
    pinned: set[int] = set()
    for index in sorted(
        range(len(messages)),
        key=lambda item: (-relevance[item]["score"], text_tokens[item], item),
    ):
        if (
            relevance[index]["score"] <= 0
            or not relevance[index]["matched_terms"]
            or not relevance[index]["pin_eligible"]
        ):
            continue
        if text_tokens[index] <= pin_budget:
            pinned.add(index)
            pin_budget -= text_tokens[index]

    unpinned = [index for index in range(len(messages)) if index not in pinned]
    remaining_budget = max(
        0, content_budget - sum(text_tokens[index] for index in pinned)
    )
    allocations = {index: text_tokens[index] for index in pinned}
    if unpinned:
        distributable = max(0, remaining_budget - len(unpinned))
        max_score = max((relevance[index]["score"] for index in unpinned), default=0.0)
        weights = {
            index: math.sqrt(text_tokens[index])
            * (1.0 + 2.0 * relevance[index]["score"] / max(1.0, max_score))
            for index in unpinned
        }
        total_weight = max(1.0, sum(weights.values()))
        for index in unpinned:
            allocations[index] = 1 + int(
                distributable * weights[index] / total_weight
            )

    result: list[dict[str, Any]] = []
    distillation_failures = 0
    for index, message in enumerate(messages):
        if index in pinned:
            result.append(copy.deepcopy(message))
            continue
        compressed_message = copy.deepcopy(message)
        locations = _compressible_text_locations(message)
        location_tokens = [max(1, _estimate_value_tokens(text)) for _, text in locations]
        total_tokens = max(1, sum(location_tokens))
        remaining = max(1, allocations[index])
        location_budgets: list[int] = []
        for location_index, token_count in enumerate(location_tokens):
            locations_left = len(location_tokens) - location_index - 1
            if locations_left == 0:
                allocated = max(1, remaining)
            else:
                proportional = max(1, int(allocations[index] * token_count / total_tokens))
                allocated = min(proportional, max(1, remaining - locations_left))
            location_budgets.append(allocated)
            remaining -= allocated

        for (block_index, raw_text), block_budget in zip(
            locations, location_budgets, strict=True
        ):
            content = raw_text
            if distill and message.get("role") == "assistant":
                try:
                    from .proxy_transform import distill_response

                    content, _, _ = distill_response(content, mode="full")
                except Exception:
                    distillation_failures += 1
            compressed_text = compress(content, budget=max(1, block_budget))
            if block_index is None:
                compressed_message["content"] = compressed_text
            else:
                compressed_message["content"][block_index]["text"] = compressed_text
        result.append(compressed_message)
    for index, item in enumerate(relevance):
        item["allocated_tokens"] = allocations[index]
    return result, pinned, relevance, distillation_failures, tool_tokens_saved


def _safe_session_name(session_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", session_id).strip("-.")
    return (safe or "session")[:80]


def _bounded_text(value: Any, limit: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()[:limit]


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _runtime_audit_metadata(request: dict[str, Any]) -> dict[str, Any]:
    """Copy only the bounded, non-secret OpenClaw runtime fields we audit."""
    raw = request.get("openclaw_runtime")
    raw = raw if isinstance(raw, dict) else {}
    runtime = raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    limits = raw.get("limits") if isinstance(raw.get("limits"), dict) else {}
    discovery = (
        raw.get("context_discovery")
        if isinstance(raw.get("context_discovery"), dict)
        else {}
    )
    resolved = _bounded_text(model.get("resolved"), 256)
    if resolved is None:
        resolved = _bounded_text(request.get("model"), 256)
    return {
        "schema_version": 1 if raw.get("schema_version") == 1 else None,
        "runtime": {
            "host": _bounded_text(runtime.get("host"), 64),
            "mode": _bounded_text(runtime.get("mode"), 64),
            "harness_id": _bounded_text(runtime.get("harness_id"), 128),
            "runtime_id": _bounded_text(runtime.get("runtime_id"), 128),
        },
        "model": {
            "requested": _bounded_text(model.get("requested"), 256),
            "resolved": resolved,
            "provider": _bounded_text(model.get("provider"), 128),
            "family": _bounded_text(model.get("family"), 128),
        },
        "limits": {
            "prompt_token_budget": _positive_int_or_none(
                limits.get("prompt_token_budget")
            ),
            "max_output_tokens": _positive_int_or_none(
                limits.get("max_output_tokens")
            ),
        },
        "context_discovery": {
            "status": _bounded_text(discovery.get("status"), 32),
            "trust": _bounded_text(discovery.get("trust"), 32),
            "model_id": _bounded_text(discovery.get("model_id"), 256),
            "exact": discovery.get("exact") if isinstance(discovery.get("exact"), bool) else None,
            "context_window": _positive_int_or_none(discovery.get("context_window")),
            "output_reserve_tokens": _positive_int_or_none(
                discovery.get("output_reserve_tokens")
            ),
            "safety_tokens": _positive_int_or_none(discovery.get("safety_tokens")),
            "registry_digest": _bounded_text(discovery.get("registry_digest"), 64),
            "source": _bounded_text(discovery.get("source"), 512),
        },
    }


def _resolve_context_budget(request: dict[str, Any]) -> dict[str, Any]:
    """Resolve a safe prompt budget from trusted, local model metadata.

    This is a fallback for OpenClaw hosts that cannot provide a finite prompt
    budget. It never accepts announced or generic fallback metadata and never
    enables remote discovery by itself. Remote discovery remains an explicit
    Entroly operator opt-in through the model-registry environment controls.
    """
    model = _bounded_text(request.get("model"), 256)
    if model is None:
        return {
            "schema_version": BRIDGE_SCHEMA,
            "ok": True,
            "status": "unavailable",
            "warning": "OpenClaw did not identify the active model route.",
        }

    from .models.registry import RegistryTrust, resolve_model

    resolution = resolve_model(model)
    capability = resolution.capability
    accepted_trust = {
        RegistryTrust.VERIFIED,
        RegistryTrust.USER,
        RegistryTrust.DISCOVERED,
    }
    if (
        capability is None
        or capability.context_window is None
        or resolution.trust not in accepted_trust
    ):
        return {
            "schema_version": BRIDGE_SCHEMA,
            "ok": True,
            "status": "unavailable",
            "model": model,
            "trust": resolution.trust.value,
            "warning": resolution.warning
            or (
                f"Model metadata for {model!r} is {resolution.trust.value}; "
                "Entroly requires verified, user-supplied, or directly discovered limits."
            ),
            "registry_digest": resolution.registry_digest,
        }

    context_window = capability.context_window
    requested_output = _positive_int_or_none(request.get("requested_output_tokens"))
    if requested_output is not None:
        output_reserve = requested_output
        if capability.max_output_tokens is not None:
            output_reserve = min(output_reserve, capability.max_output_tokens)
    elif capability.max_output_tokens is not None:
        output_reserve = capability.max_output_tokens
    else:
        # Unknown output limits must not turn the native context window into an
        # input budget. Reserve 10% (at least 4K) in addition to the 5% safety
        # margin used below.
        output_reserve = max(4096, math.ceil(context_window * 0.10))

    safety_tokens = max(512, math.ceil(context_window * 0.05))
    token_budget = context_window - output_reserve - safety_tokens
    if token_budget < 1024:
        return {
            "schema_version": BRIDGE_SCHEMA,
            "ok": True,
            "status": "unavailable",
            "model": model,
            "trust": resolution.trust.value,
            "warning": "Trusted model limits leave less than 1,024 prompt tokens after reserves.",
            "registry_digest": resolution.registry_digest,
        }

    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "status": "resolved",
        "token_budget": token_budget,
        "budget_source": "entroly_model_registry",
        "requested_model": model,
        "model_id": capability.id,
        "provider": capability.provider,
        "trust": resolution.trust.value,
        "exact": resolution.exact,
        "context_window": context_window,
        "max_output_tokens": capability.max_output_tokens,
        "output_reserve_tokens": output_reserve,
        "safety_tokens": safety_tokens,
        "registry_digest": resolution.registry_digest,
        "base_registry_digest": resolution.base_registry_digest,
        "source": capability.source,
        "verified_at": capability.verified_at,
        "observed_at": capability.observed_at,
    }


def _validate_assembly_invariants(
    source_messages: list[dict[str, Any]],
    assembled_messages: list[dict[str, Any]],
    protected_indexes: set[int],
) -> None:
    """Prove that only eligible unsigned text fields changed."""
    if len(source_messages) != len(assembled_messages):
        raise ValueError("message count changed")
    for index, (source, assembled) in enumerate(
        zip(source_messages, assembled_messages, strict=True)
    ):
        if not isinstance(assembled, dict) or source.get("role") != assembled.get("role"):
            raise ValueError(f"message role changed at index {index}")
        source_metadata = {key: value for key, value in source.items() if key != "content"}
        assembled_metadata = {
            key: value for key, value in assembled.items() if key != "content"
        }
        if _canonical_json(source_metadata) != _canonical_json(assembled_metadata):
            raise ValueError(f"message metadata changed at index {index}")

        source_content = source.get("content")
        assembled_content = assembled.get("content")
        if index in protected_indexes:
            if _canonical_json(source_content) != _canonical_json(assembled_content):
                raise ValueError(f"protected message changed at index {index}")
            continue
        if isinstance(source_content, str):
            if not isinstance(assembled_content, str):
                raise ValueError(f"text shape changed at index {index}")
            continue
        if not isinstance(source_content, list) or not isinstance(
            assembled_content, list
        ):
            if _canonical_json(source_content) != _canonical_json(assembled_content):
                raise ValueError(f"opaque content changed at index {index}")
            continue
        if len(source_content) != len(assembled_content):
            raise ValueError(f"content block count changed at index {index}")
        eligible = {
            block_index
            for block_index, _ in _compressible_text_locations(source)
            if block_index is not None
        }
        eligible_tool_results = {
            block_index for block_index, _, _ in _tool_result_text_locations(source)
        }
        for block_index, (source_block, assembled_block) in enumerate(
            zip(source_content, assembled_content, strict=True)
        ):
            if block_index in eligible:
                if not (
                    isinstance(assembled_block, dict)
                    and assembled_block.get("type") == "text"
                    and isinstance(assembled_block.get("text"), str)
                    and set(assembled_block) <= {"type", "text"}
                ):
                    raise ValueError(
                        f"text block shape changed at index {index}:{block_index}"
                    )
            elif block_index in eligible_tool_results:
                # A tool_result may lose payload text but must keep its
                # protocol identity. `tool_use_id` pairs the result with the
                # call that produced it and `is_error` is what the agent
                # branches on, so both are compared exactly; only `content`
                # may differ, and only by staying the same shape.
                if not isinstance(assembled_block, dict):
                    raise ValueError(
                        f"tool result shape changed at index {index}:{block_index}"
                    )
                source_identity = {
                    key: value
                    for key, value in source_block.items()
                    if key != "content"
                }
                assembled_identity = {
                    key: value
                    for key, value in assembled_block.items()
                    if key != "content"
                }
                if _canonical_json(source_identity) != _canonical_json(
                    assembled_identity
                ):
                    raise ValueError(
                        f"tool result identity changed at index {index}:{block_index}"
                    )
                source_inner = source_block.get("content")
                assembled_inner = assembled_block.get("content")
                if isinstance(source_inner, str):
                    if not isinstance(assembled_inner, str):
                        raise ValueError(
                            f"tool result text shape changed at index {index}:{block_index}"
                        )
                elif isinstance(source_inner, list):
                    if not isinstance(assembled_inner, list) or len(
                        source_inner
                    ) != len(assembled_inner):
                        raise ValueError(
                            f"tool result block count changed at index {index}:{block_index}"
                        )
                    for inner_index, (source_inner_block, assembled_inner_block) in (
                        enumerate(zip(source_inner, assembled_inner, strict=True))
                    ):
                        if _is_plain_text_block(source_inner_block):
                            if not _is_plain_text_block(assembled_inner_block):
                                raise ValueError(
                                    "tool result text block shape changed at "
                                    f"index {index}:{block_index}:{inner_index}"
                                )
                        elif _canonical_json(source_inner_block) != _canonical_json(
                            assembled_inner_block
                        ):
                            raise ValueError(
                                "opaque tool result block changed at "
                                f"index {index}:{block_index}:{inner_index}"
                            )
                elif _canonical_json(source_inner) != _canonical_json(assembled_inner):
                    raise ValueError(
                        f"opaque tool result changed at index {index}:{block_index}"
                    )
            elif _canonical_json(source_block) != _canonical_json(assembled_block):
                raise ValueError(
                    f"opaque block changed at index {index}:{block_index}"
                )


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _receipt_signing_key_path() -> Path:
    configured = os.environ.get("ENTROLY_OPENCLAW_RECEIPT_KEY_FILE")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError(
                "ENTROLY_OPENCLAW_RECEIPT_KEY_FILE must be an absolute path outside the receipt store"
            )
        return path.absolute()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        state_root = Path(os.environ["LOCALAPPDATA"])
    else:
        state_root = Path(
            os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
        )
    return (state_root.expanduser().absolute() / "entroly" / "openclaw-receipt.key")


@contextmanager
def _receipt_key_init_lock(parent: Path):
    lock_path = parent / ".openclaw-receipt-key.lock"
    deadline = time.monotonic() + 2.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 30:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"OpenClaw receipt signing key initialization is busy at {lock_path}"
                )
            time.sleep(0.025)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _read_receipt_signing_key(path: Path) -> bytes:
    if path.is_symlink():
        raise PermissionError(f"OpenClaw receipt signing key must not be a symlink: {path}")
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError(
            f"OpenClaw receipt signing key is corrupted at {path}; preserve the file and restore its 32-byte backup before retrying"
        )
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise PermissionError(
            f"OpenClaw receipt signing key must use 0600 permissions: {path}"
        )
    return key


def _load_receipt_signing_key(
    *forbidden_roots: Path, create: bool = True
) -> bytes:
    path = _receipt_signing_key_path()
    resolved_path = path.resolve(strict=False)
    for forbidden_root in forbidden_roots:
        resolved_root = forbidden_root.expanduser().resolve()
        if resolved_path == resolved_root or resolved_path.is_relative_to(
            resolved_root
        ):
            raise PermissionError(
                "OpenClaw receipt signing key must be stored outside the "
                f"workspace and receipt store: {path}"
            )
    parent = path.parent
    parent_existed = parent.exists()
    if not parent_existed and not create:
        raise FileNotFoundError(
            f"OpenClaw receipt signing key is missing at {path}; restore the original key before committing existing receipts"
        )
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.resolve() != parent:
        raise PermissionError("OpenClaw receipt signing key directory contains a symlink")
    if os.name != "nt":
        if not parent_existed or not os.environ.get(
            "ENTROLY_OPENCLAW_RECEIPT_KEY_FILE"
        ):
            os.chmod(parent, 0o700)
        elif parent.stat().st_mode & 0o022:
            raise PermissionError(
                f"OpenClaw receipt signing key directory is writable by other users: {parent}"
            )
    with _receipt_key_init_lock(parent):
        if path.exists():
            if path.stat().st_size != 0:
                return _read_receipt_signing_key(path)
            if not create:
                raise ValueError(
                    f"OpenClaw receipt signing key is interrupted at {path}; restore the original 32-byte key before committing existing receipts"
                )
            quarantine = path.with_name(
                f"{path.name}.incomplete-{secrets.token_hex(8)}"
            )
            os.replace(path, quarantine)
            _fsync_directory(parent)
            print(
                f"entroly: quarantined interrupted empty receipt signing key at {quarantine}; creating a new key",
                file=sys.stderr,
                flush=True,
            )

        if not create:
            raise FileNotFoundError(
                f"OpenClaw receipt signing key is missing at {path}; restore the original key before committing existing receipts"
            )

        key = secrets.token_bytes(32)
        descriptor = -1
        temporary_name = ""
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, path)
            temporary_name = ""
            if os.name != "nt":
                os.chmod(path, 0o600)
            _fsync_directory(parent)
            return key
        finally:
            if descriptor != -1:
                os.close(descriptor)
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)


def _receipt_directory(request: dict[str, Any]) -> Path:
    configured = request.get("receipt_dir")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).expanduser().resolve()
    workspace = request.get("workspace_dir")
    root = (
        Path(workspace).expanduser()
        if isinstance(workspace, str) and workspace
        else Path.cwd()
    ).resolve()
    directory = root / ".entroly" / "receipts" / "openclaw"
    current = root
    for component in (".entroly", "receipts", "openclaw"):
        current /= component
        if current.is_symlink():
            raise PermissionError(
                "default OpenClaw receipt store must not contain symlink components; "
                "configure receiptDir explicitly if an external store is intentional"
            )
    resolved = directory.resolve()
    if not resolved.is_relative_to(root):
        raise PermissionError("default OpenClaw receipt store escaped the workspace")
    return directory


def _receipt_integrity_sha256(receipt: dict[str, Any]) -> str:
    immutable = {key: value for key, value in receipt.items() if key != "proposal_sha256"}
    immutable["acceptance_status"] = "proposed"
    immutable["acceptance_signature"] = _EMPTY_ACCEPTANCE_SIGNATURE
    immutable["acceptance_commit_sha256"] = _EMPTY_ACCEPTANCE_SIGNATURE
    return _sha256_text(_canonical_json(immutable))


def _acceptance_commit_sha256(proposal_sha256: str, proof: str) -> str:
    return _sha256_text(
        f"entroly.openclaw.accept.v1:{proposal_sha256}:{proof}"
    )


def _acceptance_signature(
    signing_key: bytes,
    receipt: dict[str, Any],
    proposal_sha256: str,
    acceptance_commit_sha256: str,
) -> str:
    signed = _canonical_json(
        {
            "schema_version": receipt.get("schema_version"),
            "receipt_id": receipt.get("receipt_id"),
            "proposal_id": receipt.get("proposal_id"),
            "proposal_sha256": proposal_sha256,
            "acceptance_actor": receipt.get("acceptance_actor"),
            "acceptance_challenge_sha256": receipt.get(
                "acceptance_challenge_sha256"
            ),
            "acceptance_commit_sha256": acceptance_commit_sha256,
            "acceptance_status": "accepted",
        }
    )
    return _hmac_sha256(signing_key, "receipt_acceptance", signed)


def _validate_acceptance_state(
    receipt: dict[str, Any], proposal_sha256: str, signing_key: bytes
) -> None:
    status = receipt.get("acceptance_status")
    signature = receipt.get("acceptance_signature")
    commit_sha256 = receipt.get("acceptance_commit_sha256")
    challenge_sha256 = receipt.get("acceptance_challenge_sha256")
    if not isinstance(challenge_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", challenge_sha256
    ):
        raise ValueError("receipt proposal has an invalid acceptance challenge")
    if status == "proposed":
        if (
            signature != _EMPTY_ACCEPTANCE_SIGNATURE
            or commit_sha256 != _EMPTY_ACCEPTANCE_SIGNATURE
        ):
            raise ValueError("proposed receipt contains a forged acceptance signature")
        return
    if status != "accepted" or not isinstance(signature, str) or not re.fullmatch(
        r"[0-9a-f]{64}", signature
    ):
        raise ValueError("receipt proposal has an invalid acceptance state")
    if not isinstance(commit_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", commit_sha256
    ):
        raise ValueError("receipt acceptance commit failed integrity validation")
    expected = _acceptance_signature(
        signing_key, receipt, proposal_sha256, commit_sha256
    )
    if not hmac.compare_digest(signature, expected):
        raise ValueError("receipt acceptance signature failed integrity validation")


def _positive_receipt_limit(
    request: dict[str, Any], key: str, default: int, *, minimum: int, maximum: int
) -> int:
    value = request.get(key, default)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise ValueError(f"{key} must be an integer between {minimum} and {maximum}")
    return value


def _ensure_private_receipt_directory(
    directory: Path, *, repair_default_permissions: bool
) -> None:
    created = False
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        created = True
    except FileExistsError:
        pass
    if directory.resolve() != directory:
        raise PermissionError("OpenClaw receipt directory resolved through a symlink")
    if os.name != "nt":
        if created or repair_default_permissions:
            os.chmod(directory, 0o700)
        elif directory.stat().st_mode & 0o077:
            raise PermissionError(
                f"configured receiptDir is not private at {directory}; use a dedicated 0700 directory"
            )


@contextmanager
def _receipt_store_lock(directory: Path):
    lock_path = directory / ".receipt-store.lock"
    deadline = time.monotonic() + 1.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 30:
                    lock_path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "OpenClaw receipt store is busy; retry after the active assembly finishes"
                )
            time.sleep(0.025)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _atomic_write_private_json(destination: Path, payload: dict[str, Any]) -> None:
    descriptor = -1
    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, destination)
        temporary_name = ""
        if os.name != "nt":
            os.chmod(destination, 0o600)
            _fsync_directory(destination.parent)
    finally:
        if descriptor != -1:
            os.close(descriptor)
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def _enforce_receipt_quota(
    directory: Path, *, new_bytes: int, max_files: int, max_bytes: int
) -> None:
    receipts = [path for path in directory.glob("*.json") if path.is_file()]
    existing_bytes = sum(path.stat().st_size for path in receipts)
    if len(receipts) + 1 > max_files or existing_bytes + new_bytes > max_bytes:
        raise ValueError(
            "OpenClaw receipt quota reached; archive old receipts or increase "
            "receiptMaxFiles/receiptMaxBytes before retrying"
        )


def _write_receipt(
    request: dict[str, Any],
    source_messages: list[dict[str, Any]],
    assembled_messages: list[dict[str, Any]],
    *,
    source_tokens: int,
    assembled_tokens: int,
    warnings: list[str],
    evidence: dict[int, dict[str, Any]],
    strategy: str,
    query: str,
    budget_source: str,
    runtime_metadata: dict[str, Any],
) -> tuple[str, str | None, bool, str | None, str | None]:
    source_json = _canonical_json(source_messages)
    assembled_json = _canonical_json(assembled_messages)
    source_hash = _sha256_text(source_json)
    assembled_hash = _sha256_text(assembled_json)
    write_receipt = request.get("write_receipt", True) is not False
    directory: Path | None = None
    signing_key: bytes | None = None
    workspace_root: Path | None = None
    if write_receipt:
        directory = _receipt_directory(request)
        workspace = request.get("workspace_dir")
        workspace_root = (
            Path(workspace).expanduser()
            if isinstance(workspace, str) and workspace
            else Path.cwd()
        ).resolve()
        signing_key = _load_receipt_signing_key(directory, workspace_root)
    source_digest = (
        _hmac_sha256(signing_key, "source_context", source_json)
        if signing_key is not None
        else source_hash
    )
    assembled_digest = (
        _hmac_sha256(signing_key, "assembled_context", assembled_json)
        if signing_key is not None
        else assembled_hash
    )
    identity = _canonical_json(
        {
            "source_digest": source_digest,
            "assembled_digest": assembled_digest,
            "session_id": str(request.get("session_id") or ""),
            "budget": request.get("token_budget"),
            "budget_source": budget_source,
            "openclaw_runtime": runtime_metadata,
        }
    )
    receipt_id = "ocr_" + _sha256_text(identity)[:20]
    if not write_receipt:
        return receipt_id, None, False, None, None
    assert signing_key is not None
    assert directory is not None
    assert workspace_root is not None

    message_decisions = []
    for index, (source, assembled) in enumerate(
        zip(source_messages, assembled_messages)
    ):
        source_content = source.get("content")
        assembled_content = assembled.get("content")
        source_content_json = _canonical_json(source_content)
        assembled_content_json = _canonical_json(assembled_content)
        evidence_item = evidence.get(index, {})
        is_pinned = bool(evidence_item.get("evidence_pinned"))
        message_decisions.append(
            {
                "message_index": index,
                "role": source.get("role"),
                "action": (
                    "evidence_pinned"
                    if is_pinned
                    else "preserved"
                    if source_content_json == assembled_content_json
                    else "compressed"
                ),
                "source_content_hmac_sha256": _hmac_sha256(
                    signing_key, "source_message", source_content_json
                ),
                "assembled_content_hmac_sha256": _hmac_sha256(
                    signing_key, "assembled_message", assembled_content_json
                ),
                "source_chars": len(source_content) if isinstance(source_content, str) else None,
                "assembled_chars": (
                    len(assembled_content) if isinstance(assembled_content, str) else None
                ),
                "relevance_score": evidence_item.get("score"),
                "matched_query_term_count": len(
                    evidence_item.get("matched_terms", [])
                ),
                "allocated_tokens": evidence_item.get("allocated_tokens"),
                "pin_eligible": evidence_item.get("pin_eligible"),
                "security_flags": evidence_item.get("security_flags", []),
                "pin_blocked_reason": evidence_item.get("pin_blocked_reason"),
            }
        )
    acceptance_challenge_sha256 = request.get("receipt_commit_challenge_sha256")
    if not isinstance(acceptance_challenge_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", acceptance_challenge_sha256
    ):
        raise ValueError(
            "receipt_commit_challenge_sha256 must be a host-generated SHA-256 commitment"
        )
    proposal_id = "ocp_" + secrets.token_hex(16)
    tokens_saved = max(0, source_tokens - assembled_tokens)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "proposal_id": proposal_id,
        "session_id": str(request.get("session_id") or ""),
        "query_chars": len(query),
        "matched_query_term_storage": "count_only",
        "model": runtime_metadata["model"]["resolved"],
        "openclaw_runtime": runtime_metadata,
        "provider_mode": PROVIDER_MODE,
        "provider_independent": True,
        "budget_source": budget_source,
        "budget_authority": (
            "openclaw"
            if budget_source in {"openclaw_token_budget", "openclaw_runtime_settings"}
            else (
                "entroly_verified_registry"
                if runtime_metadata["context_discovery"]["trust"] == "verified"
                else "operator_registry"
                if runtime_metadata["context_discovery"]["trust"] == "user"
                else "local_discovery"
            )
            if budget_source == "entroly_model_registry"
            else "operator"
        ),
        "token_budget": request.get("token_budget"),
        "source_tokens_estimated": source_tokens,
        "assembled_tokens_estimated": assembled_tokens,
        "tokens_saved_estimated": tokens_saved,
        "reduction_pct_estimated": round(
            (tokens_saved / source_tokens * 100.0) if source_tokens else 0.0, 2
        ),
        "source_message_count": len(source_messages),
        "assembled_message_count": len(assembled_messages),
        "source_hmac_sha256": source_digest,
        "assembled_hmac_sha256": assembled_digest,
        "changed": source_hash != assembled_hash,
        "message_decisions": message_decisions,
        "assembly_strategy": strategy,
        "evidence_pinned_count": sum(
            1 for item in evidence.values() if item.get("evidence_pinned")
        ),
        "evidence_pin_blocked_count": sum(
            1 for item in evidence.values() if item.get("pin_blocked_reason")
        ),
        "recovery_source": "openclaw_transcript_unmodified",
        "warnings": warnings,
        "local_only": True,
        "acceptance_actor": "openclaw_plugin",
        "signing_key_id": _signing_key_id(signing_key),
        "workspace_root_hmac_sha256": _hmac_sha256(
            signing_key,
            "workspace_root",
            str(workspace_root),
        ),
        "acceptance_challenge_sha256": acceptance_challenge_sha256,
        "acceptance_status": "proposed",
        "acceptance_signature": _EMPTY_ACCEPTANCE_SIGNATURE,
        "acceptance_commit_sha256": _EMPTY_ACCEPTANCE_SIGNATURE,
    }
    proposal_sha256 = _receipt_integrity_sha256(receipt)
    receipt["proposal_sha256"] = proposal_sha256
    configured_receipt_dir = isinstance(request.get("receipt_dir"), str) and bool(
        request["receipt_dir"].strip()
    )
    _ensure_private_receipt_directory(
        directory,
        repair_default_permissions=not configured_receipt_dir,
    )
    session_name = _safe_session_name(str(request.get("session_id") or "session"))
    destination = directory / f"{session_name}-{receipt_id}-{proposal_id}.json"
    serialized_bytes = len(
        (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    )
    max_files = _positive_receipt_limit(
        request,
        "receipt_max_files",
        DEFAULT_RECEIPT_MAX_FILES,
        minimum=8,
        maximum=10000,
    )
    max_bytes = _positive_receipt_limit(
        request,
        "receipt_max_bytes",
        DEFAULT_RECEIPT_MAX_BYTES,
        minimum=1024 * 1024,
        maximum=1024 * 1024 * 1024,
    )
    with _receipt_store_lock(directory):
        if destination.exists():
            raise FileExistsError("receipt proposal identity already exists")
        _enforce_receipt_quota(
            directory,
            new_bytes=serialized_bytes,
            max_files=max_files,
            max_bytes=max_bytes,
        )
        _atomic_write_private_json(destination, receipt)
    return receipt_id, str(destination), True, proposal_id, proposal_sha256


def commit_receipt(request: dict[str, Any]) -> dict[str, Any]:
    """Idempotently validate one append-only receipt proposal."""
    receipt_id = request.get("receipt_id")
    proposal_id = request.get("proposal_id")
    proposal_sha256 = request.get("proposal_sha256")
    receipt_path = request.get("receipt_path")
    receipt_commit_token = request.get("receipt_commit_token")
    if not isinstance(receipt_id, str) or not re.fullmatch(r"ocr_[0-9a-f]{20}", receipt_id):
        raise ValueError("receipt_id must identify an OpenClaw receipt")
    if not isinstance(proposal_id, str) or not re.fullmatch(r"ocp_[0-9a-f]{32}", proposal_id):
        raise ValueError("proposal_id must identify one receipt proposal")
    if not isinstance(proposal_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", proposal_sha256
    ):
        raise ValueError("proposal_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(receipt_path, str) or not receipt_path.strip():
        raise ValueError("receipt_path is required to recover the proposal")
    if not isinstance(receipt_commit_token, str) or not re.fullmatch(
        r"[0-9a-f]{64}", receipt_commit_token
    ):
        raise ValueError("receipt_commit_token must be a 256-bit lowercase hex secret")
    raw_destination = Path(receipt_path).expanduser().absolute()
    current = Path(raw_destination.anchor)
    for component in raw_destination.parts[1:]:
        current /= component
        if current.is_symlink():
            raise PermissionError("receipt proposal path must not contain symlinks")
    destination = raw_destination.resolve(strict=True)
    expected_suffix = f"-{receipt_id}-{proposal_id}.json"
    if not destination.name.endswith(expected_suffix):
        raise ValueError("receipt proposal path does not match its identity")
    if os.name != "nt" and (
        destination.parent.stat().st_mode & 0o077
        or destination.stat().st_mode & 0o077
    ):
        raise PermissionError(
            "receipt proposal or directory is not private; require 0700/0600 permissions"
        )
    workspace = request.get("workspace_dir")
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError(
            "workspace_dir is required to bind receipt acceptance to the OpenClaw workspace"
        )
    workspace_root = Path(workspace).expanduser().resolve()
    signing_key = _load_receipt_signing_key(
        destination.parent,
        workspace_root,
        create=False,
    )
    with _receipt_store_lock(destination.parent):
        receipt = json.loads(destination.read_text(encoding="utf-8"))
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("receipt_id") != receipt_id
            or receipt.get("proposal_id") != proposal_id
            or receipt.get("proposal_sha256") != proposal_sha256
            or _receipt_integrity_sha256(receipt) != proposal_sha256
        ):
            raise ValueError("receipt proposal failed integrity validation")
        if receipt.get("signing_key_id") != _signing_key_id(signing_key):
            raise ValueError(
                "receipt proposal signing key changed; restore the original key before committing"
            )
        if receipt.get("workspace_root_hmac_sha256") != _hmac_sha256(
            signing_key,
            "workspace_root",
            str(workspace_root),
        ):
            raise PermissionError(
                "receipt proposal belongs to a different OpenClaw workspace"
            )
        _validate_acceptance_state(receipt, proposal_sha256, signing_key)
        expected_name = (
            f"{_safe_session_name(str(receipt.get('session_id') or 'session'))}"
            f"-{receipt_id}-{proposal_id}.json"
        )
        if destination.name != expected_name:
            raise ValueError("receipt proposal filename failed integrity validation")
        if _sha256_text(receipt_commit_token) != receipt.get(
            "acceptance_challenge_sha256"
        ):
            raise ValueError("receipt commit token does not match the host commitment")
        acceptance_commit_sha256 = _acceptance_commit_sha256(
            proposal_sha256, receipt_commit_token
        )
        if receipt.get("acceptance_status") != "accepted":
            receipt["acceptance_status"] = "accepted"
            receipt["acceptance_commit_sha256"] = acceptance_commit_sha256
            receipt["acceptance_signature"] = _acceptance_signature(
                signing_key,
                receipt,
                proposal_sha256,
                acceptance_commit_sha256,
            )
            _atomic_write_private_json(destination, receipt)
        elif receipt.get("acceptance_commit_sha256") != acceptance_commit_sha256:
            raise ValueError("receipt commit is not idempotent for this host proof")
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "receipt_id": receipt_id,
        "proposal_id": proposal_id,
        "proposal_sha256": proposal_sha256,
        "receipt_path": str(destination),
        "acceptance_commit_sha256": acceptance_commit_sha256,
        "committed": True,
    }


def assemble(request: dict[str, Any]) -> dict[str, Any]:
    raw_messages = request.get("messages")
    if not isinstance(raw_messages, list) or not all(
        isinstance(message, dict) for message in raw_messages
    ):
        raise ValueError("messages must be a list of objects")

    messages: list[dict[str, Any]] = copy.deepcopy(raw_messages)
    budget = request.get("token_budget")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError("token_budget must be a positive integer")
    preserve_last_n = request.get("preserve_last_n", DEFAULT_PRESERVE_LAST_N)
    if (
        isinstance(preserve_last_n, bool)
        or not isinstance(preserve_last_n, int)
        or preserve_last_n < 0
    ):
        raise ValueError("preserve_last_n must be a non-negative integer")
    budget_source = request.get("budget_source")
    if budget_source is None:
        budget_source = "openclaw_token_budget"
    if budget_source not in _BUDGET_SOURCES:
        raise ValueError("budget_source is not recognized")
    runtime_metadata = _runtime_audit_metadata(request)

    source_tokens = estimate_messages_tokens(messages)
    evidence_pinning = request.get("evidence_pinning", True) is not False
    query = _query_for_request(request, messages) if evidence_pinning else ""
    strategy = (
        "query_aware_evidence_pinning"
        if evidence_pinning
        else "uniform_budget_compression"
    )
    warnings = [
        "Token counts are deterministic estimates, not provider-billed usage."
    ]
    if budget_source == "operator_fallback":
        warnings.append(
            "OpenClaw did not provide a finite prompt budget; Entroly used the "
            "operator-configured fallbackTokenBudget."
        )
    elif budget_source == "entroly_model_registry":
        warnings.append(
            "OpenClaw did not provide a finite prompt budget; Entroly derived a "
            "conservative input ceiling from trusted model metadata and recorded "
            "its provenance."
        )
    evidence: dict[int, dict[str, Any]] = {}
    protected_indexes = {
        index for index, message in enumerate(messages) if _protected_message(message)
    }
    if preserve_last_n:
        protected_indexes.update(range(max(0, len(messages) - preserve_last_n), len(messages)))
    protected_tokens = estimate_messages_tokens(
        [messages[index] for index in sorted(protected_indexes)]
    )

    # Stays zero on both pass-through paths. Under budget the caller is
    # promised the exact original context, so tool output is not touched
    # either -- "it fits, so nothing was changed" has to keep meaning that.
    tool_tokens_saved = 0
    if source_tokens <= budget:
        assembled = messages
    elif protected_tokens >= budget:
        assembled = messages
        warnings.append(
            "Protected system, structured, and recent messages exceed the token budget; "
            "Entroly returned the exact original context."
        )
    else:
        compressible_indexes = [
            index for index in range(len(messages)) if index not in protected_indexes
        ]
        compressible = [messages[index] for index in compressible_indexes]
        fixed_tokens = sum(_fixed_message_tokens(message) for message in compressible)
        content_budget = budget - protected_tokens - fixed_tokens
        if content_budget <= 0:
            assembled = messages
            warnings.append(
                "Protected messages and message metadata exceed the token budget; "
                "Entroly returned the exact original context."
            )
        else:
            (
                compressed,
                pinned,
                relevance,
                distillation_failures,
                tool_tokens_saved,
            ) = _compress_message_bodies(
                compressible,
                content_budget=content_budget,
                distill=bool(request.get("distill", True)),
                query=query,
                compress_tool_results=bool(
                    request.get("compress_tool_results", True)
                ),
            )
            assembled = copy.deepcopy(messages)
            for local_index, (index, compressed_message) in enumerate(
                zip(compressible_indexes, compressed, strict=True)
            ):
                assembled[index] = compressed_message
                evidence[index] = {
                    **relevance[local_index],
                    "evidence_pinned": local_index in pinned,
                }
            if distillation_failures:
                warnings.append(
                    "Assistant distillation was unavailable for "
                    f"{distillation_failures} message(s); Entroly compressed their "
                    "original text instead."
                )

    try:
        _validate_assembly_invariants(messages, assembled, protected_indexes)
    except ValueError:
        assembled = messages
        evidence = {}
        warnings.append(
            "An internal preservation invariant rejected the assembled context; "
            "Entroly returned the exact original context."
        )

    assembled_tokens = estimate_messages_tokens(assembled)
    if assembled_tokens > budget and assembled != messages:
        assembled = messages
        assembled_tokens = source_tokens
        evidence = {}
        warnings.append(
            "The minimum safe structured context could not fit the token budget; "
            "Entroly returned the exact original context for OpenClaw recovery."
        )
    blocked_count = sum(
        1 for item in evidence.values() if item.get("pin_blocked_reason")
    )
    if blocked_count:
        warnings.append(
            f"Context firewall blocked verbatim evidence pinning for {blocked_count} "
            "message(s); those messages remained subject to normal compression."
        )
    (
        receipt_id,
        receipt_path,
        receipt_commit_required,
        proposal_id,
        proposal_sha256,
    ) = _write_receipt(
        request,
        messages,
        assembled,
        source_tokens=source_tokens,
        assembled_tokens=assembled_tokens,
        warnings=warnings,
        evidence=evidence,
        strategy=strategy,
        query=query,
        budget_source=budget_source,
        runtime_metadata=runtime_metadata,
    )
    pinned_indexes = sorted(
        index for index, item in evidence.items() if item.get("evidence_pinned")
    )
    discovery_metadata = runtime_metadata["context_discovery"]
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "messages": assembled,
        "estimated_tokens": assembled_tokens,
        "source_tokens": source_tokens,
        "tokens_saved": max(0, source_tokens - assembled_tokens),
        # Reported separately so a receipt reader can tell reduction that came
        # from discarding tool output from reduction that came from compressing
        # what a human or the model actually wrote.
        "tool_output_tokens_saved": tool_tokens_saved,
        "changed": assembled != messages,
        "receipt_id": receipt_id,
        "receipt_path": receipt_path,
        "receipt_commit_required": receipt_commit_required,
        "proposal_id": proposal_id,
        "proposal_sha256": proposal_sha256,
        "provider_mode": PROVIDER_MODE,
        "provider_independent": True,
        "budget_source": budget_source,
        "context_discovery_status": discovery_metadata["status"],
        "context_discovery_trust": discovery_metadata["trust"],
        "context_discovery_model": discovery_metadata["model_id"],
        "context_window": discovery_metadata["context_window"],
        "context_output_reserve": discovery_metadata["output_reserve_tokens"],
        "context_safety_tokens": discovery_metadata["safety_tokens"],
        "model": runtime_metadata["model"]["resolved"],
        "provider_hint": runtime_metadata["model"]["provider"],
        "assembly_strategy": strategy,
        "evidence_pinned": len(pinned_indexes),
        "evidence_pin_blocked": blocked_count,
        "pinned_message_indexes": pinned_indexes,
        "warnings": warnings,
    }


def verify_proof_guided_output(request: dict[str, Any]) -> dict[str, Any]:
    """Verify one OpenClaw output and recover exact omitted message evidence.

    OpenClaw owns the model transport and explicitly enables any retry. This
    bridge performs deterministic local verification only; it never calls a
    provider and never receives provider credentials.
    """
    source_messages = request.get("source_messages")
    assembled_messages = request.get("assembled_messages")
    recovered_messages = request.get("recovered_messages", [])
    output = request.get("model_output")
    if not isinstance(source_messages, list) or not all(
        isinstance(item, dict) for item in source_messages
    ):
        raise ValueError("source_messages must be a list of objects")
    if not isinstance(assembled_messages, list) or not all(
        isinstance(item, dict) for item in assembled_messages
    ):
        raise ValueError("assembled_messages must be a list of objects")
    if not isinstance(recovered_messages, list) or not all(
        isinstance(item, dict) for item in recovered_messages
    ):
        raise ValueError("recovered_messages must be a list of objects")
    if not isinstance(output, str):
        raise ValueError("model_output must be a string")
    if len(source_messages) != len(assembled_messages):
        raise ValueError("source and assembled message counts must match")

    max_messages = request.get("max_recovery_messages", 3)
    token_budget = request.get("recovery_token_budget", 1200)
    if (
        isinstance(max_messages, bool)
        or not isinstance(max_messages, int)
        or not 1 <= max_messages <= 16
    ):
        raise ValueError("max_recovery_messages must be an integer within [1, 16]")
    if (
        isinstance(token_budget, bool)
        or not isinstance(token_budget, int)
        or not 0 <= token_budget <= 100_000
    ):
        raise ValueError("recovery_token_budget must be within [0, 100000]")

    grounding_messages = [*assembled_messages, *recovered_messages]
    grounding_context = "\n\n".join(
        _message_text(item, compressible_only=False) for item in grounding_messages
    )
    from .context_firewall import scan
    from .eicv_suppressor import EICVSuppressor

    output_scan = scan(output, source="openclaw_model_output", check_repetition=False)
    suppressor = EICVSuppressor(
        profile=str(request.get("profile") or "rag"),
        mode="strict",
    )
    suppression = suppressor.suppress(grounding_context, output)
    certificates = [certificate.as_dict() for certificate in suppression.certificates]
    obligations = [
        item
        for item in certificates
        if item.get("decision") != "supported" or item.get("action") != "pass"
    ]
    n_claims = int(suppression.n_claims)
    already_recovered = {
        _sha256_text(_canonical_json(item)) for item in recovered_messages
    }
    selected: list[dict[str, Any]] = []
    selected_tokens = 0

    if obligations and token_budget > 0:
        obligation_query = " ".join(
            str(item.get("claim_text") or "") for item in obligations
        )
        relevance = _message_relevance(source_messages, obligation_query)
        candidates: list[tuple[float, int, str, int, dict[str, Any]]] = []
        for index, (source, assembled, score) in enumerate(
            zip(source_messages, assembled_messages, relevance, strict=True)
        ):
            source_text = _message_text(source, compressible_only=False)
            if not source_text.strip() or _canonical_json(source) == _canonical_json(assembled):
                continue
            fingerprint = _sha256_text(_canonical_json(source))
            if fingerprint in already_recovered or not score.get("pin_eligible", False):
                continue
            tokens = max(1, _provider_visible_message_tokens(source))
            numeric_score = float(score.get("score", 0.0) or 0.0)
            if numeric_score <= 0.0:
                continue
            candidates.append((numeric_score / tokens, tokens, fingerprint, index, source))
        for utility, tokens, fingerprint, index, source in sorted(
            candidates, key=lambda item: (-item[0], item[2])
        )[:128]:
            if len(selected) >= max_messages or selected_tokens + tokens > token_budget:
                continue
            selected.append(
                {
                    "message_index": index,
                    "message": copy.deepcopy(source),
                    "message_sha256": fingerprint,
                    "token_count_estimated": tokens,
                    "utility_per_token": round(utility, 12),
                    "verified_exact": True,
                }
            )
            selected_tokens += tokens

    if not output_scan.is_safe:
        status = "unsafe_output"
    elif n_claims == 0:
        status = "no_verifiable_claims"
    elif not obligations:
        status = "supported"
    elif selected:
        status = "retry_with_exact_evidence"
    else:
        status = "no_supporting_omitted_evidence"

    safe_output = suppression.rewritten_output
    if not output_scan.is_safe:
        safe_output = (
            "Entroly withheld this response because the local context firewall "
            "detected unsafe model output. Inspect the signed local proof receipt "
            "before retrying."
        )
    if obligations and not safe_output.strip():
        safe_output = (
            "Entroly withheld unsupported claims because the available context "
            "did not establish them. Inspect the local proof receipt for details."
        )
    recovered_exact = [item["message"] for item in selected]
    retry_instruction = None
    if selected:
        evidence_blocks = [
            "[Verified recovered OpenClaw message "
            f"{item['message_index']} sha256={item['message_sha256']}]\n"
            + _message_text(item["message"], compressible_only=False)
            for item in selected
        ]
        retry_instruction = (
            "Revise the previous answer using the exact recovered evidence below. "
            "Keep supported claims, remove unsupported claims, and state remaining "
            "uncertainty.\n\n" + "\n\n".join(evidence_blocks)
        )

    workspace = request.get("workspace_dir")
    workspace_root = (
        Path(workspace).expanduser().resolve()
        if isinstance(workspace, str) and workspace.strip()
        else Path.cwd().resolve()
    )
    from .verified_efficiency import VerifiedEfficiencyLayer

    audit_layer = VerifiedEfficiencyLayer(
        workspace_root / ".entroly" / "proof-guided-openclaw",
        context_risk_mode="audit",
    )
    audit = audit_layer._persist_audit(
        {
            "artifact_type": "openclaw_proof_guided_output",
            "session_id_hash": _sha256_text(str(request.get("session_id") or "")),
            "run_id_hash": _sha256_text(str(request.get("run_id") or "")),
            "round_index": int(request.get("round_index", 0) or 0),
            "grounding_context_hash": _sha256_text(grounding_context),
            "original_output_hash": _sha256_text(output),
            "verified_output_hash": _sha256_text(safe_output),
            "status": status,
            "counts": {
                "claims": n_claims,
                "supported": suppression.n_supported,
                "abstained": suppression.n_abstained,
                "hallucinated": suppression.n_hallucinated,
            },
            "recovered_message_commitments": [
                {
                    "message_index": item["message_index"],
                    "message_sha256": item["message_sha256"],
                    "token_count_estimated": item["token_count_estimated"],
                }
                for item in selected
            ],
            "provider_call_performed": False,
        }
    )
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "status": status,
        "verified_output": safe_output,
        "changed": safe_output != output,
        "suppression": {
            "n_claims": n_claims,
            "n_supported": suppression.n_supported,
            "n_abstained": suppression.n_abstained,
            "n_hallucinated": suppression.n_hallucinated,
            "certificates": certificates,
        },
        "recovered_messages": recovered_exact,
        "recovered_message_commitments": [
            {key: value for key, value in item.items() if key != "message"}
            for item in selected
        ],
        "recovery_tokens_used": selected_tokens,
        "retry_instruction": retry_instruction,
        "audit_artifact_id": audit.artifact_id,
        "audit_path": audit.path,
        "provider_call_performed": False,
        "local_only": True,
    }



def _communication_ingest(request: dict[str, Any]) -> dict[str, Any]:
    """Persist one privacy-scoped channel observation locally."""
    from .communication import CommunicationStore, event_from_adapter

    raw_event = request.get("event")
    if not isinstance(raw_event, dict):
        raise ValueError("communication_ingest requires an event object")
    store_path = request.get("store_path")
    if store_path is not None and not isinstance(store_path, str):
        raise ValueError("communication store_path must be a string")
    retention = request.get("retention_days")
    event = event_from_adapter(raw_event)
    correlated_action_id = None
    with CommunicationStore(store_path, retention_days=retention) as store:
        inserted = store.record_event(event)
        if event.direction == "outbound":
            correlated_action_id = store.correlate_outbound_event(
                event,
                error=str(raw_event.get("delivery_error") or ""),
            )
        stats = store.stats()
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "communication_schema": event.schema,
        "event_id": event.event_id,
        "commitment_sha256": event.commitment_sha256,
        "content_sha256": event.content_sha256,
        "identity_strength": event.identity_strength,
        "conversation_kind": event.conversation_kind,
        "inserted": inserted,
        "correlated_action_id": correlated_action_id,
        "stats": stats,
        "local_only": True,
        "provider_call_performed": False,
    }

def _communication_status(request: dict[str, Any]) -> dict[str, Any]:
    """Return scalar store health without exposing message content."""
    from .communication import CommunicationStore

    store_path = request.get("store_path")
    if store_path is not None and not isinstance(store_path, str):
        raise ValueError("communication store_path must be a string")
    retention = request.get("retention_days")
    with CommunicationStore(store_path, retention_days=retention) as store:
        stats = store.stats()
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "communication_schema": stats["schema"],
        "stats": stats,
        "local_only": True,
        "provider_call_performed": False,
    }


def _communication_digest(request: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded secretary digest from authorized local evidence."""
    from .communication import CommunicationStore, build_digest

    store_path = request.get("store_path")
    if store_path is not None and not isinstance(store_path, str):
        raise ValueError("communication store_path must be a string")
    retention = request.get("retention_days")
    channel = str(request.get("channel") or "").strip().lower()
    account_id_raw = request.get("account_id")
    account_id = None if account_id_raw is None else str(account_id_raw).strip()
    conversation_id = str(request.get("conversation_id") or "").strip()
    try:
        limit = int(request.get("limit", 1000))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("communication digest limit must be an integer") from exc
    limit = max(1, min(limit, 10_000))

    with CommunicationStore(store_path, retention_days=retention) as store:
        if conversation_id:
            if not channel:
                raise ValueError("channel is required for conversation-scoped digest")
            events = store.events_for_scope(
                channel=channel,
                account_id=account_id or "",
                conversation_id=conversation_id,
                limit=limit,
            )
            scope = "conversation"
        else:
            if request.get("owner_authorized") is not True:
                raise PermissionError(
                    "cross-conversation communication review requires trusted owner authorization"
                )
            events = store.recent_events(
                channel=channel,
                account_id=account_id,
                limit=limit,
            )
            scope = "owner_global"
        digest = build_digest(events)
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "scope": scope,
        "digest": digest,
        "local_only": True,
        "provider_call_performed": False,
    }


def _communication_assure(request: dict[str, Any]) -> dict[str, Any]:
    """Evaluate, verify, receipt, and persist one proposed external action."""
    from .communication import (
        CommunicationActionProposal,
        CommunicationPolicy,
        CommunicationReceiptLedger,
        CommunicationStore,
        action_evidence_verification,
        assess_event,
        build_action_receipt,
        combine_category,
        combine_risk,
        eicv_supports_automatic_action,
        outgoing_creates_commitment,
    )

    store_path = request.get("store_path")
    receipt_dir = request.get("receipt_dir")
    if store_path is not None and not isinstance(store_path, str):
        raise ValueError("communication store_path must be a string")
    if receipt_dir is not None and not isinstance(receipt_dir, str):
        raise ValueError("communication receipt_dir must be a string")
    if receipt_dir is None and isinstance(store_path, str) and store_path.strip():
        receipt_dir = str(Path(store_path).expanduser().absolute().parent / "receipts")
    retention = request.get("retention_days")
    action_type = str(request.get("action_type") or "").strip()
    allowed_actions = {"send_message", "reply", "react", "group_reply", "no_action"}
    if action_type not in allowed_actions:
        raise ValueError(f"unsupported communication action_type: {action_type!r}")
    channel = str(request.get("channel") or "").strip().lower()
    account_id = str(request.get("account_id") or "").strip()
    conversation_id = str(request.get("conversation_id") or "").strip()
    if not channel or not conversation_id:
        raise ValueError(
            "communication assurance requires trusted channel/conversation scope"
        )

    source_ids_raw = request.get("source_event_ids")
    if not isinstance(source_ids_raw, list):
        raise ValueError("source_event_ids must be a list")
    source_ids = tuple(
        sorted(
            {
                str(item).strip()
                for item in source_ids_raw
                if str(item).strip()
            }
        )
    )
    if action_type != "no_action" and not source_ids:
        raise ValueError("external communication actions require source_event_ids")
    payload = str(request.get("payload") or "")

    mode = str(request.get("policy_mode") or "observe").strip().lower()
    if mode not in {"observe", "suggest", "approve", "bounded"}:
        raise ValueError("invalid communication policy_mode")
    auto_actions_raw = request.get("auto_actions")
    auto_categories_raw = request.get("auto_categories")
    auto_actions = tuple(
        item
        for item in (
            str(value).strip()
            for value in (
                auto_actions_raw if isinstance(auto_actions_raw, list) else []
            )
        )
        if item in allowed_actions
    )
    auto_categories = tuple(
        sorted(
            {
                str(value).strip()
                for value in (
                    auto_categories_raw
                    if isinstance(auto_categories_raw, list)
                    else []
                )
                if str(value).strip()
            }
        )
    )

    receipt_proof: dict[str, Any] | None = None
    receipt_error: str | None = None
    with CommunicationStore(store_path, retention_days=retention) as store:
        events = []
        for event_id in source_ids:
            event = store.get_event(event_id)
            if event is None:
                raise ValueError(
                    f"unknown communication source event: {event_id}"
                )
            events.append(event)

        assessments = [assess_event(event) for event in events]
        kinds = {event.conversation_kind for event in events}
        conversation_kind = (
            next(iter(kinds)) if len(kinds) == 1 else "unknown"
        )
        category = combine_category(assessments)
        risk_class = combine_risk(assessments)
        creates_commitment = outgoing_creates_commitment(payload)
        proposal = CommunicationActionProposal.build(
            action_type=action_type,  # type: ignore[arg-type]
            channel=channel,
            account_id=account_id,
            conversation_id=conversation_id,
            conversation_kind=conversation_kind,  # type: ignore[arg-type]
            source_event_ids=source_ids,
            payload=payload,
            category=category,
            risk_class=risk_class,
            creates_commitment=creates_commitment,
        )
        policy = CommunicationPolicy(
            mode=mode,  # type: ignore[arg-type]
            auto_categories=auto_categories,
            auto_actions=auto_actions,  # type: ignore[arg-type]
        )
        decision, reasons = policy.evaluate(
            proposal,
            source_events=events,
            already_handled=store.action_is_handled(proposal.action_id),
        )
        reasons = tuple(reasons)

        evidence_verification = action_evidence_verification(
            proposal,
            events,
        )
        if (
            decision == "allow"
            and not eicv_supports_automatic_action(evidence_verification)
        ):
            decision = "approval_required"
            reasons = tuple(
                sorted(
                    {
                        *reasons,
                        "eicv:automatic_action_not_supported",
                    }
                )
            )

        receipt = build_action_receipt(
            proposal,
            decision=decision,
            reasons=reasons,
            source_events=events,
            evidence_verification=evidence_verification,
        )
        try:
            receipt_proof = CommunicationReceiptLedger(
                receipt_dir
            ).record(store, receipt)
        except Exception as exc:
            receipt_error = type(exc).__name__
            # Automatic external effects require a durable audit commitment.
            if decision == "allow":
                decision = "approval_required"
                reasons = tuple(
                    sorted(
                        {
                            *reasons,
                            "audit:signed_receipt_unavailable",
                        }
                    )
                )

        state_by_decision = {
            "allow": "assured",
            "approval_required": "awaiting_approval",
            "already_handled": "sent",
            "deny": "blocked",
            "ambiguous": "blocked",
            "insufficient_context": "blocked",
        }
        inserted = store.record_action(
            proposal,
            decision=decision,
            reasons=reasons,
            execution_state=state_by_decision[decision],
        )
        action = store.get_action(proposal.action_id)

    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "decision": decision,
        "reasons": list(reasons),
        "action_id": proposal.action_id,
        "action_type": proposal.action_type,
        "category": proposal.category,
        "risk_class": proposal.risk_class,
        "creates_commitment": proposal.creates_commitment,
        "conversation_kind": proposal.conversation_kind,
        "source_event_ids": list(proposal.source_event_ids),
        "evidence_verification": evidence_verification,
        "receipt": receipt_proof,
        "receipt_error": receipt_error,
        "inserted": inserted,
        "execution_state": (action or {}).get("execution_state"),
        "local_only": True,
        "provider_call_performed": False,
    }

def _communication_set_taste(request: dict[str, Any]) -> dict[str, Any]:
    """Replace explicit communication taste for one owner-authorized scope."""
    from .communication import CommunicationStore, CommunicationTaste

    if request.get("owner_authorized") is not True:
        raise PermissionError("explicit communication taste requires trusted owner authorization")
    store_path = request.get("store_path")
    if store_path is not None and not isinstance(store_path, str):
        raise ValueError("communication store_path must be a string")
    scope_type = str(request.get("scope_type") or "").strip()
    scope_id = str(request.get("scope_id") or "").strip()
    if scope_type not in {"owner", "contact", "group", "conversation"}:
        raise ValueError("invalid communication taste scope_type")
    if not scope_id:
        raise ValueError("communication taste scope_id is required")
    profile = request.get("taste")
    if not isinstance(profile, dict):
        raise ValueError("communication taste must be an object")

    taste = CommunicationTaste.build(
        scope_type=scope_type,  # type: ignore[arg-type]
        scope_id=scope_id,
        source="explicit",
        confidence=1.0,
        preferred_language=str(profile.get("preferred_language") or "adaptive"),
        formality=str(profile.get("formality") or "adaptive"),
        response_length=str(profile.get("response_length") or "adaptive"),
        emoji_level=str(profile.get("emoji_level") or "adaptive"),
        routine_action=str(profile.get("routine_action") or "none"),
        preferred_reaction=str(profile.get("preferred_reaction") or ""),
        greeting_style=str(profile.get("greeting_style") or ""),
        signoff_style=str(profile.get("signoff_style") or ""),
        notes=profile.get("notes") if isinstance(profile.get("notes"), dict) else {},
    )
    with CommunicationStore(store_path) as store:
        store.set_explicit_taste(taste)
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "profile": taste.to_dict(),
        "authority_expanded": False,
        "local_only": True,
        "provider_call_performed": False,
    }


def _communication_learn_taste(request: dict[str, Any]) -> dict[str, Any]:
    """Infer low-risk communication style through PRISM-selected evidence."""
    from .communication import (
        CommunicationMemory,
        CommunicationStore,
        CommunicationTasteOptimizer,
        infer_taste_from_outbound,
    )

    if request.get("owner_authorized") is not True:
        raise PermissionError(
            "communication taste learning requires trusted owner authorization"
        )
    store_path = request.get("store_path")
    memory_path = request.get("memory_path")
    learning_state_path = request.get("learning_state_path")
    learning_journal_path = request.get("learning_journal_path")
    for label, value in (
        ("store_path", store_path),
        ("memory_path", memory_path),
        ("learning_state_path", learning_state_path),
        ("learning_journal_path", learning_journal_path),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"communication {label} must be a string")
    scope_type = str(request.get("scope_type") or "").strip()
    scope_id = str(request.get("scope_id") or "").strip()
    if scope_type not in {"owner", "contact", "group", "conversation"}:
        raise ValueError("invalid communication taste scope_type")
    if not scope_id:
        raise ValueError("communication taste scope_id is required")
    channel = str(request.get("channel") or "").strip().lower()
    account_raw = request.get("account_id")
    account_id = None if account_raw is None else str(account_raw).strip()
    query = str(request.get("query") or "").strip()
    try:
        limit = max(3, min(int(request.get("limit", 500)), 5000))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("communication taste limit must be an integer") from exc

    with CommunicationStore(store_path) as store:
        if scope_type in {"group", "conversation"}:
            if not channel:
                raise ValueError(
                    "channel is required for conversation taste learning"
                )
            events = store.events_for_scope(
                channel=channel,
                account_id=account_id or "",
                conversation_id=scope_id,
                limit=limit,
            )
        else:
            events = store.recent_events(
                channel=channel,
                account_id=account_id,
                limit=limit,
            )
            if scope_type == "contact":
                events = [
                    event
                    for event in events
                    if event.sender_id == scope_id
                    or event.recipient_id == scope_id
                ]

    optimizer = CommunicationTasteOptimizer(
        learning_state_path,
        learning_journal_path,
    )
    selection = optimizer.select_examples(
        events,
        scope_type=scope_type,
        scope_id=scope_id,
        query=query,
        top_k=min(24, max(3, len(events))),
    )
    selected = set(selection.event_ids)
    selected_events = [
        event for event in events if event.event_id in selected
    ]
    taste = infer_taste_from_outbound(
        selected_events,
        scope_type=scope_type,  # type: ignore[arg-type]
        scope_id=scope_id,
    )
    if taste is None:
        return {
            "schema_version": BRIDGE_SCHEMA,
            "ok": True,
            "learned": False,
            "reason": "insufficient_prism_selected_outbound_evidence",
            "selection": selection.to_dict(),
            "authority_expanded": False,
            "local_only": True,
            "provider_call_performed": False,
        }

    memory = CommunicationMemory(memory_path)
    remembered = memory.remember_taste(taste)
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "learned": True,
        "profile": taste.to_dict(),
        "selection": selection.to_dict(),
        "learning": optimizer.stats(),
        "memory": remembered,
        "authority_expanded": False,
        "local_only": True,
        "provider_call_performed": False,
    }

def _communication_resolve_taste(request: dict[str, Any]) -> dict[str, Any]:
    """Resolve explicit policy + partitioned episodic preference memory."""
    from .communication import (
        CommunicationMemory,
        CommunicationStore,
        resolve_taste,
    )

    if request.get("owner_authorized") is not True:
        raise PermissionError("communication taste recall requires trusted owner authorization")
    store_path = request.get("store_path")
    memory_path = request.get("memory_path")
    if store_path is not None and not isinstance(store_path, str):
        raise ValueError("communication store_path must be a string")
    if memory_path is not None and not isinstance(memory_path, str):
        raise ValueError("communication memory_path must be a string")
    raw_scopes = request.get("scopes")
    if not isinstance(raw_scopes, list) or not raw_scopes:
        raise ValueError("communication taste scopes must be a non-empty list")
    if len(raw_scopes) > 8:
        raise ValueError("communication taste scopes are bounded to 8")
    scopes: list[tuple[str, str]] = []
    for raw in raw_scopes:
        if not isinstance(raw, dict):
            raise ValueError("communication taste scope must be an object")
        scope_type = str(raw.get("scope_type") or "").strip()
        scope_id = str(raw.get("scope_id") or "").strip()
        if scope_type not in {"owner", "contact", "group", "conversation"}:
            raise ValueError("invalid communication taste scope_type")
        if not scope_id:
            raise ValueError("communication taste scope_id is required")
        scopes.append((scope_type, scope_id))

    memory = CommunicationMemory(memory_path)
    profiles = []
    with CommunicationStore(store_path) as store:
        explicit_by_scope = {
            (taste.scope_type, taste.scope_id): taste
            for taste in store.explicit_tastes_for_scopes(scopes)
        }
    for scope_type, scope_id in scopes:
        profiles.extend(
            memory.recall_tastes(scope_type=scope_type, scope_id=scope_id)
        )
        explicit = explicit_by_scope.get((scope_type, scope_id))
        if explicit is not None:
            profiles.append(explicit)

    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "resolved": resolve_taste(*profiles),
        "profiles": [profile.to_dict() for profile in profiles],
        "memory_layers": [layer.as_dict() for layer in memory.fabric.capabilities()],
        "authority_expanded": False,
        "local_only": True,
        "provider_call_performed": False,
    }


def _communication_memory_status(request: dict[str, Any]) -> dict[str, Any]:
    from .communication import CommunicationMemory

    memory_path = request.get("memory_path")
    if memory_path is not None and not isinstance(memory_path, str):
        raise ValueError("communication memory_path must be a string")
    memory = CommunicationMemory(memory_path)
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "stats": memory.stats(),
        "local_only": True,
        "provider_call_performed": False,
    }


def _communication_begin_action(request: dict[str, Any]) -> dict[str, Any]:
    """Atomically claim an assured action for one external dispatch attempt."""
    from .communication import CommunicationStore

    action_id = str(request.get("action_id") or "").strip()
    if not action_id:
        raise ValueError("communication_begin_action requires action_id")
    store_path = request.get("store_path")
    if store_path is not None and not isinstance(store_path, str):
        raise ValueError("communication store_path must be a string")
    with CommunicationStore(store_path) as store:
        claimed = store.begin_action_execution(action_id)
        action = store.get_action(action_id)
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "action_id": action_id,
        "claimed": claimed,
        "execution_state": (action or {}).get("execution_state"),
        "local_only": True,
        "provider_call_performed": False,
    }


def _communication_fail_action(request: dict[str, Any]) -> dict[str, Any]:
    """Record an observed OpenClaw dispatch exception."""
    from .communication import CommunicationStore

    action_id = str(request.get("action_id") or "").strip()
    if not action_id:
        raise ValueError("communication_fail_action requires action_id")
    store_path = request.get("store_path")
    if store_path is not None and not isinstance(store_path, str):
        raise ValueError("communication store_path must be a string")
    error = str(request.get("error") or "delivery_failed")
    with CommunicationStore(store_path) as store:
        recorded = store.fail_dispatch(action_id, error=error)
        action = store.get_action(action_id)
    return {
        "schema_version": BRIDGE_SCHEMA,
        "ok": True,
        "action_id": action_id,
        "recorded": recorded,
        "execution_state": (action or {}).get("execution_state"),
        "local_only": True,
        "provider_call_performed": False,
    }

def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    operation = request.get("operation")
    if operation == "health":
        workspace = request.get("workspace_dir")
        workspace_root = (
            Path(workspace).expanduser().resolve()
            if isinstance(workspace, str) and workspace.strip()
            else None
        )
        key_path = _receipt_signing_key_path()
        initialize_key = request.get("write_receipt", True) is not False
        if workspace_root is not None and initialize_key:
            receipt_directory = _receipt_directory(request)
            _load_receipt_signing_key(receipt_directory, workspace_root)
            receipt_key_status = "ready"
        elif key_path.exists():
            forbidden_roots = (
                (_receipt_directory(request), workspace_root)
                if workspace_root is not None
                else ()
            )
            _load_receipt_signing_key(*forbidden_roots, create=False)
            receipt_key_status = "ready"
        else:
            receipt_key_status = "uninitialized"
        return {
            "schema_version": BRIDGE_SCHEMA,
            "ok": True,
            "status": "ready",
            "receipt_key_status": receipt_key_status,
            "provider_mode": PROVIDER_MODE,
            "provider_independent": True,
            "requires_host_token_budget": True,
            "supports_context_budget_discovery": True,
            "receipt_commit_protocol": "two_phase",
        }
    if operation == "resolve_context_budget":
        return _resolve_context_budget(request)
    if operation == "assemble":
        return assemble(request)
    if operation == "commit_receipt":
        return commit_receipt(request)
    if operation == "verify_proof_guided_output":
        return verify_proof_guided_output(request)
    if operation == "communication_ingest":
        return _communication_ingest(request)
    if operation == "communication_status":
        return _communication_status(request)
    if operation == "communication_digest":
        return _communication_digest(request)
    if operation == "communication_assure":
        return _communication_assure(request)
    if operation == "communication_set_taste":
        return _communication_set_taste(request)
    if operation == "communication_learn_taste":
        return _communication_learn_taste(request)
    if operation == "communication_resolve_taste":
        return _communication_resolve_taste(request)
    if operation == "communication_memory_status":
        return _communication_memory_status(request)
    if operation == "communication_begin_action":
        return _communication_begin_action(request)
    if operation == "communication_fail_action":
        return _communication_fail_action(request)
    raise ValueError(f"unsupported operation: {operation!r}")


def serve(input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> int:
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        request_id: Any = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            request_id = payload.get("request_id")
            response = handle_request(payload)
        except Exception as error:
            response = {
                "schema_version": BRIDGE_SCHEMA,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        response["request_id"] = request_id
        output_stream.write(_canonical_json(response) + "\n")
        output_stream.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="store_true", help="serve newline-delimited JSON")
    args = parser.parse_args(argv)
    if not args.jsonl:
        parser.error("--jsonl is required")
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
