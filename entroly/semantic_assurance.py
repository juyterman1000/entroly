"""Semantic assurance at Entroly's provider boundary.

The module deliberately keeps provider bytes and retrieval intent separate:
known harness boilerplate can be excluded from retrieval without mutating the
request, while final provider payloads are checked after every optimization.
Repairs are deterministic, evidence-preserving, and reported only by reason
codes and digests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger("entroly.semantic_assurance")

_REMINDER_BLOCK = re.compile(
    r"^\s*<system-reminder(?:\s[^>]*)?>.*?</system-reminder>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_REMINDER_SUFFIX = re.compile(
    r"(?:^|\n)\s*<system-reminder(?:\s[^>]*)?>.*?</system-reminder>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_TTL_RANK = {"5m": 1, "1h": 2}
_MAX_EXPLICIT_BREAKPOINTS = 4


@dataclass(frozen=True)
class IntentProjection:
    raw_text: str
    retrieval_text: str
    removed_blocks: int
    removed_chars: int
    source_sha256: str

    @property
    def changed(self) -> bool:
        return self.raw_text != self.retrieval_text


@dataclass(frozen=True)
class SemanticRepair:
    code: str
    path: str
    before_sha256: str
    after_sha256: str


@dataclass(frozen=True)
class CacheBreakpoint:
    path: str
    ttl: str


@dataclass(frozen=True)
class WireAssuranceReport:
    provider: str
    before_sha256: str
    after_sha256: str
    capability_epoch: str
    repairs: tuple[SemanticRepair, ...]

    @property
    def changed(self) -> bool:
        return self.before_sha256 != self.after_sha256

    def response_headers(self) -> dict[str, str]:
        return {
            "X-Entroly-Wire-Assurance": "repaired" if self.changed else "pass",
            "X-Entroly-Wire-Repairs": str(len(self.repairs)),
            "X-Entroly-Capability-Epoch": self.capability_epoch[:16],
        }


class SemanticWireError(ValueError):
    """A request cannot be made provider-valid without changing intent."""

    def __init__(self, code: str, path: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.path = path
        self.detail = detail


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _digest(value: Any) -> str:
    text = value if isinstance(value, str) else _json(value)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _latest_user_text_blocks(body: Mapping[str, Any], provider: str) -> list[str]:
    if "input" in body and "messages" not in body:
        value = body.get("input")
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in reversed(value):
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") == "input_text" and isinstance(item.get("text"), str):
                    return [str(item["text"])]
                if item.get("role") != "user":
                    continue
                content = item.get("content", "")
                if isinstance(content, str):
                    return [content]
                if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
                    result = [
                        str(block.get("text", ""))
                        for block in content
                        if isinstance(block, Mapping)
                        and block.get("type") in {"text", "input_text"}
                        and isinstance(block.get("text"), str)
                    ]
                    if result:
                        return result
        return []

    if provider == "gemini":
        contents = body.get("contents", [])
        if not isinstance(contents, Sequence) or isinstance(contents, (str, bytes)):
            return []
        for item in reversed(contents):
            if not isinstance(item, Mapping) or item.get("role", "user") != "user":
                continue
            parts = item.get("parts", [])
            if not isinstance(parts, Sequence) or isinstance(parts, (str, bytes)):
                continue
            result = [
                str(part.get("text", ""))
                for part in parts
                if isinstance(part, Mapping) and isinstance(part.get("text"), str)
            ]
            if result:
                return result
        return []

    messages = body.get("messages", [])
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return []
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return [content]
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            result = [
                str(block.get("text", ""))
                for block in content
                if isinstance(block, Mapping)
                and block.get("type") in {"text", "input_text"}
                and isinstance(block.get("text"), str)
            ]
            if result:
                return result
    return []


def _purify_block(text: str) -> tuple[str, int]:
    """Remove only whole/newline-suffix harness regions, never inline literals."""
    if _REMINDER_BLOCK.fullmatch(text):
        return "", 1
    current = text
    removed = 0
    while True:
        match = _REMINDER_SUFFIX.search(current)
        if match is None:
            break
        current = current[: match.start()].rstrip()
        removed += 1
    return current, removed


def project_retrieval_intent(body: Mapping[str, Any], provider: str) -> IntentProjection:
    """Project human retrieval intent without changing the provider request."""
    raw_blocks = _latest_user_text_blocks(body, provider)
    kept: list[str] = []
    removed_blocks = 0
    removed_chars = 0
    for raw in raw_blocks:
        purified, count = _purify_block(raw)
        removed_blocks += count
        removed_chars += max(0, len(raw) - len(purified))
        if purified.strip():
            kept.append(purified.strip())
    raw_text = " ".join(block.strip() for block in raw_blocks if block.strip())
    return IntentProjection(
        raw_text=raw_text,
        retrieval_text=" ".join(kept),
        removed_blocks=removed_blocks,
        removed_chars=removed_chars,
        source_sha256=_digest(raw_text),
    )


def _record_repair(
    repairs: list[SemanticRepair],
    code: str,
    path: str,
    before: Any,
    after: Any,
) -> None:
    repairs.append(SemanticRepair(code, path, _digest(before), _digest(after)))


def _as_system_blocks(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if isinstance(value, Mapping):
        return [copy.deepcopy(dict(value))]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                result.append(copy.deepcopy(dict(item)))
            elif isinstance(item, str):
                result.append({"type": "text", "text": item})
            else:
                result.append({"type": "text", "text": _json(item)})
        return result
    return [{"type": "text", "text": str(value)}]


def _relocate_system_messages(body: dict[str, Any], repairs: list[SemanticRepair]) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    moved: list[dict[str, Any]] = []
    retained: list[Any] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping) or message.get("role") != "system":
            retained.append(message)
            continue
        blocks = _as_system_blocks(message.get("content"))
        moved.extend(blocks)
        _record_repair(
            repairs,
            "anthropic_system_role_relocated",
            f"messages[{index}]",
            message,
            blocks,
        )
    if moved:
        body["system"] = [*_as_system_blocks(body.get("system")), *moved]
        body["messages"] = retained


def _advertised_tools(body: Mapping[str, Any]) -> tuple[bool, frozenset[str]]:
    if "tools" not in body:
        return False, frozenset()
    tools = body.get("tools")
    if not isinstance(tools, list):
        raise SemanticWireError("tool_contract_invalid", "tools", "tools must be an array")
    names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise SemanticWireError(
                "tool_contract_invalid", f"tools[{index}]", "tool definition must be an object"
            )
        name = tool.get("name")
        # Server tools may carry a typed contract whose name is still explicit.
        if isinstance(name, str) and name:
            names.add(name)
    return True, frozenset(names)


def _capability_epoch(provider: str, body: Mapping[str, Any]) -> str:
    try:
        authoritative, names = _advertised_tools(body)
        invalid = False
    except SemanticWireError:
        # Epoch metadata must never become a new validity gate by itself.
        authoritative, names, invalid = False, frozenset(), True
    return _digest(
        {
            "provider": provider,
            "authoritative": authoritative,
            "tools": sorted(names),
            "tool_choice": body.get("tool_choice"),
            "invalid_tool_contract": invalid,
        }
    )


def _historical_tool_use(block: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "text",
        "text": (
            "[Historical tool invocation retained as evidence; structured capability "
            "edge inactive]\n"
            + _json(
                {
                    "name": block.get("name"),
                    "id": block.get("id"),
                    "input": block.get("input"),
                }
            )
        ),
    }


def _historical_tool_result(block: Mapping[str, Any]) -> list[dict[str, Any]]:
    prefix = {
        "type": "text",
        "text": (
            "[Historical tool result retained as evidence; structured capability "
            f"edge inactive; tool_use_id={block.get('tool_use_id', 'unknown')}]"
        ),
    }
    content = block.get("content", "")
    if isinstance(content, str):
        if content:
            prefix["text"] += "\n" + content
        return [prefix]
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        output = [prefix]
        for item in content:
            if isinstance(item, Mapping):
                copied = copy.deepcopy(dict(item))
                # A nested result wrapper would still be interpreted as protocol.
                if copied.get("type") == "tool_result":
                    output.append({"type": "text", "text": _json(copied)})
                else:
                    output.append(copied)
            elif isinstance(item, str):
                output.append({"type": "text", "text": item})
            else:
                output.append({"type": "text", "text": _json(item)})
        return output
    if content not in {None, ""}:
        prefix["text"] += "\n" + _json(content)
    return [prefix]


def _reconcile_anthropic_tool_history(
    body: dict[str, Any], repairs: list[SemanticRepair]
) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    authoritative, current_names = _advertised_tools(body)
    seen_ids: set[str] = set()
    retired_ids: set[str] = set()
    pending_ids: set[str] = set()

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            pending_ids = set()
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, list):
            if role != "user":
                pending_ids = set()
            continue

        if role == "assistant":
            active: set[str] = set()
            rewritten: list[Any] = []
            for block_index, block in enumerate(content):
                if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                    rewritten.append(block)
                    continue
                call_id = str(block.get("id") or "")
                name = str(block.get("name") or "")
                stale = authoritative and name not in current_names
                duplicate = not call_id or call_id in seen_ids
                if stale or duplicate:
                    replacement = _historical_tool_use(block)
                    rewritten.append(replacement)
                    if call_id:
                        retired_ids.add(call_id)
                    _record_repair(
                        repairs,
                        "tool_use_retired" if stale else "tool_use_duplicate_retired",
                        f"messages[{message_index}].content[{block_index}]",
                        block,
                        replacement,
                    )
                    continue
                seen_ids.add(call_id)
                active.add(call_id)
                rewritten.append(block)
            message["content"] = rewritten
            pending_ids = active
            continue

        if role != "user":
            pending_ids = set()
            continue

        tool_results: list[Any] = []
        ordinary: list[Any] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, Mapping) or block.get("type") != "tool_result":
                ordinary.append(block)
                continue
            call_id = str(block.get("tool_use_id") or "")
            if call_id in pending_ids and call_id not in retired_ids:
                tool_results.append(block)
                continue
            replacement = _historical_tool_result(block)
            ordinary.extend(replacement)
            _record_repair(
                repairs,
                "tool_result_retired",
                f"messages[{message_index}].content[{block_index}]",
                block,
                replacement,
            )
        reordered = [*tool_results, *ordinary]
        if reordered != content and tool_results:
            _record_repair(
                repairs,
                "tool_results_frontloaded",
                f"messages[{message_index}].content",
                content,
                reordered,
            )
        message["content"] = reordered
        pending_ids = set()


def _cache_control(value: Any, path: str) -> str:
    if not isinstance(value, Mapping) or value.get("type") != "ephemeral":
        raise SemanticWireError(
            "cache_control_invalid", path, "cache_control must use type=ephemeral"
        )
    ttl = value.get("ttl", "5m")
    if ttl not in _TTL_RANK:
        raise SemanticWireError(
            "cache_ttl_invalid", f"{path}.ttl", "cache TTL must be 5m or 1h"
        )
    return str(ttl)


def _cacheable_objects(body: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    result: list[tuple[str, Mapping[str, Any]]] = []
    tools = body.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools):
            if isinstance(tool, Mapping):
                result.append((f"tools[{index}]", tool))
    system = body.get("system")
    if isinstance(system, Mapping):
        result.append(("system", system))
    elif isinstance(system, list):
        for index, block in enumerate(system):
            if isinstance(block, Mapping):
                result.append((f"system[{index}]", block))
    messages = body.get("messages")
    if isinstance(messages, list):
        for message_index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if isinstance(content, list):
                for block_index, block in enumerate(content):
                    if isinstance(block, Mapping):
                        result.append(
                            (f"messages[{message_index}].content[{block_index}]", block)
                        )
    return result


def validate_anthropic_cache_topology(
    body: Mapping[str, Any],
) -> tuple[CacheBreakpoint, ...]:
    """Validate cache layout in provider prompt order without changing TTL policy."""
    objects = _cacheable_objects(body)
    explicit: list[CacheBreakpoint] = []
    for path, value in objects:
        if "cache_control" in value:
            explicit.append(
                CacheBreakpoint(path, _cache_control(value["cache_control"], f"{path}.cache_control"))
            )
    if len(explicit) > _MAX_EXPLICIT_BREAKPOINTS:
        raise SemanticWireError(
            "cache_breakpoint_limit_exceeded",
            "cache_control",
            "request has more than four explicit cache breakpoints",
        )

    sequence = list(explicit)
    automatic = body.get("cache_control")
    if automatic is not None:
        auto_ttl = _cache_control(automatic, "cache_control")
        final_explicit: str | None = None
        if objects and "cache_control" in objects[-1][1]:
            final_explicit = _cache_control(
                objects[-1][1]["cache_control"], f"{objects[-1][0]}.cache_control"
            )
        if final_explicit is not None:
            if final_explicit != auto_ttl:
                raise SemanticWireError(
                    "automatic_cache_ttl_conflict",
                    "cache_control.ttl",
                    "automatic cache TTL conflicts with the final explicit breakpoint",
                )
        else:
            if len(explicit) >= _MAX_EXPLICIT_BREAKPOINTS:
                raise SemanticWireError(
                    "cache_breakpoint_limit_exceeded",
                    "cache_control",
                    "automatic caching would require an additional breakpoint",
                )
            sequence.append(CacheBreakpoint("cache_control", auto_ttl))

    previous = 10
    for breakpoint in sequence:
        rank = _TTL_RANK[breakpoint.ttl]
        if rank > previous:
            raise SemanticWireError(
                "cache_ttl_non_monotonic",
                breakpoint.path,
                "longer-lived cache breakpoints must precede shorter-lived breakpoints",
            )
        previous = rank
    return tuple(sequence)


def _validate_anthropic_roles(body: Mapping[str, Any]) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise SemanticWireError(
                "anthropic_message_invalid", f"messages[{index}]", "message must be an object"
            )
        if message.get("role") not in {"user", "assistant"}:
            raise SemanticWireError(
                "anthropic_role_invalid",
                f"messages[{index}].role",
                "Anthropic messages may contain only user and assistant roles",
            )


def assure_provider_request(
    body: Mapping[str, Any], provider: str
) -> tuple[dict[str, Any], WireAssuranceReport]:
    """Return a provider-assured deep copy or raise a bounded semantic error."""
    if not isinstance(body, Mapping):
        raise SemanticWireError("request_body_invalid", "body", "request body must be an object")
    original = copy.deepcopy(dict(body))
    candidate = copy.deepcopy(dict(body))
    repairs: list[SemanticRepair] = []
    if provider == "anthropic":
        _relocate_system_messages(candidate, repairs)
        _reconcile_anthropic_tool_history(candidate, repairs)
        _validate_anthropic_roles(candidate)
        validate_anthropic_cache_topology(candidate)
    report = WireAssuranceReport(
        provider=provider,
        before_sha256=_digest(original),
        after_sha256=_digest(candidate),
        capability_epoch=_capability_epoch(provider, candidate),
        repairs=tuple(repairs),
    )
    return candidate, report


def _copy_markers(source: Any, target: Any) -> None:
    for name in dir(source):
        if name.startswith("__entroly_"):
            try:
                setattr(target, name, getattr(source, name))
            except Exception:
                pass


def install_proxy_semantic_assurance() -> None:
    """Install intent projection and final-wire assurance once per process."""
    from starlette.responses import JSONResponse
    from . import proxy as _proxy

    current_extract = _proxy.extract_user_message
    if not hasattr(current_extract, "__entroly_intent_projection_original__"):
        def intent_extract(body: dict[str, Any], provider: str) -> str:
            try:
                projection = project_retrieval_intent(body, provider)
                if projection.changed:
                    logger.debug(
                        "Intent projection removed harness-only context: blocks=%d chars=%d sha=%s",
                        projection.removed_blocks,
                        projection.removed_chars,
                        projection.source_sha256[:12],
                    )
                return projection.retrieval_text
            except Exception as exc:
                logger.debug("Intent projection failed open: %s", type(exc).__name__)
                return current_extract(body, provider)
        intent_extract.__entroly_intent_projection_original__ = current_extract
        _copy_markers(current_extract, intent_extract)
        _proxy.extract_user_message = intent_extract

    def blocked(exc: SemanticWireError, extra: dict[str, str] | None):
        response_headers = dict(extra or {})
        response_headers.update(
            {"X-Entroly-Wire-Assurance": "blocked", "X-Entroly-Wire-Code": exc.code}
        )
        return JSONResponse(
            {
                "error": "provider_semantic_contract",
                "code": exc.code,
                "path": exc.path,
                "detail": exc.detail,
            },
            status_code=422,
            headers=response_headers,
        )

    current_forward = _proxy.PromptCompilerProxy._forward_response
    if not hasattr(current_forward, "__entroly_semantic_assurance_original__"):
        async def assured_forward(
            self: Any,
            url: str,
            headers: dict[str, str],
            body: dict[str, Any],
            selected_frag_ids: list | None = None,
            witness_context: str = "",
            provider: str = "openai",
            recoverable_fragments: list[dict[str, Any]] | None = None,
            request_id: str = "",
            recovery_depth: int = 0,
            extra_headers: dict[str, str] | None = None,
            usage_dimensions: dict[str, str] | None = None,
        ):
            try:
                assured, report = assure_provider_request(body, provider)
            except SemanticWireError as exc:
                return blocked(exc, extra_headers)
            out_headers = dict(extra_headers or {})
            out_headers.update(report.response_headers())
            return await current_forward(
                self,
                url,
                headers,
                assured,
                selected_frag_ids=selected_frag_ids,
                witness_context=witness_context,
                provider=provider,
                recoverable_fragments=recoverable_fragments,
                request_id=request_id,
                recovery_depth=recovery_depth,
                extra_headers=out_headers,
                usage_dimensions=usage_dimensions,
            )
        assured_forward.__entroly_semantic_assurance_original__ = current_forward
        _copy_markers(current_forward, assured_forward)
        _proxy.PromptCompilerProxy._forward_response = assured_forward

    current_stream = _proxy.PromptCompilerProxy._stream_response
    if not hasattr(current_stream, "__entroly_semantic_assurance_original__"):
        async def assured_stream(
            self: Any,
            url: str,
            headers: dict[str, str],
            body: dict[str, Any],
            selected_frag_ids: list | None = None,
            witness_context: str = "",
            provider: str = "openai",
            recoverable_fragments: list[dict[str, Any]] | None = None,
            request_id: str = "",
            recovery_depth: int = 0,
            extra_headers: dict[str, str] | None = None,
            usage_dimensions: dict[str, str] | None = None,
        ):
            try:
                assured, report = assure_provider_request(body, provider)
            except SemanticWireError as exc:
                return blocked(exc, extra_headers)
            out_headers = dict(extra_headers or {})
            out_headers.update(report.response_headers())
            return await current_stream(
                self,
                url,
                headers,
                assured,
                selected_frag_ids=selected_frag_ids,
                witness_context=witness_context,
                provider=provider,
                recoverable_fragments=recoverable_fragments,
                request_id=request_id,
                recovery_depth=recovery_depth,
                extra_headers=out_headers,
                usage_dimensions=usage_dimensions,
            )
        assured_stream.__entroly_semantic_assurance_original__ = current_stream
        _copy_markers(current_stream, assured_stream)
        _proxy.PromptCompilerProxy._stream_response = assured_stream


__all__ = [
    "CacheBreakpoint",
    "IntentProjection",
    "SemanticRepair",
    "SemanticWireError",
    "WireAssuranceReport",
    "assure_provider_request",
    "install_proxy_semantic_assurance",
    "project_retrieval_intent",
    "validate_anthropic_cache_topology",
]
