"""Explicit, fail-closed tool-schema deferral for proxy requests.

The proxy never guesses which tools an agent needs. A caller may provide a
bounded ``X-Entroly-Active-Tools`` allowlist; only then may named function
schemas outside that set be withheld from the provider request. Forced tool
choices are always retained, and requests with no matching active tool are
left unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_MAX_ACTIVE_TOOLS = 64


def _schema_tokens(value: object) -> int:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except Exception:
        return 0
    return (len(encoded) + 3) // 4


def _active_names(raw: object) -> set[str]:
    if not isinstance(raw, str):
        return set()
    names: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if not name or not _SAFE_NAME.fullmatch(name):
            continue
        names.append(name)
        if len(names) >= _MAX_ACTIVE_TOOLS:
            break
    return set(names)


def _forced_names(body: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    choice = body.get("tool_choice")
    if isinstance(choice, Mapping):
        function = choice.get("function")
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.add(str(function["name"]))
        elif isinstance(choice.get("name"), str):
            names.add(str(choice["name"]))

    config = body.get("toolConfig")
    if isinstance(config, Mapping):
        calling = config.get("functionCallingConfig")
        if isinstance(calling, Mapping):
            allowed = calling.get("allowedFunctionNames")
            if isinstance(allowed, list):
                names.update(str(name) for name in allowed if isinstance(name, str))
    return names


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return str(function["name"])
    if isinstance(tool.get("name"), str):
        return str(tool["name"])
    return ""


def _filter_tools(
    tools: list[Any],
    active: set[str],
) -> tuple[list[Any], int, int]:
    filtered: list[Any] = []
    before_count = 0
    after_count = 0
    for raw in tools:
        if not isinstance(raw, Mapping):
            filtered.append(raw)
            continue

        declaration_key = next(
            (
                key
                for key in ("functionDeclarations", "function_declarations")
                if isinstance(raw.get(key), list)
            ),
            "",
        )
        if declaration_key:
            declarations = raw[declaration_key]
            kept: list[Any] = []
            for declaration in declarations:
                if not isinstance(declaration, Mapping):
                    kept.append(declaration)
                    continue
                name = str(declaration.get("name") or "")
                if not name:
                    kept.append(dict(declaration))
                    continue
                before_count += 1
                if name in active:
                    kept.append(dict(declaration))
                    after_count += 1
            if kept:
                container = dict(raw)
                container[declaration_key] = kept
                filtered.append(container)
            continue

        name = _tool_name(raw)
        if not name:
            # Built-in/provider tools without a function name are not eligible.
            filtered.append(dict(raw))
            continue
        before_count += 1
        if name in active:
            filtered.append(dict(raw))
            after_count += 1
    return filtered, before_count, after_count


@dataclass(frozen=True, slots=True)
class ToolSchemaDeferral:
    body: dict[str, Any]
    changed: bool
    tokens_deferred: int
    schemas_before: int
    schemas_after: int
    reason: str


def defer_tool_schemas(
    body: Mapping[str, Any],
    active_tools_header: object,
) -> ToolSchemaDeferral:
    """Return a request with inactive named schemas removed when explicitly asked.

    The original mapping is never mutated. If the allowlist is absent, invalid,
    or matches no named schema, the request fails closed to its original tools.
    """

    original = dict(body)
    tools = body.get("tools")
    if not isinstance(tools, list) or not tools:
        return ToolSchemaDeferral(original, False, 0, 0, 0, "no_tools")

    active = _active_names(active_tools_header)
    if not active:
        return ToolSchemaDeferral(original, False, 0, 0, 0, "not_requested")
    active.update(_forced_names(body))

    filtered, before_count, after_count = _filter_tools(tools, active)
    if before_count <= 0:
        return ToolSchemaDeferral(original, False, 0, 0, 0, "no_named_schemas")
    if after_count <= 0 or after_count >= before_count:
        return ToolSchemaDeferral(
            original,
            False,
            0,
            before_count,
            before_count,
            "no_safe_reduction",
        )

    before_tokens = _schema_tokens(tools)
    after_tokens = _schema_tokens(filtered)
    deferred = max(0, before_tokens - after_tokens)
    if deferred <= 0:
        return ToolSchemaDeferral(
            original,
            False,
            0,
            before_count,
            before_count,
            "no_token_reduction",
        )

    output = dict(body)
    output["tools"] = filtered
    return ToolSchemaDeferral(
        output,
        True,
        deferred,
        before_count,
        after_count,
        "explicit_allowlist",
    )


__all__ = ["ToolSchemaDeferral", "defer_tool_schemas"]
