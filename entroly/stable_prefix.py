"""Deterministic prompt-prefix construction for provider cache reuse.

The builder separates byte-stable, cacheable context from the dynamic request
suffix. It never mutates caller-owned values and deliberately excludes the model
from the conversation anchor so routing can evaluate a model switch without
losing conversation identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

_SECTION_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


def _normalize_text(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data with a stable byte representation."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_content(value: Any) -> str:
    """Canonicalize text or structured prefix content."""
    if isinstance(value, str):
        return _normalize_text(value)
    return canonical_json(value)


@dataclass(frozen=True, slots=True)
class PrefixSection:
    name: str
    content: str
    priority: int = 100

    def __post_init__(self) -> None:
        if not _SECTION_RE.fullmatch(self.name):
            raise ValueError(
                "section name must match [a-z][a-z0-9_.-]{0,63}"
            )


@dataclass(frozen=True, slots=True)
class StablePrompt:
    """A cacheable prefix plus a deliberately separate dynamic suffix."""

    stable_prefix: str
    dynamic_tail: str
    prefix_hash: str
    version: str
    section_names: tuple[str, ...]

    @property
    def rendered(self) -> str:
        if not self.dynamic_tail:
            return self.stable_prefix
        return f"{self.stable_prefix}\n\n{self.dynamic_tail}"

    @property
    def stable_tokens_estimate(self) -> int:
        return max(1, len(self.stable_prefix) // 4)


class CanonicalPrefixBuilder:
    """Build byte-identical prefixes from semantically identical inputs."""

    def __init__(self, *, namespace: str = "entroly", version: str = "1") -> None:
        if not namespace or any(ch.isspace() for ch in namespace):
            raise ValueError("namespace must be non-empty and contain no whitespace")
        if not version:
            raise ValueError("version must be non-empty")
        self._namespace = namespace
        self._version = version
        self._sections: dict[str, PrefixSection] = {}

    def add(
        self,
        name: str,
        content: Any,
        *,
        priority: int = 100,
    ) -> "CanonicalPrefixBuilder":
        normalized = canonical_content(content)
        section = PrefixSection(name=name, content=normalized, priority=int(priority))
        previous = self._sections.get(name)
        if previous is not None and previous != section:
            raise ValueError(f"section {name!r} already has different content")
        self._sections[name] = section
        return self

    def add_tools(
        self,
        tools: Iterable[Mapping[str, Any]],
        *,
        priority: int = 30,
    ) -> "CanonicalPrefixBuilder":
        """Add tool schemas sorted by their stable semantic identity."""
        normalized = [dict(tool) for tool in tools]
        normalized.sort(
            key=lambda tool: (
                str(tool.get("name", "")),
                canonical_json(tool),
            )
        )
        return self.add("tools", normalized, priority=priority)

    def build(self, *, dynamic_tail: Any = "") -> StablePrompt:
        ordered = sorted(
            self._sections.values(),
            key=lambda section: (section.priority, section.name),
        )
        header = f"{self._namespace.upper()}-PREFIX/{self._version}"
        chunks = [header]
        for section in ordered:
            chunks.append(f"[{section.name}]\n{section.content}")
        stable = "\n\n".join(chunks)
        digest = hashlib.sha256(stable.encode("utf-8")).hexdigest()
        return StablePrompt(
            stable_prefix=stable,
            dynamic_tail=canonical_content(dynamic_tail),
            prefix_hash=digest,
            version=self._version,
            section_names=tuple(section.name for section in ordered),
        )


def conversation_anchor(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    namespace: str = "entroly-conversation-v1",
) -> str:
    """Return a stable, model-independent conversation identity.

    Only the first system message and first user message are anchors. Appending
    turns cannot change the identity. Tool names are included because changing
    the available action surface creates a materially different conversation.
    """
    anchors: list[dict[str, str]] = []
    seen_roles: set[str] = set()
    for message in messages:
        role = str(message.get("role", ""))
        if role not in {"system", "user"} or role in seen_roles:
            continue
        content = message.get("content", "")
        anchors.append({"role": role, "content": canonical_content(content)})
        seen_roles.add(role)
        if seen_roles == {"system", "user"}:
            break

    tool_names = sorted(
        str(tool.get("name", ""))
        for tool in (tools or ())
        if tool.get("name")
    )
    payload = canonical_json(
        {"namespace": namespace, "anchors": anchors, "tools": tool_names}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "CanonicalPrefixBuilder",
    "PrefixSection",
    "StablePrompt",
    "canonical_content",
    "canonical_json",
    "conversation_anchor",
]
