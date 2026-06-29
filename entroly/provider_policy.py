"""Provider capability planning and explicit gateway redaction policy.

This module is deliberately transport-free. It decides which targets are safe
for a canonical request and produces an ordered failover plan. Network retries
remain in the proxy, keeping policy deterministic and independently testable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class Capability(str, Enum):
    CHAT = "chat"
    STREAMING = "streaming"
    TOOLS = "tools"
    JSON_SCHEMA = "json_schema"
    VISION = "vision"
    REASONING = "reasoning"


@dataclass(frozen=True, slots=True)
class CanonicalGatewayRequest:
    model: str
    messages: tuple[Mapping[str, Any], ...]
    tools: tuple[Mapping[str, Any], ...] = ()
    stream: bool = False
    response_schema: Mapping[str, Any] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    requires_vision: bool = False
    requires_reasoning: bool = False

    def required_capabilities(self) -> frozenset[Capability]:
        required = {Capability.CHAT}
        if self.stream:
            required.add(Capability.STREAMING)
        if self.tools:
            required.add(Capability.TOOLS)
        if self.response_schema is not None:
            required.add(Capability.JSON_SCHEMA)
        if self.requires_vision:
            required.add(Capability.VISION)
        if self.requires_reasoning:
            required.add(Capability.REASONING)
        return frozenset(required)


@dataclass(frozen=True, slots=True)
class ProviderTarget:
    provider: str
    model: str
    capabilities: frozenset[Capability]
    priority: int = 100
    expected_input_cost_per_million: float = 0.0
    expected_latency_ms: float = 0.0
    healthy: bool = True
    circuit_open: bool = False

    def __post_init__(self) -> None:
        if not self.provider or not self.model:
            raise ValueError("provider and model are required")
        if self.expected_input_cost_per_million < 0:
            raise ValueError("expected cost must be non-negative")
        if self.expected_latency_ms < 0:
            raise ValueError("expected latency must be non-negative")

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def supports(self, required: frozenset[Capability]) -> bool:
        return required.issubset(self.capabilities)


@dataclass(frozen=True, slots=True)
class FailoverPlan:
    attempts: tuple[ProviderTarget, ...]
    excluded: Mapping[str, str]
    required_capabilities: frozenset[Capability]

    @property
    def primary(self) -> ProviderTarget:
        if not self.attempts:
            raise RuntimeError("failover plan has no eligible attempts")
        return self.attempts[0]


class ProviderFailoverPlanner:
    """Build a deterministic, capability-safe target order."""

    def plan(
        self,
        request: CanonicalGatewayRequest,
        targets: Iterable[ProviderTarget],
        *,
        preferred_key: str | None = None,
        excluded_keys: Iterable[str] = (),
    ) -> FailoverPlan:
        required = request.required_capabilities()
        explicitly_excluded = set(excluded_keys)
        eligible: list[ProviderTarget] = []
        excluded: dict[str, str] = {}

        for target in targets:
            if target.key in explicitly_excluded:
                excluded[target.key] = "explicitly_excluded"
            elif not target.healthy:
                excluded[target.key] = "unhealthy"
            elif target.circuit_open:
                excluded[target.key] = "circuit_open"
            elif not target.supports(required):
                missing = sorted(cap.value for cap in required - target.capabilities)
                excluded[target.key] = f"missing_capabilities:{','.join(missing)}"
            else:
                eligible.append(target)

        eligible.sort(
            key=lambda target: (
                0 if target.key == preferred_key else 1,
                target.priority,
                target.expected_input_cost_per_million,
                target.expected_latency_ms,
                target.key,
            )
        )
        if not eligible:
            detail = "; ".join(f"{key}={value}" for key, value in sorted(excluded.items()))
            raise RuntimeError(f"no capability-compatible provider target ({detail})")

        return FailoverPlan(
            attempts=tuple(eligible),
            excluded=excluded,
            required_capabilities=required,
        )


@dataclass(frozen=True, slots=True)
class RedactionRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str = "[REDACTED]"


@dataclass(frozen=True, slots=True)
class RedactionFinding:
    rule: str
    digest: str


@dataclass(frozen=True, slots=True)
class RedactionReceipt:
    enabled: bool
    changed: bool
    findings: tuple[RedactionFinding, ...]

    @property
    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for finding in self.findings:
            result[finding.rule] = result.get(finding.rule, 0) + 1
        return result


_DEFAULT_RULES = (
    RedactionRule(
        "openai_api_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    ),
    RedactionRule(
        "github_token",
        re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    ),
    RedactionRule(
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    RedactionRule(
        "credential_assignment",
        re.compile(
            r"(?i)\b(password|secret|api[_-]?key)\s*[:=]\s*([^\s,;]+)"
        ),
        replacement="[REDACTED_CREDENTIAL]",
    ),
    RedactionRule(
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        replacement="[REDACTED_EMAIL]",
    ),
)


class GatewayRedactionPolicy:
    """Opt-in outbound redaction that never stores the matched secret."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        rules: Sequence[RedactionRule] = _DEFAULT_RULES,
        digest_salt: str = "entroly-redaction-v1",
    ) -> None:
        self.enabled = enabled
        self.rules = tuple(rules)
        self._salt = digest_salt

    def redact_text(self, text: str) -> tuple[str, RedactionReceipt]:
        if not self.enabled:
            return text, RedactionReceipt(False, False, ())

        findings: list[RedactionFinding] = []
        redacted = text
        for rule in self.rules:
            def replace_match(match: re.Match[str]) -> str:
                digest = hashlib.sha256(
                    f"{self._salt}\0{rule.name}\0{match.group(0)}".encode("utf-8")
                ).hexdigest()[:16]
                findings.append(RedactionFinding(rule=rule.name, digest=digest))
                return rule.replacement

            redacted = rule.pattern.sub(replace_match, redacted)

        return redacted, RedactionReceipt(
            enabled=True,
            changed=bool(findings),
            findings=tuple(findings),
        )

    def redact_value(self, value: Any) -> tuple[Any, RedactionReceipt]:
        findings: list[RedactionFinding] = []

        def walk(item: Any) -> Any:
            if isinstance(item, str):
                output, receipt = self.redact_text(item)
                findings.extend(receipt.findings)
                return output
            if isinstance(item, list):
                return [walk(child) for child in item]
            if isinstance(item, tuple):
                return tuple(walk(child) for child in item)
            if isinstance(item, Mapping):
                return {key: walk(child) for key, child in item.items()}
            return item

        if not self.enabled:
            return value, RedactionReceipt(False, False, ())
        output = walk(value)
        return output, RedactionReceipt(True, bool(findings), tuple(findings))

    def apply(
        self,
        request: CanonicalGatewayRequest,
    ) -> tuple[CanonicalGatewayRequest, RedactionReceipt]:
        if not self.enabled:
            return request, RedactionReceipt(False, False, ())

        messages, message_receipt = self.redact_value(request.messages)
        tools, tool_receipt = self.redact_value(request.tools)
        schema, schema_receipt = self.redact_value(request.response_schema)
        findings = (
            message_receipt.findings
            + tool_receipt.findings
            + schema_receipt.findings
        )
        updated = replace(
            request,
            messages=tuple(messages),
            tools=tuple(tools),
            response_schema=schema,
        )
        return updated, RedactionReceipt(True, bool(findings), findings)


__all__ = [
    "CanonicalGatewayRequest",
    "Capability",
    "FailoverPlan",
    "GatewayRedactionPolicy",
    "ProviderFailoverPlanner",
    "ProviderTarget",
    "RedactionFinding",
    "RedactionReceipt",
    "RedactionRule",
]
