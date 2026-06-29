"""Bounded behavioral telemetry for agent retry and routing waste."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_ERROR_RE = re.compile(
    r"(?i)(error|exception|failed|failure|fatal|panic|timeout|denied|refused|invalid|not found)"
)
_VOLATILE_RE = re.compile(
    r"(?i)(?:0x[0-9a-f]+|\b[0-9a-f]{8}-[0-9a-f-]{27,}\b|\b\d{2,}\b)"
)


@dataclass(frozen=True, slots=True)
class WasteFinding:
    kind: str
    conversation_id: str
    occurrences: int
    estimated_wasted_tokens: int
    confidence: float
    severity: str
    tier: str = "opportunity"
    fingerprint: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _Observation:
    timestamp: float
    tool_signature: str
    error_fingerprint: str
    model: str
    tokens: int


@dataclass(slots=True)
class _ConversationState:
    observations: deque[_Observation]
    emitted: set[tuple[str, str, int]] = field(default_factory=set)
    last_seen: float = 0.0


class BehavioralWasteDetector:
    """Detect repeated calls, repeated failures, loops, and model churn.

    Findings are opportunity estimates, not realized savings. The detector is
    telemetry-only and never blocks, retries, reroutes, or calls a provider.
    """

    def __init__(
        self,
        *,
        repeat_threshold: int = 3,
        window_seconds: float = 600.0,
        max_observations: int = 64,
        max_conversations: int = 10_000,
    ) -> None:
        if repeat_threshold < 2:
            raise ValueError("repeat_threshold must be at least two")
        if window_seconds <= 0 or max_observations < 4 or max_conversations < 1:
            raise ValueError("detector bounds must be positive")
        self.repeat_threshold = repeat_threshold
        self.window_seconds = window_seconds
        self.max_observations = max_observations
        self.max_conversations = max_conversations
        self._states: OrderedDict[str, _ConversationState] = OrderedDict()
        self._lock = threading.RLock()

    def observe(
        self,
        conversation_id: str,
        *,
        tool_name: str = "",
        arguments: Mapping[str, Any] | None = None,
        result_text: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        observed_at: float | None = None,
    ) -> tuple[WasteFinding, ...]:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token counts must be non-negative")
        timestamp = time.time() if observed_at is None else float(observed_at)
        signature = _tool_signature(tool_name, arguments) if tool_name else ""
        error = _error_fingerprint(result_text)
        observation = _Observation(
            timestamp=timestamp,
            tool_signature=signature,
            error_fingerprint=error,
            model=model,
            tokens=input_tokens + output_tokens,
        )
        with self._lock:
            state = self._states.get(conversation_id)
            if state is None:
                state = _ConversationState(deque(maxlen=self.max_observations))
                self._states[conversation_id] = state
            cutoff = timestamp - self.window_seconds
            while state.observations and state.observations[0].timestamp < cutoff:
                state.observations.popleft()
            state.observations.append(observation)
            state.last_seen = timestamp
            self._states.move_to_end(conversation_id)
            while len(self._states) > self.max_conversations:
                self._states.popitem(last=False)
            findings = self._findings(conversation_id, state)
            return tuple(
                finding
                for finding in findings
                if self._mark_new(state, finding)
            )

    def _findings(
        self, conversation_id: str, state: _ConversationState
    ) -> list[WasteFinding]:
        observations = list(state.observations)
        findings: list[WasteFinding] = []
        latest = observations[-1]
        if latest.tool_signature:
            repeats = [o for o in observations if o.tool_signature == latest.tool_signature]
            if len(repeats) >= self.repeat_threshold:
                findings.append(
                    _finding(
                        "identical_tool_retry",
                        conversation_id,
                        latest.tool_signature,
                        repeats,
                    )
                )
        if latest.error_fingerprint:
            failures = [o for o in observations if o.error_fingerprint == latest.error_fingerprint]
            if len(failures) >= self.repeat_threshold:
                findings.append(
                    _finding(
                        "repeated_error",
                        conversation_id,
                        latest.error_fingerprint,
                        failures,
                    )
                )
        signatures = [o.tool_signature for o in observations if o.tool_signature]
        if len(signatures) >= 4:
            tail = signatures[-4:]
            if tail[0] == tail[2] and tail[1] == tail[3] and tail[0] != tail[1]:
                loop_obs = [o for o in observations[-4:] if o.tool_signature]
                findings.append(
                    _finding(
                        "alternating_tool_loop",
                        conversation_id,
                        _short_hash("|".join(tail)),
                        loop_obs,
                    )
                )
        models = [o.model for o in observations if o.model]
        switches = sum(a != b for a, b in zip(models, models[1:]))
        if switches >= self.repeat_threshold:
            model_obs = [o for o in observations if o.model]
            findings.append(
                _finding(
                    "model_switch_churn",
                    conversation_id,
                    _short_hash("|".join(models[-8:])),
                    model_obs,
                    occurrences=switches,
                )
            )
        return findings

    @staticmethod
    def _mark_new(state: _ConversationState, finding: WasteFinding) -> bool:
        threshold_bucket = finding.occurrences // 3
        key = (finding.kind, finding.fingerprint, threshold_bucket)
        if key in state.emitted:
            return False
        state.emitted.add(key)
        return True


def _finding(
    kind: str,
    conversation_id: str,
    fingerprint: str,
    observations: list[_Observation],
    *,
    occurrences: int | None = None,
) -> WasteFinding:
    count = occurrences if occurrences is not None else len(observations)
    wasted = sum(observation.tokens for observation in observations[1:])
    confidence = min(0.99, 0.55 + 0.10 * max(0, count - 2))
    return WasteFinding(
        kind=kind,
        conversation_id=conversation_id,
        occurrences=count,
        estimated_wasted_tokens=wasted,
        confidence=round(confidence, 3),
        severity="high" if count >= 5 else "medium",
        fingerprint=fingerprint,
    )


def _tool_signature(tool_name: str, arguments: Mapping[str, Any] | None) -> str:
    payload = json.dumps(arguments or {}, sort_keys=True, separators=(",", ":"), default=str)
    return _short_hash(f"{tool_name.casefold()}:{payload}")


def _error_fingerprint(text: str) -> str:
    matching = [line.strip() for line in text.splitlines() if _ERROR_RE.search(line)]
    if not matching:
        return ""
    normalized = " ".join(_VOLATILE_RE.sub("#", line.casefold()) for line in matching[:4])
    return _short_hash(normalized[:1000])


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def observe_canonical_messages(
    detector: BehavioralWasteDetector,
    conversation_id: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    observed_at: float | None = None,
) -> tuple[WasteFinding, ...]:
    """Extract one portable behavioral observation from canonical messages."""
    tool_name = ""
    arguments: dict[str, Any] = {}
    result_text = ""
    if messages:
        latest = messages[-1]
        role = str(latest.get("role", ""))
        tool_calls = latest.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            call = tool_calls[-1]
            if isinstance(call, Mapping):
                function = call.get("function")
                if isinstance(function, Mapping):
                    tool_name = str(function.get("name", ""))
                    arguments = {"arguments": function.get("arguments", "")}
        if role in {"tool", "function"}:
            tool_name = str(
                latest.get("name") or latest.get("tool_call_id") or role
            )
            result_text = _content_text(latest.get("content"))
            arguments = {"tool_call_id": str(latest.get("tool_call_id", ""))}
        content = latest.get("content")
        if isinstance(content, list):
            for block in reversed(content):
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "tool_result":
                    tool_name = str(block.get("tool_use_id") or "tool_result")
                    result_text = _content_text(block.get("content"))
                    arguments = {"tool_use_id": tool_name}
                    break
                if block.get("type") == "tool_use":
                    tool_name = str(block.get("name") or "tool_use")
                    raw_input = block.get("input")
                    arguments = dict(raw_input) if isinstance(raw_input, Mapping) else {}
                    break
    token_estimate = max(
        0,
        len(
            json.dumps(messages, sort_keys=True, default=str).encode(
                "utf-8", errors="replace"
            )
        )
        // 4,
    )
    return detector.observe(
        conversation_id,
        tool_name=tool_name,
        arguments=arguments,
        result_text=result_text,
        model=model,
        input_tokens=token_estimate,
        observed_at=observed_at,
    )


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_content_text(part) for part in value)
    if isinstance(value, Mapping):
        return _content_text(value.get("text", value.get("content", "")))
    return ""


__all__ = ["BehavioralWasteDetector", "WasteFinding", "observe_canonical_messages"]
