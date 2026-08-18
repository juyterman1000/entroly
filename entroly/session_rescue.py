"""Cache-stable, recoverable rescue for runaway agent conversations.

The proxy daemon sees every outbound model request.  This controller uses that
position to prevent an append-only agent loop from crossing the provider context
limit:

* soft pressure starts evidence-locked compression only when no warm provider
  cache would be sacrificed;
* a detected retry loop or the hard watermark overrides that deferral;
* once a message is compacted, its transformed bytes are frozen for the rest of
  the session, so later turns do not repeatedly rewrite the old prefix;
* every omitted span is persisted before the outbound copy is changed.

The controller does not mutate the agent application's stored transcript.  It
rescues the request sent through Entroly and returns explicit recovery handles.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import os
from pathlib import Path

from .compression_retrieval_store_secure import CompressionRetrievalStore
from .evidence_locked_compression import compress_evidence_locked, estimate_tokens


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            text for item in content if (text := _text_from_content(item))
        )
    if isinstance(content, dict):
        parts: list[str] = []
        for key in (
            "text",
            "content",
            "output",
            "result",
            "response",
            "parts",
            "functionResponse",
            "codeExecutionResult",
        ):
            value = content.get(key)
            if isinstance(value, (str, list, dict)):
                rendered = _text_from_content(value)
                if rendered:
                    parts.append(rendered)
        return "\n".join(parts)
    return ""


def estimate_message_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """Conservative, deterministic token estimate including message overhead."""
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        text = _text_from_content(message.get("content", ""))
        # ``estimate_tokens`` is the shared Entroly estimator.  The fixed
        # overhead accounts for role/name/tool-call framing.
        total += estimate_tokens(text) + 8
    return total


def _message_digest(message: dict[str, Any]) -> str:
    encoded = json.dumps(
        message,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionRescuePolicy:
    soft_watermark: float = 0.70
    hard_watermark: float = 0.88
    target_watermark: float = 0.62
    failure_watermark: float = 0.98
    loop_min_watermark: float = 0.40
    tail_messages: int = 8
    max_sessions: int = 2_000
    min_message_tokens: int = 80
    max_block_budget: int = 1_200

    def __post_init__(self) -> None:
        levels = (
            self.loop_min_watermark,
            self.target_watermark,
            self.soft_watermark,
            self.hard_watermark,
            self.failure_watermark,
        )
        if not all(0.0 < value < 1.0 for value in levels):
            raise ValueError("session rescue watermarks must be between zero and one")
        if not (
            self.loop_min_watermark
            <= self.target_watermark
            < self.soft_watermark
            < self.hard_watermark
            < self.failure_watermark
        ):
            raise ValueError("session rescue watermarks are not ordered")
        if self.tail_messages < 2:
            raise ValueError("tail_messages must be at least two")
        if self.max_sessions < 1 or self.min_message_tokens < 1:
            raise ValueError("session rescue bounds must be positive")


@dataclass(frozen=True, slots=True)
class SessionRescueResult:
    messages: list[dict[str, Any]]
    action: str
    original_tokens: int
    forwarded_tokens: int
    utilization_before: float
    utilization_after: float
    tokens_saved: int
    recovery_receipts: tuple[str, ...] = ()
    stable_prefix_messages: int = 0
    cache_deferred: bool = False
    blocked: bool = False
    error: str = ""

    def headers(self) -> dict[str, str]:
        return {
            "X-Entroly-Session-Rescue": self.action,
            "X-Entroly-Session-Original-Tokens": str(self.original_tokens),
            "X-Entroly-Session-Forwarded-Tokens": str(self.forwarded_tokens),
            "X-Entroly-Session-Tokens-Saved": str(self.tokens_saved),
            "X-Entroly-Session-Stable-Prefix-Messages": str(
                self.stable_prefix_messages
            ),
            "X-Entroly-Session-Recovery-Receipts": str(
                len(self.recovery_receipts)
            ),
        }


@dataclass(slots=True)
class _SessionState:
    frozen: dict[str, Any] = field(default_factory=dict)
    previous_output_digests: tuple[str, ...] = ()
    active: bool = False
    checkpoints: int = 0


class SessionRescueController:
    """Bounded, thread-safe controller for outbound conversation rescue."""

    def __init__(
        self,
        *,
        recovery_store: CompressionRetrievalStore,
        policy: SessionRescuePolicy | None = None,
    ) -> None:
        if recovery_store is None:
            raise ValueError("a recovery_store is required")
        self.recovery_store = recovery_store
        self.policy = policy or SessionRescuePolicy()
        self._states: OrderedDict[str, _SessionState] = OrderedDict()
        self._lock = threading.RLock()
        self._rescues = 0
        self._blocks = 0
        self._cache_deferrals = 0
        self._failures = 0

    def rescue(
        self,
        conversation_id: str,
        messages: Sequence[dict[str, Any]],
        *,
        context_window: int,
        query: str = "",
        loop_detected: bool = False,
        cache_warm: bool = False,
    ) -> SessionRescueResult:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        if context_window < 1:
            raise ValueError("context_window must be positive")
        source = [dict(message) for message in messages if isinstance(message, dict)]
        original_tokens = estimate_message_tokens(source)
        utilization = original_tokens / context_window

        with self._lock:
            state = self._states.get(conversation_id)
            if state is None:
                state = _SessionState()
                self._states[conversation_id] = state
            self._states.move_to_end(conversation_id)
            while len(self._states) > self.policy.max_sessions:
                self._states.popitem(last=False)

            should_rescue = (
                state.active
                or utilization >= self.policy.hard_watermark
                or (
                    loop_detected
                    and utilization >= self.policy.loop_min_watermark
                )
                or (
                    utilization >= self.policy.soft_watermark
                    and not cache_warm
                )
            )
            if not should_rescue:
                deferred = (
                    cache_warm
                    and utilization >= self.policy.soft_watermark
                    and utilization < self.policy.hard_watermark
                )
                if deferred:
                    self._cache_deferrals += 1
                stable = self._stable_prefix_count(state, source)
                self._remember_output(state, source)
                return SessionRescueResult(
                    messages=source,
                    action="cache-deferred" if deferred else "passthrough",
                    original_tokens=original_tokens,
                    forwarded_tokens=original_tokens,
                    utilization_before=utilization,
                    utilization_after=utilization,
                    tokens_saved=0,
                    stable_prefix_messages=stable,
                    cache_deferred=deferred,
                )

            emergency = utilization >= self.policy.hard_watermark
            target_tokens = max(1, int(context_window * self.policy.target_watermark))
            working = copy.deepcopy(source)
            receipt_ids: list[str] = []
            try:
                working, receipt_ids = self._compact_candidates(
                    state,
                    working,
                    query=query,
                    target_tokens=target_tokens,
                    include_assistant=emergency or loop_detected,
                )
            except Exception as exc:
                self._failures += 1
                stable = self._stable_prefix_count(state, source)
                self._remember_output(state, source)
                return SessionRescueResult(
                    messages=source,
                    action="failed",
                    original_tokens=original_tokens,
                    forwarded_tokens=original_tokens,
                    utilization_before=utilization,
                    utilization_after=utilization,
                    tokens_saved=0,
                    stable_prefix_messages=stable,
                    blocked=utilization >= self.policy.failure_watermark,
                    error=f"recovery persistence failed: {exc}",
                )

            forwarded_tokens = estimate_message_tokens(working)
            after = forwarded_tokens / context_window
            blocked = after >= self.policy.failure_watermark
            changed = working != source
            if blocked:
                self._blocks += 1
            else:
                state.active = True
                if changed:
                    state.checkpoints += 1
                    self._rescues += 1
            stable = self._stable_prefix_count(state, working)
            self._remember_output(state, working)
            return SessionRescueResult(
                messages=working,
                action=(
                    "blocked"
                    if blocked
                    else "pressure-observed"
                    if not changed
                    else "loop-rescue"
                    if loop_detected
                    else "emergency-rescue"
                    if emergency
                    else "high-water-rescue"
                ),
                original_tokens=original_tokens,
                forwarded_tokens=forwarded_tokens,
                utilization_before=utilization,
                utilization_after=after,
                tokens_saved=max(0, original_tokens - forwarded_tokens),
                recovery_receipts=tuple(receipt_ids),
                stable_prefix_messages=stable,
                blocked=blocked,
                error=(
                    "request still exceeds the safe context watermark after "
                    "recoverable compression"
                    if blocked
                    else ""
                ),
            )

    def _compact_candidates(
        self,
        state: _SessionState,
        messages: list[dict[str, Any]],
        *,
        query: str,
        target_tokens: int,
        include_assistant: bool,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        cutoff = max(0, len(messages) - self.policy.tail_messages)
        tool_indices = [
            index
            for index, message in enumerate(messages[:cutoff])
            if str(message.get("role", "")).lower() in {"tool", "function"}
            or self._has_tool_result(message.get("content"))
        ]
        assistant_indices = [
            index
            for index, message in enumerate(messages[:cutoff])
            if str(message.get("role", "")).lower() == "assistant"
            and index not in tool_indices
        ]
        candidates = tool_indices + (assistant_indices if include_assistant else [])
        receipts: list[str] = []
        for index in candidates:
            if estimate_message_tokens(messages) <= target_tokens:
                break
            message = messages[index]
            digest = _message_digest(message)
            frozen = state.frozen.get(digest)
            if frozen is not None:
                messages[index] = copy.deepcopy(frozen)
                continue
            # Frozen historical bytes must not depend on the newest user
            # query. Otherwise a daemon restart or state eviction can
            # recompress the same old tool output differently and churn the
            # provider-cache prefix. Session rescue is structural, so use a
            # stable query-independent compression contract.
            compacted, message_receipts = self._compact_message(
                message,
                query="",
            )
            if compacted == message:
                continue
            state.frozen[digest] = copy.deepcopy(compacted)
            messages[index] = compacted
            receipts.extend(message_receipts)
        return messages, receipts

    def _compact_message(
        self,
        message: dict[str, Any],
        *,
        query: str,
    ) -> tuple[dict[str, Any], list[str]]:
        content = message.get("content")
        if isinstance(content, str):
            compacted, receipt_id = self._compact_text(content, query=query)
            if receipt_id is None:
                return message, []
            updated = dict(message)
            updated["content"] = compacted
            return updated, [receipt_id]
        if isinstance(content, list):
            updated_blocks: list[Any] = []
            receipt_ids: list[str] = []
            changed = False
            for block in content:
                if not isinstance(block, dict):
                    updated_blocks.append(block)
                    continue
                if block.get("type") == "tool_result":
                    inner = block.get("content")
                    if not isinstance(inner, str):
                        updated_blocks.append(block)
                        continue
                    compacted, receipt_id = self._compact_text(inner, query=query)
                    if receipt_id is None:
                        updated_blocks.append(block)
                        continue
                    updated = dict(block)
                    updated["content"] = compacted
                    updated_blocks.append(updated)
                    receipt_ids.append(receipt_id)
                    changed = True
                    continue
                updated, block_receipts = self._compact_gemini_part(
                    block,
                    query=query,
                )
                if not block_receipts:
                    updated_blocks.append(block)
                    continue
                updated_blocks.append(updated)
                receipt_ids.extend(block_receipts)
                changed = True
            if changed:
                updated_message = dict(message)
                updated_message["content"] = updated_blocks
                return updated_message, receipt_ids
        return message, []

    def _compact_gemini_part(
        self,
        block: dict[str, Any],
        *,
        query: str,
    ) -> tuple[dict[str, Any], list[str]]:
        """Compress known textual Gemini tool outputs without touching IDs."""
        updated = copy.deepcopy(block)
        targets: list[tuple[dict[str, Any], str]] = []
        function_response = updated.get("functionResponse")
        if isinstance(function_response, dict):
            response = function_response.get("response")
            if isinstance(response, dict):
                targets.extend(
                    (response, key)
                    for key in ("output", "result", "content", "text")
                    if isinstance(response.get(key), str)
                )
        code_result = updated.get("codeExecutionResult")
        if isinstance(code_result, dict) and isinstance(
            code_result.get("output"),
            str,
        ):
            targets.append((code_result, "output"))

        receipts: list[str] = []
        for container, key in targets:
            compacted, receipt_id = self._compact_text(
                container[key],
                query=query,
            )
            if receipt_id is None:
                continue
            container[key] = compacted
            receipts.append(receipt_id)
        return (updated, receipts) if receipts else (block, [])

    def _compact_text(self, text: str, *, query: str) -> tuple[str, str | None]:
        original_tokens = estimate_tokens(text)
        if original_tokens < self.policy.min_message_tokens:
            return text, None
        budget = min(
            self.policy.max_block_budget,
            max(64, int(original_tokens * 0.30)),
        )
        result = compress_evidence_locked(
            text,
            query=query,
            budget_tokens=budget,
            min_savings=0.08,
        )
        if not result.changed:
            return text, None
        receipt = result.receipt.as_dict()
        # A session-rescue handle is a last-resort integrity boundary, not just
        # an optimization receipt. Persist the complete original as one span so
        # recovery never depends on a compressor's omission bookkeeping being
        # exhaustive. The compacted request still carries the exact retained
        # evidence and the receipt's content hash.
        line_count = max(1, text.count("\n") + 1)
        receipt["omitted_spans"] = [
            {
                "start_line": 1,
                "end_line": line_count,
                "line_count": line_count,
                "reason": "session_rescue_full_recovery",
            }
        ]
        stored = self.recovery_store.put(
            original_text=text,
            compressed_text=result.compressed,
            receipt=receipt,
            metadata={
                "component": "session_rescue",
                "query_sha256": hashlib.sha256(
                    query.encode("utf-8", "ignore")
                ).hexdigest(),
            },
        )
        if len(stored.spans) != 1:
            raise RuntimeError(
                "session rescue requires one exact full-original recovery span"
            )
        span_id = stored.spans[0].span_id
        marker = f"[entroly-recovery:{stored.receipt_id}:{span_id}]"
        rendered = f"{result.with_receipt_header()}\n{marker}"
        if estimate_tokens(rendered) >= original_tokens:
            return text, None
        return rendered, stored.receipt_id

    @staticmethod
    def _has_tool_result(content: Any) -> bool:
        return isinstance(content, list) and any(
            isinstance(block, dict)
            and (
                block.get("type") == "tool_result"
                or "functionResponse" in block
                or "codeExecutionResult" in block
            )
            for block in content
        )

    @staticmethod
    def _stable_prefix_count(
        state: _SessionState,
        messages: Sequence[dict[str, Any]],
    ) -> int:
        current = tuple(_message_digest(message) for message in messages)
        stable = 0
        for previous, candidate in zip(state.previous_output_digests, current):
            if previous != candidate:
                break
            stable += 1
        return stable

    @staticmethod
    def _remember_output(
        state: _SessionState,
        messages: Sequence[dict[str, Any]],
    ) -> None:
        state.previous_output_digests = tuple(
            _message_digest(message) for message in messages
        )

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "tracked_sessions": len(self._states),
                "rescues": self._rescues,
                "blocked": self._blocks,
                "cache_deferrals": self._cache_deferrals,
                "failures": self._failures,
                "active_sessions": sum(
                    1 for state in self._states.values() if state.active
                ),
            }


# ── surface-neutral entry point ──────────────────────────────────────────────
#
# The controller above is pure policy: it takes a message list and returns a
# transformed one. Nothing in it is HTTP-aware. It was reachable only through
# `entroly proxy` for no better reason than that being its first caller, which
# left every pip, npm, SDK and provider-SDK user without a capability their
# install already contained.
#
# What *is* proxy-specific is being automatic. Rescue must run on the outbound
# request, and the proxy is the only surface Entroly owns that sees one. Any
# caller that can hand over its conversation can drive the same policy; this is
# that entry point.

_DEFAULT_CONTROLLER: SessionRescueController | None = None
_DEFAULT_CONTROLLER_LOCK = threading.Lock()


def default_controller() -> SessionRescueController:
    """The process-wide controller, created on first use.

    Sharing one controller per process is load-bearing, not convenience. The
    freeze semantics that keep the prompt prefix byte-stable live in its
    per-conversation state: once a message has been compacted, its transformed
    bytes are reused for the rest of the session instead of being recomputed.
    A caller that built a fresh controller per turn would recompress the old
    prefix every time, changing bytes the provider cache is keyed on -- turning
    a cache-preserving rescue into a cache-destroying one.

    Honours the same `ENTROLY_DIR` / `ENTROLY_SESSION_RESCUE_STORE` settings as
    the proxy, so a machine running both writes recovery records to one store
    and `entroly recover` finds spans from either surface.
    """
    global _DEFAULT_CONTROLLER
    if _DEFAULT_CONTROLLER is not None:
        return _DEFAULT_CONTROLLER
    with _DEFAULT_CONTROLLER_LOCK:
        if _DEFAULT_CONTROLLER is None:
            root = Path(os.environ.get("ENTROLY_DIR", str(Path.home() / ".entroly")))
            path = Path(
                os.environ.get(
                    "ENTROLY_SESSION_RESCUE_STORE",
                    str(root / "session_rescue_recovery.json"),
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            _DEFAULT_CONTROLLER = SessionRescueController(
                recovery_store=CompressionRetrievalStore(path)
            )
    return _DEFAULT_CONTROLLER


def rescue_session(
    conversation_id: str,
    messages: Sequence[dict[str, Any]],
    *,
    context_window: int,
    query: str = "",
    loop_detected: bool = False,
    cache_warm: bool = False,
    controller: SessionRescueController | None = None,
) -> SessionRescueResult:
    """Compact a conversation that is approaching the provider context limit.

    The same policy the proxy runs, callable from anywhere that assembles a
    prompt -- the Python SDK, a provider-SDK wrapper, a CLI pipeline, or an MCP
    host that chooses to pass its transcript in::

        from entroly.session_rescue import rescue_session

        result = rescue_session(
            conversation_id=session_id,
            messages=messages,
            context_window=200_000,
            cache_warm=True,          # defer while a warm prefix would be lost
        )
        messages = result.messages    # unchanged unless a watermark was crossed

    Below the soft watermark this returns the conversation untouched, so it is
    safe to call on every turn; that is how it is meant to be used, because the
    watermark policy needs to see the growth to act on it.

    Omitted spans are persisted to the recovery store *before* the returned copy
    is changed, so nothing is dropped that cannot be recovered.

    `cache_warm=True` tells the policy that a warm provider cache would be
    sacrificed by compacting now; it defers under soft pressure and overrides
    only at the hard watermark or on a detected retry loop. Callers that cannot
    tell should leave it False.

    **What actually gets compacted, and what does not.** Candidates are tool and
    function messages (and, once past the hard watermark or on a detected loop,
    older assistant turns) outside the protected tail. Within those, only
    content the evidence-locked compressor recognises as compressible is
    touched -- bulky structured output: logs, JSON, tables, build chatter. Plain
    prose reasoning is left alone rather than paraphrased, which is the point:
    this compacts machine output, it does not summarise your conversation.

    A consequence worth stating plainly, because a silent no-op is the failure
    mode this project keeps having: a conversation over the watermark whose bulk
    is prose comes back with ``action="pressure-observed"`` and
    ``tokens_saved=0``. That is the honest answer -- nothing safely compressible
    was found -- not a malfunction. Check ``result.action`` rather than assuming
    a call did something.

    ``query`` is accepted for interface parity with the proxy and is **not** used
    to choose what to drop. Compaction is deliberately query-independent so that
    the same historical message always compresses to the same bytes; letting the
    newest question reshape old content would churn the very prefix the provider
    cache is keyed on.
    """
    return (controller or default_controller()).rescue(
        conversation_id,
        messages,
        context_window=context_window,
        query=query,
        loop_detected=loop_detected,
        cache_warm=cache_warm,
    )


__all__ = [
    "SessionRescueController",
    "SessionRescuePolicy",
    "SessionRescueResult",
    "default_controller",
    "estimate_message_tokens",
    "rescue_session",
]
