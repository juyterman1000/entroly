"""Current Hermes ContextEngine compatibility layer.

Hermes added per-request context selection, post-turn observation, model updates,
status reporting, and engine-owned tools after Entroly's original adapter was
written. This mixin implements those optional contracts without making
``hermes-agent`` a runtime dependency.
"""

from __future__ import annotations

import json
from typing import Any

from ..exact_recovery import (
    ExactRecoveryError,
    exact_recovery_tool_schema,
    retrieve_exact,
)

_RECOVERY_MARKER = "<entroly_exact_recovery"


def _canonical_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _is_recovery_marker(message: dict[str, Any]) -> bool:
    return (
        message.get("role") == "system"
        and isinstance(message.get("content"), str)
        and _RECOVERY_MARKER in message["content"]
    )


def _insert_after_leading_systems(
    messages: list[dict[str, Any]], marker: dict[str, Any]
) -> list[dict[str, Any]]:
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        index += 1
    return [*messages[:index], marker, *messages[index:]]


class ModernHermesContextMixin:
    """Optional Hermes hooks plus proof-carrying exact recovery."""

    _model: str = ""
    _last_recovery_handle: str | None = None

    def _attach_exact_recovery(
        self,
        source_messages: list[dict[str, Any]],
        selected_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        previous_markers = [message for message in source_messages if _is_recovery_marker(message)]
        original = [message for message in source_messages if not _is_recovery_marker(message)]
        selected = [message for message in selected_messages if not _is_recovery_marker(message)]
        if _canonical_messages(original) == _canonical_messages(selected):
            result = list(selected)
            for marker in previous_markers:
                result = _insert_after_leading_systems(result, marker)
            return result

        from ...ccr import get_ccr_store

        original_text = _canonical_messages(original)
        compressed_text = _canonical_messages(selected)
        source = f"hermes:session:{getattr(self, '_session_id', None) or 'anonymous'}"
        handle = get_ccr_store().store(
            source=source,
            original_content=original_text,
            compressed_content=compressed_text,
            resolution="conversation",
            original_tokens=max(1, (len(original_text) + 3) // 4),
            compressed_tokens=max(1, (len(compressed_text) + 3) // 4),
        )
        self._last_recovery_handle = handle
        marker = {
            "role": "system",
            "content": (
                f'<entroly_exact_recovery hash="{handle}">'
                "The exact pre-compression conversation is available through the "
                "entroly_retrieve tool. Pass only this hash; do not invent a query "
                "or source path.</entroly_exact_recovery>"
            ),
        }
        return _insert_after_leading_systems(selected, marker)

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> list[dict[str, Any]]:
        """Use the existing Entroly compressor and attach one exact replay handle."""

        selected = super().compress(messages, current_tokens, focus_topic)  # type: ignore[misc]
        if not isinstance(selected, list) or not all(isinstance(item, dict) for item in selected):
            return messages
        try:
            return self._attach_exact_recovery(messages, selected)
        except Exception:
            # Exact recovery is an assurance enhancement, never a reason to break
            # a Hermes turn. The inherited compressor already follows fail-open.
            return selected

    def select_context(
        self,
        request_messages: list[dict[str, Any]],
        *,
        conversation_messages: list[dict[str, Any]] | None = None,
        incoming_message: dict[str, Any] | None = None,
        budget_tokens: int = 0,
    ) -> list[dict[str, Any]] | None:
        """Select request-only context using Hermes's current pre-dispatch hook.

        Returning ``None`` preserves a byte-identical request when the current
        messages already fit. Persisted Hermes history is never mutated.
        """

        if (
            not isinstance(request_messages, list)
            or not request_messages
            or not all(isinstance(item, dict) for item in request_messages)
            or isinstance(budget_tokens, bool)
            or int(budget_tokens) <= 0
        ):
            return None
        estimated = max(1, (len(_canonical_messages(request_messages)) + 3) // 4)
        if estimated <= int(budget_tokens):
            return None
        try:
            from ..hermes import safe_compress_hermes

            selected = safe_compress_hermes(
                [dict(item) for item in request_messages],
                budget=int(budget_tokens),
                preserve_last_n=max(2, int(getattr(self, "protect_last_n", 6))),
            )
            if not isinstance(selected, list) or not all(
                isinstance(item, dict) for item in selected
            ):
                return None
            return self._attach_exact_recovery(request_messages, selected)
        except Exception:
            return None

    def on_turn_complete(
        self,
        messages: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Observe canonical usage after a completed Hermes turn."""

        if isinstance(usage, dict):
            self.update_from_response(usage)

    def update_model(
        self,
        model: str = "",
        context_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Recalculate the compaction threshold after a Hermes model switch."""

        self._model = str(model or "")
        if isinstance(context_length, int) and not isinstance(context_length, bool):
            if context_length > 0:
                self.context_length = context_length
        threshold = kwargs.get("threshold_percent", getattr(self, "threshold_percent", 0.75))
        try:
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            threshold_value = 0.75
        self.threshold_percent = min(0.95, max(0.10, threshold_value))
        self.threshold_tokens = int(self.context_length * self.threshold_percent)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Expose one native Hermes tool with a hash-only input surface."""

        return [exact_recovery_tool_schema()]

    def handle_tool_call(
        self,
        name: str,
        args: dict[str, Any] | None,
        **kwargs: Any,
    ) -> str:
        """Dispatch native Hermes exact-recovery calls."""

        if name != "entroly_retrieve":
            return json.dumps({"status": "error", "error": f"unknown tool: {name}"})
        if not isinstance(args, dict) or set(args) != {"hash"}:
            return json.dumps(
                {
                    "status": "error",
                    "error": "entroly_retrieve accepts exactly one argument: hash",
                }
            )
        try:
            return json.dumps(retrieve_exact(str(args["hash"])), ensure_ascii=False)
        except ExactRecoveryError as error:
            return json.dumps({"status": "not_found", "error": str(error)})

    def get_status(self) -> dict[str, Any]:
        """Return bounded status fields consumed by current Hermes builds."""

        return {
            "name": self.name,
            "model": self._model,
            "context_length": int(getattr(self, "context_length", 0)),
            "threshold_tokens": int(getattr(self, "threshold_tokens", 0)),
            "last_prompt_tokens": int(getattr(self, "last_prompt_tokens", 0)),
            "last_completion_tokens": int(getattr(self, "last_completion_tokens", 0)),
            "last_total_tokens": int(getattr(self, "last_total_tokens", 0)),
            "compression_count": int(getattr(self, "compression_count", 0)),
            "exact_recovery": {
                "tool": "entroly_retrieve",
                "lookup": "hash_only",
                "full_content": True,
                "last_handle": self._last_recovery_handle,
            },
        }
