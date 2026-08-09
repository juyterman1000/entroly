"""Dependency-free LiteLLM Proxy pre-call hook."""
from __future__ import annotations

from typing import Any

from .request_adapter import compress_request_payload


class EntrolyLiteLLMCallback:
    """Duck-typed LiteLLM ``async_pre_call_hook`` implementation.

    Register ``entroly.integrations.litellm.proxy_handler_instance`` in
    ``litellm_settings.callbacks``. The hook preserves all fields other than
    compressible string message content, including tool calls and controls.
    """

    def __init__(self, budget: int = 50_000, preserve_last_n: int = 4):
        self.budget = budget
        self.preserve_last_n = preserve_last_n
        self.last_result = None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        if call_type not in {
            "completion", "acompletion", "responses", "aresponses",
            "anthropic_messages", "text_completion",
        }:
            return data
        result = compress_request_payload(
            data,
            budget=self.budget,
            preserve_last_n=self.preserve_last_n,
        )
        self.last_result = result
        return result.payload


proxy_handler_instance = EntrolyLiteLLMCallback()


__all__ = ["EntrolyLiteLLMCallback", "proxy_handler_instance"]
