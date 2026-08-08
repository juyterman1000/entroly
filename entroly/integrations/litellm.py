"""LiteLLM Proxy pre-call hook backed by Entroly exact recovery.

The hook signature follows LiteLLM's documented ``async_pre_call_hook`` proxy
contract. LiteLLM remains optional; importing this module does not install or
start it.
"""

from __future__ import annotations

import collections
import contextvars
import threading
from typing import Any

from .gateway import CompressionGatewayClient, GatewayReceipt

try:  # pragma: no cover - optional dependency
    from litellm.integrations.custom_logger import CustomLogger as _CustomLogger
except ImportError:  # pragma: no cover - exercised through the fallback tests
    class _CustomLogger:  # type: ignore[no-redef]
        pass


class EntrolyLiteLLMHook(_CustomLogger):
    """Modify completion requests before LiteLLM sends them upstream."""

    def __init__(self, gateway: CompressionGatewayClient | None = None) -> None:
        try:
            super().__init__()
        except TypeError:
            pass
        self.gateway = gateway or CompressionGatewayClient()
        self._request_receipt: contextvars.ContextVar[GatewayReceipt | None] = (
            contextvars.ContextVar("entroly_litellm_receipt", default=None)
        )
        self._receipt_lock = threading.Lock()
        self._receipts_by_payload: collections.OrderedDict[
            int, tuple[dict[str, Any], GatewayReceipt]
        ] = collections.OrderedDict()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        del user_api_key_dict, cache
        if call_type not in {"completion", "text_completion"}:
            return data
        result = await self.gateway.async_compress_payload(data)
        self._request_receipt.set(result.receipt)
        with self._receipt_lock:
            self._receipts_by_payload[id(result.payload)] = (
                result.payload,
                result.receipt,
            )
            self._receipts_by_payload.move_to_end(id(result.payload))
            while len(self._receipts_by_payload) > 2048:
                self._receipts_by_payload.popitem(last=False)
        return result.payload

    async def async_post_call_response_headers_hook(
        self,
        data: dict[str, Any],
        user_api_key_dict: Any,
        response: Any,
        request_headers: dict[str, str] | None = None,
    ) -> dict[str, str] | None:
        del user_api_key_dict, response, request_headers
        with self._receipt_lock:
            stored = self._receipts_by_payload.pop(id(data), None)
        receipt = stored[1] if stored is not None and stored[0] is data else None
        if receipt is None:
            receipt = self._request_receipt.get()
        if receipt is None:
            return None
        self._request_receipt.set(None)
        return dict(receipt.headers)


__all__ = ["EntrolyLiteLLMHook"]
