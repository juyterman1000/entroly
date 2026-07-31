"""Direct HTTP proxy integration helper for ELC.

This is the non-monkey-patch integration surface intended for `proxy.py`:
call this from the request handler after aged tool pruning and before normal
conversation compression.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .compression_proxy import ProxyCompressionResult, _env_budget_tokens, compress_proxy_payload
from .compression_retrieval_store import CompressionRetrievalStore
from .proxy_transform import extract_user_message


def apply_elc_to_proxy_body(
    body: dict[str, Any],
    *,
    provider: str,
    budget_tokens: int | None = None,
) -> ProxyCompressionResult:
    """Apply Evidence-Locked Compression directly to a proxy request body.

    This function is side-effect free and returns the transformed body plus
    receipt/headers. It is safe for the live HTTP proxy to call directly instead
    of monkey-patching `proxy_transform.compress_tool_messages`.
    """
    mode = os.environ.get("ENTROLY_COMPRESSION_PROXY_MODE", "off").strip().lower()
    budget = budget_tokens if budget_tokens is not None else _env_budget_tokens()
    store_path = os.environ.get("ENTROLY_COMPRESSION_STORE")
    store = CompressionRetrievalStore(Path(store_path)) if store_path else None
    query = _extract_query(body)
    return compress_proxy_payload(
        body,
        provider=provider,
        query=query,
        budget_tokens=budget,
        mode="elc" if mode == "elc" else "off",
        retrieval_store=store,
        compress_user_messages=os.environ.get("ENTROLY_ELC_COMPRESS_USER", "0").lower()
        in {"1", "true", "yes", "on"},
    )


def _extract_query(body: dict[str, Any]) -> str:
    try:
        return extract_user_message(body)
    except Exception:
        messages = body.get("messages")
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str):
                        return content[:500]
        return ""


__all__ = ["apply_elc_to_proxy_body"]
