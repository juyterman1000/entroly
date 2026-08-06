"""Live HTTP proxy installer for Evidence-Locked Compression.

The existing HTTP proxy imports ``compress_tool_messages`` from
``entroly.proxy_transform`` inside the request handler. Replacing that large
module through the GitHub contents API would be fragile, so this installer uses a
feature-flagged, explicit monkey patch instead.

When ``ENTROLY_COMPRESSION_PROXY_MODE=elc`` is set, the old tool-output
compressor is replaced with the Evidence-Locked Compression proxy surface. When
the env var is absent or different, nothing changes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

# This module is itself optional and imported behind an ImportError guard by the
# package root. Activating proxy hardening here keeps base MCP/recovery installs
# independent of HTTP extras while protecting every live proxy construction path.
# Import order is a security and observability contract: transport and access
# hardening are installed before the non-authoritative gateway shadow wrapper.
from . import proxy as _proxy
from . import proxy_transport_safe as _proxy_transport_safe  # noqa: F401
from . import proxy_transport_final as _proxy_transport_final  # noqa: F401
from . import proxy_control_plane_safe as _proxy_control_plane_safe  # noqa: F401
from . import proxy_access_security as _proxy_access_security
from . import proxy_gateway_shadow as _proxy_gateway_shadow
from .compression_proxy import compress_proxy_payload_from_env

_INSTALLED = False
_ORIGINAL: Callable[..., tuple[list[dict[str, Any]], int]] | None = None


def _install_gateway_shadow_factory_seam() -> None:
    """Install the shadow route without replacing the public security factory.

    ``proxy_access_security.create_proxy_app`` is an intentional identity and
    trust contract.  Its implementation resolves ``_ORIGINAL_CREATE_PROXY_APP``
    at call time, so the shadow route belongs behind that private seam rather
    than above the public function.
    """

    current = _proxy_access_security._ORIGINAL_CREATE_PROXY_APP
    original = getattr(
        current,
        "__entroly_gateway_shadow_original__",
        current,
    )

    def shadow_inner_factory(*args: Any, **kwargs: Any):
        app = original(*args, **kwargs)
        _proxy_gateway_shadow._install_route(app)
        return app

    shadow_inner_factory.__entroly_gateway_shadow_original__ = original
    _proxy_access_security._ORIGINAL_CREATE_PROXY_APP = shadow_inner_factory
    _proxy.create_proxy_app = _proxy_access_security.create_proxy_app


_install_gateway_shadow_factory_seam()


def install_live_compression_proxy() -> bool:
    """Install ELC into the live HTTP proxy path when the env flag is enabled.

    Returns True when the patch is active. The operation is idempotent and safe
    to call during package import.
    """
    global _INSTALLED, _ORIGINAL
    if _INSTALLED:
        return True
    if os.environ.get("ENTROLY_COMPRESSION_PROXY_MODE", "").strip().lower() != "elc":
        return False

    try:
        from . import proxy_transform
    except Exception:
        return False

    _ORIGINAL = proxy_transform.compress_tool_messages

    def _elc_compress_tool_messages(
        messages: list[dict[str, Any]],
        *,
        policy: str = "compress",
        excluded_tools: str | set[str] | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        original = _ORIGINAL
        if original is None:
            return messages, 0
        # Honor explicit preserve/off modes used by existing proxy config.
        if str(policy).lower() in {"off", "0", "false", "protect", "exact", "preserve"}:
            return original(messages, policy=policy, excluded_tools=excluded_tools)
        query = _last_user_query(messages)
        result = compress_proxy_payload_from_env(
            {"messages": messages},
            provider="openai",
            query=query,
        )
        if not result.changed:
            return original(messages, policy=policy, excluded_tools=excluded_tools)
        return result.body.get("messages", messages), result.receipt.tokens_saved

    proxy_transform.compress_tool_messages = _elc_compress_tool_messages
    _INSTALLED = True
    return True


def reset_live_compression_proxy() -> None:
    """Restore the original compressor after tests or embedded use."""
    global _INSTALLED, _ORIGINAL
    if not _INSTALLED or _ORIGINAL is None:
        return
    try:
        from . import proxy_transform

        proxy_transform.compress_tool_messages = _ORIGINAL
    finally:
        _ORIGINAL = None
        _INSTALLED = False


def _last_user_query(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = _content_to_text(msg.get("content"))
        if text:
            return text[:500]
    return ""


def _content_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_content_to_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "input"):
            if key in value:
                text = _content_to_text(value[key])
                if text:
                    return text
    return ""


__all__ = ["install_live_compression_proxy", "reset_live_compression_proxy"]
