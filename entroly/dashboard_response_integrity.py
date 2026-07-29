"""Cross-platform response integrity for rejected dashboard control requests.

macOS sends a TCP reset when a server closes a connection while declared request
bytes remain unread. The dashboard correctly rejected missing capabilities, but
its early 403 path left the small JSON body pending, so clients could observe a
connection error instead of the auditable denial response.

This boundary drains only one valid, bounded Content-Length body before an early
rejection. It never parses or dispatches rejected content, never accepts chunked
framing, and never reads beyond the dashboard's existing 64 KiB control limit.
"""

from __future__ import annotations

from typing import Any

from . import dashboard_security as _security

_MAX_DRAIN_BYTES = int(
    getattr(_security._ORIGINAL_HANDLER, "_MAX_CONTROL_BODY_BYTES", 64 * 1024)
)
_ORIGINAL_SAFE_DO_POST = _security.SafeDashboardHandler.do_POST


def _bounded_declared_length(handler: Any) -> int | None:
    """Return one canonical bounded Content-Length, or ``None`` if unsafe."""
    values = handler.headers.get_all("Content-Length", [])
    if not values:
        return 0
    if len(values) != 1:
        return None
    raw = str(values[0]).strip()
    if not raw or len(raw) > 20 or not raw.isascii() or not raw.isdecimal():
        return None
    value = int(raw)
    if value > _MAX_DRAIN_BYTES:
        return None
    return value


def _drain_rejected_body(handler: Any) -> None:
    """Consume a small declared body so the denial response reaches the client."""
    if handler.headers.get("Transfer-Encoding"):
        handler.close_connection = True
        return
    length = _bounded_declared_length(handler)
    if length is None:
        handler.close_connection = True
        return
    remaining = length
    try:
        while remaining:
            chunk = handler.rfile.read(min(remaining, 8192))
            if not chunk:
                handler.close_connection = True
                return
            remaining -= len(chunk)
    except (OSError, TimeoutError):
        handler.close_connection = True


def _reject_after_bounded_drain(handler: Any, status: int, error: str) -> None:
    _drain_rejected_body(handler)
    handler._reject(status, error)


def _integrity_safe_do_post(self) -> None:
    """Preserve security checks while making early denials transport-complete."""
    if len(self.path) > _security._MAX_REQUEST_TARGET_CHARS:
        _reject_after_bounded_drain(self, 414, "request target too long")
        return
    if not self._trusted_request_context():
        _reject_after_bounded_drain(self, 403, "untrusted dashboard request context")
        return
    if not _security._request_token_is_valid(self):
        _reject_after_bounded_drain(self, 403, "invalid control capability")
        return
    if self.headers.get("Transfer-Encoding"):
        self.close_connection = True
        self._reject(400, "chunked control requests are unsupported")
        return
    content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()
    if content_type != "application/json":
        _reject_after_bounded_drain(
            self,
            415,
            "control requests require application/json",
        )
        return
    _ORIGINAL_SAFE_DO_POST(self)


_security.SafeDashboardHandler.do_POST = _integrity_safe_do_post

__all__ = ["_bounded_declared_length", "_drain_rejected_body"]
