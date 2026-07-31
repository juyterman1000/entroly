"""Cross-platform HTTP framing integrity for the localhost dashboard.

Early control-request rejection must remain auditable on every supported OS.
Leaving request bytes unread can reset a client while it is still transmitting,
and omitting ``Content-Length`` makes response completion depend on platform-
specific connection-close behavior.

This boundary therefore:

* writes every dashboard response with an exact byte length;
* drains one valid, bounded ``Content-Length`` or chunked request body before an
  early rejection;
* never parses rejected JSON or dispatches rejected content;
* caps chunk count, chunk-line size, trailers, and total discarded bytes;
* fails closed and closes the connection for malformed or oversized framing.
"""

from __future__ import annotations

import re
from typing import Any

from . import dashboard_security as _security

_MAX_DRAIN_BYTES = int(
    getattr(_security._ORIGINAL_HANDLER, "_MAX_CONTROL_BODY_BYTES", 64 * 1024)
)
_MAX_CHUNKS = 256
_MAX_CHUNK_LINE_BYTES = 128
_MAX_TRAILER_BYTES = 8 * 1024
_HEX_SIZE_RE = re.compile(br"^[0-9A-Fa-f]+$")
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


def _read_exact(handler: Any, size: int) -> bytes | None:
    """Read exactly ``size`` bytes without allowing an unbounded allocation."""
    remaining = size
    parts: list[bytes] = []
    try:
        while remaining:
            chunk = handler.rfile.read(min(remaining, 8192))
            if not chunk:
                return None
            parts.append(chunk)
            remaining -= len(chunk)
    except (OSError, TimeoutError):
        return None
    return b"".join(parts)


def _drain_content_length_body(handler: Any) -> bool:
    length = _bounded_declared_length(handler)
    if length is None:
        return False
    return _read_exact(handler, length) is not None


def _bounded_line(handler: Any, maximum: int) -> bytes | None:
    try:
        line = handler.rfile.readline(maximum + 1)
    except (OSError, TimeoutError):
        return None
    if not line or len(line) > maximum or not line.endswith(b"\r\n"):
        return None
    return line


def _drain_chunked_body(handler: Any) -> bool:
    """Discard one syntactically valid, bounded HTTP/1.1 chunked body."""
    total = 0
    for _chunk_index in range(_MAX_CHUNKS):
        line = _bounded_line(handler, _MAX_CHUNK_LINE_BYTES)
        if line is None:
            return False
        size_token = line[:-2].split(b";", 1)[0].strip()
        if not size_token or not _HEX_SIZE_RE.fullmatch(size_token):
            return False
        size = int(size_token, 16)
        if size == 0:
            trailer_total = 0
            while True:
                trailer = _bounded_line(handler, _MAX_CHUNK_LINE_BYTES)
                if trailer is None:
                    return False
                if trailer == b"\r\n":
                    return True
                trailer_total += len(trailer)
                if trailer_total > _MAX_TRAILER_BYTES:
                    return False
        if size > _MAX_DRAIN_BYTES - total:
            return False
        framed = _read_exact(handler, size + 2)
        if framed is None or not framed.endswith(b"\r\n"):
            return False
        total += size
    return False


def _drain_rejected_body(handler: Any) -> bool:
    """Consume only bounded, canonical framing before sending a denial."""
    transfer_values = handler.headers.get_all("Transfer-Encoding", [])
    if transfer_values:
        if len(transfer_values) != 1:
            handler.close_connection = True
            return False
        encodings = [
            token.strip().casefold()
            for token in str(transfer_values[0]).split(",")
            if token.strip()
        ]
        if encodings != ["chunked"]:
            handler.close_connection = True
            return False
        drained = _drain_chunked_body(handler)
    else:
        drained = _drain_content_length_body(handler)
    if not drained:
        handler.close_connection = True
    return drained


def _integrity_respond(
    self,
    status: int,
    content_type: str,
    body: bytes,
    *,
    no_cache: bool = False,
    cors_origin: str | None = None,
) -> None:
    """Write one coherently framed response without relying on socket teardown."""
    payload = bytes(body)
    self.send_response(status)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(payload)))
    if cors_origin:
        self.send_header("Access-Control-Allow-Origin", cors_origin)
        self.send_header("Vary", "Origin")
    self._send_security_headers()
    if self.close_connection:
        self.send_header("Connection", "close")
    self.end_headers()
    if payload:
        try:
            self.wfile.write(payload)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close_connection = True


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
        _reject_after_bounded_drain(self, 400, "chunked control requests are unsupported")
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


_security.SafeDashboardHandler._respond = _integrity_respond
_security.SafeDashboardHandler.do_POST = _integrity_safe_do_post

__all__ = [
    "_bounded_declared_length",
    "_drain_chunked_body",
    "_drain_rejected_body",
]
