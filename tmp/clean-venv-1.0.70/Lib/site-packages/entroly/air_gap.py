"""Air-gap mode: a hard, checkable no-outbound guarantee.

Entroly is local-first by design — ranking, receipts, verification, and
diagnostics never require a remote call, and federation is opt-in and off by
default. Air-gap mode turns that design property into an *enforced, testable*
guarantee for regulated and offline deployments:

    ENTROLY_AIR_GAP=1

installs a process-wide socket guard that refuses any connection to a
non-loopback address, so even a misconfigured feature, a third-party plugin,
or a transitive dependency cannot exfiltrate context. Loopback (127.0.0.0/8,
::1, ``localhost``) stays allowed so the local proxy, dashboard, and MCP stdio
paths keep working.

Scope, stated honestly: the guard wraps ``socket.socket.connect`` /
``connect_ex``, which covers the standard library and every common HTTP client
(urllib, requests, httpx, aiohttp). It is a strong safety net, not a kernel
firewall — code using raw OS syscalls could bypass it. For hostile isolation,
pair it with an OS/network-level control. ``air_gap_status()`` reports whether
the guard is active so it can be surfaced in receipts and ``verify`` output.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Any


class AirGapViolation(RuntimeError):
    """Raised when air-gap mode blocks an outbound network connection."""


_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain", "ip6-localhost"})


def air_gap_enabled() -> bool:
    return os.environ.get("ENTROLY_AIR_GAP", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _host_of(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # A non-literal hostname could resolve anywhere — deny under air-gap.
        return False


# Module-level install state so the guard is idempotent and reversible (tests).
_installed = False
_orig_connect = None
_orig_connect_ex = None


def install_air_gap_guard(*, force: bool = False) -> bool:
    """Install the socket guard when air-gap is enabled. Returns True if active.

    Idempotent: a second call is a no-op. ``force=True`` installs regardless of
    the env var (used by tests).
    """
    global _installed, _orig_connect, _orig_connect_ex
    if _installed:
        return True
    if not (force or air_gap_enabled()):
        return False

    _orig_connect = socket.socket.connect
    _orig_connect_ex = socket.socket.connect_ex

    def _guarded_connect(self: socket.socket, address: Any):
        host = _host_of(address)
        if not _is_loopback(host):
            raise AirGapViolation(
                f"ENTROLY_AIR_GAP is set: refused outbound connection to {host!r}. "
                "Unset ENTROLY_AIR_GAP to allow network access."
            )
        return _orig_connect(self, address)

    def _guarded_connect_ex(self: socket.socket, address: Any):
        host = _host_of(address)
        if not _is_loopback(host):
            raise AirGapViolation(
                f"ENTROLY_AIR_GAP is set: refused outbound connection to {host!r}."
            )
        return _orig_connect_ex(self, address)

    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    _installed = True
    return True


def uninstall_air_gap_guard() -> None:
    """Restore the original socket methods (primarily for tests)."""
    global _installed, _orig_connect, _orig_connect_ex
    if not _installed:
        return
    if _orig_connect is not None:
        socket.socket.connect = _orig_connect  # type: ignore[method-assign]
    if _orig_connect_ex is not None:
        socket.socket.connect_ex = _orig_connect_ex  # type: ignore[method-assign]
    _installed = False
    _orig_connect = None
    _orig_connect_ex = None


def air_gap_status() -> dict[str, Any]:
    """Machine-auditable air-gap state for receipts / verify surfaces."""
    return {
        "requested": air_gap_enabled(),
        "enforced": _installed,
        "allowed": "loopback-only",
        "scope": "socket.connect/connect_ex (stdlib + common HTTP clients)",
    }
