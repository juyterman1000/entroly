"""Fail-closed network boundary for the unified Entroly daemon.

The daemon shares one live engine between the provider proxy, dashboard, MCP SSE
server, watcher, and learning workers. Exposing that engine on a wildcard, LAN,
or container-network address would let another host reach context-bearing proxy
routes and the full MCP tool surface. The MCP SSE transport currently has no
independent authentication contract, so the unified daemon is deliberately
loopback-only.

Standalone proxy mode has a separate capability-authenticated remote contract in
``proxy_access_security``. Remote MCP use should go through the existing scoped
attach/grant flow or an operator-managed tunnel, not an unauthenticated SSE bind.
"""

from __future__ import annotations

import ipaddress
import math
import re
from typing import Any

from . import daemon as _daemon

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9.-]{1,253}$")


def normalize_loopback_host(host: object) -> str:
    """Return a canonical literal loopback bind or raise before any side effect."""
    if not isinstance(host, str):
        raise ValueError("daemon host must be a loopback address")
    value = host.strip()
    if (
        not value
        or len(value) > 253
        or any(character.isspace() or ord(character) < 32 for character in value)
        or "://" in value
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("daemon host must be a safe loopback address")
    if value.casefold().rstrip(".") == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        if _HOSTNAME_RE.fullmatch(value):
            raise ValueError(
                "daemon host must be a literal loopback address; remote and DNS "
                "binds are not supported by the shared MCP daemon"
            ) from exc
        raise ValueError("daemon host must be a safe loopback address") from exc
    if not address.is_loopback:
        raise ValueError(
            "daemon host must be loopback-only; use standalone proxy mode with "
            "ENTROLY_ALLOW_REMOTE_PROXY=1 and an access token for remote HTTP"
        )
    return address.compressed


def _validated_flag(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _validated_port(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer between 1 and 65535")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer between 1 and 65535")
    if isinstance(value, str):
        if not value.isascii() or not value.isdecimal():
            raise ValueError(f"{name} must be an integer between 1 and 65535")
    try:
        port = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer between 1 and 65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be an integer between 1 and 65535")
    return port


def _validate_service_ports(
    *,
    proxy_port: object,
    dashboard_port: object,
    mcp_port: object,
    enable_proxy: bool,
    enable_mcp: bool,
) -> tuple[int, int, int]:
    proxy = _validated_port(proxy_port, name="proxy_port")
    dashboard = _validated_port(dashboard_port, name="dashboard_port")
    mcp = _validated_port(mcp_port, name="mcp_port")
    active = [("dashboard", dashboard)]
    if enable_proxy:
        active.append(("proxy", proxy))
    if enable_mcp:
        active.append(("mcp", mcp))
    by_port: dict[int, list[str]] = {}
    for service, port in active:
        by_port.setdefault(port, []).append(service)
    collisions = [
        f"{port} ({', '.join(services)})"
        for port, services in by_port.items()
        if len(services) > 1
    ]
    if collisions:
        raise ValueError(
            "enabled daemon services must use distinct ports: " + "; ".join(collisions)
        )
    return proxy, dashboard, mcp


_ORIGINAL_DAEMON = _daemon.EntrolyDaemon
_ORIGINAL_INIT = _ORIGINAL_DAEMON.__init__
_ORIGINAL_START_PROXY = _ORIGINAL_DAEMON._start_proxy_worker
_ORIGINAL_START_MCP = _ORIGINAL_DAEMON._start_mcp_worker


class EntrolyDaemon(_ORIGINAL_DAEMON):
    """Unified supervisor with a validated loopback-only listener contract."""

    def __init__(
        self,
        proxy_port: int = 9377,
        dashboard_port: int = 9378,
        mcp_port: int = 9379,
        host: str = "127.0.0.1",
        enable_proxy: bool = True,
        enable_mcp: bool = True,
        quality: str = "balanced",
        repo_paths: list[str] | None = None,
    ) -> None:
        normalized_host = normalize_loopback_host(host)
        proxy_enabled = _validated_flag(enable_proxy, name="enable_proxy")
        mcp_enabled = _validated_flag(enable_mcp, name="enable_mcp")
        proxy, dashboard, mcp = _validate_service_ports(
            proxy_port=proxy_port,
            dashboard_port=dashboard_port,
            mcp_port=mcp_port,
            enable_proxy=proxy_enabled,
            enable_mcp=mcp_enabled,
        )
        _ORIGINAL_INIT(
            self,
            proxy_port=proxy,
            dashboard_port=dashboard,
            mcp_port=mcp,
            host=normalized_host,
            enable_proxy=proxy_enabled,
            enable_mcp=mcp_enabled,
            quality=quality,
            repo_paths=repo_paths,
        )

    def _validated_worker_host(self) -> str:
        host = normalize_loopback_host(getattr(self, "_host", ""))
        self._host = host
        return host

    def _start_proxy_worker(self) -> Any:
        self._validated_worker_host()
        return _ORIGINAL_START_PROXY(self)

    def _start_mcp_worker(self) -> Any:
        self._validated_worker_host()
        return _ORIGINAL_START_MCP(self)


# Keep historical imports aligned. CLI and dashboard code import the class lazily
# from this module, so replacing the public name protects every normal entry path.
_daemon.EntrolyDaemon = EntrolyDaemon

__all__ = [
    "EntrolyDaemon",
    "normalize_loopback_host",
]
