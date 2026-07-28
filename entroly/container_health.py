"""Mode-aware Docker health probe for stdio MCP and HTTP proxy containers."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _pid_one_args() -> list[str]:
    try:
        raw = Path("/proc/1/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _proxy_mode() -> bool:
    configured = os.environ.get("ENTROLY_CONTAINER_MODE", "").strip().casefold()
    if configured:
        return configured == "proxy"
    return "--proxy" in _pid_one_args()


def _proxy_healthy() -> bool:
    port = os.environ.get("ENTROLY_PROXY_PORT", "9377")
    if not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535:
        return False
    request = urllib.request.Request(f"http://127.0.0.1:{int(port)}/health")
    token = os.environ.get("ENTROLY_PROXY_ACCESS_TOKEN", "")
    if token:
        request.add_header("X-Entroly-Access-Token", token)
    try:
        with urllib.request.urlopen(request, timeout=3) as response:
            payload = json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return (
        response.status == 200
        and isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("service") == "entroly-proxy"
    )


def _mcp_process_healthy() -> bool:
    try:
        os.kill(1, 0)
        import entroly  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def main() -> int:
    healthy = _proxy_healthy() if _proxy_mode() else _mcp_process_healthy()
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
