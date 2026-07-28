"""Truthful cross-platform launcher for the public ``entroly`` command.

The legacy launcher remains responsible for version-pinned Docker MCP execution.
Proxy mode is deliberately native and loopback-only: Docker bridge publication
would otherwise require every client to understand an additional access header,
which is neither zero-friction nor supported by all OpenAI-compatible clients.
"""

from __future__ import annotations

import os
import sys

from . import _docker_launcher as _legacy


def _proxy_requested(argv: list[str]) -> bool:
    return "--proxy" in argv or os.environ.get("ENTROLY_PROXY") == "1"


def _apply_proxy_cli_overrides(argv: list[str]) -> None:
    """Translate the small documented serve-proxy CLI surface into env config."""
    remaining = list(argv)
    if remaining and remaining[0] == "serve":
        remaining.pop(0)
    remaining = [value for value in remaining if value != "--proxy"]

    index = 0
    while index < len(remaining):
        item = remaining[index]
        if item in {"--port", "--host"}:
            if index + 1 >= len(remaining):
                raise ValueError(f"{item} requires a value")
            value = remaining[index + 1]
            if item == "--port":
                os.environ["ENTROLY_PROXY_PORT"] = value
            else:
                os.environ["ENTROLY_PROXY_HOST"] = value
            index += 2
            continue
        raise ValueError(
            f"unsupported proxy serve argument: {item!r}; use --port or --host"
        )


def _run_native_proxy(argv: list[str]) -> None:
    from .container_proxy import main as proxy_main

    _apply_proxy_cli_overrides(argv)
    os.environ.setdefault("ENTROLY_PROXY_HOST", "127.0.0.1")
    try:
        proxy_main()
    except (RuntimeError, ValueError) as exc:
        print(f"[entroly] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def launch() -> None:
    """Launch local commands, native proxy mode, or version-pinned Docker MCP."""
    argv = sys.argv[1:]
    if argv and argv[0] == "serve" and _proxy_requested(argv):
        _run_native_proxy(argv)
        return
    _legacy.launch()


__all__ = ["launch"]
