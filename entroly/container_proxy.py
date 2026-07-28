"""Dedicated HTTP proxy process for native and container launchers.

Historically ``python -m entroly.server --proxy`` silently started stdio MCP
because ``server.main`` recognized only ``--sse``. This entrypoint owns the
actual proxy lifecycle and deliberately reuses every hardened proxy boundary:
transport limits, sidecar authorization, catch-all validation, and remote-bind
capability authentication.
"""

from __future__ import annotations

import atexit
import logging
import math
import os
import signal
import sys
import threading
from typing import Any

# Import order is part of the security contract. Each layer wraps the factory
# produced by the previous layer; remote access authentication must be outermost.
from . import proxy_transport_safe as _proxy_transport_safe  # noqa: F401
from . import proxy_transport_final as _proxy_transport_final  # noqa: F401
from . import proxy_control_plane_safe as _proxy_control_plane_safe  # noqa: F401
from . import proxy_access_security as _proxy_access_security  # noqa: F401
from .proxy import create_proxy_app
from .proxy_config import ProxyConfig
from .server import EntrolyEngine, _start_background_services

logger = logging.getLogger("entroly.container_proxy")


def _validated_port(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("proxy port must be an integer between 1 and 65535")
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("proxy port must be an integer between 1 and 65535")
    if isinstance(value, str):
        if not value.isascii() or not value.isdecimal():
            raise ValueError("proxy port must be an integer between 1 and 65535")
    try:
        port = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("proxy port must be an integer between 1 and 65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError("proxy port must be an integer between 1 and 65535")
    return port


def _checkpoint_once(engine: Any):
    lock = threading.Lock()
    completed = False

    def _checkpoint(*_args: object) -> None:
        nonlocal completed
        with lock:
            if completed:
                return
            completed = True
        try:
            engine.checkpoint()
            logger.info("Proxy state persisted successfully")
        except Exception as exc:
            logger.warning("Failed to persist proxy state on shutdown: %s", exc)

    return _checkpoint


def main() -> None:
    """Start the hardened HTTP proxy and block until its Uvicorn server exits."""
    try:
        import uvicorn
    except ImportError:
        raise RuntimeError(
            "HTTP proxy dependencies are unavailable. Install entroly[proxy]."
        ) from None

    config = ProxyConfig.from_env()
    config.host = os.environ.get("ENTROLY_PROXY_HOST", config.host)
    config.port = _validated_port(os.environ.get("ENTROLY_PROXY_PORT", config.port))

    engine = EntrolyEngine()
    checkpoint = _checkpoint_once(engine)
    atexit.register(checkpoint)
    try:
        signal.signal(
            signal.SIGTERM,
            lambda _signal, _frame: (checkpoint(), sys.exit(0)),
        )
    except (AttributeError, OSError, ValueError):
        pass

    _start_background_services(engine)
    app = create_proxy_app(
        engine,
        config,
        start_dashboard=bool(config.enable_dashboard),
        start_autotune=False,
    )
    logger.info(
        "Starting hardened Entroly proxy on %s:%d",
        config.host,
        config.port,
    )
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()


__all__ = ["main"]
