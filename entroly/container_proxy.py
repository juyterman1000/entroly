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
from .proxy_routing_safety import (
    configure_proxy_routing_safety,
    install_routing_safety,
    validate_routing_environment,
)
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


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean flag")


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
            "HTTP proxy dependencies are unavailable. Reinstall or upgrade entroly."
        ) from None

    config = ProxyConfig.from_env()
    config.host = os.environ.get("ENTROLY_PROXY_HOST", config.host)
    config.port = _validated_port(os.environ.get("ENTROLY_PROXY_PORT", config.port))
    dashboard_enabled = _env_flag("ENTROLY_PROXY_DASHBOARD", default=False)

    # Validate the entire routing deployment contract before constructing the
    # engine, indexing the repository, or starting any background thread.
    routing_safety = validate_routing_environment(
        proxy_config=config,
        host=config.host,
    )
    install_routing_safety()

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

    # Build and validate the complete security boundary before indexing, watcher,
    # or learning threads can create side effects. Remote-bind policy failures are
    # therefore clean startup failures, not partially initialized services.
    app = create_proxy_app(
        engine,
        config,
        start_dashboard=dashboard_enabled,
        start_autotune=False,
    )
    proxy = getattr(getattr(app, "state", None), "proxy", None)
    if proxy is not None:
        configure_proxy_routing_safety(proxy, routing_safety)

    _start_background_services(engine)
    if routing_safety.enabled:
        logger.info(
            "Routing authority %s: providers=%s models=%d origins=%d pricing=%s loopback_only=%s",
            routing_safety.mode.upper(),
            ",".join(sorted(routing_safety.allowed_providers)),
            len(routing_safety.allowed_models),
            len(routing_safety.allowed_origins),
            routing_safety.pricing_catalog_name or "not-required-in-observe",
            routing_safety.loopback_only,
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
