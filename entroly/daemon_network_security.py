"""Fail-closed network and lifecycle boundary for the unified Entroly daemon.

The daemon shares one live engine between the provider proxy, dashboard, MCP SSE
server, watcher, and learning workers. Exposing that engine on a wildcard, LAN,
or container-network address would let another host reach context-bearing proxy
routes and the full MCP tool surface. The MCP SSE transport therefore remains
loopback-only.

Startup and shutdown are treated as auditable transactions. The daemon does not
claim ``running`` until every required listener accepts a connection. It does not
claim ``stopped`` while a proxy or MCP worker is still alive. FastMCP's SSE app is
mounted into an Entroly-owned Uvicorn server so shutdown has an explicit handle;
a graceful stop is followed by a bounded force-exit attempt, then a visible
``stop_failed`` state if a worker remains stuck.
"""

from __future__ import annotations

import ipaddress
import logging
import math
import os
import re
import socket
import threading
import time
from typing import Any

from . import daemon as _daemon

logger = logging.getLogger("entroly.daemon.security")
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9.-]{1,253}$")
_DEFAULT_START_TIMEOUT_S = 60.0
_DEFAULT_STOP_TIMEOUT_S = 5.0


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


def _bounded_timeout(
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value) or not minimum <= value <= maximum:
        return default
    return value


def _startup_timeout() -> float:
    return _bounded_timeout(
        "ENTROLY_DAEMON_START_TIMEOUT",
        _DEFAULT_START_TIMEOUT_S,
        minimum=1.0,
        maximum=300.0,
    )


def _stop_timeout() -> float:
    return _bounded_timeout(
        "ENTROLY_DAEMON_STOP_TIMEOUT",
        _DEFAULT_STOP_TIMEOUT_S,
        minimum=0.5,
        maximum=60.0,
    )


def _socket_family(host: str) -> socket.AddressFamily:
    return socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET


def _assert_bind_available(host: str, port: int, *, service: str) -> None:
    family = _socket_family(host)
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((host, port))
    except OSError as exc:
        raise RuntimeError(
            f"{service} cannot start because {host}:{port} is unavailable"
        ) from exc
    finally:
        probe.close()


def _listener_accepts(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


_ORIGINAL_DAEMON = _daemon.EntrolyDaemon
_ORIGINAL_INIT = _ORIGINAL_DAEMON.__init__
_ORIGINAL_START_PROXY = _ORIGINAL_DAEMON._start_proxy_worker


class EntrolyDaemon(_ORIGINAL_DAEMON):
    """Unified supervisor with validated listeners and truthful lifecycle state."""

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
        self._startup_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._mcp_server: Any = None

    def _validated_worker_host(self) -> str:
        host = normalize_loopback_host(getattr(self, "_host", ""))
        self._host = host
        return host

    def _start_proxy_worker(self) -> Any:
        self._validated_worker_host()
        return _ORIGINAL_START_PROXY(self)

    def _start_mcp_worker(self) -> None:
        """Run FastMCP's SSE app in an Entroly-owned Uvicorn server thread."""
        host = self._validated_worker_host()
        port = int(self.state.mcp.port)

        def _run_mcp() -> None:
            try:
                from entroly.server import create_mcp_server
                import uvicorn

                mcp, _engine = create_mcp_server(engine=self._engine)
                try:
                    mcp.settings.host = host
                    mcp.settings.port = port
                except Exception:
                    pass
                if not hasattr(mcp, "sse_app"):
                    raise RuntimeError(
                        "installed MCP SDK does not expose FastMCP.sse_app()"
                    )
                app = mcp.sse_app()
                config = uvicorn.Config(
                    app,
                    host=host,
                    port=port,
                    log_level="warning",
                    timeout_graceful_shutdown=_stop_timeout(),
                )
                server = uvicorn.Server(config)
                self._mcp_server = server
                self.state.mcp.started_at = time.time()
                self.state.mcp.running = True
                server.run()
            except BaseException as exc:
                self.state.mcp.error = str(exc)[:1000]
                logger.exception("MCP server failed: %s", exc)
            finally:
                self.state.mcp.running = False
                self._mcp_server = None

        worker = threading.Thread(
            target=_run_mcp,
            daemon=True,
            name="entroly-mcp",
        )
        worker.start()
        self._workers["mcp"] = worker

    def _preflight_listeners(self) -> None:
        _assert_bind_available(
            "127.0.0.1",
            int(self.state.dashboard.port),
            service="dashboard",
        )
        if self._enable_proxy:
            _assert_bind_available(
                self._validated_worker_host(),
                int(self.state.proxy.port),
                service="proxy",
            )
        if self._enable_mcp:
            _assert_bind_available(
                self._validated_worker_host(),
                int(self.state.mcp.port),
                service="mcp",
            )

    def _wait_listener(self, service: str, *, host: str, port: int) -> None:
        timeout = _startup_timeout()
        deadline = time.monotonic() + timeout
        state = getattr(self.state, service)
        while time.monotonic() < deadline:
            error = getattr(state, "error", None)
            if error:
                raise RuntimeError(f"{service} failed during startup: {error}")
            if service != "dashboard":
                worker = self._workers.get(service)
                if worker is None:
                    raise RuntimeError(f"{service} worker was not registered")
                if not worker.is_alive():
                    raise RuntimeError(f"{service} worker exited before readiness")
            if getattr(state, "running", False) and _listener_accepts(host, port):
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"{service} did not accept connections on {host}:{port} "
            f"within {timeout:.1f}s"
        )

    def _reset_worker_state_for_start(self) -> None:
        for state in (self.state.proxy, self.state.dashboard, self.state.mcp):
            state.running = False
            state.started_at = None
            state.error = None

    def _request_server_exit(self, server: Any) -> None:
        if server is None:
            return
        try:
            server.should_exit = True
        except Exception:
            pass

    def _rollback_startup(self) -> None:
        self._shutdown.set()
        self._learning_wake.set()
        self._request_server_exit(self._proxy_server)
        self._request_server_exit(self._mcp_server)
        if self._dashboard_server is not None:
            try:
                self._dashboard_server.shutdown()
            except Exception:
                pass
        timeout = min(2.0, _stop_timeout())
        for worker in list(self._workers.values()):
            if worker is threading.current_thread():
                continue
            worker.join(timeout=timeout)
        self.state.proxy.running = False
        self.state.dashboard.running = False
        self.state.mcp.running = False
        self.state.status = "stopped"
        self.state.started_at = None

    def start(self) -> None:
        """Start all required services and claim success only after readiness."""
        if not self._startup_lock.acquire(blocking=False):
            raise RuntimeError("daemon startup is already in progress")
        try:
            if self.state.status != "stopped":
                raise RuntimeError(
                    f"daemon cannot start from state {self.state.status!r}"
                )
            self._preflight_listeners()
            self._shutdown.clear()
            self._learning_wake.clear()
            self._reset_worker_state_for_start()
            self.state.status = "starting"
            self.state.started_at = time.time()
            logger.info("Entroly daemon starting with readiness gates...")

            try:
                from entroly.server import EntrolyEngine

                self._engine = EntrolyEngine()
                self._index_repos()

                self._start_dashboard_worker()
                self._wait_listener(
                    "dashboard",
                    host="127.0.0.1",
                    port=int(self.state.dashboard.port),
                )

                if self._enable_proxy:
                    self._start_proxy_worker()
                    self._wait_listener(
                        "proxy",
                        host=self._validated_worker_host(),
                        port=int(self.state.proxy.port),
                    )

                if self._enable_mcp:
                    self._start_mcp_worker()
                    self._wait_listener(
                        "mcp",
                        host=self._validated_worker_host(),
                        port=int(self.state.mcp.port),
                    )

                self._start_watcher()
                self._start_learning_loop()
                self.state.status = "running"
            except BaseException:
                self._rollback_startup()
                raise

            try:
                import webbrowser

                webbrowser.open(f"http://localhost:{self.state.dashboard.port}")
            except Exception:
                pass
            logger.info(
                "Entroly daemon running — proxy:%s dashboard:%s mcp:%s learning:%s",
                self.state.proxy.port if self._enable_proxy else "off",
                self.state.dashboard.port,
                self.state.mcp.port if self._enable_mcp else "off",
                "ON" if self.state.learning_enabled else "OFF",
            )
        finally:
            self._startup_lock.release()

    def stop(self) -> None:
        """Stop owned servers and report failure while any worker remains alive."""
        if not self._stop_lock.acquire(blocking=False):
            raise RuntimeError("daemon shutdown is already in progress")
        try:
            if self.state.status == "stopped":
                return
            self.state.status = "stopping"
            self._shutdown.set()
            self._learning_wake.set()
            self._request_server_exit(self._proxy_server)
            self._request_server_exit(self._mcp_server)
            if self._dashboard_server is not None:
                try:
                    self._dashboard_server.shutdown()
                except Exception as exc:
                    self.state.dashboard.error = str(exc)[:1000]

            timeout = _stop_timeout()
            alive: list[str] = []
            for name, worker in list(self._workers.items()):
                if worker is threading.current_thread():
                    continue
                worker.join(timeout=timeout)
                if worker.is_alive():
                    server = (
                        self._proxy_server if name == "proxy" else self._mcp_server
                    )
                    if server is not None:
                        try:
                            server.force_exit = True
                            server.should_exit = True
                        except Exception:
                            pass
                    worker.join(timeout=min(2.0, timeout))
                if worker.is_alive():
                    alive.append(name)
                    worker_state = getattr(self.state, name, None)
                    if worker_state is not None:
                        worker_state.error = "worker did not stop within timeout"

            self.state.proxy.running = False
            self.state.dashboard.running = False
            self.state.mcp.running = False
            if alive:
                self.state.status = "stop_failed"
                raise RuntimeError(
                    "daemon shutdown incomplete; workers still alive: "
                    + ", ".join(sorted(alive))
                )
            self.state.status = "stopped"
            self.state.started_at = None
            logger.info("Entroly daemon stopped")
        finally:
            self._stop_lock.release()


# Keep historical imports aligned. CLI and dashboard code import the class lazily
# from this module, so replacing the public name protects every normal entry path.
_daemon.EntrolyDaemon = EntrolyDaemon

__all__ = [
    "EntrolyDaemon",
    "normalize_loopback_host",
]
