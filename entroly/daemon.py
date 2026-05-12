"""
Entroly Daemon — Single Supervisor Process
============================================

One process that owns all state and manages all workers:
  - Proxy server (:9377)
  - Dashboard + Control API (:9378)
  - MCP server (:9379 or stdio)
  - Repo file watcher
  - Local learning loop
  - Optional federation worker

Backward compatible: existing `entroly proxy`, `entroly serve`,
`entroly dashboard` commands continue to work standalone.
The daemon is a NEW `entroly daemon` command that unifies them.

Usage:
    entroly daemon              # start everything
    entroly daemon --no-proxy   # dashboard + MCP only
    entroly daemon stop         # stop via control API
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("entroly.daemon")


# ── Daemon State (single source of truth) ──────────────────────────────


@dataclass
class WorkerState:
    """State of a managed worker."""
    name: str
    running: bool = False
    port: int | None = None
    transport: str | None = None  # for MCP: "sse" or "stdio"
    pid: int | None = None
    started_at: float | None = None
    error: str | None = None


@dataclass
class RepoState:
    """State of a watched repository."""
    path: str
    watching: bool = False
    indexed_files: int = 0
    total_tokens: int = 0
    last_sync: float | None = None


@dataclass
class EntrolyDaemonState:
    """
    The single source of truth for the entire daemon.

    Every controller reads and mutates this state.
    The dashboard polls it via /api/control/status.
    """
    status: str = "stopped"  # stopped | starting | running | stopping
    version: str = "0.18.0"
    started_at: float | None = None

    # Feature flags
    optimization_enabled: bool = True
    bypass_mode: bool = False
    quality_mode: str = "balanced"  # fast | balanced | max

    # Workers
    proxy: WorkerState = field(
        default_factory=lambda: WorkerState("proxy", port=9377)
    )
    dashboard: WorkerState = field(
        default_factory=lambda: WorkerState("dashboard", port=9378)
    )
    mcp: WorkerState = field(
        default_factory=lambda: WorkerState("mcp", port=9379, transport="sse")
    )

    # Repos
    repos: list[RepoState] = field(default_factory=list)

    # Learning
    learning_enabled: bool = True
    autotune_enabled: bool = True

    # Federation
    federation_enabled: bool = False
    federation_mode: str = "off"  # off | preview | anonymous | full

    def to_dict(self) -> dict:
        """JSON-serializable snapshot."""
        d = {
            "status": self.status,
            "version": self.version,
            "started_at": self.started_at,
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0,
            "optimization": {
                "enabled": self.optimization_enabled,
                "bypass": self.bypass_mode,
                "quality": self.quality_mode,
            },
            "proxy": asdict(self.proxy),
            "dashboard": asdict(self.dashboard),
            "mcp": asdict(self.mcp),
            "repos": [asdict(r) for r in self.repos],
            "learning": {
                "local_enabled": self.learning_enabled,
                "autotune_enabled": self.autotune_enabled,
            },
            "federation": {
                "enabled": self.federation_enabled,
                "mode": self.federation_mode,
            },
        }
        return d


# ── Daemon Supervisor ──────────────────────────────────────────────────


class EntrolyDaemon:
    """
    Supervisor that starts/stops all workers and owns the state.

    Design rules:
    1. State is owned here, not in the workers.
    2. Workers are threads (not processes) for shared-memory access.
    3. Fail-closed: if a worker crashes, log it but don't take down others.
    4. Backward compatible: uses the same EntrolyEngine/proxy/dashboard code.
    """

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
    ):
        self.state = EntrolyDaemonState()
        self.state.proxy.port = proxy_port
        self.state.dashboard.port = dashboard_port
        self.state.mcp.port = mcp_port
        self.state.quality_mode = quality

        self._host = host
        self._enable_proxy = enable_proxy
        self._enable_mcp = enable_mcp
        self._repo_paths = repo_paths or [os.getcwd()]

        self._engine: Any = None
        self._proxy_server: Any = None
        self._dashboard_server: Any = None
        self._workers: dict[str, threading.Thread] = {}
        self._proxy_config: Any = None  # live ProxyConfig ref for quality toggle
        self._shutdown = threading.Event()
        self._lock = threading.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self):
        """Start the daemon and all workers."""
        self.state.status = "starting"
        self.state.started_at = time.time()
        logger.info("Entroly daemon starting...")

        # 1. Initialize engine (same as cmd_proxy does)
        try:
            from entroly.server import EntrolyEngine
            self._engine = EntrolyEngine()
        except Exception as e:
            logger.error(f"Failed to create engine: {e}")
            self.state.status = "stopped"
            raise

        # 2. Auto-index repos
        self._index_repos()

        # 3. Start workers
        self._start_dashboard_worker()

        if self._enable_proxy:
            self._start_proxy_worker()

        # 4. Start file watcher
        self._start_watcher()

        self.state.status = "running"

        # Auto-open dashboard in browser
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.state.dashboard.port}")
        except Exception:
            pass

        logger.info(
            f"Entroly daemon running — "
            f"proxy:{self.state.proxy.port} "
            f"dashboard:{self.state.dashboard.port}"
        )

    def stop(self):
        """Gracefully stop all workers."""
        self.state.status = "stopping"
        logger.info("Entroly daemon stopping...")
        self._shutdown.set()

        # Stop proxy
        if self._proxy_server:
            try:
                self._proxy_server.should_exit = True
            except Exception:
                pass
            self.state.proxy.running = False

        # Stop dashboard
        if self._dashboard_server:
            try:
                self._dashboard_server.shutdown()
            except Exception:
                pass
            self.state.dashboard.running = False

        # Wait for threads
        for name, t in self._workers.items():
            t.join(timeout=5)
            if t.is_alive():
                logger.warning(f"Worker {name} did not stop cleanly")

        self.state.status = "stopped"
        logger.info("Entroly daemon stopped")

    def run_forever(self):
        """Block until shutdown signal."""
        # Handle Ctrl+C
        def _sighandler(signum, frame):
            self.stop()

        signal.signal(signal.SIGINT, _sighandler)
        signal.signal(signal.SIGTERM, _sighandler)

        try:
            while not self._shutdown.is_set():
                self._shutdown.wait(timeout=1.0)
        except KeyboardInterrupt:
            self.stop()

    # ── Worker launchers ───────────────────────────────────────────

    def _index_repos(self):
        """Auto-index all configured repos."""
        from entroly.auto_index import auto_index

        for repo_path in self._repo_paths:
            try:
                old_cwd = os.getcwd()
                os.chdir(repo_path)
                result = auto_index(self._engine)
                repo = RepoState(
                    path=repo_path,
                    watching=True,
                    indexed_files=result.get("files_indexed", 0),
                    total_tokens=result.get("total_tokens", 0),
                    last_sync=time.time(),
                )
                self.state.repos.append(repo)
                os.chdir(old_cwd)
                logger.info(
                    f"Indexed {repo.indexed_files} files from {repo_path}"
                )
            except Exception as e:
                logger.error(f"Failed to index {repo_path}: {e}")
                self.state.repos.append(
                    RepoState(path=repo_path, watching=False)
                )

        # Warm up engine subsystems
        try:
            self._engine.optimize_context(
                token_budget=128000, query="project overview"
            )
        except Exception as e:
            logger.warning(f"Warm-up optimize failed: {e}")

    def _start_dashboard_worker(self):
        """Start dashboard + control API on :9378."""
        from entroly.dashboard import start_dashboard

        try:
            # Wire the control API into the dashboard
            _register_control_api(self)
            self._dashboard_server = start_dashboard(
                engine=self._engine,
                port=self.state.dashboard.port,
                daemon=True,
            )
            self.state.dashboard.running = True
            self.state.dashboard.started_at = time.time()
            logger.info(
                f"Dashboard live at http://localhost:{self.state.dashboard.port}"
            )
        except Exception as e:
            self.state.dashboard.error = str(e)
            logger.error(f"Dashboard failed to start: {e}")

    def _start_proxy_worker(self):
        """Start proxy server on :9377 in a background thread."""

        def _run_proxy():
            try:
                from entroly.proxy import create_proxy_app
                from entroly.proxy_config import ProxyConfig, resolve_quality

                config = ProxyConfig.from_env()
                config.port = self.state.proxy.port
                config.host = self._host

                quality_val = resolve_quality(self.state.quality_mode)
                config.quality = quality_val
                config._apply_quality_dial(quality_val)
                self._proxy_config = config  # store ref for live quality toggle

                if self.state.bypass_mode:
                    os.environ["ENTROLY_BYPASS"] = "1"

                app = create_proxy_app(
                    self._engine, config, start_dashboard=False
                )
                self.state.proxy.running = True
                self.state.proxy.started_at = time.time()

                import uvicorn

                uconfig = uvicorn.Config(
                    app,
                    host=self._host,
                    port=self.state.proxy.port,
                    log_level="warning",
                )
                server = uvicorn.Server(uconfig)
                self._proxy_server = server
                server.run()
            except Exception as e:
                self.state.proxy.error = str(e)
                self.state.proxy.running = False
                logger.error(f"Proxy failed: {e}")

        t = threading.Thread(target=_run_proxy, daemon=True, name="entroly-proxy")
        t.start()
        self._workers["proxy"] = t

    def _start_watcher(self):
        """Start incremental file watcher."""
        try:
            from entroly.auto_index import start_incremental_watcher
            start_incremental_watcher(self._engine)
        except Exception as e:
            logger.warning(f"File watcher failed to start: {e}")

    # ── Control methods (called by control API) ────────────────────

    def set_optimization(self, enabled: bool):
        self.state.optimization_enabled = enabled
        if not enabled:
            os.environ["ENTROLY_BYPASS"] = "1"
        else:
            os.environ.pop("ENTROLY_BYPASS", None)
        self.state.bypass_mode = not enabled

    def set_bypass(self, enabled: bool):
        self.state.bypass_mode = enabled
        if enabled:
            os.environ["ENTROLY_BYPASS"] = "1"
        else:
            os.environ.pop("ENTROLY_BYPASS", None)

    def set_quality(self, mode: str):
        if mode not in ("fast", "balanced", "max"):
            raise ValueError(f"Invalid quality mode: {mode}")
        self.state.quality_mode = mode
        # Apply quality dial to the live proxy config so it takes effect
        # immediately on the next request (not just cosmetic state).
        from entroly.proxy_config import resolve_quality
        quality_val = resolve_quality(mode)
        os.environ["ENTROLY_QUALITY"] = str(quality_val)
        # Update the live proxy config in-place
        if self._proxy_config is not None:
            self._proxy_config._apply_quality_dial(quality_val)
            logger.info("Quality dial applied: %s → %.2f", mode, quality_val)

    def get_learning_weights(self) -> dict:
        """Get current PRISM RL weights."""
        if self._engine and hasattr(self._engine, "_rust"):
            rust = self._engine._rust
            return {
                "recency": round(getattr(rust, "w_recency", 0.3), 4),
                "frequency": round(getattr(rust, "w_frequency", 0.25), 4),
                "semantic": round(getattr(rust, "w_semantic", 0.25), 4),
                "entropy": round(getattr(rust, "w_entropy", 0.2), 4),
            }
        return {}

    def reset_learning(self):
        """Reset PRISM weights to defaults."""
        if self._engine and hasattr(self._engine, "_rust"):
            try:
                self._engine._rust.reset_weights()
            except AttributeError:
                pass
        self.state.learning_enabled = True

    def reindex_repo(self, path: str | None = None):
        """Re-index a specific repo or all repos."""
        from entroly.auto_index import auto_index

        targets = [path] if path else [r.path for r in self.state.repos]
        for rpath in targets:
            try:
                old_cwd = os.getcwd()
                os.chdir(rpath)
                result = auto_index(self._engine, force=True)
                os.chdir(old_cwd)

                # Update state
                for r in self.state.repos:
                    if r.path == rpath:
                        r.indexed_files = result.get("files_indexed", 0)
                        r.total_tokens = result.get("total_tokens", 0)
                        r.last_sync = time.time()
                        break
            except Exception as e:
                logger.error(f"Reindex failed for {rpath}: {e}")

    def get_last_context(self) -> dict:
        """Get the last injected context (knapsack explain)."""
        if self._engine and hasattr(self._engine, "_rust"):
            try:
                explain = self._engine._rust.explain_selection()
                return dict(explain)
            except Exception:
                pass
        return {}

    def get_logs(self, n: int = 50) -> list[str]:
        """Get recent log lines."""
        # Read from the logging handler buffer if available
        handler = _get_log_buffer()
        if handler:
            return handler.get_lines(n)
        return []


# ── Log buffer (ring buffer for observability) ─────────────────────────


class _RingLogHandler(logging.Handler):
    """Keeps last N log lines in memory for the dashboard."""

    def __init__(self, capacity: int = 200):
        super().__init__()
        self._lines: list[str] = []
        self._capacity = capacity
        self._lock = threading.Lock()

    def emit(self, record):
        msg = self.format(record)
        with self._lock:
            self._lines.append(msg)
            if len(self._lines) > self._capacity:
                del self._lines[:len(self._lines) - self._capacity]

    def get_lines(self, n: int = 50) -> list[str]:
        with self._lock:
            return list(self._lines[-n:])


_log_buffer: _RingLogHandler | None = None


def _get_log_buffer() -> _RingLogHandler | None:
    return _log_buffer


def _install_log_buffer():
    global _log_buffer
    if _log_buffer is None:
        _log_buffer = _RingLogHandler(200)
        _log_buffer.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
        )
        logging.getLogger("entroly").addHandler(_log_buffer)


# ── Control API registration ──────────────────────────────────────────

# Global reference so the dashboard handler can access the daemon
_daemon: EntrolyDaemon | None = None


def _register_control_api(daemon: EntrolyDaemon):
    """Register the daemon instance for the control API routes."""
    global _daemon
    _daemon = daemon


def get_daemon() -> EntrolyDaemon | None:
    """Get the running daemon instance (used by dashboard handler)."""
    return _daemon
