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
from pathlib import Path
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
    version: str = "1.0.13"
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
    last_activity_at: float | None = None
    last_feedback_at: float | None = None
    last_dream_at: float | None = None
    dream_cycles: int = 0
    dream_improvements: int = 0

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
                "dreaming_active": self.autotune_enabled and self.learning_enabled,
                "last_activity_at": self.last_activity_at,
                "last_feedback_at": self.last_feedback_at,
                "last_dream_at": self.last_dream_at,
                "dream_cycles": self.dream_cycles,
                "dream_improvements": self.dream_improvements,
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
        self.state.federation_enabled = os.environ.get("ENTROLY_FEDERATION", "0") == "1"
        self.state.federation_mode = (
            "anonymous" if self.state.federation_enabled else "off"
        )

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
        self._learning_wake = threading.Event()
        self._lock = threading.Lock()
        self._learning_interval_s = 30.0
        self._learning_last_tick_at: float | None = None
        self._learning_last_tick_status: str = "not_started"  # ok|error|disabled|not_started
        self._learning_last_error: str | None = None
        self._learning_last_tick_saw_new_feedback: bool | None = None
        self._learning_last_tick_optimized_profiles: bool | None = None
        self._learning_last_tick_dreamed: bool | None = None
        self._learning_next_interval_s = self._learning_interval_s
        self._learning_last_interval_reason = "startup"
        self._learning_journal_log_attempts = 0
        self._learning_journal_log_failures = 0
        self._learning_last_journal_log_status = "not_started"
        self._learning_last_journal_log_error: str | None = None
        self._learning_wakeups = 0
        self._learning_last_wake_at: float | None = None
        self._learning_last_wake_reason: str | None = None

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

        if self._enable_mcp:
            self._start_mcp_worker()

        # 4. Start file watcher
        self._start_watcher()

        # 5. Start learning loop (DreamingLoop + FeedbackJournal + PRISM)
        self._start_learning_loop()

        self.state.status = "running"

        # Auto-open dashboard in browser
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.state.dashboard.port}")
        except Exception:
            pass

        logger.info(
            f"Entroly daemon running — "
            f"proxy:{self.state.proxy.port if self._enable_proxy else 'off'} "
            f"dashboard:{self.state.dashboard.port} "
            f"mcp:{self.state.mcp.port if self._enable_mcp else 'off'} "
            f"learning:{'ON' if self.state.learning_enabled else 'OFF'}"
        )

    def stop(self):
        """Gracefully stop all workers."""
        self.state.status = "stopping"
        logger.info("Entroly daemon stopping...")
        self._shutdown.set()
        self._learning_wake.set()

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
            old_cwd = os.getcwd()
            try:
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
                logger.info(
                    f"Indexed {repo.indexed_files} files from {repo_path}"
                )
            except Exception as e:
                logger.error(f"Failed to index {repo_path}: {e}")
                self.state.repos.append(
                    RepoState(path=repo_path, watching=False)
                )
            finally:
                os.chdir(old_cwd)

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
                    self._engine,
                    config,
                    start_dashboard=False,
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
                logger.exception("Proxy failed: %s", e)
            finally:
                self.state.proxy.running = False

        t = threading.Thread(target=_run_proxy, daemon=True, name="entroly-proxy")
        t.start()
        self._workers["proxy"] = t

    def _start_mcp_worker(self):
        """Start MCP server on :9379 (SSE transport) in a background thread.

        SSE is the only transport that can run inside the daemon — stdio
        would conflict with the daemon's stdout and prevent multiple IDE
        clients from connecting. The MCP worker reuses the daemon's own
        EntrolyEngine via create_mcp_server() so all on-disk state is shared.
        """

        def _run_mcp():
            try:
                from entroly.server import create_mcp_server

                mcp, _engine = create_mcp_server(engine=self._engine)
                mcp.settings.port = self.state.mcp.port
                # Bind to the daemon's host so external clients on the same
                # machine can reach the SSE endpoint at /sse.
                try:
                    mcp.settings.host = self._host
                except Exception:
                    pass  # older FastMCP may not expose host setting

                self.state.mcp.running = True
                self.state.mcp.started_at = time.time()

                # mcp.run(transport="sse") blocks running its own asyncio
                # loop inside this thread — exactly what we want here.
                try:
                    mcp.run(transport="sse")
                except TypeError:
                    # Older MCP SDK doesn't accept transport kwarg;
                    # fall back to whatever default the SDK provides.
                    mcp.run()
            except Exception as e:
                self.state.mcp.error = str(e)
                logger.exception("MCP server failed: %s", e)
            finally:
                self.state.mcp.running = False

        t = threading.Thread(target=_run_mcp, daemon=True, name="entroly-mcp")
        t.start()
        self._workers["mcp"] = t

    def _start_watcher(self):
        """Start incremental file watcher."""
        try:
            from entroly.auto_index import start_incremental_watcher
            start_incremental_watcher(self._engine)
        except Exception as e:
            logger.warning(f"File watcher failed to start: {e}")

    def _start_learning_loop(self):
        """Start the self-learning background worker.

        Integrates three learning systems:
          1. OnlinePrism (inline) — updates PRISM 5D weights on every
             optimize_context() call via Dirichlet posterior updates.
             Already wired in EntrolyEngine.__init__ — no daemon action needed.

          2. FeedbackJournal + TaskProfileOptimizer — persists (weights, reward)
             episodes to disk, builds per-task-type weight profiles.
             Wired here: journal → optimizer → profiles.

          3. DreamingLoop — during idle periods (>60s), generates synthetic
             queries from journal history, tests counterfactual weight
             perturbations, and keeps only monotonic improvements.
             Runs in a background thread, yields to user queries.

        All computation is local — zero tokens, zero API calls.
        """
        if not self.state.learning_enabled:
            logger.info("Learning loop disabled — skipping")
            return

        try:
            from entroly.autotune import (
                DreamingLoop,
                FeedbackJournal,
                TaskProfileOptimizer,
            )

            checkpoint_dir = os.environ.get(
                "ENTROLY_DIR",
                os.path.join(os.getcwd(), ".entroly"),
            )

            # Initialize the feedback journal (cross-session persistence)
            self._feedback_journal = FeedbackJournal(checkpoint_dir)
            try:
                from entroly.privacy import retention_days

                days = retention_days()
                if days > 0:
                    self._feedback_journal.prune(max_age=days * 24 * 60 * 60)
            except Exception:
                pass

            # Task-conditioned weight profiles
            self._task_profiles = TaskProfileOptimizer(self._feedback_journal)
            self._task_profiles.optimize_all()

            # Wire journal callback into the engine so every outcome signal
            # logs a (weights, reward) episode and resets the idle timer.
            if self._engine and hasattr(self._engine, "set_journal_callback"):
                self._engine.set_journal_callback(self._log_learning_episode)

            # DreamingLoop: autonomous self-play during idle. Pass the live
            # engine reference so per-archetype w_resonance priors actually
            # flow into the Rust 5D PRISM optimizer when DreamingLoop finds
            # an improvement (otherwise Rust cold-starts at 0 every restart
            # and ignores any Python-side resonance prior).
            self._dreaming_loop = DreamingLoop(
                journal=self._feedback_journal,
                max_iterations=10,
                engine=self._engine,
            )

            # Background thread that periodically checks idle → dream
            def _learning_worker():
                interval_s = 30.0
                min_interval_s = 10.0
                max_interval_s = 120.0
                last_episode_count = self._feedback_journal.count()
                self._last_profile_optimize_episode_count = last_episode_count
                self._last_profile_optimize_at = None
                self._profile_optimize_runs = 0
                self._learning_interval_s = interval_s
                self._last_retention_prune_at = None

                while not self._shutdown.is_set():
                    self._learning_interval_s = interval_s
                    woke_early = self._learning_wake.wait(timeout=interval_s)
                    if woke_early:
                        self._learning_wake.clear()
                    if self._shutdown.is_set():
                        break

                    if not self.state.learning_enabled:
                        self._learning_last_tick_at = time.time()
                        self._learning_last_tick_status = "disabled"
                        interval_s = max_interval_s
                        self._learning_next_interval_s = interval_s
                        self._learning_last_interval_reason = "disabled"
                        continue

                    try:
                        self._learning_last_tick_at = time.time()
                        # Periodically prune journal to enforce retention.
                        try:
                            from entroly.privacy import retention_days

                            days = retention_days()
                            if days > 0:
                                now = time.time()
                                last = self._last_retention_prune_at
                                if last is None or (now - float(last)) >= 3600.0:
                                    self._feedback_journal.prune(max_age=days * 24 * 60 * 60)
                                    self._last_retention_prune_at = now
                        except Exception:
                            pass
                        # 0. Detect new feedback since last loop
                        episode_count = self._feedback_journal.count()
                        saw_new_feedback = episode_count > last_episode_count
                        last_episode_count = episode_count

                        # 1. Re-optimize task profiles when feedback arrives (or
                        # periodically as a safety net), instead of on every tick.
                        optimized_profiles = self._maybe_optimize_task_profiles(
                            episode_count=episode_count,
                            now=time.time(),
                        )

                        # 2. If idle, run a dream cycle
                        dreamed = False
                        if (
                            self.state.autotune_enabled
                            and self._dreaming_loop.should_dream()
                        ):
                            result = self._dreaming_loop.run_dream_cycle()
                            dreamed = result.get("status") == "completed"
                            if result.get("status") == "completed":
                                self.state.last_dream_at = time.time()
                                self.state.dream_cycles += 1
                                self.state.dream_improvements += int(
                                    result.get("improvements", 0) or 0
                                )
                                improvements = result.get("improvements", 0)
                                if improvements > 0:
                                    logger.info(
                                        "DreamingLoop: %d improvements in "
                                        "cycle #%d (eff=%.6f)",
                                        improvements,
                                        result.get("dream_id", 0),
                                        result.get("best_efficiency", 0),
                                    )
                                    # Apply improved weights to live engine
                                    self._apply_dreamed_weights()

                        self._learning_last_tick_saw_new_feedback = bool(saw_new_feedback)
                        self._learning_last_tick_optimized_profiles = bool(optimized_profiles)
                        self._learning_last_tick_dreamed = bool(dreamed)
                        self._learning_last_tick_status = "ok"
                        self._learning_last_error = None

                        interval_s, interval_reason = self._next_learning_interval(
                            current_interval_s=interval_s,
                            saw_new_feedback=saw_new_feedback,
                            dreamed=dreamed,
                            optimized_profiles=optimized_profiles,
                            min_interval_s=min_interval_s,
                            max_interval_s=max_interval_s,
                        )
                        self._learning_next_interval_s = interval_s
                        self._learning_last_interval_reason = interval_reason
                    except Exception as e:
                        self._learning_last_tick_status = "error"
                        self._learning_last_error = str(e)
                        logger.debug(f"Learning loop error: {e}")
                        interval_s = min(max_interval_s, interval_s * 1.5)
                        self._learning_next_interval_s = interval_s
                        self._learning_last_interval_reason = "error_backoff"

            t = threading.Thread(
                target=_learning_worker,
                daemon=True,
                name="entroly-learning",
            )
            t.start()
            self._workers["learning"] = t
            logger.info(
                "Learning loop started: journal=%d episodes, "
                "dreaming=idle>60s, profiles=%d task types",
                self._feedback_journal.count(),
                len(self._task_profiles._profiles),
            )

        except Exception as e:
            logger.warning(f"Learning loop failed to start: {e}")
            self._feedback_journal = None
            self._task_profiles = None
            self._dreaming_loop = None

    def _next_learning_interval(
        self,
        *,
        current_interval_s: float,
        saw_new_feedback: bool,
        dreamed: bool,
        optimized_profiles: bool,
        min_interval_s: float = 10.0,
        max_interval_s: float = 120.0,
    ) -> tuple[float, str]:
        """Pick the next learning-loop sleep and explain why it changed."""
        lower_bound = min(min_interval_s, max_interval_s)
        upper_bound = max(min_interval_s, max_interval_s)

        def bounded_interval(target_s: float) -> float:
            return min(upper_bound, max(lower_bound, target_s))

        if saw_new_feedback:
            return bounded_interval(min_interval_s), "new_feedback"
        if dreamed:
            return bounded_interval(30.0), "dreamed"
        if optimized_profiles:
            return bounded_interval(30.0), "optimized_profiles"
        return bounded_interval(current_interval_s * 1.5), "idle_backoff"

    def _maybe_optimize_task_profiles(self, episode_count: int, now: float) -> bool:
        """Optimize task profiles when it is likely to be useful.

        The TaskProfileOptimizer can be relatively expensive; running it every
        learning tick wastes CPU when there is no new feedback. We optimize on:
          - New feedback episodes; or
          - A periodic safety refresh (default ~5 minutes).
        """
        profiles = getattr(self, "_task_profiles", None)
        journal = getattr(self, "_feedback_journal", None)
        if not profiles or not journal:
            return False

        last_opt_ep = getattr(self, "_last_profile_optimize_episode_count", 0)
        last_opt_at = getattr(self, "_last_profile_optimize_at", None)
        refresh_s = 300.0

        should_optimize = episode_count > last_opt_ep
        if not should_optimize and last_opt_at is not None:
            should_optimize = (now - last_opt_at) >= refresh_s
        elif not should_optimize and last_opt_at is None:
            # First tick after startup: only optimize if we have any data.
            should_optimize = episode_count > 0

        if not should_optimize:
            return False

        profiles.optimize_all()
        self._last_profile_optimize_episode_count = episode_count
        self._last_profile_optimize_at = now
        self._profile_optimize_runs = int(getattr(self, "_profile_optimize_runs", 0)) + 1
        return True

    def _apply_dreamed_weights(self):
        """Apply DreamingLoop improvements to the live engine."""
        if not self._engine:
            return
        try:
            from entroly.autotune import load_config
            config = load_config()
            if self._engine._use_rust:
                self._engine._rust.set_weights(
                    config.get("weight_recency", 0.30),
                    config.get("weight_frequency", 0.25),
                    config.get("weight_semantic_sim", 0.25),
                    config.get("weight_entropy", 0.20),
                )
            else:
                self._engine.config.weight_recency = config.get(
                    "weight_recency", 0.30
                )
                self._engine.config.weight_frequency = config.get(
                    "weight_frequency", 0.25
                )
                self._engine.config.weight_semantic_sim = config.get(
                    "weight_semantic_sim", 0.25
                )
                self._engine.config.weight_entropy = config.get(
                    "weight_entropy", 0.20
                )
        except Exception as e:
            logger.debug(f"Failed to apply dreamed weights: {e}")

    # ── Control methods (called by control API) ────────────────────

    def _log_learning_episode(self, **episode: Any) -> None:
        """Persist an outcome episode and mark the daemon as active."""
        self.state.last_feedback_at = time.time()
        self.record_activity()
        self._learning_journal_log_attempts = int(
            getattr(self, "_learning_journal_log_attempts", 0) or 0
        ) + 1
        journal = getattr(self, "_feedback_journal", None)
        if not journal:
            self._learning_last_journal_log_status = "skipped_no_journal"
            self._learning_last_journal_log_error = None
            return

        try:
            journal.log(**episode)
            self._learning_last_journal_log_status = "ok"
            self._learning_last_journal_log_error = None
            self._wake_learning_loop("feedback")
        except Exception as e:
            self._learning_journal_log_failures = int(
                getattr(self, "_learning_journal_log_failures", 0) or 0
            ) + 1
            self._learning_last_journal_log_status = "error"
            self._learning_last_journal_log_error = str(e)
            logger.debug("Learning episode journal log failed: %s", e)

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

    def set_learning_enabled(self, enabled: bool):
        self.state.learning_enabled = bool(enabled)
        self._wake_learning_loop(
            "learning_enabled" if self.state.learning_enabled else "learning_disabled"
        )

    def trigger_autotune(self):
        self.state.autotune_enabled = True
        self._wake_learning_loop("autotune")

    def _wake_learning_loop(self, reason: str):
        self._learning_wakeups = int(getattr(self, "_learning_wakeups", 0) or 0) + 1
        self._learning_last_wake_at = time.time()
        self._learning_last_wake_reason = reason
        self._learning_wake.set()

    def set_federation_enabled(self, enabled: bool) -> None:
        """Reflect startup federation state; live reconfiguration is unsupported."""
        active = os.environ.get("ENTROLY_FEDERATION", "0") == "1"
        self.state.federation_enabled = active
        self.state.federation_mode = "anonymous" if active else "off"
        if enabled != active:
            desired = "with ENTROLY_FEDERATION=1" if enabled else "without ENTROLY_FEDERATION=1"
            raise RuntimeError(
                f"Live federation changes are unsupported; restart the daemon {desired}"
            )

    def get_learning_weights(self) -> dict:
        """Get current PRISM 5D weights + OnlinePrism state.

        Returns weights from three sources (priority order):
          1. OnlinePrism Dirichlet posterior (live, most accurate)
          2. Rust engine's current weights (if no OnlinePrism)
          3. Config defaults (fallback)
        """
        result = {
            "source": "defaults",
            "weights": {
                "recency": 0.30,
                "frequency": 0.25,
                "semantic": 0.25,
                "entropy": 0.20,
            },
        }

        # Try OnlinePrism first (most accurate — live posterior mean)
        if self._engine and hasattr(self._engine, "_online_prism"):
            try:
                prism = self._engine._online_prism
                prism_w = prism.weights()
                prism_stats = prism.stats()
                result = {
                    "source": "online_prism",
                    "weights": {
                        "recency": round(prism_w.get("w_recency", 0.30), 4),
                        "frequency": round(prism_w.get("w_frequency", 0.25), 4),
                        "semantic": round(prism_w.get("w_semantic", 0.25), 4),
                        "entropy": round(prism_w.get("w_entropy", 0.20), 4),
                    },
                    "online_prism": {
                        "n_observations": prism_stats.get("n_observations", 0),
                        "phase": prism_stats.get("phase", "warmup"),
                        "reward_ema": prism_stats.get("reward_ema", 0),
                        "avg_reward": prism_stats.get("avg_reward", 0),
                        "best_reward": prism_stats.get("best_reward", 0),
                        "learning_rate": prism_stats.get("learning_rate", 0),
                    },
                }
                # Add resonance if available (5th PRISM dimension)
                if "w_resonance" in prism_w:
                    result["weights"]["resonance"] = round(
                        prism_w["w_resonance"], 4
                    )
                return result
            except Exception:
                pass

        # Fallback: Rust engine direct
        if self._engine and hasattr(self._engine, "_rust"):
            try:
                rust = self._engine._rust
                result = {
                    "source": "rust_engine",
                    "weights": {
                        "recency": round(getattr(rust, "w_recency", 0.3), 4),
                        "frequency": round(getattr(rust, "w_frequency", 0.25), 4),
                        "semantic": round(getattr(rust, "w_semantic", 0.25), 4),
                        "entropy": round(getattr(rust, "w_entropy", 0.2), 4),
                    },
                }
            except Exception:
                pass

        return result

    def get_learning_stats(self) -> dict:
        """Get comprehensive learning loop stats for the dashboard.

        Aggregates telemetry from all three learning systems:
          - OnlinePrism: live Dirichlet posterior state
          - FeedbackJournal: cross-session episode persistence
          - DreamingLoop: idle-time counterfactual self-play
          - TaskProfileOptimizer: per-task-type weight profiles
        """
        stats: dict = {
            "learning_enabled": self.state.learning_enabled,
            "autotune_enabled": self.state.autotune_enabled,
            "learning_interval_s": round(
                getattr(self, "_learning_interval_s", 30.0), 1
            ),
        }
        stats["learning_loop"] = {
            "status": getattr(self, "_learning_last_tick_status", "not_started"),
            "last_tick_at": getattr(self, "_learning_last_tick_at", None),
            "last_error": getattr(self, "_learning_last_error", None),
            "next_interval_s": round(
                getattr(
                    self,
                    "_learning_next_interval_s",
                    getattr(self, "_learning_interval_s", 30.0),
                ),
                1,
            ),
            "interval_reason": getattr(
                self, "_learning_last_interval_reason", "unknown"
            ),
            "wakeups": int(getattr(self, "_learning_wakeups", 0) or 0),
            "last_wake_at": getattr(self, "_learning_last_wake_at", None),
            "last_wake_reason": getattr(self, "_learning_last_wake_reason", None),
            "last_tick": {
                "saw_new_feedback": getattr(
                    self, "_learning_last_tick_saw_new_feedback", None
                ),
                "optimized_profiles": getattr(
                    self, "_learning_last_tick_optimized_profiles", None
                ),
                "dreamed": getattr(self, "_learning_last_tick_dreamed", None),
            },
            "journal_callback": {
                "attempts": int(
                    getattr(self, "_learning_journal_log_attempts", 0) or 0
                ),
                "failures": int(
                    getattr(self, "_learning_journal_log_failures", 0) or 0
                ),
                "last_status": getattr(
                    self, "_learning_last_journal_log_status", "unknown"
                ),
                "last_error": getattr(
                    self, "_learning_last_journal_log_error", None
                ),
            },
        }
        stats["daemon"] = {
            "last_activity_at": self.state.last_activity_at,
            "last_feedback_at": self.state.last_feedback_at,
            "last_dream_at": self.state.last_dream_at,
            "dream_cycles": self.state.dream_cycles,
            "dream_improvements": self.state.dream_improvements,
        }

        # PRISM weights
        stats["prism"] = self.get_learning_weights()

        # FeedbackJournal
        journal = getattr(self, "_feedback_journal", None)
        if journal:
            try:
                stats["journal"] = journal.stats()
            except Exception as e:
                stats["journal"] = {"status": "error", "error": str(e)}
        else:
            stats["journal"] = {"episodes": 0, "status": "not_initialized"}

        # DreamingLoop
        dreaming = getattr(self, "_dreaming_loop", None)
        if dreaming:
            stats["dreaming"] = dreaming.stats()
        else:
            stats["dreaming"] = {"status": "not_initialized"}

        # TaskProfileOptimizer
        profiles = getattr(self, "_task_profiles", None)
        if profiles:
            profile_data = {}
            for task_type, profile in profiles._profiles.items():
                profile_data[task_type] = {
                    "confidence": profile.get("confidence", 0),
                    "episodes": profile.get("episodes", 0),
                }
            stats["task_profiles"] = profile_data
        else:
            stats["task_profiles"] = {}

        stats["task_profile_optimizer"] = {
            "optimize_runs": int(getattr(self, "_profile_optimize_runs", 0) or 0),
            "last_optimize_at": getattr(self, "_last_profile_optimize_at", None),
            "last_optimized_episode_count": int(
                getattr(self, "_last_profile_optimize_episode_count", 0) or 0
            ),
        }

        return stats

    def reset_learning(self):
        """Reset PRISM weights to defaults and clear journal."""
        if self._engine and hasattr(self._engine, "_online_prism"):
            try:
                self._engine._online_prism.reset_to_prior(
                    {"w_recency": 0.30, "w_frequency": 0.25,
                     "w_semantic": 0.25, "w_entropy": 0.20},
                    prior_strength=20.0,
                )
                logger.info("OnlinePrism reset to default prior")
            except Exception:
                pass
        if self._engine and hasattr(self._engine, "_rust"):
            try:
                self._engine._rust.reset_weights()
            except AttributeError:
                pass
        self.state.learning_enabled = True

    def record_activity(self):
        """Record user activity (resets the DreamingLoop idle timer)."""
        self.state.last_activity_at = time.time()
        dreaming = getattr(self, "_dreaming_loop", None)
        if dreaming:
            dreaming.record_activity()

    def reindex_repo(self, path: str | None = None) -> bool:
        """Re-index a specific repo or all repos."""
        from entroly.auto_index import auto_index

        registered = {
            str(Path(repo.path).resolve()): repo.path
            for repo in self.state.repos
        }
        if path:
            resolved = str(Path(path).resolve())
            if resolved not in registered:
                logger.warning("Rejected reindex request for unregistered repo: %s", path)
                return False
            targets = [registered[resolved]]
        else:
            targets = list(registered.values())

        succeeded = True
        for rpath in targets:
            old_cwd = os.getcwd()
            try:
                os.chdir(rpath)
                result = auto_index(self._engine, force=True)

                # Update state
                for r in self.state.repos:
                    if r.path == rpath:
                        r.indexed_files = result.get("files_indexed", 0)
                        r.total_tokens = result.get("total_tokens", 0)
                        r.last_sync = time.time()
                        break
            except Exception as e:
                logger.error(f"Reindex failed for {rpath}: {e}")
                succeeded = False
            finally:
                os.chdir(old_cwd)
        return succeeded

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
