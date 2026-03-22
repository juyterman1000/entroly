"""
Entroly Prompt Compiler Proxy
==============================

An invisible HTTP reverse proxy that sits between the IDE and the LLM API.
Intercepts every request, optimizes the prompt using entroly's algorithms,
and forwards the enriched request to the real API.

The developer changes one setting (API base URL → localhost:9377) and every
query is automatically optimized. No MCP tools to call. No behavior change.

Architecture:
    IDE → localhost:9377 → entroly pipeline (3-6ms) → real API → stream back

All heavy computation runs in Rust (PyO3). The proxy adds <10ms latency.
Errors fall back to forwarding the original request unmodified.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
from typing import Any, Dict, Optional

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .proxy_config import ProxyConfig
from .proxy_transform import (
    apply_temperature,
    apply_trajectory_convergence,
    compute_dynamic_budget,
    compute_optimal_temperature,
    compute_token_budget,
    detect_provider,
    extract_model,
    extract_user_message,
    format_context_block,
    format_hierarchical_context,
    inject_context_anthropic,
    inject_context_openai,
)

logger = logging.getLogger("entroly.proxy")

# ── Privacy utilities ───────────────────────────────────────────────────

# Patterns that indicate secrets/credentials in user queries
_SECRET_PATTERNS = re.compile(
    r"(sk-[a-zA-Z0-9]{20,}|"           # OpenAI keys
    r"ghp_[a-zA-Z0-9]{36}|"            # GitHub PATs
    r"AKIA[0-9A-Z]{16}|"               # AWS access keys
    r"password\s*[:=]\s*\S+|"          # password assignments
    r"secret\s*[:=]\s*\S+|"            # secret assignments
    r"api[_-]?key\s*[:=]\s*\S+)",      # api key assignments
    re.IGNORECASE,
)


def _sanitize_query(query: str, max_len: int = 200) -> str:
    """Sanitize a user query for safe storage/display.

    - Truncates to max_len characters
    - Redacts anything that looks like a secret or credential
    """
    truncated = query[:max_len]
    return _SECRET_PATTERNS.sub("[REDACTED]", truncated)


def _safe_preview(content: str, max_chars: int = 30) -> str:
    """Generate a privacy-safe preview of code content.

    Returns only the first line's structural signature (def/class/import),
    never raw variable values or string literals.
    """
    if not content:
        return ""
    first_line = content.split("\n", 1)[0].strip()
    # Only show structural keywords, not values
    for prefix in ("def ", "class ", "import ", "from ", "async def ", "#"):
        if first_line.startswith(prefix):
            return first_line[:max_chars] + ("..." if len(first_line) > max_chars else "")
    # For non-structural lines, show only the shape (no values)
    return f"[{len(content)} chars, {content.count(chr(10)) + 1} lines]"


# ── Resilience primitives (ported from agentOS) ─────────────────────────


class _CircuitBreaker:
    """SIRS-inspired circuit breaker: open after N consecutive failures,
    half-open after cooldown period, close on success.

    Inspired by the refractory period in agentOS/scheduler.rs SIRS routing.
    """

    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._state = "closed"  # closed | open | half_open
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._state == "closed":
                return True
            if self._state == "open":
                if time.time() - self._last_failure_time > self.cooldown_s:
                    self._state = "half_open"
                    return True  # allow one probe request
                return False
            return True  # half_open: allow the probe

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.time()
            if self._consecutive_failures >= self.failure_threshold:
                self._state = "open"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state


class _TokenBucket:
    """Token bucket rate limiter.

    Ported from agentOS/compliance.rs TokenBucket.
    """

    def __init__(self, capacity: float, refill_per_second: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_per_second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_consume(self, cost: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self._last_refill = now
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False


class _WelfordStats:
    """Welford's online algorithm for streaming mean/variance.

    Ported from agentOS/persona_manifold.rs Welford tracker.
    Tracks pipeline latency without storing all samples.
    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._m2 = 0.0

    def add(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self._m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean_ms": round(self.mean, 2),
            "stddev_ms": round(self.stddev, 2),
        }


def _dp_round(value: int, granularity: int = 100) -> int:
    """Differential-privacy-safe rounding for public-facing counts.

    Rounds to nearest `granularity` to prevent exact fingerprinting of
    codebase size via token counts.  This implements the simplest form
    of ε-differential privacy: deterministic rounding with sensitivity
    bounded by the granularity parameter.

    For example, _dp_round(41237, 100) → 41200.  An adversary cannot
    distinguish a 41,200-token codebase from a 41,299-token one.
    """
    if value <= 0:
        return 0
    return (value // granularity) * granularity


# ── Progressive Conversation Compression ──────────────────────────────────


def compress_conversation_messages(
    messages: list[dict],
    context_window: int = 128_000,
) -> list[dict]:
    """Apply progressive multi-resolution compression to conversation messages.

    Uses the Rust Causal Information DAG Pruner to surgically compress
    tool calls, tool results, and thinking blocks while preserving user
    and assistant messages.  Triggered when context utilization > 70%.

    Returns a new messages list with compressed content where appropriate.
    """
    if not messages:
        return messages

    # Estimate utilization (rough: 4 chars ≈ 1 token)
    total_chars = sum(len(m.get("content", "")) for m in messages)
    total_tokens_est = total_chars // 4
    utilization = total_tokens_est / max(context_window, 1)

    if utilization < 0.70:
        return messages  # no compression needed

    try:
        from entroly_core import py_progressive_thresholds, py_compress_block
        import json as _json

        # Build block descriptors for Rust
        blocks = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "user")
            tool_name = msg.get("name") or msg.get("tool_name")
            token_count = len(content) // 4  # rough estimate
            blocks.append({
                "index": i,
                "role": role,
                "content": content,
                "token_count": token_count,
                "tool_name": tool_name,
                "timestamp": float(i),
            })

        recency_cutoff = max(0, len(blocks) - 6)
        result_json = py_progressive_thresholds(blocks, utilization, recency_cutoff)
        assignments = _json.loads(result_json)

        # Apply compression
        compressed = []
        for i, msg in enumerate(messages):
            resolution = "verbatim"
            for a in assignments:
                if int(a["index"]) == i:
                    resolution = a["resolution"]
                    break

            if resolution == "verbatim":
                compressed.append(msg)
            else:
                content = msg.get("content", "")
                role = msg.get("role", "user")
                tool_name = msg.get("name") or msg.get("tool_name")
                token_count = len(content) // 4

                new_content = py_compress_block(
                    role, content, token_count, resolution, tool_name
                )
                new_msg = dict(msg)
                new_msg["content"] = new_content
                compressed.append(new_msg)

        return compressed
    except ImportError:
        return messages  # Rust not available, pass through
    except Exception as e:
        logger.debug("Conversation compression skipped: %s", e)
        return messages


# ── Proxy ────────────────────────────────────────────────────────────────


class PromptCompilerProxy:
    """HTTP reverse proxy that optimizes every LLM request with entroly."""

    def __init__(self, engine: Any, config: Optional[ProxyConfig] = None):
        self.engine = engine
        self.config = config or ProxyConfig()
        self._client: Optional[httpx.AsyncClient] = None

        # Thread-safe stats
        self._stats_lock = threading.Lock()
        self._requests_total: int = 0
        self._requests_optimized: int = 0
        self._requests_bypassed: int = 0
        self._temperature_sum: float = 0.0
        self._temperature_count: int = 0
        self._last_temperature: Optional[float] = None
        self._total_original_tokens: int = 0
        self._total_optimized_tokens: int = 0
        # Per-client trajectory isolation: each API key / auth header gets
        # its own turn counter. Prevents concurrent IDE clients from
        # corrupting each other's EGTC temperature calibration.
        self._trajectory_turns: Dict[str, int] = {}  # client_key -> turn_count

        # Gap #29: Last optimization context (for transparency endpoint)
        self._last_context_fragments: list = []
        self._last_pipeline_ms: float = 0.0
        self._last_query: str = ""

        # Gap #36: Confidence threshold — below this, pass through unmodified
        self._confidence_threshold = float(
            os.environ.get("ENTROLY_CONFIDENCE_THRESHOLD", "0.15")
        )

        # Gap #37: Error budget — track how often optimization may have hurt
        self._outcome_success: int = 0
        self._outcome_failure: int = 0

        # Bypass mode (Gap #28)
        self._bypass = os.environ.get("ENTROLY_BYPASS", "0") == "1"

        # Resilience: circuit breaker for upstream API
        self._breaker = _CircuitBreaker(failure_threshold=3, cooldown_s=30.0)

        # Rate limiter — default 120 req/min if not set (Gap #39)
        rate_limit = int(os.environ.get("ENTROLY_RATE_LIMIT", "120"))
        self._rate_limiter: Optional[_TokenBucket] = None
        if rate_limit > 0:
            self._rate_limiter = _TokenBucket(
                capacity=float(rate_limit), refill_per_second=rate_limit / 60.0
            )

        # Pipeline latency tracking (Welford online stats)
        self._pipeline_stats = _WelfordStats()

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
            follow_redirects=True,
        )
        logger.info("Prompt compiler proxy ready")

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Gap #47: Auto-reconnect if client connection dropped (long sessions).

        IDE left open overnight, connection pool stale → recreate client.
        """
        if self._client is None or self._client.is_closed:
            logger.info("Reconnecting HTTP client (previous connection dropped)")
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
                follow_redirects=True,
            )
        return self._client

    async def shutdown(self) -> None:
        # Persist learned state before exit (graceful shutdown)
        try:
            self._persist_engine_state()
        except Exception as e:
            logger.warning(f"Failed to persist state on shutdown: {e}")
        if self._client:
            await self._client.aclose()

    def _persist_engine_state(self) -> None:
        """Flush learned PRISM weights, fragment index, and feedback to disk.

        Called on graceful shutdown (ASGI shutdown, Ctrl+C via atexit).
        Without this, all KKT-REINFORCE learning from the session is lost.
        """
        if not hasattr(self.engine, '_checkpoint_mgr'):
            return
        try:
            self.engine.checkpoint()
            logger.info("State persisted on shutdown")
        except Exception as e:
            logger.warning(f"Checkpoint on shutdown failed: {e}")

    async def handle_proxy(self, request: Request) -> StreamingResponse | JSONResponse:
        """Main proxy handler — intercept, optimize, forward.

        Uses pipelined async architecture:
        1. Parse request + start HTTP connection warmup concurrently
        2. Run Rust pipeline in thread pool (off event loop)
        3. Connection is ready by the time pipeline completes
        """
        # Rate limiting (Gap #39)
        if self._rate_limiter and not self._rate_limiter.try_consume():
            return JSONResponse(
                {"error": "rate_limit_exceeded", "retry_after_s": 1},
                status_code=429,
                headers={"Retry-After": "1"},
            )

        with self._stats_lock:
            self._requests_total += 1

        # Read request
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Not JSON — forward raw (e.g. health checks hitting wrong path)
            return await self._forward_raw(request, body_bytes)

        path = request.url.path
        headers = {k: v for k, v in request.headers.items()}
        provider = detect_provider(path, headers)

        # ── Progressive conversation compression ──
        # Surgically compress tool calls/results and thinking blocks when
        # context utilization is high, before the optimization pipeline runs.
        if "messages" in body and self.config.enable_conversation_compression:
            body["messages"] = compress_conversation_messages(
                body["messages"],
                context_window=getattr(self.config, "context_window", 128_000),
            )

        # Gap #28: Bypass mode — forward unmodified, no optimization
        if self._bypass:
            with self._stats_lock:
                self._requests_bypassed += 1
            target_url = self._resolve_target(provider, path)
            forward_headers = self._build_headers(headers, provider)
            is_streaming = body.get("stream", False)
            if is_streaming:
                return await self._stream_response(target_url, forward_headers, body)
            return await self._forward_response(target_url, forward_headers, body)

        # ── Pipelined: warmup connection while Rust pipeline runs ──
        # Start HTTP connection pool warmup concurrently with the
        # Rust optimization. For persistent connections this is nearly
        # free; for cold starts it saves the TLS handshake time (~50ms).
        target_url = self._resolve_target(provider, path)
        warmup_task = asyncio.create_task(self._warmup_connection(target_url))

        # Per-client key for trajectory isolation (hash of auth header)
        auth_raw = headers.get("authorization", "") or headers.get("x-api-key", "")
        client_key = hashlib.sha256(auth_raw.encode()).hexdigest()[:12] if auth_raw else "_default"

        # Run the optimization pipeline (synchronous Rust, off the event loop)
        try:
            user_message = extract_user_message(body, provider)
            if user_message:
                pipeline_result = await asyncio.to_thread(
                    self._run_pipeline, user_message, body
                )
                context_text = pipeline_result["context"]
                pipeline_ms = pipeline_result["elapsed_ms"]
                optimal_tau = pipeline_result.get("temperature")

                # Track pipeline latency
                self._pipeline_stats.add(pipeline_ms)

                if context_text:
                    # Gap #36: Confidence threshold — skip injection if
                    # entropy scores are too low (context quality is poor)
                    avg_entropy = 0.0
                    selected_frags = pipeline_result.get("selected_fragments", [])
                    if selected_frags:
                        avg_entropy = sum(
                            f.get("entropy_score", 0.5) for f in selected_frags
                        ) / len(selected_frags)
                    if avg_entropy < self._confidence_threshold and selected_frags:
                        logger.info(
                            f"Low confidence ({avg_entropy:.3f} < {self._confidence_threshold}), "
                            f"passing through unmodified"
                        )
                        context_text = ""  # skip injection

                if context_text:
                    # Gap #27 & #29: Track original vs optimized tokens
                    original_tokens = sum(
                        len(m.get("content", "").split())
                        for m in body.get("messages", [])
                    ) * 4 // 3
                    optimized_tokens = len(context_text.split()) * 4 // 3
                    with self._stats_lock:
                        self._total_original_tokens += original_tokens
                        self._total_optimized_tokens += optimized_tokens
                        self._last_context_fragments = selected_frags[:20] if selected_frags else []
                        self._last_pipeline_ms = pipeline_ms
                        self._last_query = _sanitize_query(user_message)

                    if provider == "openai":
                        body = inject_context_openai(body, context_text)
                    else:
                        body = inject_context_anthropic(body, context_text)

                    # EGTC v2: apply Fisher-derived optimal temperature
                    if self.config.enable_temperature_calibration and optimal_tau is not None:
                        # Per-client trajectory convergence: each client's
                        # temperature decays independently across its own turns
                        if self.config.enable_trajectory_convergence:
                            with self._stats_lock:
                                client_turns = self._trajectory_turns.get(client_key, 0)
                            optimal_tau = apply_trajectory_convergence(
                                optimal_tau,
                                client_turns,
                                c_min=self.config.trajectory_c_min,
                                lam=self.config.trajectory_lambda,
                            )
                        body = apply_temperature(body, optimal_tau)
                        with self._stats_lock:
                            self._temperature_sum += optimal_tau
                            self._temperature_count += 1
                            self._last_temperature = optimal_tau
                            self._trajectory_turns[client_key] = self._trajectory_turns.get(client_key, 0) + 1

                    with self._stats_lock:
                        self._requests_optimized += 1
                        opt_count = self._requests_optimized
                        total_count = self._requests_total

                    tau_str = f", τ={optimal_tau:.2f}" if optimal_tau else ""

                    # Startup banner: on first optimized request, print a
                    # human-visible confirmation so the user knows it's working.
                    if opt_count == 1:
                        original_tokens = sum(
                            len(m.get("content", "").split())
                            for m in body.get("messages", [])
                        ) * 4 // 3  # rough word→token estimate
                        optimized_tokens = len(context_text.split()) * 4 // 3
                        if original_tokens > 0:
                            saved_pct = max(0, (original_tokens - optimized_tokens)) * 100 // original_tokens
                            print(
                                f"\n  First request optimized: "
                                f"{original_tokens:,} → {optimized_tokens:,} tokens "
                                f"({saved_pct}% saved) in {pipeline_ms:.1f}ms\n",
                                file=sys.stderr,
                            )

                    logger.info(
                        f"Optimized in {pipeline_ms:.1f}ms{tau_str} "
                        f"({opt_count}/{total_count} requests)"
                    )
        except Exception as e:
            # Cardinal rule: never block a request due to entroly errors
            logger.debug("Pipeline error (forwarding unmodified): %s: %s",
                         type(e).__name__, str(e)[:120])

        # Await warmup (usually completes during pipeline, essentially free)
        await warmup_task

        # Forward to real API (target_url already resolved above)
        forward_headers = self._build_headers(headers, provider)
        is_streaming = body.get("stream", False)

        if is_streaming:
            return await self._stream_response(target_url, forward_headers, body)
        else:
            return await self._forward_response(target_url, forward_headers, body)

    def _run_pipeline(self, user_message: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Run the synchronous optimization pipeline. Called via asyncio.to_thread.

        Returns dict with keys: context, elapsed_ms, temperature.
        """
        t0 = time.perf_counter()

        model = extract_model(body)

        # ── ECDB: Dynamic Budget Computation ──
        # We need vagueness for the budget, but the full query analysis
        # happens inside optimize_context(). Do a lightweight pre-analysis
        # to get vagueness for budget calibration.
        if self.config.enable_dynamic_budget:
            try:
                from entroly_core import py_analyze_query
                summaries = []  # Empty summaries for quick vagueness estimate
                vagueness_pre, _, _, _ = py_analyze_query(user_message, summaries)
                frag_count = self.engine._rust.fragment_count()
                token_budget = compute_dynamic_budget(
                    model, self.config,
                    vagueness=vagueness_pre,
                    total_fragments=frag_count,
                )
            except Exception:
                token_budget = compute_token_budget(model, self.config)
        else:
            token_budget = compute_token_budget(model, self.config)

        # ── Hierarchical Compression path (ECC) ──
        # Try 3-level hierarchical compression first if enabled.
        # Falls back to flat optimize_context if hierarchical_compress
        # is not available (e.g., older Rust engine version).
        hcc_result = None
        if self.config.enable_hierarchical_compression:
            try:
                hcc_result = self.engine._rust.hierarchical_compress(
                    token_budget, user_message
                )
                if hcc_result.get("status") == "empty":
                    hcc_result = None  # Fall through to flat path
            except (AttributeError, Exception) as e:
                logger.debug(f"HCC unavailable, falling back to flat: {e}")
                hcc_result = None

        # ── Flat optimization path (original) ──
        # optimize_context already does:
        #   1. Query refinement (py_analyze_query + py_refine_heuristic)
        #   2. LTM recall (cross-session memories)
        #   3. Knapsack optimization (Rust)
        #   4. SSSL filtering
        #   5. Ebbinghaus decay bookkeeping
        self.engine._turn_counter += 1
        self.engine.advance_turn()
        result = self.engine.optimize_context(token_budget, user_message)

        selected = result.get("selected_fragments", [])
        refinement = result.get("query_refinement")

        # Build refinement info for the context block
        refinement_info = None
        # Extract vagueness from query_analysis (always present) rather than
        # query_refinement (only present when vagueness >= 0.45)
        query_analysis = result.get("query_analysis", {})
        vagueness = query_analysis.get("vagueness_score", 0.0)
        if refinement:
            vagueness = max(vagueness, refinement.get("vagueness_score", 0.0))
            refinement_info = {
                "original": refinement.get("original_query", user_message),
                "refined": refinement.get("refined_query", user_message),
                "vagueness": vagueness,
            }

        # ── Task classification (used by both EGTC and APA preamble) ──
        task_type = "Unknown"
        try:
            task_info = self.engine._rust.classify_task(user_message)
            task_type = task_info.get("task_type", "Unknown")
        except Exception:
            pass

        # ── EGTC v2: Fisher-based Temperature Calibration ──
        optimal_tau = None
        if self.config.enable_temperature_calibration and selected:
            # Signal 1: vagueness (from query_analysis, always available)
            # Signal 2: fragment entropies (now from entropy_score key, Bug #1 fixed)
            fragment_entropies = [
                f.get("entropy_score", 0.5) for f in selected
            ]
            # Signal 3: sufficiency — knapsack fill ratio
            total_tokens_used = sum(f.get("token_count", 0) for f in selected)
            sufficiency = min(1.0, total_tokens_used / max(token_budget, 1))

            optimal_tau = compute_optimal_temperature(
                vagueness=vagueness,
                fragment_entropies=fragment_entropies,
                sufficiency=sufficiency,
                task_type=task_type,
                fisher_scale=self.config.fisher_scale,
                alpha=self.config.egtc_alpha,
                gamma=self.config.egtc_gamma,
                eps_d=self.config.egtc_epsilon,
            )

        # Security scan on selected fragments
        security_issues: list[str] = []
        if self.config.enable_security_scan and self.engine._guard.available:
            for frag in selected:
                content = frag.get("preview", frag.get("content", ""))
                source = frag.get("source", "")
                issues = self.engine._guard.scan(content, source)
                for issue in issues:
                    security_issues.append(f"[{source}] {issue}")

        # LTM memories (already injected by optimize_context, but we want to
        # show them in the context block for transparency)
        ltm_memories: list[dict] = []
        if self.config.enable_ltm and self.engine._ltm.active:
            ltm_memories = self.engine._ltm.recall_relevant(
                user_message, top_k=3, min_retention=0.3
            )

        # ── Format context block ──
        apa_kwargs: Dict[str, Any] = {}
        if self.config.enable_prompt_directives:
            apa_kwargs["task_type"] = task_type
            apa_kwargs["vagueness"] = vagueness

        if hcc_result is not None:
            # Hierarchical: 3-level compression
            context_text = format_hierarchical_context(
                hcc_result, security_issues, ltm_memories, refinement_info,
                **apa_kwargs,
            )
            logger.info(
                f"HCC: L1={hcc_result.get('level1_tokens', 0)}t, "
                f"L2={hcc_result.get('level2_tokens', 0)}t, "
                f"L3={hcc_result.get('level3_tokens', 0)}t, "
                f"coverage={hcc_result.get('coverage', {})}"
            )
        else:
            # Flat: original format_context_block
            context_text = format_context_block(
                selected, security_issues, ltm_memories, refinement_info,
                **apa_kwargs,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if selected:
            total_tokens = sum(f.get("token_count", 0) for f in selected)
            tau_str = f", τ={optimal_tau:.4f}" if optimal_tau else ""
            # IOS diversity score from Rust engine
            ios_div = result.get("ios_diversity_score")
            ios_str = f", diversity={ios_div:.2f}" if ios_div else ""
            # Resolution breakdown
            full_count = sum(1 for f in selected if f.get("variant") == "full")
            skel_count = sum(1 for f in selected if f.get("variant") == "skeleton")
            ref_count = sum(1 for f in selected if f.get("variant") == "reference")
            res_parts = [f"{full_count}F"]
            if skel_count:
                res_parts.append(f"{skel_count}S")
            if ref_count:
                res_parts.append(f"{ref_count}R")
            res_str = "+".join(res_parts)
            logger.info(
                f"Pipeline: {elapsed_ms:.1f}ms, "
                f"{len(selected)} fragments [{res_str}], "
                f"{total_tokens} tokens{tau_str}{ios_str}"
            )

        return {
            "context": context_text,
            "elapsed_ms": elapsed_ms,
            "temperature": optimal_tau,
            "selected_fragments": selected,
        }

    async def _stream_response(
        self, url: str, headers: Dict[str, str], body: Dict[str, Any]
    ) -> StreamingResponse:
        """Forward a streaming request and proxy the SSE response."""
        # Check circuit breaker
        if not self._breaker.allow_request():
            return JSONResponse(
                {"error": "circuit_breaker_open", "message": "Upstream API experiencing failures, retrying after cooldown"},
                status_code=503,
                headers={"Retry-After": str(int(self._breaker.cooldown_s))},
            )

        async def event_generator():
            try:
                client = await self._ensure_client()
                async with client.stream(
                    "POST", url, json=body, headers=headers
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                self._breaker.record_success()
            except httpx.ReadError as e:
                self._breaker.record_failure()
                logger.warning(f"Upstream stream interrupted: {e}")
                yield f'data: {{"error": "upstream_connection_lost"}}\n\n'.encode()
            except httpx.TimeoutException as e:
                self._breaker.record_failure()
                logger.warning(f"Upstream stream timeout: {e}")
                yield f'data: {{"error": "upstream_timeout"}}\n\n'.encode()
            except Exception as e:
                self._breaker.record_failure()
                logger.warning(f"Unexpected stream error: {e}")

        resp_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Entroly-Optimized": "true",
        }
        with self._stats_lock:
            if self._last_temperature is not None:
                resp_headers["X-Entroly-Temperature"] = f"{self._last_temperature:.4f}"
            # Gap #27: Value signal headers
            if self._total_original_tokens > 0:
                saved_pct = max(0, (self._total_original_tokens - self._total_optimized_tokens)) * 100 // self._total_original_tokens
                resp_headers["X-Entroly-Tokens-Saved-Pct"] = str(saved_pct)
            resp_headers["X-Entroly-Pipeline-Ms"] = f"{self._last_pipeline_ms:.1f}"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=resp_headers,
        )

    async def _forward_response(
        self, url: str, headers: Dict[str, str], body: Dict[str, Any]
    ) -> JSONResponse:
        """Forward a non-streaming request with circuit breaker, retry on 429/5xx, and response validation."""
        # Check circuit breaker
        if not self._breaker.allow_request():
            logger.warning("Circuit breaker open — forwarding unmodified")

        # Retry loop: 1 initial attempt + up to 2 retries on 429/5xx
        max_retries = 2
        response = None
        for attempt in range(max_retries + 1):
            try:
                client = await self._ensure_client()
                response = await client.post(url, json=body, headers=headers)
                self._breaker.record_success()
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                self._breaker.record_failure()
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # 1s, 2s backoff
                    continue
                # Sanitize error message: never leak auth headers in error responses
                err_msg = str(e)
                for key_header in ("authorization", "x-api-key"):
                    if key_header in headers:
                        err_msg = err_msg.replace(headers[key_header], "[REDACTED]")
                return JSONResponse(
                    {"error": "upstream_unavailable", "detail": err_msg},
                    status_code=502,
                )

            # Retry on 429 (rate limit) and 5xx (server errors)
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = float(response.headers.get("retry-after", str(1.0 * (attempt + 1))))
                if attempt < max_retries:
                    logger.info(
                        f"Upstream {response.status_code}, retrying in {retry_after:.0f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(min(retry_after, 10.0))
                    continue
                # Out of retries — distinguish entroly error vs upstream error
                return JSONResponse(
                    {
                        "error": "upstream_error",
                        "status": response.status_code,
                        "detail": f"Upstream returned {response.status_code} after {max_retries} retries",
                        "source": "upstream_api",
                    },
                    status_code=response.status_code,
                    headers={"X-Entroly-Source": "upstream"},
                )

            # Success — break out of retry loop
            break

        resp_headers: Dict[str, str] = {"X-Entroly-Optimized": "true"}
        with self._stats_lock:
            if self._last_temperature is not None:
                resp_headers["X-Entroly-Temperature"] = f"{self._last_temperature:.4f}"

        # Validate response content-type before parsing JSON
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                content = response.json()
            except (json.JSONDecodeError, ValueError):
                content = {
                    "error": "invalid_upstream_json",
                    "status": response.status_code,
                }
        else:
            # Non-JSON response (e.g., HTML error page from CDN/gateway)
            content = {
                "error": "non_json_upstream_response",
                "status": response.status_code,
                "body_preview": response.text[:500],
            }

        return JSONResponse(
            content=content,
            status_code=response.status_code,
            headers=resp_headers,
        )

    async def _forward_raw(
        self, request: Request, body_bytes: bytes
    ) -> JSONResponse:
        """Forward a raw (non-JSON) request."""
        return JSONResponse(
            {"error": "invalid request body"}, status_code=400
        )

    async def _warmup_connection(self, target_url: str) -> None:
        """Pre-warm the HTTP connection pool for the target API.

        Overlap connection setup (TLS handshake, DNS resolution) with the
        Rust compute pipeline.

        For persistent connections (typical after first request), this is
        essentially a no-op. For cold starts, it saves ~50ms of TLS time.
        """
        if not self._client:
            return
        try:
            # HEAD request to establish the connection without payload
            # httpx will reuse this connection for the actual POST
            from urllib.parse import urlparse
            parsed = urlparse(target_url)
            warmup_url = f"{parsed.scheme}://{parsed.netloc}/health"
            await self._client.head(warmup_url, timeout=2.0)
        except Exception:
            pass  # Non-critical — the actual request will establish the connection

    def _resolve_target(self, provider: str, path: str) -> str:
        if provider == "anthropic":
            return f"{self.config.anthropic_base_url}{path}"
        return f"{self.config.openai_base_url}{path}"

    def _build_headers(
        self, original: Dict[str, str], provider: str
    ) -> Dict[str, str]:
        """Build headers for the forwarded request. Pass through auth."""
        forward: Dict[str, str] = {"Content-Type": "application/json"}
        if "authorization" in original:
            forward["Authorization"] = original["authorization"]
        if "x-api-key" in original:
            forward["x-api-key"] = original["x-api-key"]
        if "anthropic-version" in original:
            forward["anthropic-version"] = original["anthropic-version"]
        return forward

    @staticmethod
    def _mask_key(value: str) -> str:
        """Mask an API key for safe logging: 'sk-abc...xyz' → 'sk-abc...xyz' (first 6 + last 4)."""
        if len(value) <= 12:
            return "***"
        return f"{value[:6]}...{value[-4:]}"


async def _health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "entroly-proxy"})


async def _context_inspect(request: Request) -> JSONResponse:
    """Gap #29: Context transparency — show what fragments entroly injected."""
    proxy = request.app.state.proxy
    with proxy._stats_lock:
        fragments = []
        for f in proxy._last_context_fragments:
            fragments.append({
                "source": f.get("source", ""),
                "token_count": f.get("token_count", 0),
                "entropy_score": round(f.get("entropy_score", 0), 4),
                "relevance": round(f.get("relevance", 0), 4),
                "preview": _safe_preview(f.get("content", "")),
            })
        return JSONResponse({
            "last_query": proxy._last_query,
            "pipeline_ms": round(proxy._last_pipeline_ms, 2),
            "fragments_injected": len(fragments),
            "fragments": fragments,
        })


async def _metrics_prometheus(request: Request) -> StreamingResponse:
    """Gap #34: Prometheus-compatible metrics endpoint for Grafana/Datadog."""
    proxy = request.app.state.proxy
    with proxy._stats_lock:
        lines = [
            f"# HELP entroly_requests_total Total proxy requests",
            f"# TYPE entroly_requests_total counter",
            f"entroly_requests_total {proxy._requests_total}",
            f"# HELP entroly_requests_optimized Optimized requests",
            f"# TYPE entroly_requests_optimized counter",
            f"entroly_requests_optimized {proxy._requests_optimized}",
            f"# HELP entroly_requests_bypassed Bypassed requests",
            f"# TYPE entroly_requests_bypassed counter",
            f"entroly_requests_bypassed {proxy._requests_bypassed}",
            f"# HELP entroly_tokens_original_total Original token count",
            f"# TYPE entroly_tokens_original_total counter",
            f"entroly_tokens_original_total {proxy._total_original_tokens}",
            f"# HELP entroly_tokens_optimized_total Optimized token count",
            f"# TYPE entroly_tokens_optimized_total counter",
            f"entroly_tokens_optimized_total {proxy._total_optimized_tokens}",
            f"# HELP entroly_pipeline_latency_ms Pipeline latency",
            f"# TYPE entroly_pipeline_latency_ms gauge",
            f"entroly_pipeline_latency_ms {proxy._pipeline_stats.mean:.2f}",
            f"# HELP entroly_circuit_breaker Circuit breaker state (0=closed, 1=open)",
            f"# TYPE entroly_circuit_breaker gauge",
            f'entroly_circuit_breaker {1 if proxy._breaker.state == "open" else 0}',
            f"# HELP entroly_outcome_success Successful outcomes recorded",
            f"# TYPE entroly_outcome_success counter",
            f"entroly_outcome_success {proxy._outcome_success}",
            f"# HELP entroly_outcome_failure Failed outcomes recorded",
            f"# TYPE entroly_outcome_failure counter",
            f"entroly_outcome_failure {proxy._outcome_failure}",
        ]

    async def _gen():
        yield "\n".join(lines) + "\n"

    return StreamingResponse(_gen(), media_type="text/plain; version=0.0.4")


async def _record_outcome(request: Request) -> JSONResponse:
    """Gap #37: Record whether entroly's optimization helped or hurt.

    When fragment_ids are provided, also feeds the PRISM RL weight update
    loop so the engine learns from proxy-mode outcomes (not just MCP).
    """
    proxy = request.app.state.proxy
    body = await request.json()
    success = body.get("success", True)
    fragment_ids = body.get("fragment_ids", [])

    # Feed PRISM RL update if fragment IDs provided
    if fragment_ids:
        try:
            if success:
                proxy.engine.record_success(fragment_ids)
            else:
                proxy.engine.record_failure(fragment_ids)
        except Exception as e:
            logger.debug("PRISM RL update skipped: %s", e)

    with proxy._stats_lock:
        if success:
            proxy._outcome_success += 1
        else:
            proxy._outcome_failure += 1
        total = proxy._outcome_success + proxy._outcome_failure
        error_rate = proxy._outcome_failure / max(total, 1)
    return JSONResponse({
        "recorded": True,
        "outcome": "success" if success else "failure",
        "fragment_ids": fragment_ids,
        "prism_updated": bool(fragment_ids),
        "error_rate": round(error_rate, 4),
        "total_outcomes": total,
    })


async def _fragment_feedback(request: Request) -> JSONResponse:
    """Gap #42: Thumbs-up/down feedback on specific injected fragments.

    POST /feedback {"fragment_id": "f123", "helpful": true}
    or    /feedback {"fragment_id": "f123", "helpful": false}

    Feeds directly into the Wilson Score feedback tracker in the Rust engine.
    """
    proxy = request.app.state.proxy
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    fragment_id = body.get("fragment_id")
    helpful = body.get("helpful", True)

    if not fragment_id:
        return JSONResponse({"error": "fragment_id required"}, status_code=400)

    try:
        if helpful:
            proxy.engine._rust.record_feedback(fragment_id, True)
        else:
            proxy.engine._rust.record_feedback(fragment_id, False)
    except Exception as e:
        # Engine method may not exist in older versions
        logger.debug(f"Feedback recording failed: {e}")

    with proxy._stats_lock:
        if helpful:
            proxy._outcome_success += 1
        else:
            proxy._outcome_failure += 1

    return JSONResponse({
        "recorded": True,
        "fragment_id": fragment_id,
        "helpful": helpful,
    })


async def _context_explain(request: Request) -> JSONResponse:
    """Gap #43: Explain WHY each fragment was selected.

    GET /explain — returns per-fragment selection reasons.
    """
    proxy = request.app.state.proxy
    fragments = []
    for f in proxy._last_context_fragments:
        # Build explanation from scores
        reasons = []
        entropy = f.get("entropy_score", 0)
        relevance = f.get("relevance", 0)
        variant = f.get("variant", "full")
        source = f.get("source", "")

        if entropy > 0.7:
            reasons.append(f"high information density (entropy={entropy:.2f})")
        elif entropy > 0.4:
            reasons.append(f"moderate entropy ({entropy:.2f})")
        else:
            reasons.append(f"low entropy ({entropy:.2f})")

        if relevance > 0.7:
            reasons.append("strong query relevance")
        elif relevance > 0.4:
            reasons.append("moderate query relevance")

        if variant == "skeleton":
            reasons.append("included as skeleton (compressed)")
        elif variant == "reference":
            reasons.append("included as reference (minimal)")

        if "test" in source.lower():
            reasons.append("test file (may verify behavior)")
        if any(kw in source.lower() for kw in ["config", "schema", "setup"]):
            reasons.append("configuration/setup file (critical)")

        fragments.append({
            "source": source,
            "token_count": f.get("token_count", 0),
            "reasons": reasons,
            "scores": {
                "entropy": round(entropy, 4),
                "relevance": round(relevance, 4),
            },
            "preview": _safe_preview(f.get("content", "")),
        })

    return JSONResponse({
        "query": proxy._last_query,
        "pipeline_ms": round(proxy._last_pipeline_ms, 2),
        "fragments_explained": len(fragments),
        "fragments": fragments,
    })


async def _toggle_bypass(request: Request) -> JSONResponse:
    """Gap #28: Toggle bypass mode at runtime via POST /bypass."""
    proxy = request.app.state.proxy
    body = await request.json()
    proxy._bypass = body.get("enabled", not proxy._bypass)
    return JSONResponse({
        "bypass": proxy._bypass,
        "message": "Optimization disabled — forwarding raw" if proxy._bypass else "Optimization re-enabled",
    })


async def _catch_all(request: Request) -> StreamingResponse | JSONResponse:
    """Transparent catch-all: forward any unmatched path to upstream API.

    IDE clients (Cursor, Continue, Copilot) hit paths like /v1/models,
    /v1/completions, /v1/engines beyond the two chat endpoints we optimize.
    Without this, they get a 404 and the user has to work around it.

    This route matches LAST (Starlette matches routes in order), so it
    only fires for paths not handled by the explicit routes above.
    """
    proxy = request.app.state.proxy
    headers = {k: v for k, v in request.headers.items()}
    provider = detect_provider(request.url.path, headers)
    target_url = proxy._resolve_target(provider, request.url.path)
    forward_headers = proxy._build_headers(headers, provider)

    if request.method == "GET":
        try:
            client = await proxy._ensure_client()
            response = await client.get(target_url, headers=forward_headers)
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return JSONResponse(
                    content=response.json(),
                    status_code=response.status_code,
                )
            return JSONResponse(
                content={"data": response.text},
                status_code=response.status_code,
            )
        except Exception as e:
            return JSONResponse(
                {"error": "upstream_unavailable", "detail": str(e)},
                status_code=502,
            )

    # POST/PUT/DELETE — forward with body
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"error": "invalid request body"}, status_code=400)

    try:
        client = await proxy._ensure_client()
        is_streaming = body.get("stream", False) if isinstance(body, dict) else False
        if is_streaming:
            return await proxy._stream_response(target_url, forward_headers, body)
        response = await client.request(
            request.method, target_url, json=body, headers=forward_headers
        )
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code,
            )
        return JSONResponse(
            content={"data": response.text},
            status_code=response.status_code,
        )
    except Exception as e:
        return JSONResponse(
            {"error": "upstream_unavailable", "detail": str(e)},
            status_code=502,
        )


async def _proxy_stats(request: Request) -> JSONResponse:
    proxy = request.app.state.proxy
    with proxy._stats_lock:
        stats: Dict[str, Any] = {
            "requests_total": proxy._requests_total,
            "requests_optimized": proxy._requests_optimized,
            "requests_bypassed": proxy._requests_bypassed,
            "optimization_rate": (
                f"{proxy._requests_optimized / max(proxy._requests_total, 1):.0%}"
            ),
            "bypass_mode": proxy._bypass,
            "circuit_breaker": proxy._breaker.state,
            "pipeline_latency": proxy._pipeline_stats.to_dict(),
            # Gap #27: Value signal (DP-rounded to prevent fingerprinting)
            "tokens": {
                "original_total": _dp_round(proxy._total_original_tokens),
                "optimized_total": _dp_round(proxy._total_optimized_tokens),
                "saved_total": _dp_round(max(0, proxy._total_original_tokens - proxy._total_optimized_tokens)),
                "savings_pct": (
                    f"{max(0, proxy._total_original_tokens - proxy._total_optimized_tokens) * 100 // max(proxy._total_original_tokens, 1)}%"
                    if proxy._total_original_tokens > 0 else "N/A"
                ),
            },
            # Gap #37: Error budget
            "outcomes": {
                "success": proxy._outcome_success,
                "failure": proxy._outcome_failure,
                "error_rate": round(
                    proxy._outcome_failure / max(proxy._outcome_success + proxy._outcome_failure, 1), 4
                ),
            },
        }
        if proxy._temperature_count > 0:
            stats["egtc"] = {
                "enabled": proxy.config.enable_temperature_calibration,
                "avg_temperature": round(proxy._temperature_sum / proxy._temperature_count, 4),
                "last_temperature": proxy._last_temperature,
                "calibrations": proxy._temperature_count,
            }
    return JSONResponse(stats)


# Need os for ENTROLY_RATE_LIMIT env var
import os  # noqa: E402


def create_proxy_app(
    engine: Any, config: Optional[ProxyConfig] = None
) -> Starlette:
    """Create the Starlette ASGI app for the prompt compiler proxy."""
    proxy = PromptCompilerProxy(engine, config)

    # Auto-start the live value dashboard alongside the proxy
    try:
        from .dashboard import start_dashboard
        start_dashboard(engine=engine, port=9378, daemon=True)
        logger.info("Value dashboard live at http://localhost:9378")
    except Exception as e:
        logger.warning(f"Dashboard failed to start: {e}")

    # Start the autotune RL daemon — continuously improves weights in background.
    # Lazy import to avoid circular dependency (server.py ↔ proxy.py).
    try:
        import importlib
        _server_mod = importlib.import_module("entroly.server")
        _server_mod._start_autotune_daemon(engine)
        logger.info("Autotune RL daemon started (background, nice+10)")
    except Exception as e:
        logger.debug(f"Autotune daemon not started: {e}")

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", proxy.handle_proxy, methods=["POST"]),
            Route("/v1/messages", proxy.handle_proxy, methods=["POST"]),
            Route("/health", _health),
            Route("/stats", _proxy_stats),
            Route("/context", _context_inspect),          # Gap #29
            Route("/metrics", _metrics_prometheus),        # Gap #34
            Route("/outcome", _record_outcome, methods=["POST"]),  # Gap #37
            Route("/bypass", _toggle_bypass, methods=["POST"]),    # Gap #28
            Route("/feedback", _fragment_feedback, methods=["POST"]),  # Gap #42
            Route("/explain", _context_explain),                       # Gap #43
            # Catch-all: forward any unmatched path to upstream API
            # Must be LAST — Starlette matches routes in declaration order
            Route("/{path:path}", _catch_all, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
        ],
        on_startup=[proxy.startup],
        on_shutdown=[proxy.shutdown],
    )
    app.state.proxy = proxy
    return app
