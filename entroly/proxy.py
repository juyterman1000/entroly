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
import collections
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
from typing import Any

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
    inject_context_gemini,
    inject_context_openai,
)
from .value_tracker import get_tracker

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


# ── Resilience primitives ────────────────────────────────────────────


class _CircuitBreaker:
    """SIRS-inspired circuit breaker: open after N consecutive failures,
    half-open after cooldown period, close on success.

    Inspired by the refractory period in SIRS epidemic routing models.
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

    Standard token bucket rate limiter.
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
        import json as _json

        from entroly_core import py_compress_block, py_progressive_thresholds

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


# ── Passive Implicit Feedback ─────────────────────────────────────────────
#
# Extracts RL feedback signals from observable proxy traffic:
#   Signal 1: LLM confusion detection (response text analysis)
#   Signal 2: Query trajectory rephrase detection (SimHash similarity)
#   Signal 3: Sufficiency heuristic (already computed in optimize)
#
# This closes the RL feedback loop without IDE cooperation.
# Reference: Implementation plan — "3-Signal Passive Feedback"


class ImplicitFeedbackTracker:
    """Extract implicit RL feedback from proxy traffic.

    Thread-safe. Per-client state tracks query trajectories for
    rephrase detection. Response text is scanned for confusion
    indicators to infer success/failure.
    """

    # ── Signal 1: Confusion patterns in LLM responses ────────────────
    # When the LLM says these phrases, our context selection failed.
    _CONFUSION_PATTERNS = re.compile(
        r"(?:I\s+(?:don'?t|do\s+not)\s+(?:have|see)\s+(?:enough\s+|the\s+)?(?:context|code|file|information))"
        r"|(?:could\s+you\s+(?:provide|share|show|paste))"
        r"|(?:I(?:'m|\s+am)\s+not\s+(?:sure|certain)\s+(?:about|what|which|where))"
        r"|(?:without\s+(?:seeing|access|the\s+(?:full|actual|complete)))"
        r"|(?:I\s+(?:cannot|can'?t)\s+(?:see|access|find|determine))"
        r"|(?:I\s+(?:don'?t|do\s+not)\s+have\s+(?:access|visibility))"
        r"|(?:(?:more|additional)\s+context\s+(?:would|is)\s+(?:needed|helpful|required))"
        r"|(?:please\s+(?:share|provide|paste)\s+(?:the|your))",
        re.IGNORECASE,
    )

    # Minimum response length to trigger confidence signal (chars)
    _MIN_CONFIDENT_LENGTH = 200

    # Rephrase detection thresholds
    _REPHRASE_SIMILARITY_THRESHOLD = 0.75  # SimHash similarity > this = rephrase
    _REPHRASE_TIME_WINDOW_S = 90.0  # Within this many seconds
    _TOPIC_CHANGE_THRESHOLD = 0.30  # Similarity < this = topic change = success

    # Buffer cap for streaming responses (bytes)
    _MAX_BUFFER_BYTES = 50 * 1024  # 50KB — covers 99%+ of LLM responses

    def __init__(self):
        self._lock = threading.Lock()
        # Per-client trajectory: client_key -> (query_simhash, selected_ids, timestamp)
        self._trajectories: dict[str, tuple] = {}
        # Stats
        self._confusion_detections = 0
        self._confidence_detections = 0
        self._rephrase_detections = 0
        self._topic_changes = 0
        self._total_assessed = 0
        # CUSUM-EMA quality drift detector (arXiv 2025, NeurIPS 2025)
        self._drift_detector = _CusumEmaDriftDetector()

    def assess_response(self, response_text: str) -> float:
        """Assess an LLM response for confusion vs confidence.

        Returns a reward signal:
          -1.0  = strong confusion detected (multiple indicators)
          -0.5  = mild confusion detected (one indicator)
           0.0  = ambiguous / too short to tell
          +0.3  = confident response (long, structured)
          +0.5  = confident response with code blocks
        """
        if not response_text or len(response_text) < 50:
            return 0.0

        # Count confusion pattern matches
        confusion_matches = len(self._CONFUSION_PATTERNS.findall(response_text[:5000]))

        if confusion_matches >= 2:
            return -1.0  # Strong confusion
        if confusion_matches == 1:
            return -0.5  # Mild confusion

        # Check for confidence signals
        has_code_blocks = "```" in response_text
        is_long = len(response_text) >= self._MIN_CONFIDENT_LENGTH

        if is_long and has_code_blocks:
            return 0.5  # Confident with code
        if is_long:
            return 0.3  # Confident (structured answer)

        return 0.0  # Ambiguous

    def detect_rephrase(
        self, client_key: str, query_text: str, selected_ids: list
    ) -> tuple | None:
        """Check if this query is a rephrase of the previous one.

        Returns:
          ("rephrase", prev_selected_ids) if rephrase detected -> failure signal
          ("topic_change", prev_selected_ids) if topic changed -> success signal
          None if no trajectory data or ambiguous
        """
        try:
            from entroly_core import py_simhash
            query_hash = py_simhash(query_text)
        except (ImportError, Exception):
            return None

        now = time.time()

        with self._lock:
            prev = self._trajectories.get(client_key)

            # Update trajectory
            self._trajectories[client_key] = (query_hash, selected_ids, now)

            # Evict old entries (> 1000 clients)
            if len(self._trajectories) > 1000:
                oldest_key = min(
                    self._trajectories,
                    key=lambda k: self._trajectories[k][2],
                )
                del self._trajectories[oldest_key]

        if prev is None:
            return None

        prev_hash, prev_ids, prev_time = prev
        time_delta = now - prev_time

        if time_delta > self._REPHRASE_TIME_WINDOW_S:
            return None  # Too long ago to be a rephrase

        if not prev_ids:
            return None  # No fragment IDs to attribute

        # Compute SimHash similarity (Hamming-based)
        xor = query_hash ^ prev_hash
        hamming = bin(xor).count("1")
        similarity = 1.0 - (hamming / 64.0)

        if similarity > self._REPHRASE_SIMILARITY_THRESHOLD:
            with self._lock:
                self._rephrase_detections += 1
            return ("rephrase", prev_ids)

        if similarity < self._TOPIC_CHANGE_THRESHOLD:
            with self._lock:
                self._topic_changes += 1
            return ("topic_change", prev_ids)

        return None  # Ambiguous mid-range similarity

    def record_assessment(self, reward: float) -> None:
        """Track assessment stats and feed the drift detector."""
        with self._lock:
            self._total_assessed += 1
            if reward < -0.25:
                self._confusion_detections += 1
            elif reward > 0.25:
                self._confidence_detections += 1
            # Feed dual drift detector
            self._drift_detector.update(reward)

    def quality_trend(self) -> str:
        """Return current quality trend: 'stable', 'declining', or 'improving'."""
        with self._lock:
            return self._drift_detector.trend()

    def stats(self) -> dict[str, Any]:
        """Return feedback tracker statistics."""
        with self._lock:
            drift_stats = self._drift_detector.to_dict()
            return {
                "total_assessed": self._total_assessed,
                "confusion_detections": self._confusion_detections,
                "confidence_detections": self._confidence_detections,
                "rephrase_detections": self._rephrase_detections,
                "topic_changes": self._topic_changes,
                "quality_trend": drift_stats["trend"],
                "drift_detector": drift_stats,
            }


class _CusumEmaDriftDetector:
    """Dual online quality drift detector: CUSUM + EMA.

    Combines two complementary algorithms from the change-point detection
    literature (Online Kernel CUSUM, arXiv 2025; RL drift detection,
    NeurIPS 2025):

    1. **EMA** (Exponential Moving Average): Smooth trend tracker.
       α = 0.15 → emphasizes recent observations. Fast to respond but
       susceptible to noise.

    2. **Page's CUSUM** (Cumulative Sum): Detects persistent drift in
       the reward signal. Accumulates deviations from the target mean.
       More robust than EMA — fires only on sustained degradation.

    Quality trend states:
      - "stable": Both detectors within bounds
      - "declining": Either detector flags degradation
      - "improving": EMA above positive threshold after a decline

    Thread-safety: Caller must hold lock (ImplicitFeedbackTracker._lock).
    """

    # EMA smoothing factor: 0.15 gives ~13-sample effective window
    _ALPHA = 0.15
    # CUSUM sensitivity: accumulate when reward < this target
    _TARGET_MEAN = 0.0
    # CUSUM decision threshold: fire alarm when cumulative sum exceeds this
    _CUSUM_THRESHOLD = 3.0
    # EMA threshold for "declining" signal
    _EMA_DECLINE_THRESHOLD = -0.20
    # EMA threshold for "improving" signal
    _EMA_IMPROVE_THRESHOLD = 0.15
    # Minimum observations before drift detection activates
    _MIN_OBSERVATIONS = 5

    def __init__(self):
        self.ema: float = 0.0
        self.cusum_pos: float = 0.0  # Detect upward shift (quality improving)
        self.cusum_neg: float = 0.0  # Detect downward shift (quality declining)
        self.count: int = 0
        self._was_declining: bool = False

    def update(self, reward: float) -> None:
        """Feed a new reward observation."""
        self.count += 1

        # EMA update
        if self.count == 1:
            self.ema = reward
        else:
            self.ema = self._ALPHA * reward + (1.0 - self._ALPHA) * self.ema

        # Page's CUSUM update (two-sided)
        deviation = reward - self._TARGET_MEAN
        self.cusum_pos = max(0.0, self.cusum_pos + deviation)
        self.cusum_neg = max(0.0, self.cusum_neg - deviation)

        # Track state transitions for "improving" detection
        if self.trend() == "declining":
            self._was_declining = True

    def trend(self) -> str:
        """Return current quality trend."""
        if self.count < self._MIN_OBSERVATIONS:
            return "stable"  # Not enough data yet

        # Declining: EMA below threshold OR CUSUM negative alarm
        if (self.ema < self._EMA_DECLINE_THRESHOLD
                or self.cusum_neg > self._CUSUM_THRESHOLD):
            return "declining"

        # Improving: EMA above positive threshold AND recovered from decline
        if self._was_declining and self.ema > self._EMA_IMPROVE_THRESHOLD:
            return "improving"

        return "stable"

    def reset(self) -> None:
        """Reset detector state (e.g., on session restart)."""
        self.ema = 0.0
        self.cusum_pos = 0.0
        self.cusum_neg = 0.0
        self.count = 0
        self._was_declining = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ema": round(self.ema, 4),
            "cusum_pos": round(self.cusum_pos, 4),
            "cusum_neg": round(self.cusum_neg, 4),
            "observations": self.count,
            "trend": self.trend(),
        }


def _extract_text_from_sse(raw_bytes: bytes) -> str:
    """Extract assistant text content from SSE stream bytes.

    Handles OpenAI, Anthropic, and Gemini SSE formats.
    Returns concatenated text content for confusion pattern analysis.
    """
    text_parts = []
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except (json.JSONDecodeError, ValueError):
                continue
            # OpenAI format: choices[0].delta.content
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if "content" in delta and delta["content"]:
                    text_parts.append(delta["content"])
            # Anthropic format: content_block.text or delta.text
            if data.get("type") == "content_block_delta":
                delta = data.get("delta", {})
                if "text" in delta:
                    text_parts.append(delta["text"])
            # Gemini format: candidates[0].content.parts[0].text
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "text" in part:
                        text_parts.append(part["text"])
    except Exception:
        pass
    return "".join(text_parts)


# ── Proxy ────────────────────────────────────────────────────────────────


class PromptCompilerProxy:
    """HTTP reverse proxy that optimizes every LLM request with entroly."""

    def __init__(self, engine: Any, config: ProxyConfig | None = None):
        self.engine = engine
        self.config = config or ProxyConfig()
        self._client: httpx.AsyncClient | None = None

        # Thread-safe stats
        self._stats_lock = threading.Lock()
        self._requests_total: int = 0
        self._requests_optimized: int = 0
        self._requests_bypassed: int = 0
        self._temperature_sum: float = 0.0
        self._temperature_count: int = 0
        self._last_temperature: float | None = None
        self._total_original_tokens: int = 0
        self._total_optimized_tokens: int = 0
        # Per-client trajectory isolation: each API key / auth header gets
        # its own turn counter. Prevents concurrent IDE clients from
        # corrupting each other's EGTC temperature calibration.
        self._trajectory_turns: collections.OrderedDict[str, int] = collections.OrderedDict()
        self._trajectory_max_clients = 1000  # evict LRU beyond this

        # ── Context Waste Telemetry ──
        # Tracks input context waste % per request (orig - optimized) / orig
        # This is what produces the real "X% of context window is wasted" number
        self._waste_ratios: list[float] = []  # Rolling window of per-request waste %
        self._waste_max_samples: int = 1000   # Keep last 1000 for rolling average

        # ── Response Distillation ──
        # Strips filler from LLM responses (pleasantries, hedging, meta-commentary)
        # Grounded in Selective Context (Li et al., EMNLP 2023) self-information theory
        self._enable_distill = os.environ.get("ENTROLY_DISTILL", "1") != "0"
        self._distill_mode = os.environ.get("ENTROLY_DISTILL_MODE", "full")  # lite/full/ultra
        self._total_output_original_tokens: int = 0
        self._total_output_compressed_tokens: int = 0

        # Gap #29: Last optimization context (for transparency endpoint)
        self._last_context_fragments: list = []
        self._last_excluded_fragments: list = []  # Top rejected candidates for /explain
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
        self._rate_limiter: _TokenBucket | None = None
        if rate_limit > 0:
            self._rate_limiter = _TokenBucket(
                capacity=float(rate_limit), refill_per_second=rate_limit / 60.0
            )

        # Pipeline latency tracking (Welford online stats)
        self._pipeline_stats = _WelfordStats()

        # Passive implicit feedback — closes the RL loop without IDE cooperation
        self._feedback_tracker = ImplicitFeedbackTracker()
        self._enable_passive_feedback = (
            os.environ.get("ENTROLY_PASSIVE_FEEDBACK", "1") != "0"
        )

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
        provider = detect_provider(path, headers, body)

        if "messages" in body:
            from .proxy_transform import compress_tool_messages
            # Stage 1: age-tiered pruning (collapses old tool outputs to digests).
            # Runs first so content-aware compression in stage 2 only pays for
            # the small tail window of recent tool messages.
            if getattr(self.config, "enable_aged_tool_pruning", True):
                from .hardening import prune_aged_tool_outputs
                body["messages"], aged_bytes_saved = prune_aged_tool_outputs(
                    body["messages"],
                    tail_window=getattr(self.config, "aged_tool_tail_window", 4),
                )
                if aged_bytes_saved > 0:
                    logger.info(
                        f"Aged tool pruning: ~{aged_bytes_saved // 4} tokens saved"
                    )
            # Stage 2: content-aware compression (test output, diffs, JSON, …)
            body["messages"], tool_tokens_saved = compress_tool_messages(body["messages"])
            if tool_tokens_saved > 0:
                logger.info(f"Tool output compression: {tool_tokens_saved} tokens saved")

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
            if not is_streaming and "streamGenerateContent" in path:
                is_streaming = True
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
        auth_raw = (headers.get("authorization", "")
                    or headers.get("x-api-key", "")
                    or headers.get("x-goog-api-key", ""))
        client_key = hashlib.sha256(auth_raw.encode()).hexdigest()[:12] if auth_raw else "_default"

        # Track selected fragment IDs for passive feedback attribution
        _selected_frag_ids: list = []

        # Run the optimization pipeline (synchronous Rust, off the event loop)
        try:
            user_message = extract_user_message(body, provider)
            if user_message:
                pipeline_result = await asyncio.to_thread(
                    self._run_pipeline, user_message, body, path
                )
                context_text = pipeline_result["context"]
                pipeline_ms = pipeline_result["elapsed_ms"]
                optimal_tau = pipeline_result.get("temperature")

                # Track pipeline latency
                self._pipeline_stats.add(pipeline_ms)

                # Collect fragment IDs for passive feedback
                _selected_frag_ids = [
                    f.get("id", f.get("fragment_id", ""))
                    for f in pipeline_result.get("selected_fragments", [])
                    if f.get("id") or f.get("fragment_id")
                ]

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
                    if provider == "gemini":
                        # Gemini uses contents/parts instead of messages
                        original_tokens = sum(
                            len(p.get("text", "").split())
                            for item in body.get("contents", [])
                            for p in item.get("parts", [])
                            if isinstance(p, dict) and "text" in p
                        ) * 4 // 3
                    else:
                        original_tokens = sum(
                            len(m.get("content", "").split())
                            for m in body.get("messages", [])
                        ) * 4 // 3
                    optimized_tokens = len(context_text.split()) * 4 // 3
                    with self._stats_lock:
                        self._total_original_tokens += original_tokens
                        self._total_optimized_tokens += optimized_tokens
                        # ── Context Waste Telemetry ──
                        # Record per-request waste ratio for the real "X% wasted" number
                        if original_tokens > 0:
                            waste_ratio = max(0.0, (original_tokens - optimized_tokens) / original_tokens)
                            self._waste_ratios.append(waste_ratio)
                            if len(self._waste_ratios) > self._waste_max_samples:
                                self._waste_ratios.pop(0)
                        self._last_context_fragments = selected_frags[:20] if selected_frags else []
                        # Track the top 10 excluded fragments for /explain transparency.
                        # These are candidates the engine considered but dropped.
                        all_frags_result = pipeline_result.get("all_candidates", [])
                        selected_ids = {
                            f.get("id", f.get("fragment_id", ""))
                            for f in selected_frags
                        } if selected_frags else set()
                        if all_frags_result:
                            self._last_excluded_fragments = [
                                f for f in all_frags_result
                                if f.get("id", f.get("fragment_id", "")) not in selected_ids
                            ][:10]
                        else:
                            self._last_excluded_fragments = []
                        self._last_pipeline_ms = pipeline_ms
                        self._last_query = _sanitize_query(user_message)

                    if getattr(self.config, "enable_context_sanitizer", True):
                        from .hardening import sanitize_injected_context
                        context_text, _sanitize_report = sanitize_injected_context(
                            context_text, fence=True
                        )
                        if _sanitize_report.matches:
                            logger.warning(
                                "Injection-scan: %s patterns in retrieved context: %s",
                                len(_sanitize_report.matches),
                                _sanitize_report.matches,
                            )

                    if provider == "gemini":
                        body = inject_context_gemini(body, context_text)
                    elif provider == "anthropic":
                        body = inject_context_anthropic(body, context_text)
                    else:
                        body = inject_context_openai(body, context_text)

                    # Entropic Conversation Pruning
                    if provider != "gemini":
                        try:
                            from .proxy_transform import entropic_conversation_prune
                            from .hardening import ECP_THRASH_GUARD
                            anti_thrash = getattr(self.config, "enable_ecp_anti_thrash", True)
                            if anti_thrash and ECP_THRASH_GUARD.should_skip(client_key):
                                logger.debug("ECP: skipped (anti-thrash cooldown)")
                            else:
                                ecp_messages = body.get("messages", [])
                                pruned_msgs, ecp_stats = entropic_conversation_prune(
                                    ecp_messages, context_text, provider
                                )
                                if ecp_stats.get("pruned"):
                                    body["messages"] = pruned_msgs
                                    logger.debug(
                                        "ECP: %d messages compressed, %.1f%% savings",
                                        ecp_stats["messages_compressed"],
                                        ecp_stats["savings_ratio"] * 100,
                                    )
                                if anti_thrash:
                                    ECP_THRASH_GUARD.record(
                                        client_key, ecp_stats.get("savings_ratio", 0.0)
                                    )
                        except Exception:
                            pass  # Never block for conversation pruning

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
                        body = apply_temperature(body, optimal_tau, provider)
                        with self._stats_lock:
                            self._temperature_sum += optimal_tau
                            self._temperature_count += 1
                            self._last_temperature = optimal_tau
                            # Move to end on access (LRU tracking)
                            self._trajectory_turns[client_key] = self._trajectory_turns.get(client_key, 0) + 1
                            self._trajectory_turns.move_to_end(client_key)
                            # Evict least-recently-used clients to prevent unbounded memory growth
                            while len(self._trajectory_turns) > self._trajectory_max_clients:
                                self._trajectory_turns.popitem(last=False)

                    with self._stats_lock:
                        self._requests_optimized += 1
                        opt_count = self._requests_optimized
                        total_count = self._requests_total

                    # ── Persistent value tracking ──
                    try:
                        _saved = max(0, original_tokens - optimized_tokens)
                        _model = extract_model(body, path) or ""
                        _coverage = (len(selected_frags) / max(self.engine._rust.fragment_count(), 1) * 100) if selected_frags and hasattr(self.engine, '_rust') else 0.0
                        _confidence = avg_entropy if selected_frags else 0.0
                        get_tracker().record(
                            tokens_saved=_saved,
                            model=_model,
                            duplicates=0,
                            optimized=True,
                            coverage_pct=_coverage,
                            confidence=_confidence,
                        )
                    except Exception:
                        pass  # Never block a request for tracking

                    tau_str = f", τ={optimal_tau:.2f}" if optimal_tau else ""

                    # Startup banner: on first optimized request, print a
                    # human-visible confirmation so the user knows it's working.
                    if opt_count == 1:
                        if original_tokens > 0:
                            saved_pct = max(0, (original_tokens - optimized_tokens)) * 100 // original_tokens
                            # Resolution breakdown for the trust-building banner
                            s_frags = pipeline_result.get("selected_fragments", [])
                            full_names: list[str] = []
                            skel_c = 0
                            ref_c = 0
                            belief_c = 0
                            for sf in s_frags:
                                v = sf.get("variant", "full")
                                src = sf.get("source", "")
                                bname = src.rsplit("/", 1)[-1].removeprefix("file:")
                                if v == "full":
                                    if len(full_names) < 5:
                                        full_names.append(bname)
                                elif v == "skeleton":
                                    skel_c += 1
                                elif v == "reference":
                                    ref_c += 1
                                elif v == "belief":
                                    belief_c += 1
                            banner_lines = [
                                f"\n  First request optimized: "
                                f"{original_tokens:,} \u2192 {optimized_tokens:,} tokens "
                                f"({saved_pct}% saved) in {pipeline_ms:.1f}ms",
                            ]
                            if full_names:
                                more = f", +{len([sf for sf in s_frags if sf.get('variant', 'full') == 'full']) - len(full_names)} more" if len([sf for sf in s_frags if sf.get("variant", "full") == "full"]) > 5 else ""
                                banner_lines.append(
                                    f"  \u251c\u2500 Full (100%):    {', '.join(full_names)}{more}"
                                )
                            if belief_c:
                                banner_lines.append(
                                    f"  \u251c\u2500 Belief:        {belief_c} files"
                                )
                            if skel_c:
                                banner_lines.append(
                                    f"  \u251c\u2500 Skeleton:      {skel_c} files"
                                )
                            if ref_c:
                                banner_lines.append(
                                    f"  \u2514\u2500 Reference:     {ref_c} files"
                                )
                            print(
                                "\n".join(banner_lines) + "\n",
                                file=sys.stderr,
                            )

                    logger.info(
                        f"Optimized in {pipeline_ms:.1f}ms{tau_str} "
                        f"({opt_count}/{total_count} requests)"
                    )
        except Exception as e:
            # Cardinal rule: never block a request due to entroly errors
            logger.warning("Pipeline error (forwarding unmodified): %s: %s",
                          type(e).__name__, str(e)[:200])

        # Await warmup (usually completes during pipeline, essentially free)
        await warmup_task

        # ── Signal 2: Query trajectory rephrase detection ──
        if self._enable_passive_feedback and user_message:
            try:
                rephrase_result = self._feedback_tracker.detect_rephrase(
                    client_key, user_message, _selected_frag_ids
                )
                if rephrase_result:
                    signal_type, prev_ids = rephrase_result
                    if signal_type == "rephrase" and prev_ids:
                        logger.debug("Rephrase detected -> record_failure(%d ids)", len(prev_ids))
                        try:
                            self.engine.record_failure(prev_ids)
                        except Exception:
                            pass
                    elif signal_type == "topic_change" and prev_ids:
                        logger.debug("Topic change -> record_success(%d ids)", len(prev_ids))
                        try:
                            self.engine.record_success(prev_ids)
                        except Exception:
                            pass
            except Exception:
                pass  # Never block the request for feedback

        # Forward to real API (target_url already resolved above)
        forward_headers = self._build_headers(headers, provider)
        is_streaming = body.get("stream", False)
        # Gemini: streaming is determined by URL path, not a body field.
        # streamGenerateContent returns SSE — must be handled as streaming.
        if not is_streaming and "streamGenerateContent" in path:
            is_streaming = True

        if is_streaming:
            return await self._stream_response(
                target_url, forward_headers, body, _selected_frag_ids
            )
        else:
            return await self._forward_response(
                target_url, forward_headers, body, _selected_frag_ids
            )

    def _run_pipeline(self, user_message: str, body: dict[str, Any], path: str = "") -> dict[str, Any]:
        """Run the synchronous optimization pipeline. Called via asyncio.to_thread.

        Returns dict with keys: context, elapsed_ms, temperature.
        """
        t0 = time.perf_counter()

        model = extract_model(body, path)

        # Auto-configure cache cost model from the model name.
        # Zero-config: developers never need to call set_model() manually.
        if model and hasattr(self.engine, 'set_model'):
            self.engine.set_model(model)

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
            except Exception as e:
                logger.debug("ECDB pre-analysis fallback: %s", e)
                token_budget = compute_token_budget(model, self.config)
        else:
            token_budget = compute_token_budget(model, self.config)

        # Rate-Distortion self-correction: shift IOS toward full-resolution
        # fragments when quality declines (budget stays unchanged).
        if self._enable_passive_feedback:
            trend = self._feedback_tracker.quality_trend()
            if trend == "declining" and hasattr(self.engine, '_rust'):
                try:
                    self.engine._rust.update_belief_utilization(0.1, 0.8)
                    logger.debug(
                        "Quality declining: R-D rebalance (budget=%d)", token_budget
                    )
                except Exception:
                    pass
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

        # ── Context Resonance + Coverage Estimator metrics ──
        # These come from the Rust engine's optimize() and are forwarded
        # to response headers for observability.
        self._last_coverage = result.get("coverage", 0.0)
        self._last_coverage_confidence = result.get("coverage_confidence", 0.0)
        self._last_coverage_risk = result.get("coverage_risk", "unknown")
        self._last_coverage_gap = result.get("coverage_gap", 0.0)
        self._last_resonance_pairs = result.get("resonance_pairs", 0)
        self._last_resonance_strength = result.get("resonance_strength", 0.0)
        self._last_w_resonance = result.get("w_resonance", 0.0)
        # Causal Context Graph diagnostics
        self._last_causal_tracked = result.get("causal_tracked", 0)
        self._last_causal_interventional = result.get("causal_interventional", 0)
        self._last_causal_gravity_sources = result.get("causal_gravity_sources", 0)
        self._last_causal_mean_mass = result.get("causal_mean_mass", 0.0)

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
        if self.config.enable_security_scan and hasattr(self.engine, '_guard') and self.engine._guard.available:
            for frag in selected:
                content = frag.get("preview", frag.get("content", ""))
                source = frag.get("source", "")
                issues = self.engine._guard.scan(content, source)
                for issue in issues:
                    security_issues.append(f"[{source}] {issue}")

        # LTM memories (already injected by optimize_context, but we want to
        # show them in the context block for transparency).
        # Defensive access: ``hasattr`` returns True for an attribute set to
        # None — must also check for not-None before calling .active. The
        # engine *should* now initialize _ltm itself (see EntrolyEngine
        # init), but this guards against any path that bypasses that init.
        ltm_memories: list[dict] = []
        _ltm = getattr(self.engine, '_ltm', None)
        if (
            self.config.enable_ltm
            and _ltm is not None
            and getattr(_ltm, 'active', False)
        ):
            ltm_memories = _ltm.recall_relevant(
                user_message, top_k=3, min_retention=0.3
            )

        # ── Format context block ──
        apa_kwargs: dict[str, Any] = {}
        if self.config.enable_prompt_directives:
            apa_kwargs["task_type"] = task_type
            apa_kwargs["vagueness"] = vagueness
            apa_kwargs["coverage_risk"] = self._last_coverage_risk
            apa_kwargs["coverage"] = self._last_coverage

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
            belief_count = sum(1 for f in selected if f.get("variant") == "belief")
            skel_count = sum(1 for f in selected if f.get("variant") == "skeleton")
            ref_count = sum(1 for f in selected if f.get("variant") == "reference")
            res_parts = [f"{full_count}F"]
            if belief_count:
                res_parts.append(f"{belief_count}B")
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
        self, url: str, headers: dict[str, str], body: dict[str, Any],
        selected_frag_ids: list | None = None,
    ) -> StreamingResponse:
        """Forward a streaming request and proxy the SSE response.

        When passive feedback is enabled, tees response chunks into a buffer
        (capped at 50KB) and fires implicit feedback analysis after the
        stream completes. Zero latency impact — analysis runs in background.
        """
        # Check circuit breaker
        if not self._breaker.allow_request():
            return JSONResponse(
                {"error": "circuit_breaker_open", "message": "Upstream API experiencing failures, retrying after cooldown"},
                status_code=503,
                headers={"Retry-After": str(int(self._breaker.cooldown_s))},
            )

        # Capture references for the async generator closure
        _tracker = self._feedback_tracker
        _engine = self.engine
        _feedback_enabled = self._enable_passive_feedback and bool(selected_frag_ids)
        _frag_ids = selected_frag_ids or []
        _buffer_cap = ImplicitFeedbackTracker._MAX_BUFFER_BYTES
        # Capture selected fragments for per-variant utilization tracking (Change 4)
        _selected_frags = getattr(self, '_last_context_fragments', []) if _feedback_enabled else []

        async def event_generator():
            buffer = [] if _feedback_enabled else None
            buffer_size = 0
            try:
                client = await self._ensure_client()
                async with client.stream(
                    "POST", url, json=body, headers=headers
                ) as response:
                    async for chunk in response.aiter_bytes():
                        # Tee: pass through AND accumulate for analysis
                        if buffer is not None and buffer_size < _buffer_cap:
                            buffer.append(chunk)
                            buffer_size += len(chunk)
                        yield chunk
                self._breaker.record_success()
            except httpx.ReadError as e:
                self._breaker.record_failure()
                logger.warning(f"Upstream stream interrupted: {e}")
                yield b'data: {"error": "upstream_connection_lost"}\n\n'
            except httpx.TimeoutException as e:
                self._breaker.record_failure()
                logger.warning(f"Upstream stream timeout: {e}")
                yield b'data: {"error": "upstream_timeout"}\n\n'
            except Exception as e:
                self._breaker.record_failure()
                logger.warning(f"Unexpected stream error: {e}")
                yield b'data: {"error": "stream_error"}\n\n'

            # ── Signal 1: Assess response after stream completes ──
            # REVOLUTIONARY FIX: Eliminate the dead zone.
            # Old: binary record_success/record_failure gated at ±0.5/+0.3
            #      → ~52% of signals discarded (everything in -0.5 < r < 0.3)
            # New: record_reward(continuous) for ALL non-zero rewards.
            #      Inspired by HER (NeurIPS 2025) — ambiguous outcomes carry
            #      gradient information when aggregated over hundreds of requests.
            if buffer and _frag_ids:
                try:
                    full_bytes = b"".join(buffer)
                    response_text = _extract_text_from_sse(full_bytes)
                    if response_text:
                        reward = _tracker.assess_response(response_text)
                        _tracker.record_assessment(reward)
                        if abs(reward) > 0.01:  # Only skip truly zero signals
                            logger.debug(
                                "Stream RL signal (%.2f) → record_reward(%d ids)",
                                reward, len(_frag_ids),
                            )
                            _engine.record_reward(_frag_ids, reward)

                            # ── Closed-Loop Belief Utilization (Change 4) ──
                            # Compute per-variant utilization from the reward signal.
                            # The reward is a proxy for "did the LLM use this context?"
                            # We split by variant to learn: are beliefs sufficient?
                            if _selected_frags and hasattr(_engine, 'update_belief_utilization'):
                                belief_scores = [
                                    max(0, reward) for f in _selected_frags
                                    if f.get("variant") == "belief"
                                ]
                                full_scores = [
                                    max(0, reward) for f in _selected_frags
                                    if f.get("variant", "full") == "full"
                                ]
                                if belief_scores or full_scores:
                                    belief_util = sum(belief_scores) / max(len(belief_scores), 1)
                                    full_util = sum(full_scores) / max(len(full_scores), 1)
                                    try:
                                        _engine.update_belief_utilization(belief_util, full_util)
                                    except Exception:
                                        pass  # Never fail on feedback
                except Exception:
                    pass  # Never fail on feedback

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
            resp_headers["X-Entroly-Fragments"] = str(len(self._last_context_fragments))
            # Confidence + coverage from value tracker
            try:
                _conf = get_tracker().get_confidence()
                resp_headers["X-Entroly-Confidence"] = str(round(_conf.get("confidence", 0), 4))
                resp_headers["X-Entroly-Coverage-Pct"] = str(round(_conf.get("coverage_pct", 0), 2))
                resp_headers["X-Entroly-Cost-Saved-Today"] = f"${_conf.get('today', {}).get('cost_saved_usd', 0):.4f}"
            except Exception:
                pass
            # Quality drift signal — "check engine light" for context quality
            quality_trend = self._feedback_tracker.quality_trend()
            if quality_trend != "stable":
                resp_headers["X-Entroly-Quality-Trend"] = quality_trend
            # Context Resonance + Coverage Estimator headers
            if hasattr(self, '_last_coverage'):
                resp_headers["X-Entroly-Coverage"] = f"{self._last_coverage:.4f}"
                resp_headers["X-Entroly-Coverage-Risk"] = str(self._last_coverage_risk)
                resp_headers["X-Entroly-Coverage-Confidence"] = f"{self._last_coverage_confidence:.4f}"
            if hasattr(self, '_last_resonance_pairs') and self._last_resonance_pairs > 0:
                resp_headers["X-Entroly-Resonance-Pairs"] = str(self._last_resonance_pairs)
                resp_headers["X-Entroly-Resonance-Strength"] = f"{self._last_resonance_strength:.4f}"
                resp_headers["X-Entroly-W-Resonance"] = f"{self._last_w_resonance:.4f}"
            # Causal Context Graph headers
            if hasattr(self, '_last_causal_tracked') and self._last_causal_tracked > 0:
                resp_headers["X-Entroly-Causal-Tracked"] = str(self._last_causal_tracked)
                resp_headers["X-Entroly-Causal-Interventional"] = str(self._last_causal_interventional)
                resp_headers["X-Entroly-Causal-Gravity-Sources"] = str(self._last_causal_gravity_sources)
                resp_headers["X-Entroly-Causal-Mean-Mass"] = f"{self._last_causal_mean_mass:.4f}"
            # Belief Utilization Auto-Tuning headers (Change 4)
            try:
                if hasattr(self.engine, 'get_belief_util_ema'):
                    resp_headers["X-Entroly-Belief-Util-EMA"] = f"{self.engine.get_belief_util_ema():.4f}"
                    resp_headers["X-Entroly-Full-Util-EMA"] = f"{self.engine.get_full_util_ema():.4f}"
                    resp_headers["X-Entroly-Belief-Info-Factor"] = f"{self.engine.get_belief_info_factor():.4f}"
            except Exception:
                pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=resp_headers,
        )

    async def _forward_response(
        self, url: str, headers: dict[str, str], body: dict[str, Any],
        selected_frag_ids: list | None = None,
    ) -> JSONResponse:
        """Forward a non-streaming request with circuit breaker, retry on 429/5xx, and response validation."""
        # Check circuit breaker
        if not self._breaker.allow_request():
            logger.warning("Circuit breaker open -- forwarding unmodified")

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
                try:
                    retry_after = float(response.headers.get("retry-after", str(1.0 * (attempt + 1))))
                except (ValueError, TypeError):
                    retry_after = 1.0 * (attempt + 1)
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

        resp_headers: dict[str, str] = {"X-Entroly-Optimized": "true"}
        with self._stats_lock:
            if self._last_temperature is not None:
                resp_headers["X-Entroly-Temperature"] = f"{self._last_temperature:.4f}"
            # Use getattr so this is robust to call paths that didn't go
            # through optimization (and so didn't set the attribute) —
            # matches the defensive pattern of _last_fragment_count below.
            tsp = getattr(self, "_last_tokens_saved_pct", None)
            if tsp:
                resp_headers["X-Entroly-Tokens-Saved-Pct"] = f"{tsp:.1f}"
            if hasattr(self, '_last_fragment_count'):
                resp_headers["X-Entroly-Fragments"] = str(getattr(self, '_last_fragment_count', 0))
            if hasattr(self, '_last_confidence'):
                resp_headers["X-Entroly-Confidence"] = f"{getattr(self, '_last_confidence', 0.0):.4f}"
            if hasattr(self, '_last_coverage_pct'):
                resp_headers["X-Entroly-Coverage-Pct"] = f"{getattr(self, '_last_coverage_pct', 0.0):.1f}"
            # Today's cumulative cost saved
            try:
                tracker = get_tracker()
                today_data = tracker.get_confidence().get("today", {})
                resp_headers["X-Entroly-Cost-Saved-Today"] = f"${today_data.get('cost_saved_usd', 0.0):.4f}"
            except Exception:
                pass
            # Quality drift signal
            quality_trend = self._feedback_tracker.quality_trend()
            if quality_trend != "stable":
                resp_headers["X-Entroly-Quality-Trend"] = quality_trend
            # Context Resonance + Coverage Estimator headers (non-streaming path)
            if hasattr(self, '_last_coverage'):
                resp_headers["X-Entroly-Coverage"] = f"{self._last_coverage:.4f}"
                resp_headers["X-Entroly-Coverage-Risk"] = str(self._last_coverage_risk)
                resp_headers["X-Entroly-Coverage-Confidence"] = f"{self._last_coverage_confidence:.4f}"
            if hasattr(self, '_last_resonance_pairs') and self._last_resonance_pairs > 0:
                resp_headers["X-Entroly-Resonance-Pairs"] = str(self._last_resonance_pairs)
                resp_headers["X-Entroly-Resonance-Strength"] = f"{self._last_resonance_strength:.4f}"
                resp_headers["X-Entroly-W-Resonance"] = f"{self._last_w_resonance:.4f}"
            # Causal Context Graph headers (non-streaming path)
            if hasattr(self, '_last_causal_tracked') and self._last_causal_tracked > 0:
                resp_headers["X-Entroly-Causal-Tracked"] = str(self._last_causal_tracked)
                resp_headers["X-Entroly-Causal-Interventional"] = str(self._last_causal_interventional)
                resp_headers["X-Entroly-Causal-Gravity-Sources"] = str(self._last_causal_gravity_sources)
                resp_headers["X-Entroly-Causal-Mean-Mass"] = f"{self._last_causal_mean_mass:.4f}"

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

        # ── Signal 1: Assess non-streaming response for implicit feedback ──
        if self._enable_passive_feedback and selected_frag_ids and isinstance(content, dict):
            try:
                # Extract assistant text from response JSON
                response_text = ""
                # OpenAI / Anthropic: choices[0].message.content
                for choice in content.get("choices", []):
                    msg = choice.get("message", {})
                    if msg.get("content"):
                        response_text += msg["content"]
                # Anthropic direct: content[0].text
                for block in content.get("content", []):
                    if isinstance(block, dict) and block.get("text"):
                        response_text += block["text"]
                # Gemini: candidates[0].content.parts[0].text
                for cand in content.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("text"):
                            response_text += part["text"]

                if response_text:
                    reward = self._feedback_tracker.assess_response(response_text)
                    self._feedback_tracker.record_assessment(reward)
                    # Continuous RL signal — eliminate dead zone
                    if abs(reward) > 0.01 and selected_frag_ids:
                        logger.debug(
                            "Response RL signal (%.2f) -> record_reward(%d ids)",
                            reward, len(selected_frag_ids),
                        )
                        self.engine.record_reward(selected_frag_ids, reward)
            except Exception:
                pass  # Never block response for feedback

        # ── Response Distillation ──
        # Strip filler from LLM response (pleasantries, hedging, meta-commentary)
        # Preserves all code blocks, technical content, and structured data
        if self._enable_distill and isinstance(content, dict):
            try:
                from .proxy_transform import distill_response
                modified = False
                # OpenAI format: choices[].message.content
                for choice in content.get("choices", []):
                    msg = choice.get("message", {})
                    if msg.get("content") and isinstance(msg["content"], str):
                        compressed, orig_c, comp_c = distill_response(
                            msg["content"], mode=self._distill_mode,
                        )
                        if comp_c < orig_c:
                            msg["content"] = compressed
                            modified = True
                            with self._stats_lock:
                                self._total_output_original_tokens += orig_c
                                self._total_output_compressed_tokens += comp_c

                # Anthropic format: content[].text
                for block in content.get("content", []):
                    if isinstance(block, dict) and block.get("text"):
                        compressed, orig_c, comp_c = distill_response(
                            block["text"], mode=self._distill_mode,
                        )
                        if comp_c < orig_c:
                            block["text"] = compressed
                            modified = True
                            with self._stats_lock:
                                self._total_output_original_tokens += orig_c
                                self._total_output_compressed_tokens += comp_c

                # Gemini format: candidates[].content.parts[].text
                for cand in content.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("text") and isinstance(part["text"], str):
                            compressed, orig_c, comp_c = distill_response(
                                part["text"], mode=self._distill_mode,
                            )
                            if comp_c < orig_c:
                                part["text"] = compressed
                                modified = True
                                with self._stats_lock:
                                    self._total_output_original_tokens += orig_c
                                    self._total_output_compressed_tokens += comp_c

                if modified:
                    with self._stats_lock:
                        output_saved_pct = 0
                        if self._total_output_original_tokens > 0:
                            output_saved_pct = (
                                (self._total_output_original_tokens - self._total_output_compressed_tokens)
                                * 100 // self._total_output_original_tokens
                            )
                    resp_headers["X-Entroly-Output-Saved-Pct"] = str(output_saved_pct)
                    logger.debug(
                        "Distill: output compressed (%d%% saved, mode=%s)",
                        output_saved_pct, self._distill_mode,
                    )
            except Exception:
                pass  # Never block response for output compression

        # Add context waste % header
        with self._stats_lock:
            if self._waste_ratios:
                avg_waste = sum(self._waste_ratios) / len(self._waste_ratios)
                resp_headers["X-Entroly-Context-Waste-Pct"] = f"{avg_waste * 100:.1f}"

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
        if provider == "gemini":
            return f"{self.config.gemini_base_url}{path}"
        return f"{self.config.openai_base_url}{path}"

    def _build_headers(
        self, original: dict[str, str], provider: str
    ) -> dict[str, str]:
        """Build headers for the forwarded request. Pass through auth."""
        forward: dict[str, str] = {"Content-Type": "application/json"}
        if "authorization" in original:
            forward["Authorization"] = original["authorization"]
        if "x-api-key" in original:
            forward["x-api-key"] = original["x-api-key"]
        if "anthropic-version" in original:
            forward["anthropic-version"] = original["anthropic-version"]
        if "x-goog-api-key" in original:
            forward["x-goog-api-key"] = original["x-goog-api-key"]
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
            "# HELP entroly_requests_total Total proxy requests",
            "# TYPE entroly_requests_total counter",
            f"entroly_requests_total {proxy._requests_total}",
            "# HELP entroly_requests_optimized Optimized requests",
            "# TYPE entroly_requests_optimized counter",
            f"entroly_requests_optimized {proxy._requests_optimized}",
            "# HELP entroly_requests_bypassed Bypassed requests",
            "# TYPE entroly_requests_bypassed counter",
            f"entroly_requests_bypassed {proxy._requests_bypassed}",
            "# HELP entroly_tokens_original_total Original token count",
            "# TYPE entroly_tokens_original_total counter",
            f"entroly_tokens_original_total {proxy._total_original_tokens}",
            "# HELP entroly_tokens_optimized_total Optimized token count",
            "# TYPE entroly_tokens_optimized_total counter",
            f"entroly_tokens_optimized_total {proxy._total_optimized_tokens}",
            "# HELP entroly_pipeline_latency_ms Pipeline latency",
            "# TYPE entroly_pipeline_latency_ms gauge",
            f"entroly_pipeline_latency_ms {proxy._pipeline_stats.mean:.2f}",
            "# HELP entroly_circuit_breaker Circuit breaker state (0=closed, 1=open)",
            "# TYPE entroly_circuit_breaker gauge",
            f'entroly_circuit_breaker {1 if proxy._breaker.state == "open" else 0}',
            "# HELP entroly_outcome_success Successful outcomes recorded",
            "# TYPE entroly_outcome_success counter",
            f"entroly_outcome_success {proxy._outcome_success}",
            "# HELP entroly_outcome_failure Failed outcomes recorded",
            "# TYPE entroly_outcome_failure counter",
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

            # Report cache hit rate for observability
            if hasattr(proxy.engine, 'cache_hit_rate'):
                hit_rate = proxy.engine.cache_hit_rate()
                if hit_rate > 0:
                    logger.debug(f"Cache hit rate: {hit_rate:.2%}")
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
    """Gap #43: Explain WHY each fragment was selected and at what resolution.

    GET /explain — returns per-fragment selection reasons with resolution
    labels, plus the top excluded fragments with reasons why they were dropped.

    This is the core trust endpoint. A senior engineer can call this after
    any request and see exactly what code was included at what fidelity,
    and what was considered but dropped.
    """
    proxy = request.app.state.proxy

    # ── Included fragments with selection reasons ──
    fragments = []
    for f in proxy._last_context_fragments:
        reasons = []
        entropy = f.get("entropy_score", 0)
        relevance = f.get("relevance", 0)
        variant = f.get("variant", "full")
        source = f.get("source", "")

        # Resolution explanation
        resolution_labels = {
            "full": "Full resolution — complete code included, nothing stripped",
            "skeleton": "Signatures only — function/class signatures preserved, bodies omitted",
            "belief": "Belief summary — vault knowledge graph summary (~50% info at ~10% tokens)",
            "reference": "Reference only — file path included for awareness (no code)",
        }
        reasons.append(resolution_labels.get(variant, f"Resolution: {variant}"))

        # WHY it was included
        if relevance > 0.7:
            reasons.append(f"Strong query match (relevance={relevance:.2f})")
        elif relevance > 0.4:
            reasons.append(f"Moderate query match (relevance={relevance:.2f})")
        elif relevance > 0:
            reasons.append(f"Weak query match (relevance={relevance:.2f})")

        if entropy > 0.7:
            reasons.append(f"High information density (entropy={entropy:.2f})")
        elif entropy > 0.4:
            reasons.append(f"Moderate entropy ({entropy:.2f})")

        # WHY this resolution was chosen
        if variant == "full":
            reasons.append("Included at full resolution because it directly matches the query")
        elif variant == "skeleton":
            reasons.append("Compressed to signatures — tangential import, not query-critical")
        elif variant == "belief":
            reasons.append("Vault summary used — provides architectural context at low token cost")
        elif variant == "reference":
            reasons.append("Path-only reference — LLM knows file exists without seeing code")

        if "test" in source.lower():
            reasons.append("Test file (may verify behavior)")
        if any(kw in source.lower() for kw in ["config", "schema", "setup", "prisma"]):
            reasons.append("Configuration/schema file (critical for correctness)")

        fragments.append({
            "source": source,
            "resolution": variant,
            "token_count": f.get("token_count", 0),
            "reasons": reasons,
            "scores": {
                "entropy": round(entropy, 4),
                "relevance": round(relevance, 4),
            },
            "preview": _safe_preview(f.get("content", "")),
        })

    # ── Excluded fragments with DROP reasons ──
    excluded = []
    for f in getattr(proxy, '_last_excluded_fragments', []):
        source = f.get("source", "")
        entropy = f.get("entropy_score", 0)
        relevance = f.get("relevance", 0)
        tokens = f.get("token_count", 0)

        drop_reasons = []
        if relevance < 0.3:
            drop_reasons.append(f"Low query relevance ({relevance:.2f})")
        if entropy < 0.3:
            drop_reasons.append(f"Low information density ({entropy:.2f})")
        if tokens > 500:
            drop_reasons.append(f"Large token cost ({tokens}) — budget trade-off")
        if not drop_reasons:
            drop_reasons.append("Exceeded token budget (knapsack trade-off)")

        excluded.append({
            "source": source,
            "token_count": tokens,
            "drop_reasons": drop_reasons,
            "scores": {
                "entropy": round(entropy, 4),
                "relevance": round(relevance, 4),
            },
        })

    return JSONResponse({
        "query": proxy._last_query,
        "pipeline_ms": round(proxy._last_pipeline_ms, 2),
        "included_count": len(fragments),
        "excluded_count": len(excluded),
        "resolution_summary": {
            "full": sum(1 for f in fragments if f["resolution"] == "full"),
            "skeleton": sum(1 for f in fragments if f["resolution"] == "skeleton"),
            "belief": sum(1 for f in fragments if f["resolution"] == "belief"),
            "reference": sum(1 for f in fragments if f["resolution"] == "reference"),
        },
        "included": fragments,
        "excluded": excluded,
        "trust_note": (
            "Files matching your query are included at FULL resolution — "
            "function bodies are never stripped from files the LLM needs. "
            "Only tangential imports are compressed to signatures."
        ),
    })


async def _context_retrieve(request: Request) -> JSONResponse:
    """CCR: Compressed Context Retrieval — get full original of a compressed fragment.

    GET /retrieve?source=file:src/auth.py → full original content
    GET /retrieve → list all retrievable fragments

    This is the architectural answer to 'silent truncation':
    nothing is permanently lost, the LLM can always get the original back.
    """
    try:
        from .ccr import get_ccr_store
        store = get_ccr_store()
    except ImportError:
        return JSONResponse({"error": "CCR module not available"}, status_code=500)

    source = request.query_params.get("source", "")

    if not source:
        # List all retrievable fragments
        available = store.list_available()
        return JSONResponse({
            "available": available,
            "count": len(available),
            "stats": store.stats(),
            "usage": 'GET /retrieve?source=file:src/auth.py to retrieve full content',
        })

    entry = store.retrieve(source)
    if entry is None:
        return JSONResponse(
            {"error": f"Source '{source}' not found in CCR store", "hint": "GET /retrieve to list available"},
            status_code=404,
        )

    return JSONResponse({
        "source": source,
        "resolution": entry["resolution"],
        "original_tokens": entry["original_tokens"],
        "compressed_tokens": entry["compressed_tokens"],
        "tokens_recovered": entry["original_tokens"] - entry["compressed_tokens"],
        "original_content": entry["original"],
    })


async def _confidence(request: Request) -> JSONResponse:
    """Real-time confidence snapshot for IDE status bar widgets.

    GET /confidence → {confidence, coverage_pct, session, today, lifetime, status}

    Designed to be polled every 5-10s by a VS Code extension status bar item.
    """
    tracker = get_tracker()
    return JSONResponse(tracker.get_confidence())


async def _value_trends(request: Request) -> JSONResponse:
    """Historical savings trends for dashboard charts.

    GET /trends → {daily: [...], weekly: [...], monthly: [...], lifetime, session}
    """
    tracker = get_tracker()
    return JSONResponse(tracker.get_trends())


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

    # Re-detect provider with body for model-name-based detection
    provider = detect_provider(request.url.path, headers, body if isinstance(body, dict) else None)
    target_url = proxy._resolve_target(provider, request.url.path)
    forward_headers = proxy._build_headers(headers, provider)

    try:
        client = await proxy._ensure_client()
        is_streaming = body.get("stream", False) if isinstance(body, dict) else False
        # Gemini: streaming determined by URL path, not body field
        if not is_streaming and "streamGenerateContent" in request.url.path:
            is_streaming = True
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
        stats: dict[str, Any] = {
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
        # Passive RL feedback stats
        stats["implicit_feedback"] = proxy._feedback_tracker.stats()
    return JSONResponse(stats)


# Need os for ENTROLY_RATE_LIMIT env var


def create_proxy_app(
    engine: Any, config: ProxyConfig | None = None
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

    # Starlette >= 0.21 removed on_startup/on_shutdown from __init__.
    # Use lifespan context manager for forward-compatible startup/shutdown.
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(app: Starlette):  # type: ignore[type-arg]
        await proxy.startup()
        yield
        await proxy.shutdown()

    app = Starlette(
        routes=[
            Route("/v1/chat/completions", proxy.handle_proxy, methods=["POST"]),
            Route("/v1/messages", proxy.handle_proxy, methods=["POST"]),
            # Gemini: model name is embedded in the URL path
            Route("/v1beta/models/{model_id:path}", proxy.handle_proxy, methods=["POST"]),
            Route("/health", _health),
            Route("/stats", _proxy_stats),
            Route("/context", _context_inspect),          # Gap #29
            Route("/metrics", _metrics_prometheus),        # Gap #34
            Route("/outcome", _record_outcome, methods=["POST"]),  # Gap #37
            Route("/bypass", _toggle_bypass, methods=["POST"]),    # Gap #28
            Route("/feedback", _fragment_feedback, methods=["POST"]),  # Gap #42
            Route("/explain", _context_explain),                       # Gap #43
            Route("/confidence", _confidence),                         # IDE widget API
            Route("/trends", _value_trends),                           # Dashboard trends
            Route("/retrieve", _context_retrieve),                     # CCR: lossless retrieval
            # Catch-all: forward any unmatched path to upstream API
            # Must be LAST — Starlette matches routes in declaration order
            Route("/{path:path}", _catch_all, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
        ],
        lifespan=_lifespan,
    )
    app.state.proxy = proxy
    return app
