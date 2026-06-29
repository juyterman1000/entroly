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
import copy
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .adaptive_budget import AdaptiveBudgetModel, extract_features
from .context_scaffold import generate_scaffold
from .proxy_config import ProxyConfig, context_window_for_model, provider_capability
from .proxy_transform import (
    compute_dynamic_budget,
    compute_token_budget,
    detect_provider,
    extract_model,
    extract_user_message,
    format_context_block,
    format_hierarchical_context,
    inject_context_anthropic,
    inject_context_gemini,
    inject_context_openai,
    inject_context_responses,
    strip_anthropic_unsupported_params,
)
from .cache_aligner import CacheAligner
from .control_plane import (
    ControlAudit,
    ControlPlaneDecision,
    audit_request_transform,
    plan_request,
    stable_request_fingerprint,
)
from .provider_policy import GatewayRedactionPolicy
from .value_tracker import get_tracker

logger = logging.getLogger("entroly.proxy")

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}

_COMMON_PROVIDER_HEADERS = {
    "accept",
    "accept-language",
    "content-type",
    "traceparent",
    "tracestate",
    "baggage",
    "user-agent",
}

_CERT_ENV_VARS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS")

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


def _content_to_text(content: Any) -> str:
    """Best-effort text extraction from provider message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _content_to_text(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "input"):
            value = content.get(key)
            if isinstance(value, (str, list, dict)):
                text = _content_to_text(value)
                if text:
                    return text
        parts = content.get("parts")
        if isinstance(parts, list):
            return _content_to_text(parts)
        return ""
    return str(content)


def _entroly_tags_to_headers(tags: dict[str, str]) -> dict[str, str]:
    """Convert safe control-plane tags to bounded response headers."""
    headers: dict[str, str] = {}
    for key, value in tags.items():
        if not key.startswith("entroly_"):
            continue
        name = "X-Entroly-" + key.removeprefix("entroly_").replace("_", "-").title()
        safe = str(value).replace("\r", " ").replace("\n", " ").strip()
        headers[name] = safe[:512]
    return headers


def _estimate_message_tokens(messages: Any) -> int:
    """Rough token estimate for OpenAI/Anthropic message arrays."""
    if not isinstance(messages, list):
        return 0
    words = 0
    for msg in messages:
        if isinstance(msg, dict):
            words += len(_content_to_text(msg.get("content", "")).split())
    return words * 4 // 3


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


def _parse_retry_after(raw: str | None) -> float | None:
    """Parse the upstream Retry-After header per RFC 7231 §7.1.3.

    Returns the cooldown in seconds, or None if the header is absent or
    unparseable. Accepts both forms permitted by the spec:
      1. Integer seconds: ``Retry-After: 30``
      2. HTTP-date:       ``Retry-After: Wed, 21 Oct 2026 07:28:00 GMT``

    Negative or zero values are normalized to 0 (already-elapsed cooldown).
    Returning None signals to the caller "no upstream signal — fall back
    to client-side policy" rather than guessing a value.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    # Form 1: delta-seconds.
    try:
        secs = float(raw)
        return max(0.0, secs)
    except ValueError:
        pass
    # Form 2: HTTP-date. Use stdlib parser; both RFC 1123 and obsolete
    # RFC 850 formats are supported by parsedate_to_datetime.
    try:
        from email.utils import parsedate_to_datetime
        target = parsedate_to_datetime(raw)
        if target is None:
            return None
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc) if target.tzinfo else datetime.utcnow()
        delta = (target - now).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError, OverflowError):
        return None


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
    total_chars = sum(
        len(_content_to_text(m.get("content", "")))
        for m in messages
        if isinstance(m, dict)
    )
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
            raw_content = msg.get("content", "")
            content = _content_to_text(raw_content)
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
                if not isinstance(content, str):
                    compressed.append(msg)
                    continue
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

        # Rust owns the classification math; Python keeps only per-client
        # trajectory state and feedback side effects.
        try:
            from entroly_core import py_classify_query_transition

            transition = py_classify_query_transition(
                prev_hash,
                query_text,
                time_delta,
                self._REPHRASE_TIME_WINDOW_S,
                self._REPHRASE_SIMILARITY_THRESHOLD,
                self._TOPIC_CHANGE_THRESHOLD,
            )
            status = transition.get("status")
        except Exception:
            # Fallback for older wheels: same thresholds, same Hamming math.
            xor = query_hash ^ prev_hash
            hamming = bin(xor).count("1")
            similarity = 1.0 - (hamming / 64.0)
            if similarity > self._REPHRASE_SIMILARITY_THRESHOLD:
                status = "rephrase"
            elif similarity < self._TOPIC_CHANGE_THRESHOLD:
                status = "topic_change"
            else:
                status = "ambiguous"

        if status == "rephrase":
            with self._lock:
                self._rephrase_detections += 1
            return ("rephrase", prev_ids)

        if status == "topic_change":
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


def _extract_logprobs_from_sse(
    raw_bytes: bytes,
) -> tuple[list[float], list[str]]:
    """Extract per-token logprobs + token text from SSE stream bytes.

    Grounded in HALT (arXiv:2602.02888) and EPR research (2025-2026):
    logprobs from a single generation pass contain direct uncertainty
    information at zero extra API cost.

    Handles OpenAI format: choices[0].logprobs.content[i].{logprob, token}

    Returns:
        (logprobs, token_texts) — aligned lists.
        Empty lists if logprobs not present in the stream.
    """
    logprobs: list[float] = []
    token_texts: list[str] = []
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
            # OpenAI streaming format:
            # choices[0].logprobs.content[i].{token, logprob}
            for choice in data.get("choices", []):
                lp_obj = choice.get("logprobs")
                if not lp_obj or not isinstance(lp_obj, dict):
                    continue
                content = lp_obj.get("content")
                if not content or not isinstance(content, list):
                    continue
                for item in content:
                    if isinstance(item, dict):
                        tok = item.get("token", "")
                        lp = item.get("logprob")
                        if lp is not None and tok:
                            logprobs.append(float(lp))
                            token_texts.append(str(tok))
    except Exception:
        pass
    return logprobs, token_texts



def _looks_like_structured_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        json.loads(stripped)
        return True
    except Exception:
        return False


def _host_without_port(value: str | None) -> str:
    if not value:
        return ""
    parsed = value.strip()
    if parsed.startswith("[") and "]" in parsed:
        return parsed[1:parsed.index("]")]
    if parsed.count(":") == 1:
        return parsed.rsplit(":", 1)[0]
    return parsed


def _port_from_host(value: str | None) -> int | None:
    if not value:
        return None
    parsed = value.strip()
    if parsed.startswith("[") and "]" in parsed:
        suffix = parsed[parsed.index("]") + 1:]
        return int(suffix[1:]) if suffix.startswith(":") and suffix[1:].isdigit() else None
    if parsed.count(":") == 1:
        port = parsed.rsplit(":", 1)[1]
        return int(port) if port.isdigit() else None
    return None


def _is_loopback_host(value: str | None) -> bool:
    host = _host_without_port(value).lower()
    if host in {"localhost", "test", "testserver"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _origin_matches_sidecar(request: Request, header_name: str) -> bool:
    origin = request.headers.get(header_name)
    if not origin:
        return True
    parsed = urlparse(origin)
    if not _is_loopback_host(parsed.hostname):
        return False
    request_host = request.headers.get("host", "")
    if not _is_loopback_host(request_host):
        return False
    request_port = _port_from_host(request_host) or request.url.port
    origin_port = parsed.port
    return request_port == origin_port


def _is_trusted_sidecar_request(request: Request) -> bool:
    client_host = request.client.host if request.client else ""
    if client_host and not _is_loopback_host(client_host):
        return False
    if not client_host and not _is_loopback_host(request.headers.get("host")):
        return False
    return (
        _origin_matches_sidecar(request, "origin")
        and _origin_matches_sidecar(request, "referer")
    )


def _sidecar_guard(handler):
    async def _guarded(request: Request):
        if not _is_trusted_sidecar_request(request):
            return JSONResponse(
                {
                    "error": "sidecar_forbidden",
                    "detail": "Entroly sidecar endpoints only accept local same-origin requests.",
                },
                status_code=403,
            )
        return await handler(request)

    return _guarded


def _resolve_ca_bundle_from_env() -> str | None:
    for name in _CERT_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _http_client_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
        "follow_redirects": True,
        "trust_env": True,
    }
    ca_bundle = _resolve_ca_bundle_from_env()
    if ca_bundle:
        kwargs["verify"] = ca_bundle
    return kwargs


async def _bytes_iter(data: bytes):
    yield data


# ── Proxy ────────────────────────────────────────────────────────────────


class PromptCompilerProxy:
    """HTTP reverse proxy that optimizes every LLM request with entroly."""

    def __init__(self, engine: Any, config: ProxyConfig | None = None):
        self.engine = engine
        self.config = config or ProxyConfig()
        self._client: httpx.AsyncClient | None = None

        # ── System 1 ↔ System 2 coupling (opt-in via ENTROLY_VAULT_COUPLING=1) ──
        # When enabled, the proxy reads verified beliefs from the vault on each
        # request, injects them as engine fragments, and feeds outcomes back as
        # Bayesian updates. See entroly/coupling.py.
        self._vault: Any = None
        self._last_injected_claim_ids: list[str] = []
        try:
            from . import coupling
            if coupling.is_enabled():
                from .vault import VaultManager, VaultConfig
                self._vault = VaultManager(VaultConfig())
                logger.info("Vault coupling enabled (ENTROLY_VAULT_COUPLING=1)")
        except Exception as e:
            logger.debug("Vault coupling unavailable: %s", e)
            self._vault = None

        # ── EGSC persistent-cache warm-start snapshot ─────────────────────
        # Capture cache state at proxy startup so we can report how many
        # entries were restored from disk (cross-session reuse) vs admitted
        # in the current session. This surfaces a feature that already ships
        # (EgscCache → CacheSnapshot → engine state → ~/.entroly/checkpoints/)
        # but had no visibility on the wire. See README "Persistent
        # Cross-Session Cache" for the chain.
        self._cache_warm_restored: int = 0
        self._cache_warm_started_at: float = time.time()
        try:
            _stats0 = self.engine.stats() if hasattr(self.engine, "stats") else None
            if isinstance(_stats0, dict):
                self._cache_warm_restored = int(
                    _stats0.get("cache", {}).get("entries", 0)
                )
        except Exception:
            pass

        # Thread-safe stats
        self._stats_lock = threading.Lock()
        self._requests_total: int = 0
        self._requests_optimized: int = 0
        self._requests_bypassed: int = 0
        self._requests_subscription_blocked: int = 0
        self._total_original_tokens: int = 0
        self._total_optimized_tokens: int = 0
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

        # ── Cache Aligner ──
        # Stabilizes Entroly's injected CONTEXT block across turns of the same
        # conversation so the provider's prefix/KV cache keeps hitting
        # (Anthropic 90% / OpenAI 50% read discount) on chatty, stable-context
        # sessions — the dominant cost on large repos. Context-only: it does
        # not alter the model, generation params, tools, or the user's messages.
        # Provider terms and data-handling rules still apply to whatever the
        # user sends through their configured provider. Disable with
        # ENTROLY_CACHE_ALIGN=0. Fail-open.
        self._cache_align_enabled = os.environ.get("ENTROLY_CACHE_ALIGN", "1") != "0"
        self._cache_aligner = CacheAligner() if self._cache_align_enabled else None

        # Optional enterprise outbound policy. It is applied to the final JSON
        # payload immediately before transport, including bypass/catch-all paths.
        self._gateway_redaction = GatewayRedactionPolicy(
            enabled=os.environ.get("ENTROLY_GATEWAY_REDACTION", "0").lower()
            in {"1", "true", "yes", "on"}
        )

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

        # ACB: Adaptive Compression Budget — per-query learned budget predictor
        self._acb = AdaptiveBudgetModel()


        # Passive implicit feedback — closes the RL loop without IDE cooperation
        self._feedback_tracker = ImplicitFeedbackTracker()
        self._enable_passive_feedback = (
            os.environ.get("ENTROLY_PASSIVE_FEEDBACK", "1") != "0"
        )

        # RAVS Bayesian Router -- routes cheap tasks to cheaper models only
        # when explicitly enabled and hook-captured event data proves it is safe.
        # Silent model substitution must be opt-in because it can change provider,
        # capability, cost, and data-handling semantics.
        self._ravs_router_enabled = (
            os.environ.get("ENTROLY_RAVS_ROUTER", "0") == "1"
        )
        try:
            from .ravs.router import BayesianRouter
            self._ravs_router = BayesianRouter(
                enabled=self._ravs_router_enabled,
                min_samples=int(os.environ.get("ENTROLY_RAVS_MIN_SAMPLES", "10")),
                ci_threshold=float(os.environ.get("ENTROLY_RAVS_CI_THRESHOLD", "0.80")),
            )
        except Exception as e:
            logger.debug("RAVS router init skipped: %s", e)
            self._ravs_router = None

        # ECE: Epistemic Cascade Engine (RAVS V5) — mathematical
        # uncertainty verification for routed responses. Uses Fisher
        # curvature, adaptive Rényi divergence, and Lyapunov-stable
        # thresholding to detect hallucination without an LLM judge.
        self._ece = None
        self._ece_enabled = (
            os.environ.get("ENTROLY_ECE", "1") != "0"
        )
        if self._ece_enabled:
            try:
                from .ravs.ece import EpistemicCascadeEngine
                self._ece = EpistemicCascadeEngine(
                    curvature_threshold=float(
                        os.environ.get("ENTROLY_ECE_CURVATURE_THRESHOLD", "0.4")
                    ),
                    enable_lyapunov=True,
                )
                logger.info("ECE v6 epistemic cascade engine initialized")
            except Exception as e:
                logger.debug("ECE init skipped: %s", e)
                self._ece = None

        # WITNESS: proof-carrying factuality gateway. Disabled by default
        # unless ENTROLY_WITNESS_MODE or CLI --witness is set.
        # P1: WITNESS defaults to audit mode — zero visible change,
        # but every response gets certificate headers + observability.
        # Opt OUT with ENTROLY_WITNESS=0 or witness_mode=off.
        self._witness_mode = (getattr(self.config, "witness_mode", "") or "").lower()
        if not self._witness_mode or self._witness_mode == "off":
            # Default: audit unless explicitly disabled
            if os.environ.get("ENTROLY_WITNESS", "1") == "0":
                self._witness_mode = "off"
            else:
                self._witness_mode = "audit"
        self._witness_enabled = self._witness_mode != "off"
        self._witness_use_nli = bool(getattr(self.config, "witness_use_nli", False))
        self._witness_profile = (getattr(self.config, "witness_profile", "auto") or "auto").lower()
        self._witness_analyzer: Any = None
        if self._witness_enabled:
            try:
                from .witness import WitnessAnalyzer
                self._witness_analyzer = WitnessAnalyzer(
                    use_nli=self._witness_use_nli,
                    profile=self._witness_profile,
                )
                logger.info(
                    "WITNESS enabled (mode=%s, profile=%s, nli=%s)",
                    self._witness_mode,
                    self._witness_profile,
                    self._witness_use_nli,
                )
            except Exception as e:
                logger.warning("WITNESS init skipped: %s", e)
                self._witness_enabled = False
        self._witness_total = 0
        self._witness_flagged = 0
        self._witness_rewritten = 0
        self._witness_last: dict[str, Any] | None = None
        self._witness_embed = bool(getattr(self.config, "witness_embed", False))
        self._witness_certificates: collections.OrderedDict[str, dict[str, Any]] = collections.OrderedDict()
        self._witness_store_max = int(os.environ.get("ENTROLY_WITNESS_STORE_MAX", "500"))
        self._witness_feedback: dict[str, int] = collections.Counter()

        # If verification rejects an answer produced from compressed context,
        # retrieve exact CCR originals and retry once with a bounded expansion.
        self._auto_recovery_enabled = os.environ.get("ENTROLY_AUTO_RECOVERY", "1") != "0"
        self._auto_recovery_max_fragments = int(
            os.environ.get("ENTROLY_AUTO_RECOVERY_MAX_FRAGMENTS", "6")
        )
        self._auto_recovery_max_candidates = int(
            os.environ.get("ENTROLY_AUTO_RECOVERY_MAX_CANDIDATES", "8")
        )
        self._auto_recovery_max_tokens = int(
            os.environ.get("ENTROLY_AUTO_RECOVERY_MAX_TOKENS", "12000")
        )
        self._auto_recovery_attempted = 0
        self._auto_recovery_succeeded = 0
        self._auto_recovery_failed = 0
        self._auto_recovery_last: dict[str, Any] | None = None

        # P2: Conformal cascade observability counters
        # Track post-response WITNESS→ECE→escalation.py decisions
        self._cascade_total = 0
        self._cascade_escalations = 0
        self._cascade_last: dict[str, Any] | None = None

        # ── EICV: Evidence-Invariant Causal Verification (auto-suppression) ─
        # Runs AFTER WITNESS as the final hallucination guard. Deterministic
        # local computation (no neural model, no LLM calls). Defaults to
        # audit mode — zero output change, just emits X-Entroly-EICV-*
        # headers for observability. Set ENTROLY_EICV_MODE to "annotate" or
        # "strict" to enable response rewriting. See benchmarks/results/
        # for accuracy numbers on the test sets we evaluated.
        self._eicv_mode = (
            os.environ.get("ENTROLY_EICV_MODE", "")
            or getattr(self.config, "eicv_mode", "")
            or ""
        ).lower()
        if not self._eicv_mode or self._eicv_mode == "off":
            self._eicv_mode = "off" if os.environ.get("ENTROLY_EICV", "1") == "0" else "audit"
        self._eicv_enabled = self._eicv_mode != "off"
        self._eicv_profile = (
            os.environ.get("ENTROLY_EICV_PROFILE", "")
            or getattr(self.config, "eicv_profile", "rag")
            or "rag"
        ).lower()
        self._eicv_suppressor: Any = None
        if self._eicv_enabled:
            try:
                from .eicv_suppressor import EICVSuppressor
                self._eicv_suppressor = EICVSuppressor(
                    profile=self._eicv_profile,
                    mode=self._eicv_mode,
                )
                logger.info(
                    "EICV enabled (mode=%s, profile=%s) -- auto-suppression on every response",
                    self._eicv_mode, self._eicv_profile,
                )
            except Exception as e:
                logger.warning("EICV init skipped: %s", e)
                self._eicv_enabled = False
        self._eicv_total = 0
        self._eicv_hallucinated_total = 0
        self._eicv_suppressed_total = 0
        self._eicv_last: dict[str, Any] | None = None

        # ── Active Escalation Engine ──────────────────────────────────
        # When ENTROLY_ESCALATION_MODE=active, the proxy will re-issue
        # flagged requests to a stronger model when the 4-signal fusion
        # risk exceeds the escalation threshold. This adds latency
        # (~200-500ms) for escalated requests but catches hallucinations
        # that the original model would miss.
        #
        # Modes:
        #   observe (default) — log escalation decisions, never re-route
        #   active            — actually re-issue to a stronger model
        #   shadow            — re-issue in background, compare (no block)
        #
        # Safety: max_depth=1 prevents recursive escalation chains.
        # The escalated model's response is NEVER re-escalated.
        self._escalation_mode = os.environ.get(
            "ENTROLY_ESCALATION_MODE", "observe"
        ).lower().strip()
        self._escalation_max_depth = 1  # never escalate the escalation
        self._escalation_total = 0
        self._escalation_actual = 0
        self._escalation_saved_tokens = 0
        self._escalation_last: dict[str, Any] | None = None

        # Model escalation ladder: current_model → stronger_model
        # Only populated models trigger escalation. Unknown models
        # stay as-is (fail-open). Each mapping includes approximate
        # cost multiplier for ROI tracking.
        self._escalation_ladder: dict[str, tuple[str, float]] = {
            # OpenAI
            "gpt-4o-mini": ("gpt-4o", 6.0),
            "gpt-4o-mini-2024-07-18": ("gpt-4o", 6.0),
            "gpt-3.5-turbo": ("gpt-4o-mini", 3.0),
            "gpt-3.5-turbo-0125": ("gpt-4o-mini", 3.0),
            # Anthropic
            "claude-3-5-haiku-20241022": ("claude-sonnet-4-20250514", 5.0),
            "claude-3-5-haiku-latest": ("claude-sonnet-4-20250514", 5.0),
            "claude-sonnet-4-20250514": ("claude-opus-4-20250514", 5.0),
            # Gemini
            "gemini-2.0-flash": ("gemini-2.5-pro-preview-05-06", 8.0),
            "gemini-1.5-flash": ("gemini-2.5-pro-preview-05-06", 10.0),
            "gemini-2.0-flash-lite": ("gemini-2.0-flash", 3.0),
        }

        if self._escalation_mode != "observe":
            logger.info(
                "Escalation mode: %s (ladder: %d models)",
                self._escalation_mode,
                len(self._escalation_ladder),
            )

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(**_http_client_kwargs())
        logger.info("Prompt compiler proxy ready")

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Gap #47: Auto-reconnect if client connection dropped (long sessions).

        IDE left open overnight, connection pool stale → recreate client.
        """
        if self._client is None or self._client.is_closed:
            logger.info("Reconnecting HTTP client (previous connection dropped)")
            self._client = httpx.AsyncClient(**_http_client_kwargs())
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

    def _control_token_budget(self, body: dict[str, Any], path: str) -> int | None:
        try:
            model = extract_model(body, path)
            return compute_token_budget(model, self.config)
        except Exception:
            window = getattr(self.config, "context_window", None)
            return int(window) if isinstance(window, int) and window > 0 else None

    def _control_headers(
        self,
        decision: ControlPlaneDecision | None,
        audit: ControlAudit | None = None,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        provider: str = "unknown",
        path: str = "",
        outcome: str = "received",
    ) -> dict[str, str]:
        tags: dict[str, str] = {"entroly_outcome": outcome}
        if decision is not None:
            tags.update(decision.to_tags())
        if audit is not None:
            tags.update(audit.to_tags())
        if body is not None:
            try:
                tags.update(
                    stable_request_fingerprint(
                        body,
                        headers=headers,
                        provider=provider,  # type: ignore[arg-type]
                        path=path,
                    )
                )
            except Exception:
                pass
        return _entroly_tags_to_headers(tags)

    def _subscription_guard(
        self, provider: str, headers: dict
    ) -> JSONResponse | None:
        """Keep Claude Pro/Max subscription logins on their supported setup path.

        A subscription sends a first-party OAuth bearer (``Authorization: Bearer
        sk-ant-oat…``) with **no** ``x-api-key``. The public Anthropic API accepts
        pay-as-you-go API keys, and subscription tokens are intended for
        first-party use — so the supported setups for a subscription are the MCP
        integration (Claude Code stays the client) or a pay-as-you-go API key.
        Entroly detects the subscription bearer up front and returns ONE friendly,
        actionable response pointing there, instead of forwarding a request that
        wouldn't work and isn't the intended path. (Maintainer note: per
        Anthropic's Consumer Terms, subscription OAuth tokens are for first-party
        use only — which is exactly why there is no bypass below.)

        Design (why this is correct, not a heuristic):
          • **Provider-scoped to Anthropic.** For OpenAI/Gemini, ``Authorization:
            Bearer <key>`` IS the valid API key — never block those.
          • **Precise, fail-open.** Blocks ONLY the positively-identified OAuth
            prefix ``sk-ant-oat``. API keys sent as a bearer (``sk-ant-api…``),
            ``x-api-key`` clients, and any unrecognized form are forwarded
            untouched — zero false positives.
          • **Covers every entry path** (wrap, bare ``entroly proxy``, manual
            base-url) and the case the CLI pre-flight misses: a user who has
            ``ANTHROPIC_API_KEY`` exported but whose Claude Code is logged in via
            Pro/Max OAuth.
          • **No bypass switch.** ``sk-ant-oat`` tokens are for first-party use
            only, so the supported setups are MCP or a pay-as-you-go API key —
            Entroly deliberately offers no "proxy a subscription anyway" flag.
        """
        if provider != "anthropic":
            return None
        h = {k.lower(): v for k, v in headers.items()}
        if h.get("x-api-key"):
            return None  # pay-as-you-go API-key client — forwards fine
        auth = h.get("authorization", "") or ""
        if auth[:7].lower() != "bearer ":
            return None
        token = auth[7:].strip()
        if not token.startswith("sk-ant-oat"):
            return None  # API key as bearer / unknown form → forward (fail-open)

        with self._stats_lock:
            self._requests_subscription_blocked += 1
        logger.warning(
            "Blocked a Claude Pro/Max subscription (OAuth) request: the public "
            "Anthropic API rejects first-party tokens. Routed user to MCP."
        )
        return JSONResponse(
            {
                "error": "subscription_not_proxyable",
                "status": 400,
                "detail": (
                    "You're signed in with a Claude Pro/Max subscription. For "
                    "subscriptions, the smoothest Entroly setup is the MCP integration "
                    "— Claude Code stays your client and Entroly adds its tools locally: "
                    "`claude mcp add entroly -- entroly`. Prefer proxy mode? Use a "
                    "pay-as-you-go API key (set ANTHROPIC_API_KEY). Want to see your "
                    "savings first, with no API call? Run `entroly simulate`."
                ),
                "source": "entroly_proxy",
                "remedy": {
                    "mcp": "claude mcp add entroly -- entroly",
                    "simulate": "entroly simulate",
                },
            },
            status_code=400,
        )

    def _apply_outbound_redaction(
        self,
        body: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Apply the final outbound policy and emit value-free audit headers."""
        if not self._gateway_redaction.enabled:
            return body, {}
        redacted, receipt = self._gateway_redaction.redact_value(body)
        if not isinstance(redacted, dict):
            raise TypeError("gateway redaction must preserve the JSON object shape")
        headers = {
            "X-Entroly-Redaction": "changed" if receipt.changed else "clean",
            "X-Entroly-Redaction-Count": str(len(receipt.findings)),
        }
        return redacted, headers

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
            if self._gateway_redaction.enabled:
                return JSONResponse(
                    {
                        "error": "outbound_redaction_requires_json",
                        "detail": "Enterprise redaction cannot inspect a non-JSON payload.",
                    },
                    status_code=415,
                )
            # Not JSON — forward raw (e.g. health checks hitting wrong path)
            return await self._forward_raw(request, body_bytes)

        path = request.url.path
        headers = {k: v for k, v in request.headers.items()}
        provider = detect_provider(path, headers, body)
        # Authoritative subscription-auth guard — fail fast with actionable guidance
        # before any forwarding, instead of a confusing upstream 401/429.
        _sub_block = self._subscription_guard(provider, headers)
        if _sub_block is not None:
            return _sub_block
        control_before = copy.deepcopy(body)
        control_decision: ControlPlaneDecision | None = None
        control_headers: dict[str, str] = {}
        try:
            control_decision = plan_request(
                control_before,
                headers=headers,
                provider=provider,  # type: ignore[arg-type]
                path=path,
                token_budget=self._control_token_budget(control_before, path),
                compression_enabled=not self._bypass,
                cache_alignment_enabled=bool(self._cache_aligner),
            )
            control_headers = self._control_headers(
                control_decision,
                body=control_before,
                headers=headers,
                provider=provider,
                path=path,
                outcome="received",
            )
        except Exception as e:
            logger.debug("Control-plane planning skipped: %s", e)

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
            body["messages"], tool_tokens_saved = compress_tool_messages(
                body["messages"],
                policy=getattr(self.config, "tool_result_policy", "auto"),
                excluded_tools=getattr(self.config, "tool_result_excluded_tools", ""),
            )
            if tool_tokens_saved > 0:
                logger.info(f"Tool output compression: {tool_tokens_saved} tokens saved")

        if "messages" in body and self.config.enable_conversation_compression:
            body["messages"] = compress_conversation_messages(
                body["messages"],
                context_window=getattr(self.config, "context_window", 128_000),
            )

        # Detect if this is a Responses API request (uses `input` instead of `messages`)
        _is_responses_api = "input" in body and "messages" not in body

        # Gap #28: Bypass mode — forward unmodified, no optimization
        if self._bypass:
            body, redaction_headers = self._apply_outbound_redaction(body)
            control_headers.update(redaction_headers)
            with self._stats_lock:
                self._requests_bypassed += 1
            target_url = self._resolve_target(provider, path)
            forward_headers = self._build_headers(headers, provider)
            is_streaming = body.get("stream", False)
            if not is_streaming and "streamGenerateContent" in path:
                is_streaming = True
            bypass_headers = {
                **control_headers,
                "X-Entroly-Optimized": "false",
                "X-Entroly-Outcome": "passthrough",
            }
            if is_streaming:
                return await self._stream_response(
                    target_url,
                    forward_headers,
                    body,
                    provider=provider,
                    extra_headers=bypass_headers,
                )
            return await self._forward_response(
                target_url,
                forward_headers,
                body,
                extra_headers=bypass_headers,
            )

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
        _selected_fragments: list[dict[str, Any]] = []
        _recoverable_fragments: list[dict[str, Any]] = []
        user_message = ""
        witness_context = ""
        request_id = headers.get("x-request-id") or uuid.uuid4().hex[:12]

        # Run the optimization pipeline (synchronous Rust, off the event loop)
        try:
            user_message = extract_user_message(body, provider)
            witness_context = user_message
            if user_message:
                pipeline_result = await asyncio.to_thread(
                    self._run_pipeline, user_message, body, path, request_id
                )
                context_text = pipeline_result["context"]
                pipeline_ms = pipeline_result["elapsed_ms"]
                # Track pipeline latency
                self._pipeline_stats.add(pipeline_ms)

                # Collect fragment IDs for passive feedback
                _selected_frag_ids = [
                    f.get("id", f.get("fragment_id", ""))
                    for f in pipeline_result.get("selected_fragments", [])
                    if f.get("id") or f.get("fragment_id")
                ]

                if context_text:
                    witness_context = f"{user_message}\n\n{context_text}" if user_message else context_text
                    # Gap #36: Confidence threshold — skip injection if
                    # entropy scores are too low (context quality is poor)
                    avg_entropy = 0.0
                    selected_frags = pipeline_result.get("selected_fragments", [])
                    _selected_fragments = selected_frags
                    _recoverable_fragments = pipeline_result.get(
                        "recoverable_fragments", selected_frags
                    )
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
                        witness_context = user_message

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
                    elif _is_responses_api:
                        # Responses API: token count from `input` + `instructions`
                        _inp = body.get("input", "")
                        if isinstance(_inp, str):
                            _inp_text = _inp
                        elif isinstance(_inp, list):
                            _inp_parts = []
                            for _item in _inp:
                                if isinstance(_item, dict):
                                    _inp_parts.append(_item.get("text", _item.get("content", "")))
                                elif isinstance(_item, str):
                                    _inp_parts.append(_item)
                            _inp_text = " ".join(str(p) for p in _inp_parts)
                        else:
                            _inp_text = str(_inp)
                        _instr_text = body.get("instructions", "")
                        original_tokens = (len(_inp_text.split()) + len(_instr_text.split())) * 4 // 3
                    else:
                        original_tokens = _estimate_message_tokens(
                            body.get("messages", [])
                        )
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
                    elif _is_responses_api:
                        body = inject_context_responses(body, context_text)
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
                        f"Optimized in {pipeline_ms:.1f}ms "
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

        # ── RAVS Bayesian Router: model swap if safe ──────────────
        _ravs_swapped = False
        _ravs_original_model = ""
        _ravs_archetype = ""
        if self._ravs_router and user_message:
            try:
                from .proxy_transform import extract_model as _extract_model
                from .ravs.router import classify_archetype
                _current_model = _extract_model(body, path) or ""
                _ravs_archetype = classify_archetype(user_message)

                # ── Implicit feedback → RAVS cells ──
                # If the PREVIOUS request was routed and we now detect
                # rephrase (failure) or topic change (success), record
                # that outcome so prompt-level archetypes accumulate
                # Bayesian evidence without needing shell exit codes.
                prev_routed = getattr(self, "_ravs_prev_routed", None)
                if prev_routed and hasattr(self, "_ravs_prev_client"):
                    prev_client, prev_arch, prev_model = prev_routed
                    if prev_client == client_key:
                        try:
                            from entroly_core import py_simhash
                            prev_query_data = self._feedback_tracker._trajectories.get(client_key)
                            if prev_query_data:
                                prev_hash = prev_query_data[0]
                                curr_hash = py_simhash(user_message)
                                xor = curr_hash ^ prev_hash
                                hamming = bin(xor).count("1")
                                similarity = 1.0 - (hamming / 64.0)

                                verdict = None
                                if similarity > 0.75:
                                    verdict = "fail"  # rephrase = routed model failed
                                elif similarity < 0.30:
                                    verdict = "pass"  # topic change = routed model succeeded

                                if verdict:
                                    try:
                                        from .ravs.capture import capture_from_args
                                        capture_from_args(
                                            command=f"[proxy:{prev_arch}]",
                                            exit_code=0 if verdict == "pass" else 1,
                                            stdout_tail=f"implicit_feedback:{verdict}",
                                            source="proxy_feedback",
                                            source_strength=0.4,
                                        )
                                        logger.debug(
                                            "RAVS feedback: %s → %s (sim=%.2f)",
                                            prev_arch, verdict, similarity,
                                        )
                                    except Exception:
                                        pass
                        except (ImportError, Exception):
                            pass  # No simhash = no feedback, safe to skip

                if _current_model:
                    decision = self._ravs_router.route(_current_model, user_message)
                    if not decision.use_original and decision.recommended_model:
                        # ── ECE Pre-Screen (V5): Tier 0 ambiguity filter only ──
                        # Tier 0 is the ONLY pre-routing check that works without
                        # response text (regex-based open-ended query detection).
                        # Full ECE (Fisher curvature on response text) runs
                        # POST-response alongside WITNESS — see P0 in
                        # _run_post_response_verification().
                        _ece_blocked = False
                        if self._ece and self._ece._is_open_ended(user_message):
                            # Open-ended queries: allow swap (aleatoric, not epistemic)
                            pass  # Allow the swap — open-ended = cheap model is fine
                        elif self._ece:
                            # For factual queries, allow RAVS swap but flag for
                            # post-response ECE verification. The real curvature
                            # check happens after we get the actual response.
                            pass

                        _model_specific_fields = {
                            "thinking",
                            "reasoning_effort",
                            "effort",
                        }
                        if any(field in body for field in _model_specific_fields):
                            _ece_blocked = True
                            logger.info(
                                "RAVS: kept original model because request has model-specific controls"
                            )

                        if not _ece_blocked:
                            from .ravs.router import swap_model_in_body
                            _ravs_original_model = _current_model
                            body = swap_model_in_body(body, decision.recommended_model)
                            _ravs_swapped = True
                            logger.info(
                                "RAVS: %s -> %s (%s)",
                                _current_model, decision.recommended_model, decision.reason,
                            )

                # Store for next-request feedback attribution
                if _ravs_swapped:
                    self._ravs_prev_routed = (client_key, _ravs_archetype, _current_model)
                else:
                    self._ravs_prev_routed = None
            except Exception as e:
                logger.debug("RAVS router error (forwarding original): %s", e)

        # Forward to real API (target_url already resolved above). Apply
        # provider compatibility cleanup even when RAVS did not swap models.
        if provider == "anthropic":
            body = strip_anthropic_unsupported_params(body)

        body, redaction_headers = self._apply_outbound_redaction(body)
        control_headers.update(redaction_headers)

        try:
            control_audit = audit_request_transform(
                control_before,
                body,
                provider=provider,  # type: ignore[arg-type]
                path=path,
                headers=headers,
                allow_model_change=_ravs_swapped,
            )
            control_outcome = "optimized" if body != control_before else "observed"
            control_headers = self._control_headers(
                control_decision,
                control_audit,
                body=body,
                headers=headers,
                provider=provider,
                path=path,
                outcome=control_outcome,
            )
            if not control_audit.compliant:
                logger.warning(
                    "Control-plane audit violation: provider=%s controls=%d tools=%d",
                    provider,
                    len(control_audit.provider_control_violations),
                    len(control_audit.tool_contract_violations),
                )
        except Exception as e:
            logger.debug("Control-plane audit skipped: %s", e)

        forward_headers = self._build_headers(headers, provider)
        is_streaming = body.get("stream", False)
        # Gemini: streaming is determined by URL path, not a body field.
        # streamGenerateContent returns SSE — must be handled as streaming.
        if not is_streaming and "streamGenerateContent" in path:
            is_streaming = True

        if is_streaming:
            return await self._stream_response(
                target_url, forward_headers, body, _selected_frag_ids,
                witness_context, provider, _recoverable_fragments, request_id,
                extra_headers=control_headers,
            )
        else:
            return await self._forward_response(
                target_url, forward_headers, body, _selected_frag_ids,
                witness_context, provider, _recoverable_fragments, request_id,
                extra_headers=control_headers,
            )

    @staticmethod
    def _conversation_key(body: dict[str, Any]) -> str:
        """Stable per-conversation key for cache-alignment, derived from the
        request's anchor (model + first system/user message). Read-only: used
        solely to scope reuse of Entroly's injected context block; it never
        affects the outbound request. Returns "" when no anchor is available
        (which disables alignment for that request — fail-open)."""
        if not isinstance(body, dict):
            return ""
        msgs = body.get("messages")
        if not isinstance(msgs, list):
            return ""
        model = str(body.get("model", ""))
        for m in msgs:
            if isinstance(m, dict) and m.get("role") in ("system", "user"):
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    return hashlib.sha256(
                        (model + "\x00" + c[:2000]).encode("utf-8", "ignore")
                    ).hexdigest()[:24]
        return ""

    def _run_pipeline(self, user_message: str, body: dict[str, Any], path: str = "", request_id: str = "") -> dict[str, Any]:
        """Run the synchronous optimization pipeline. Called via asyncio.to_thread.

        Returns dict with keys: context and elapsed_ms.
        """
        t0 = time.perf_counter()

        model = extract_model(body, path)

        # Auto-configure cache cost model from the model name.
        # Zero-config: developers never need to call set_model() manually.
        if model and hasattr(self.engine, 'set_model'):
            self.engine.set_model(model)

        # ── Budget Computation: ACB → ECDB → Static ──
        # Priority: (1) ACB learned per-query budget when confident,
        # (2) ECDB sigmoid-based dynamic budget, (3) static fallback.
        token_budget = None
        acb_prediction = None

        # (1) ACB: Adaptive Compression Budget
        if self.config.enable_adaptive_budget:
            try:
                ctx_text = ""  # lightweight — no full context assembly yet
                acb_features = extract_features(
                    user_message, ctx_text, task_type=None,
                )
                ctx_window = context_window_for_model(model or "")
                acb_prediction = self._acb.predict(acb_features)
                if acb_prediction.get("fallback") is None:
                    # Model is confident — use its learned budget
                    ratio = acb_prediction["budget_used"]
                    token_budget = max(200, int(ctx_window * ratio))
                    logger.debug(
                        "ACB: budget_ratio=%.3f → %d tokens (se=%.4f, n=%d)",
                        ratio, token_budget,
                        acb_prediction.get("budget_se") or 0,
                        acb_prediction.get("n_training", 0),
                    )
            except Exception as e:
                logger.debug("ACB fallback: %s", e)

        # (2) ECDB: Entropy-Calibrated Dynamic Budget
        if token_budget is None and self.config.enable_dynamic_budget:
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

        # (3) Static fallback
        if token_budget is None:
            token_budget = compute_token_budget(model, self.config)

        # (4) Cost Cortex: price-aware clamp on Entroly's injected-context
        # budget. A cheap long-context model can otherwise permit a runaway
        # injection (e.g. ~629K tokens); clamp by a hard token cap
        # (ENTROLY_MAX_CONTEXT_TOKENS, default 256K) and an optional dollar
        # ceiling (ENTROLY_MAX_CONTEXT_USD), priced from the single
        # value_tracker source. This only ever LOWERS the budget for Entroly's
        # OWN injected context — it never touches the user's request, model, or
        # generation params.
        try:
            from .cost_cortex import clamp_injected_budget
            _clamped, _why = clamp_injected_budget(model or "", int(token_budget))
            if _clamped < token_budget:
                logger.info(
                    "Cost Cortex: clamped injected-context budget %d -> %d (%s)",
                    token_budget, _clamped, _why,
                )
                token_budget = _clamped
        except Exception as e:
            logger.debug("Cost Cortex clamp skipped: %s", e)

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
        # ── Flat optimization path (original) ──
        # optimize_context already does:
        #   1. Query refinement (py_analyze_query + py_refine_heuristic)
        #   2. LTM recall (cross-session memories)
        #   3. Knapsack optimization (Rust)
        #   4. SSSL filtering
        #   5. Ebbinghaus decay bookkeeping
        self.engine._turn_counter += 1
        self.engine.advance_turn()

        # ── System 2 → System 1 coupling (opt-in: ENTROLY_VAULT_COUPLING=1) ──
        # Project verified vault beliefs into the engine as fragment candidates
        # (so the knapsack weighs them against IDE-supplied fragments) AND
        # discount candidate fragments that merely restate those beliefs
        # (belief-conditioned compression). Both directions share one belief
        # projection. See entroly/coupling.py for the math.
        injected_claim_ids: list[str] = []
        if self._vault is not None:
            try:
                from . import coupling
                injected_claim_ids = coupling.couple_beliefs(
                    self.engine, self._vault, user_message,
                )
            except Exception as e:
                logger.debug("Vault coupling injection skipped: %s", e)
        # Stash for outcome attribution at /outcome time
        self._last_injected_claim_ids = injected_claim_ids

        result = self.engine.optimize_context(token_budget, user_message)

        if "online_prism" in result and hasattr(self.engine, "_outcome_bridge") and self.engine._outcome_bridge is not None and request_id:
            op = result["online_prism"]
            try:
                self.engine._outcome_bridge.cache_observation(
                    request_id=request_id,
                    implicit_reward=op.get("reward", 0.0),
                    implicit_advantage=op.get("implicit_advantage", 0.0),
                    contributions=op.get("contributions", {}),
                    weights=op.get("weights", {}),
                )
            except Exception as e:
                logger.debug("OutcomeBridge caching failed: %s", e)

        selected = result.get("selected_fragments", [])
        # Render the hierarchy from the query-conditioned BM25+PRISM selection.
        # The old path seeded HCC independently with SimHash similarity, which
        # discarded the stronger optimizer signal already computed here.
        hcc_result = None
        if self.config.enable_hierarchical_compression:
            try:
                hcc_seed_ids = [
                    fragment.get("id", fragment.get("fragment_id", ""))
                    for fragment in selected
                    if fragment.get("id") or fragment.get("fragment_id")
                ]
                hcc_result = self.engine._rust.hierarchical_compress(
                    token_budget, user_message, hcc_seed_ids
                )
                if hcc_result.get("status") == "empty":
                    hcc_result = None
            except (AttributeError, Exception) as e:
                logger.debug(f"HCC unavailable, falling back to flat: {e}")
                hcc_result = None
        if hcc_result is not None:
            # HCC renders its own query-conditioned hierarchy. Use the
            # fragments from that rendered hierarchy for recovery, feedback,
            # scaffolding, and security scanning so exact replay expands what
            # the model actually saw instead of a parallel flat selection.
            from .ccr import hierarchical_context_fragments
            selected = hierarchical_context_fragments(hcc_result)
        refinement = result.get("query_refinement")

        # CCR: keep exact originals behind content-addressed handles whenever
        # IOS selects a compressed resolution. This turns skeleton/reference
        # output into recoverable context instead of silent truncation.
        try:
            from .ccr import capture_recoverable_fragments
            capture_recoverable_fragments(selected, self.engine._get_fragment)
        except Exception as e:
            logger.debug("CCR capture skipped: %s", e)

        # Standby exact replay for selection misses. These omitted candidates
        # are never rendered into the first prompt; they become eligible only
        # after verification rejects the first answer.
        recovery_candidates: list[dict[str, Any]] = []
        try:
            from .ccr import capture_ranked_recovery_candidates

            def fragment_key(fragment: dict[str, Any]) -> str:
                return (
                    fragment.get("id")
                    or fragment.get("fragment_id")
                    or fragment.get("retrieval_handle")
                    or fragment.get("source", "")
                )

            selected_keys = {
                key for fragment in selected
                if (key := fragment_key(fragment))
            }
            recall_hits = self.engine._rust.recall_auto(
                user_message,
                self._auto_recovery_max_candidates,
            )
            recalled = [
                {
                    "id": hit.get("id", hit.get("fragment_id", "")),
                    "source": hit.get("source", ""),
                    "scores": {
                        "semantic": hit.get("relevance", 0.0),
                        "composite": hit.get("relevance", 0.0),
                    },
                }
                for hit in recall_hits
            ]
            recovery_candidates = capture_ranked_recovery_candidates(
                recalled,
                self.engine._get_fragment,
                selected_keys=selected_keys,
                max_candidates=self._auto_recovery_max_candidates,
            )
        except Exception as e:
            logger.debug("CCR native recall capture skipped: %s", e)
        try:
            from .ccr import capture_ranked_recovery_candidates
            explanation = self.engine.explain_selection()
            captured_keys = {
                key for fragment in [*selected, *recovery_candidates]
                if (key := (
                    fragment.get("id")
                    or fragment.get("fragment_id")
                    or fragment.get("retrieval_handle")
                    or fragment.get("source", "")
                ))
            }
            recovery_candidates.extend(capture_ranked_recovery_candidates(
                explanation.get("excluded", []),
                self.engine._get_fragment,
                selected_keys=captured_keys,
                max_candidates=(
                    self._auto_recovery_max_candidates - len(recovery_candidates)
                ),
            ))
        except Exception as e:
            logger.debug("CCR optimizer-omission capture skipped: %s", e)

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

        # ── Context Scaffolding Engine (CSE) ──
        # Generate a structural dependency preamble that shows the LLM
        # how selected files relate to each other. Based on:
        #   - GRACG (NeurIPS 2025): heterogeneous code graph rendering
        #   - Scaffold Reasoning (arxiv 2025): structured reasoning streams
        #   - S2LPP (arxiv 2025): prompt strategy transfer across model sizes
        # Cost: ~200 tokens. Benefit: enables Haiku to match Opus quality
        # by pre-connecting cross-file relationships.
        scaffold = ""
        if self.config.enable_context_scaffold and selected:
            try:
                scaffold = generate_scaffold(
                    selected,
                    task_type=task_type,
                    max_tokens=self.config.scaffold_max_tokens,
                )
                if scaffold:
                    scaffold_tokens = len(scaffold) // 4
                    logger.debug(
                        "CSE: scaffold generated, ~%d tokens, task=%s",
                        scaffold_tokens, task_type,
                    )
            except Exception as e:
                logger.debug("CSE scaffold generation failed: %s", e)

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
                scaffold=scaffold,
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
                scaffold=scaffold,
                **apa_kwargs,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if selected:
            total_tokens = sum(f.get("token_count", 0) for f in selected)
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
                f"{total_tokens} tokens{ios_str}"
            )

        # ── Cache-aligned context reuse (provider prefix-cache hits) ──
        # When this turn's injected context block is >=90% similar to the
        # previous one for the same conversation, reuse the previous block
        # verbatim so the provider's cached prefix keeps hitting (Anthropic 90%
        # / OpenAI 50% read discount) — the dominant cost on chatty large-repo
        # sessions. This rewrites ONLY Entroly's own injected context string;
        # it never mutates the model, generation params, tools, or the user's
        # messages. Provider terms and data-handling rules still apply.
        # Fail-open: on any error the freshly optimized context is used.
        if context_text and self._cache_aligner is not None:
            try:
                _ckey = self._conversation_key(body)
                if _ckey:
                    context_text, _ = self._cache_aligner.align(_ckey, context_text)
            except Exception as e:
                logger.debug("Context cache-align skipped: %s", e)

        return {
            "context": context_text,
            "elapsed_ms": elapsed_ms,
            "selected_fragments": selected,
            "recoverable_fragments": [*selected, *recovery_candidates],
        }

    async def _buffered_witness_stream_response(
        self,
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        *,
        selected_frag_ids: list | None = None,
        witness_context: str = "",
        provider: str = "openai",
        recoverable_fragments: list[dict[str, Any]] | None = None,
        recovery_depth: int = 0,
        request_id: str = "",
        extra_headers: dict[str, str] | None = None,
    ) -> StreamingResponse | JSONResponse:
        """Buffer a streaming upstream response so WITNESS can enforce before display."""
        if not self._breaker.allow_request():
            return JSONResponse(
                {"error": "circuit_breaker_open", "message": "Upstream API experiencing failures, retrying after cooldown"},
                status_code=503,
                headers={"Retry-After": str(int(self._breaker.cooldown_s))},
            )

        max_bytes = int(os.environ.get("ENTROLY_WITNESS_STREAM_MAX_BYTES", str(2 * 1024 * 1024)))
        try:
            client = await self._ensure_client()
            chunks: list[bytes] = []
            total = 0
            status_code = 200
            content_type = "text/event-stream"
            async with client.stream("POST", url, json=body, headers=headers) as response:
                status_code = response.status_code
                content_type = response.headers.get("content-type", content_type)
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        self._breaker.record_failure()
                        return JSONResponse(
                            {"error": "witness_stream_too_large", "limit_bytes": max_bytes},
                            status_code=413,
                        )
                    chunks.append(chunk)
        except httpx.TimeoutException:
            self._breaker.record_failure()
            return JSONResponse({"error": "upstream_timeout"}, status_code=502)
        except httpx.ConnectError as e:
            self._breaker.record_failure()
            return JSONResponse({"error": "upstream_unavailable", "detail": str(e)}, status_code=502)
        except Exception as e:
            self._breaker.record_failure()
            return JSONResponse({"error": "stream_error", "detail": str(e)[:200]}, status_code=502)

        raw = b"".join(chunks)

        # ── Pre-flight: handle upstream errors before WITNESS ──
        # Must check status BEFORE recording success to the circuit breaker.
        if status_code == 429:
            self._breaker.record_failure()

            # Re-parse Retry-After from raw chunks is lossy; use stored status.
            # The response object is out of scope here, so propagate a
            # conservative default if the upstream header wasn't captured.
            return StreamingResponse(
                _bytes_iter(raw),
                status_code=429,
                media_type=content_type,
                headers={"X-Entroly-Witness": "skipped-rate-limited"},
            )
        if status_code >= 400:
            self._breaker.record_failure()
            return StreamingResponse(
                _bytes_iter(raw),
                status_code=status_code,
                media_type=content_type,
                headers={"X-Entroly-Witness": "skipped-upstream-error"},
            )
        self._breaker.record_success()

        response_text = _extract_text_from_sse(raw)
        if not response_text or not self._witness_analyzer:
            return StreamingResponse(
                _bytes_iter(raw),
                status_code=status_code,
                media_type=content_type,
                headers={"X-Entroly-Witness": "no-text"},
            )

        result, rewrite = self._witness_analyzer.analyze_and_rewrite(
            witness_context,
            response_text,
            mode=self._witness_mode,
        )
        self._record_resolution_feedback(
            recoverable_fragments or [],
            success=not bool(result.flagged()),
        )
        failed_recovery_sources: list[str] = []
        retry = self._prepare_auto_recovery_retry(
            body,
            provider,
            witness_context,
            recoverable_fragments or [],
            result,
            recovery_depth=recovery_depth,
        )
        if retry is not None:
            retry_body, retry_context, recovered_sources = retry
            logger.info(
                "Auto-recovery retry: %d exact fragment(s), buffered stream",
                len(recovered_sources),
            )
            recovered = await self._buffered_witness_stream_response(
                url,
                headers,
                retry_body,
                selected_frag_ids=selected_frag_ids,
                witness_context=retry_context,
                provider=provider,
                recoverable_fragments=[],
                recovery_depth=recovery_depth + 1,
                request_id=request_id,
                extra_headers=extra_headers,
            )
            recovered_ok = (
                recovered.status_code < 400
                and recovered.headers.get("X-Entroly-Witness") == "pass"
            )
            self._record_auto_recovery(
                recovered_sources=recovered_sources,
                success=recovered_ok,
                trigger="verification",
                request_id=request_id,
            )
            if recovered_ok:
                self._mark_auto_recovery_headers(recovered, recovered_sources)
                return recovered
            failed_recovery_sources = recovered_sources
            logger.warning("Auto-recovery retry failed; serving first buffered response")

        structured_output = _looks_like_structured_text(response_text)
        self._record_witness_result(result, changed=rewrite.changed and not structured_output)
        if request_id and hasattr(self.engine, "_outcome_bridge") and self.engine._outcome_bridge is not None:
            try:
                self.engine._outcome_bridge.on_honest_outcome(
                    request_id=request_id,
                    event_type="verification_result",
                    value="failed" if result.flagged() else "passed",
                    strength="strong"
                )
            except Exception as e:
                logger.debug("OutcomeBridge verification outcome failed: %s", e)
        witness_id = self._store_witness_certificate(result, rewrite)

        # P0+P2: Post-response ECE + conformal cascade (buffered path)
        self._run_post_response_verification(
            response_text, witness_context, result,
        )

        # ── Active Escalation: re-issue to stronger model if flagged ──
        # Only in buffered path (annotate/strict) — streaming can't be
        # intercepted. This is the key cost-saving loop:
        #   cheap model → hallucination detected → re-issue to expensive
        #   model → serve the better response → learn from the event.
        #
        # The alternative (always use the expensive model) costs ~5-10x
        # more. Active escalation only pays the premium on the ~5-15%
        # of requests that trigger hallucination risk.
        escalation_attempted = False
        if (
            self._escalation_mode == "active"
            and self._cascade_last
            and self._cascade_last.get("would_escalate")
            and not body.get("_entroly_escalation_depth", 0)
        ):
            original_model = str(body.get("model", ""))
            ladder_entry = self._escalation_ladder.get(original_model)
            if ladder_entry:
                escalated_model, cost_mult = ladder_entry
                try:
                    # Build escalated request: same body, stronger model
                    escalated_body = dict(body)
                    escalated_body["model"] = escalated_model
                    escalated_body["_entroly_escalation_depth"] = 1

                    logger.info(
                        "ESCALATING: %s → %s (fused_risk=%.3f, "
                        "entity_gap=%.3f, cost_mult=%.1fx)",
                        original_model,
                        escalated_model,
                        self._cascade_last.get("fused_risk", 0),
                        self._cascade_last.get("entity_gap", 0),
                        cost_mult,
                    )

                    # Re-issue to the stronger model
                    esc_client = await self._ensure_client()
                    esc_chunks: list[bytes] = []
                    max_bytes = int(os.environ.get(
                        "ENTROLY_WITNESS_STREAM_MAX_BYTES",
                        str(2 * 1024 * 1024),
                    ))
                    esc_total = 0
                    async with esc_client.stream(
                        "POST", url, json=escalated_body, headers=headers
                    ) as esc_response:
                        if esc_response.status_code < 400:
                            async for chunk in esc_response.aiter_bytes():
                                esc_total += len(chunk)
                                if esc_total > max_bytes:
                                    break
                                esc_chunks.append(chunk)

                    if esc_chunks:
                        esc_raw = b"".join(esc_chunks)
                        esc_text = _extract_text_from_sse(esc_raw)
                        if esc_text:
                            # Success: replace response with escalated version
                            raw = esc_raw
                            response_text = esc_text
                            rewrite = self._witness_analyzer.analyze_and_rewrite(
                                witness_context, esc_text,
                                mode=self._witness_mode,
                            )[1]  # just the rewrite
                            escalation_attempted = True

                            with self._stats_lock:
                                self._escalation_total += 1
                                self._escalation_actual += 1
                                self._escalation_last = {
                                    "from": original_model,
                                    "to": escalated_model,
                                    "fused_risk": self._cascade_last.get(
                                        "fused_risk", 0
                                    ),
                                    "cost_multiplier": cost_mult,
                                    "timestamp": time.time(),
                                }

                            logger.info(
                                "Escalation successful: %s → %s "
                                "(%d bytes)",
                                original_model,
                                escalated_model,
                                len(esc_raw),
                            )
                except Exception as e:
                    # Fail-open: serve original response
                    logger.warning(
                        "Escalation failed (%s → %s): %s",
                        original_model,
                        ladder_entry[0] if ladder_entry else "?",
                        str(e)[:200],
                    )
                    with self._stats_lock:
                        self._escalation_total += 1

        if self._enable_passive_feedback and selected_frag_ids:
            try:
                reward = self._feedback_tracker.assess_response(rewrite.output)
                self._feedback_tracker.record_assessment(reward)
                if abs(reward) > 0.01:
                    self.engine.record_reward(selected_frag_ids, reward)
            except Exception:
                pass

        resp_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Entroly-Optimized": "true",
            "X-Entroly-Witness": "flagged" if result.flagged() else "pass",
            "X-Entroly-Witness-Buffered": "true",
            "X-Entroly-Witness-Id": witness_id,
            "X-Entroly-Witness-Mode": self._witness_mode,
            "X-Entroly-Witness-Score": f"{result.summary_score:.4f}",
            "X-Entroly-Witness-Flagged": str(len(result.flagged())),
            "X-Entroly-Witness-Suppressed": str(getattr(rewrite, "suppressed_count", 0)),
            "X-Entroly-Witness-Warned": str(getattr(rewrite, "warned_count", 0)),
        }
        resp_headers.update(extra_headers or {})
        if failed_recovery_sources:
            self._mark_auto_recovery_failed_headers(
                resp_headers, failed_recovery_sources
            )
        resp_headers.update(self._cache_headers())
        # Escalation telemetry headers
        if escalation_attempted and self._escalation_last:
            resp_headers["X-Entroly-Escalated"] = "true"
            resp_headers["X-Entroly-Escalated-From"] = self._escalation_last.get("from", "")
            resp_headers["X-Entroly-Escalated-To"] = self._escalation_last.get("to", "")
            resp_headers["X-Entroly-Escalated-Risk"] = str(
                round(self._escalation_last.get("fused_risk", 0), 4)
            )
        if structured_output:
            resp_headers["X-Entroly-Witness-Rewrite-Skipped"] = "structured-output"
            return StreamingResponse(
                _bytes_iter(raw),
                media_type="text/event-stream",
                headers=resp_headers,
            )

        sse = self._format_witness_sse(provider, body, rewrite.output)
        if rewrite.changed:
            resp_headers["X-Entroly-Witness-Rewritten"] = "true"

        return StreamingResponse(
            _bytes_iter(sse),
            media_type="text/event-stream",
            headers=resp_headers,
        )

    def _format_witness_sse(self, provider: str, body: dict[str, Any], text: str) -> bytes:
        model = str(body.get("model") or "")
        now = int(time.time())
        if provider == "anthropic":
            events = [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_entroly_witness",
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                ),
                ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 0}}),
                ("message_stop", {"type": "message_stop"}),
            ]
            return "".join(f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in events).encode()

        if provider == "gemini":
            payload = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": text}]},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ]
            }
            return f"data: {json.dumps(payload)}\n\n".encode()

        chunk = {
            "id": "chatcmpl-entroly-witness",
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        }
        final = {
            "id": "chatcmpl-entroly-witness",
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode()

    async def _stream_response(
        self, url: str, headers: dict[str, str], body: dict[str, Any],
        selected_frag_ids: list | None = None,
        witness_context: str = "",
        provider: str = "openai",
        recoverable_fragments: list[dict[str, Any]] | None = None,
        request_id: str = "",
        recovery_depth: int = 0,
        extra_headers: dict[str, str] | None = None,
    ) -> StreamingResponse:
        """Forward a streaming request and proxy the SSE response.

        When passive feedback is enabled, tees response chunks into a buffer
        (capped at 50KB) and fires implicit feedback analysis after the
        stream completes. Zero latency impact — analysis runs in background.
        """
        if self._witness_enabled and self._witness_mode in {"annotate", "strict"}:
            return await self._buffered_witness_stream_response(
                url,
                headers,
                body,
                selected_frag_ids=selected_frag_ids,
                witness_context=witness_context,
                provider=provider,
                recoverable_fragments=recoverable_fragments,
                recovery_depth=recovery_depth,
                request_id=request_id,
                extra_headers=extra_headers,
            )

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
        _witness_enabled = self._witness_enabled and bool(self._witness_analyzer)
        _frag_ids = selected_frag_ids or []
        _buffer_cap = ImplicitFeedbackTracker._MAX_BUFFER_BYTES
        # Capture selected fragments for per-variant utilization tracking (Change 4)
        _selected_frags = getattr(self, '_last_context_fragments', []) if _feedback_enabled else []

        async def event_generator():
            buffer = [] if (_feedback_enabled or _witness_enabled) else None
            buffer_size = 0
            try:
                client = await self._ensure_client()
                async with client.stream(
                    "POST", url, json=body, headers=headers
                ) as response:
                    # ── Pre-flight status check before streaming ──
                    # httpx streams don't raise on 4xx/5xx — we must
                    # check before blindly iterating chunks.
                    if response.status_code == 429:
                        self._breaker.record_failure()
                        retry_after = _parse_retry_after(
                            response.headers.get("retry-after")
                        )

                        logger.warning(
                            "Upstream 429 on streaming request (Retry-After=%s)",
                            retry_after,
                        )
                        import json as _json
                        err_body = _json.dumps({
                            "error": "rate_limited",
                            "status": 429,
                            "detail": "Upstream rate-limited; honor Retry-After.",
                            "retry_after_s": retry_after,
                            "source": "upstream_api",
                        })
                        yield (
                            f'data: {err_body}\n\n'
                            f'data: [DONE]\n\n'
                        ).encode()
                        return
                    if response.status_code >= 500:
                        self._breaker.record_failure()
                        logger.warning(
                            "Upstream %d on streaming request",
                            response.status_code,
                        )
                        import json as _json
                        err_body = _json.dumps({
                            "error": "upstream_error",
                            "status": response.status_code,
                            "detail": f"Upstream returned {response.status_code}",
                            "source": "upstream_api",
                        })
                        yield (
                            f'data: {err_body}\n\n'
                            f'data: [DONE]\n\n'
                        ).encode()
                        return
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

            if buffer and _witness_enabled:
                try:
                    full_bytes = b"".join(buffer)
                    response_text = _extract_text_from_sse(full_bytes)
                    if response_text:
                        result = self._witness_analyzer.analyze(witness_context, response_text)
                        self._record_witness_result(result, changed=False)
                        # P0+P2: Post-response ECE + conformal cascade
                        self._run_post_response_verification(
                            response_text, witness_context, result,
                        )
                except Exception:
                    logger.debug("WITNESS stream audit failed", exc_info=True)

        resp_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Entroly-Optimized": "true",
        }
        resp_headers.update(extra_headers or {})
        with self._stats_lock:
            if self._witness_enabled:
                resp_headers["X-Entroly-Witness"] = "streaming-audit"
                resp_headers["X-Entroly-Witness-Mode"] = self._witness_mode
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
            resp_headers.update(self._cache_headers())

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers=resp_headers,
        )

    def _cache_headers(self) -> dict[str, str]:
        """Surface EGSC persistent-cache state on every response.

        The cache is already persisted across sessions via the
        EngineState.cache_snapshot path (see entroly-core/src/lib.rs).
        These headers expose what was warm-restored from disk on this
        process vs admitted live, so callers can tell when a hit came
        from a previous session rather than the current one.
        """
        out: dict[str, str] = {}
        try:
            stats = self.engine.stats() if hasattr(self.engine, "stats") else None
            cs = stats.get("cache", {}) if isinstance(stats, dict) else {}
            entries = int(cs.get("entries", 0))
            out["X-Entroly-Cache-Entries"] = str(entries)
            out["X-Entroly-Cache-Hit-Rate"] = f"{float(cs.get('hit_rate', 0.0)):.4f}"
            out["X-Entroly-Cache-Hits-Exact"] = str(int(cs.get("exact_hits", 0)))
            out["X-Entroly-Cache-Hits-Semantic"] = str(int(cs.get("semantic_hits", 0)))
            out["X-Entroly-Cache-Tokens-Saved"] = str(int(cs.get("tokens_saved", 0)))
            out["X-Entroly-Cache-Warm-Restored"] = str(self._cache_warm_restored)
            age_s = max(0.0, time.time() - self._cache_warm_started_at)
            out["X-Entroly-Cache-Warm-Age-S"] = f"{age_s:.0f}"
            # Cache source classification — quick "where did the hits come from"
            # signal. If warm_restored == entries, no admissions happened yet
            # in this session, so any hit is necessarily cross-session.
            if self._cache_warm_restored > 0 and entries <= self._cache_warm_restored:
                out["X-Entroly-Cache-Source"] = "persistent"
            elif self._cache_warm_restored > 0:
                out["X-Entroly-Cache-Source"] = "mixed"
            else:
                out["X-Entroly-Cache-Source"] = "session"
        except Exception:
            pass
        return out

    def _extract_response_text(self, content: dict[str, Any]) -> str:
        """Extract assistant text from OpenAI, Anthropic, or Gemini JSON."""
        parts: list[str] = []
        for choice in content.get("choices", []):
            msg = choice.get("message", {})
            text = msg.get("content")
            if isinstance(text, str):
                parts.append(text)
        for block in content.get("content", []):
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        for cand in content.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
        return "\n".join(parts)

    def _replace_response_text(self, content: dict[str, Any], new_text: str) -> bool:
        """Replace the first assistant text slot while preserving provider shape."""
        for choice in content.get("choices", []):
            msg = choice.get("message", {})
            if isinstance(msg.get("content"), str):
                msg["content"] = new_text
                return True
        for block in content.get("content", []):
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                block["text"] = new_text
                return True
        for cand in content.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if isinstance(part.get("text"), str):
                    part["text"] = new_text
                return True
        return False

    def _build_recovery_context(
        self,
        fragments: list[dict[str, Any]],
        query: str = "",
    ) -> tuple[str, list[str]]:
        """Materialize exact originals or marked exact excerpts within a cap."""
        try:
            from .ccr import get_ccr_store, slice_recovery_content
            store = get_ccr_store()
        except Exception:
            return "", []

        fragment_lookup = getattr(self.engine, "_get_fragment", None)
        candidates = [
            fragment for fragment in fragments
            if fragment.get("variant", "full") != "full"
        ]
        # Recover omitted detail from rendered selections first. Query-ranked
        # standby fragments repair a different failure class and may consume
        # only residual capacity, so they can add recall without displacing
        # the exact selected-source replay path.
        candidates.sort(
            key=lambda fragment: (
                bool(fragment.get("recovery_candidate")),
                -float(fragment.get("relevance", 0.0) or 0.0),
            ),
        )

        parts = ["--- Exact Recovery Context ---", ""]
        recovered_sources: list[str] = []
        recovered_keys: set[str] = set()
        recovered_tokens = 0
        for fragment in candidates:
            if len(recovered_sources) >= self._auto_recovery_max_fragments:
                break
            source = fragment.get("source", "")
            key = fragment.get("retrieval_handle") or source
            if not key or key in recovered_keys:
                continue
            entry = store.retrieve_or_materialize(key, fragment_lookup)
            if not entry:
                continue
            original = entry.get("original", "")
            if not original:
                continue
            remaining_tokens = self._auto_recovery_max_tokens - recovered_tokens
            if remaining_tokens <= 0:
                break
            original_tokens = max(
                int(entry.get("original_tokens", 0) or 0),
                max(1, (len(original) + 3) // 4),
            )
            recovered_content = original
            sliced = False
            if original_tokens > remaining_tokens:
                recovered_content, sliced = slice_recovery_content(
                    original,
                    query,
                    remaining_tokens,
                )
            if not recovered_content:
                continue
            tokens = max(1, (len(recovered_content) + 3) // 4)

            recovered_sources.append(entry["source"])
            recovered_keys.add(key)
            recovered_tokens += tokens
            if sliced:
                parts.append(
                    f"## {entry['source']} "
                    "(exact bounded excerpts; omitted gaps marked; "
                    f"full_sha256={entry['content_sha256']}; "
                    f"retrieve={entry['retrieval_handle']})"
                )
            else:
                parts.append(
                    f"## {entry['source']} "
                    f"(exact original; sha256={entry['content_sha256']})"
                )
            parts.append(recovered_content.rstrip())
            parts.append("")

        if not recovered_sources:
            return "", []
        parts.append("--- End Exact Recovery Context ---")
        return "\n".join(parts), recovered_sources

    def _prepare_auto_recovery_retry(
        self,
        body: dict[str, Any],
        provider: str,
        witness_context: str,
        fragments: list[dict[str, Any]],
        witness_result: Any,
        *,
        recovery_depth: int,
    ) -> tuple[dict[str, Any], str, list[str]] | None:
        """Build a single bounded retry request after verification rejects output."""
        if (
            not self._auto_recovery_enabled
            or recovery_depth > 0
            or not self._witness_enabled
            or not self._witness_analyzer
            or not witness_result
            or not witness_result.flagged()
        ):
            return None

        recovery_query = getattr(self, "_last_query", "") or witness_context
        recovery_context, recovered_sources = self._build_recovery_context(
            fragments,
            query=recovery_query,
        )
        if not recovery_context:
            return None

        if getattr(self.config, "enable_context_sanitizer", True):
            try:
                from .hardening import sanitize_injected_context
                recovery_context, _ = sanitize_injected_context(
                    recovery_context, fence=True
                )
            except Exception:
                pass

        if provider == "gemini":
            retry_body = inject_context_gemini(body, recovery_context)
        elif provider == "anthropic":
            retry_body = inject_context_anthropic(body, recovery_context)
        elif "input" in body and "messages" not in body:
            retry_body = inject_context_responses(body, recovery_context)
        else:
            retry_body = inject_context_openai(body, recovery_context)

        expanded_witness_context = (
            f"{witness_context}\n\n{recovery_context}"
            if witness_context else recovery_context
        )
        return retry_body, expanded_witness_context, recovered_sources

    def _record_auto_recovery(
        self,
        *,
        recovered_sources: list[str],
        success: bool,
        trigger: str,
        request_id: str = "",
    ) -> None:
        with self._stats_lock:
            self._auto_recovery_attempted += 1
            if success:
                self._auto_recovery_succeeded += 1
            else:
                self._auto_recovery_failed += 1
            self._auto_recovery_last = {
                "success": success,
                "trigger": trigger,
                "sources": recovered_sources,
                "coverage_risk": getattr(self, "_last_coverage_risk", "unknown"),
                "timestamp": time.time(),
            }

        if request_id and hasattr(self.engine, "_outcome_bridge") and self.engine._outcome_bridge is not None:
            try:
                self.engine._outcome_bridge.on_honest_outcome(
                    request_id=request_id,
                    event_type="recovery_event",
                    value="success" if success else "failure",
                    strength="strong"
                )
            except Exception as e:
                logger.debug("OutcomeBridge recovery outcome failed: %s", e)

    def _record_resolution_feedback(
        self,
        fragments: list[dict[str, Any]],
        *,
        success: bool,
    ) -> None:
        """Teach IOS whether the selected compressed resolutions were sufficient."""
        resolutions = sorted({
            fragment.get("variant", "full")
            for fragment in fragments
            if (
                fragment.get("variant", "full") != "full"
                and not fragment.get("recovery_candidate")
            )
        })
        if not resolutions:
            return
        try:
            if hasattr(self.engine, "record_resolution_outcome"):
                self.engine.record_resolution_outcome(resolutions, success)
            elif hasattr(self.engine, "_rust") and hasattr(
                self.engine._rust, "record_resolution_outcome"
            ):
                self.engine._rust.record_resolution_outcome(resolutions, success)
        except Exception:
            logger.debug("Resolution feedback skipped", exc_info=True)

    @staticmethod
    def _mark_auto_recovery_headers(
        response: StreamingResponse | JSONResponse,
        recovered_sources: list[str],
    ) -> None:
        response.headers["X-Entroly-Recovery-Attempted"] = "true"
        response.headers["X-Entroly-Recovery-Verified"] = "true"
        response.headers["X-Entroly-Recovery-Fragments"] = str(
            len(recovered_sources)
        )
        response.headers["X-Entroly-Recovered"] = "true"
        response.headers["X-Entroly-Recovered-Fragments"] = str(
            len(recovered_sources)
        )
        response.headers["X-Entroly-Recovery-Trigger"] = "verification"

    @staticmethod
    def _mark_auto_recovery_failed_headers(
        headers: dict[str, str],
        recovered_sources: list[str],
    ) -> None:
        headers["X-Entroly-Recovery-Attempted"] = "true"
        headers["X-Entroly-Recovery-Verified"] = "false"
        headers["X-Entroly-Recovery-Fragments"] = str(len(recovered_sources))
        headers["X-Entroly-Recovery-Trigger"] = "verification"

    def _apply_witness_gateway(
        self,
        content: dict[str, Any],
        witness_context: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        headers: dict[str, str] = {}
        if not self._witness_enabled or not self._witness_analyzer:
            return content, headers
        response_text = self._extract_response_text(content)
        if not response_text:
            return content, {"X-Entroly-Witness": "no-text"}
        try:
            result, rewrite = self._witness_analyzer.analyze_and_rewrite(
                witness_context,
                response_text,
                mode=self._witness_mode,
            )
            structured_output = _looks_like_structured_text(response_text)
            changed = False
            if rewrite.changed and not structured_output:
                changed = self._replace_response_text(content, rewrite.output)
            self._record_witness_result(result, changed=changed)
            witness_id = self._store_witness_certificate(result, rewrite)
            # P0+P2: Post-response ECE + conformal cascade (non-streaming)
            self._run_post_response_verification(
                response_text, witness_context, result,
            )
            if self._witness_embed:
                content["entroly_witness"] = result.as_dict()
                content["entroly_witness"]["policy"] = rewrite.as_dict()
                content["entroly_witness"]["id"] = witness_id
            headers.update({
                "X-Entroly-Witness": "flagged" if result.flagged() else "pass",
                "X-Entroly-Witness-Id": witness_id,
                "X-Entroly-Witness-Mode": self._witness_mode,
                "X-Entroly-Witness-Score": f"{result.summary_score:.4f}",
                "X-Entroly-Witness-Flagged": str(len(result.flagged())),
                "X-Entroly-Witness-Suppressed": str(getattr(rewrite, "suppressed_count", 0)),
                "X-Entroly-Witness-Warned": str(getattr(rewrite, "warned_count", 0)),
            })
            if changed:
                headers["X-Entroly-Witness-Rewritten"] = "true"
            elif structured_output and rewrite.changed:
                headers["X-Entroly-Witness-Rewrite-Skipped"] = "structured-output"
            return content, headers
        except Exception as e:
            logger.debug("WITNESS gateway failed: %s", e, exc_info=True)
            return content, {"X-Entroly-Witness": "error"}

    def _apply_eicv_gateway(
        self,
        content: dict[str, Any],
        eicv_context: str,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """EICV auto-suppression gateway — Phase 9.

        Runs AFTER WITNESS. Verifies the (possibly already-rewritten)
        response against the retrieved context with the EICV pipeline and
        optionally rewrites/annotates depending on mode.

        Modes:
          audit    — zero output change, headers only
          annotate — keeps body, appends "[Entroly EICV] verification warnings"
          strict   — per-claim 4-action rewrite (suppress hallucinated,
                     hedge abstained, pass supported)

        Fail-open: any exception leaves the response untouched and emits
        X-Entroly-EICV: error. Configured via ENTROLY_EICV{,_MODE,_PROFILE}.
        """
        headers: dict[str, str] = {}
        if not self._eicv_enabled or not self._eicv_suppressor:
            return content, headers
        response_text = self._extract_response_text(content)
        if not response_text:
            return content, {"X-Entroly-EICV": "no-text"}
        if not eicv_context:
            return content, {"X-Entroly-EICV": "no-context"}

        try:
            result = self._eicv_suppressor.suppress(eicv_context, response_text)

            # Counters
            self._eicv_total += 1
            self._eicv_hallucinated_total += result.n_hallucinated
            self._eicv_suppressed_total += result.suppressed_count
            self._eicv_last = {
                "phi_mean": (
                    sum(v.phi for v in result.certificates) / len(result.certificates)
                    if result.certificates else 1.0
                ),
                "n_claims": result.n_claims,
                "n_hallucinated": result.n_hallucinated,
                "suppressed_count": result.suppressed_count,
                "latency_ms": result.latency_ms,
            }

            # Apply rewrite if changed and the response isn't structured (e.g. JSON)
            structured_output = _looks_like_structured_text(response_text)
            changed = False
            if (result.mode == "strict" or result.mode == "annotate") \
                    and result.changed and not structured_output:
                changed = self._replace_response_text(content, result.rewritten_output)

            # Optionally embed certificates in the response payload
            if getattr(self.config, "eicv_embed", False) or \
                    os.environ.get("ENTROLY_EICV_EMBED", "") == "1":
                content["entroly_eicv"] = result.as_dict()

            phi_mean = self._eicv_last["phi_mean"]
            headers.update({
                "X-Entroly-EICV": "hallucinated" if result.n_hallucinated > 0 else "clean",
                "X-Entroly-EICV-Mode": result.mode,
                "X-Entroly-EICV-Profile": result.profile,
                "X-Entroly-EICV-Phi-Mean": f"{phi_mean:.4f}",
                "X-Entroly-EICV-Claims": str(result.n_claims),
                "X-Entroly-EICV-Hallucinated": str(result.n_hallucinated),
                "X-Entroly-EICV-Abstained": str(result.n_abstained),
                "X-Entroly-EICV-Supported": str(result.n_supported),
                "X-Entroly-EICV-Suppressed": str(result.suppressed_count),
                "X-Entroly-EICV-Warned": str(result.warned_count),
                "X-Entroly-EICV-Latency-Ms": f"{result.latency_ms:.2f}",
            })
            if changed:
                headers["X-Entroly-EICV-Rewritten"] = "true"
            elif structured_output and result.changed:
                headers["X-Entroly-EICV-Rewrite-Skipped"] = "structured-output"
            return content, headers
        except Exception as e:
            logger.debug("EICV gateway failed: %s", e, exc_info=True)
            return content, {"X-Entroly-EICV": "error"}

    # ── P0 + P2: Post-Response Verification ─────────────────────────
    #
    # This is the architectural fix: ECE evaluates REAL response text
    # (not empty strings) and the conformal cascade ties WITNESS risk
    # to escalation.py's rule (★). Runs after every response,
    # alongside WITNESS. Zero extra latency for the user (runs after
    # stream completion / in the buffered path).

    def _run_post_response_verification(
        self,
        response_text: str,
        witness_context: str,
        witness_result: Any,
    ) -> None:
        """P0+P2: Post-response ECE + conformal cascade verification.

        Called after WITNESS has already analyzed the response. This:
        1. Runs ECE Fisher curvature on the ACTUAL response text (P0)
        2. Feeds WITNESS risk into the conformal cascade (P2)
        3. Records the cascade decision for observability

        Never blocks the response — this is post-response telemetry.
        Fail-open: any error here is logged and swallowed.
        """
        # ── P0: ECE on real response text ──
        ece_signal = None
        if self._ece and response_text:
            try:
                from .ravs.router import classify_risk as _cr
                risk = _cr(witness_context[:500] if witness_context else "").value
                ece_signal = self._ece.evaluate_uncertainty(
                    query=witness_context[:500] if witness_context else "",
                    response_text=response_text,
                    risk_level=risk,
                )
                logger.debug(
                    "ECE post-response: tier=%d kappa=%.3f U_e=%.3f (%s)",
                    ece_signal.tier_used,
                    ece_signal.fisher_curvature,
                    ece_signal.epistemic_uncertainty,
                    ece_signal.reason,
                )
            except Exception:
                pass  # ECE failure = no telemetry, never blocks

        # ── P2: 4-Signal Fusion Cascade Decision ──
        # Previously used single-signal WITNESS risk (AUROC 0.80).
        # Now applies benchmark-optimized 4-signal fusion (AUROC 0.90):
        #   fused_risk = 0.05*witness + 0.05*ece + 0.80*entity_gap + 0.10*spectral
        # Weights from grid search on HaluEval-QA calibration (n=4000),
        # validated on test split (n=16000): AUROC 0.9003.
        if witness_result is not None:
            try:
                witness_risk = 1.0 - float(witness_result.summary_score)
                from .conformal_cascade import ACCEPT, FLAG, ESCALATE
                from .escalation import should_escalate as _se

                # ── Signal 2: ECE curvature (already computed above) ──
                ece_curvature = 0.0
                if ece_signal:
                    ece_curvature = min(1.0, ece_signal.fisher_curvature * 2.5)

                # ── Signal 3: Entity coverage gap ──
                entity_gap = 0.0
                try:
                    _ent_pats = [
                        re.compile(r'\b\d+\.?\d*\b'),
                        re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'),
                    ]
                    ans_ents = set()
                    ctx_lower = (witness_context or "").lower()
                    for _pat in _ent_pats:
                        for _m in _pat.finditer(response_text):
                            ans_ents.add(_m.group().lower())
                    if ans_ents:
                        missing = sum(1 for e in ans_ents if e not in ctx_lower)
                        entity_gap = missing / len(ans_ents)
                except Exception:
                    pass

                # ── Signal 4: Spectral consistency ──
                spectral_risk = 0.0
                try:
                    from .ravs.spectral import compute_spectral_consistency
                    spec = compute_spectral_consistency(
                        witness_context or "", response_text
                    )
                    spectral_risk = 1.0 - spec.score
                except Exception:
                    pass

                # ── 4-signal fusion (benchmark-optimized weights) ──
                # Weights: W=0.05, E=0.05, G=0.80, S=0.10
                # Source: benchmarks/results/fusion4_optimized.json
                fused_risk = min(1.0, max(0.0, (
                    0.05 * witness_risk
                    + 0.05 * ece_curvature
                    + 0.80 * entity_gap
                    + 0.10 * spectral_risk
                )))

                # Escalation rule (★): escalate iff fused_risk > r_floor + c/q
                _c_exp = 0.05   # 5% of q as escalation cost ratio
                _q = 1.0        # normalized hallucination cost
                _r_floor = 0.0  # no irreducible error assumed
                would_escalate = _se(fused_risk, _c_exp, _q, _r_floor)

                cascade_decision = {
                    "witness_risk": round(witness_risk, 4),
                    "ece_curvature": round(ece_curvature, 4),
                    "entity_gap": round(entity_gap, 4),
                    "spectral_risk": round(spectral_risk, 4),
                    "fused_risk": round(fused_risk, 4),
                    "would_escalate": would_escalate,
                    "rule_threshold": round(_r_floor + _c_exp / _q, 4),
                    "ece_epistemic_u": round(
                        ece_signal.epistemic_uncertainty if ece_signal else 0.0, 4
                    ),
                    "ece_tier": ece_signal.tier_used if ece_signal else -1,
                    "fusion_weights": "W=0.05,E=0.05,G=0.80,S=0.10",
                }

                with self._stats_lock:
                    self._cascade_last = cascade_decision
                    self._cascade_total += 1
                    if would_escalate:
                        self._cascade_escalations += 1

                logger.debug(
                    "Cascade: fused=%.3f (W=%.3f E=%.3f G=%.3f S=%.3f) "
                    "threshold=%.3f -> %s",
                    fused_risk, witness_risk, ece_curvature,
                    entity_gap, spectral_risk,
                    _r_floor + _c_exp / _q,
                    "ESCALATE" if would_escalate else "ACCEPT",
                )
            except Exception:
                pass  # Cascade failure = no telemetry, never blocks

    def _store_witness_certificate(self, result: Any, rewrite: Any) -> str:
        raw = f"{time.time_ns()}:{result.summary_score}:{len(result.certificates)}:{rewrite.mode}"
        witness_id = hashlib.sha256(raw.encode()).hexdigest()[:16]
        payload = result.as_dict()
        payload["policy"] = rewrite.as_dict()
        payload["id"] = witness_id
        with self._stats_lock:
            self._witness_certificates[witness_id] = payload
            self._witness_certificates.move_to_end(witness_id)
            while len(self._witness_certificates) > self._witness_store_max:
                self._witness_certificates.popitem(last=False)
        return witness_id

    def _record_witness_result(self, result: Any, *, changed: bool) -> None:
        flagged = len(result.flagged())
        with self._stats_lock:
            self._witness_total += 1
            self._witness_flagged += flagged
            if changed:
                self._witness_rewritten += 1
            self._witness_last = {
                "score": round(result.summary_score, 4),
                "claims": len(result.certificates),
                "flagged": flagged,
                "grounded": result.n_grounded,
                "unsupported": result.n_unsupported,
                "contradicted": result.n_contradicted,
                "unknown": result.n_unknown,
                "latency_ms": round(result.latency_ms, 1),
            }
        # Real hallucination value: when WITNESS actually rewrote the
        # response, the unsupported/contradicted claims it removed are
        # hallucinations that never reached the user. Record to the
        # shared sink so the dashboard's "Hallucinations Blocked" tile
        # shows REAL data across pip/proxy/MCP. Fail-open: telemetry must
        # never alter or break the response path.
        if changed:
            try:
                blocked = int(result.n_unsupported) + int(
                    result.n_contradicted)
                if blocked > 0:
                    from entroly.value_tracker import get_tracker
                    get_tracker().record_hallucination_blocked(
                        blocked, source="witness:proxy",
                        detail=(f"WITNESS blocked {blocked} unsupported/"
                                f"contradicted claim(s)"),
                    )
            except Exception:
                pass

    async def _forward_response(
        self, url: str, headers: dict[str, str], body: dict[str, Any],
        selected_frag_ids: list | None = None,
        witness_context: str = "",
        provider: str = "openai",
        recoverable_fragments: list[dict[str, Any]] | None = None,
        request_id: str = "",
        recovery_depth: int = 0,
        extra_headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        """Forward a non-streaming request with circuit breaker, retry on 429/5xx, and response validation.

        Retry policy (RFC 6585 §4 + RFC 7231 §7.1.3 + Anthropic API conventions):

        - On 429 with Retry-After ≤ RATE_LIMIT_INLINE_WAIT_S and budget remaining,
          we sleep the FULL Retry-After duration (no silent cap) and retry once.
          Truncating Retry-After would violate the upstream's explicit cooldown
          signal and can prolong the rate-limit window.
        - On 429 with a longer Retry-After (or none), we DO NOT retry inline —
          we surface the 429 immediately to the client with the upstream
          Retry-After header propagated so the client decides whether to wait.
          This keeps interactive sessions (Claude Code, Cursor) responsive and
          lets the agent's own backoff handle the longer cooldown.
        - On 5xx, we use exponential backoff (1s, 2s) up to 2 retries. These are
          server errors, not deliberate cooldowns, so retrying is appropriate.
        - Total inline retry wall-time is bounded by RATE_LIMIT_TOTAL_BUDGET_S
          to avoid blocking the caller's request past their own timeout.
        """
        # Check circuit breaker
        if not self._breaker.allow_request():
            logger.warning("Circuit breaker open -- forwarding unmodified")

        # Policy thresholds. Inline-wait threshold is intentionally short
        # because Claude Code / Cursor / agent clients are interactive — a
        # multi-second inline wait kills UX. Longer cooldowns are surfaced
        # to the client immediately so its own retry logic applies.
        RATE_LIMIT_INLINE_WAIT_S = 5.0
        RATE_LIMIT_TOTAL_BUDGET_S = 15.0
        SERVER_ERROR_MAX_RETRIES = 2

        # Per-status retry budget. 429s respect upstream cooldown; 5xx
        # uses exponential backoff up to SERVER_ERROR_MAX_RETRIES.
        response = None
        attempts = 0
        total_slept = 0.0
        # Hard upper bound on iterations so a misbehaving upstream that
        # always returns 429 with tiny Retry-After can't infinite-loop us.
        max_iterations = SERVER_ERROR_MAX_RETRIES + 1 + 1  # +1 reserved for one 429 retry
        while attempts < max_iterations:
            try:
                client = await self._ensure_client()
                response = await client.post(url, json=body, headers=headers)
                self._breaker.record_success()
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                self._breaker.record_failure()
                if attempts < SERVER_ERROR_MAX_RETRIES:
                    await asyncio.sleep(1.0 * (attempts + 1))
                    attempts += 1
                    total_slept += 1.0 * attempts
                    continue
                # Sanitize error message: never leak auth headers in error responses
                err_msg = str(e)
                for key_header in ("authorization", "x-api-key"):
                    if key_header in headers:
                        err_msg = err_msg.replace(headers[key_header], "[REDACTED]")
                out_headers = dict(extra_headers or {})
                return JSONResponse(
                    {"error": "upstream_unavailable", "detail": err_msg},
                    status_code=502,
                    headers=out_headers,
                )

            # ── 429: explicit upstream cooldown signal ──
            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("retry-after"))
                # If the upstream tells us to wait longer than we can safely
                # block, surface 429 immediately with the header propagated.
                # This is spec-compliant: the client is informed of the
                # cooldown and can honor it itself.
                if (retry_after is None
                        or retry_after > RATE_LIMIT_INLINE_WAIT_S
                        or total_slept + retry_after > RATE_LIMIT_TOTAL_BUDGET_S):
                    logger.info(
                        "Upstream 429 with Retry-After=%s; surfacing to client "
                        "(inline_wait_threshold=%.1fs, budget_left=%.1fs)",
                        retry_after, RATE_LIMIT_INLINE_WAIT_S,
                        RATE_LIMIT_TOTAL_BUDGET_S - total_slept,
                    )
                    out_headers = {"X-Entroly-Source": "upstream"}
                    out_headers.update(extra_headers or {})
                    upstream_ra = response.headers.get("retry-after")
                    if upstream_ra:
                        # Propagate Retry-After verbatim so the client honors
                        # the upstream's exact cooldown directive.
                        out_headers["Retry-After"] = upstream_ra
                    return JSONResponse(
                        {
                            "error": "rate_limited",
                            "status": 429,
                            "detail": "Upstream rate-limited this request; client should honor Retry-After.",
                            "retry_after_s": retry_after,
                            "source": "upstream_api",
                        },
                        status_code=429,
                        headers=out_headers,
                    )
                # Short cooldown: wait the FULL duration (no silent cap) and retry once.
                logger.info(
                    "Upstream 429, honoring Retry-After=%.1fs (attempt %d, total_slept=%.1fs)",
                    retry_after, attempts + 1, total_slept,
                )
                await asyncio.sleep(retry_after)
                total_slept += retry_after
                attempts += 1
                # 429 gets a single inline retry — Anthropic asked us to wait
                # once, not loop. Cap further 429s to surface-immediately.
                max_iterations = attempts + 1
                continue

            # ── 5xx: server-side error, retry with exponential backoff ──
            if response.status_code >= 500:
                if attempts < SERVER_ERROR_MAX_RETRIES:
                    backoff = 1.0 * (attempts + 1)
                    logger.info(
                        "Upstream %d, retrying in %.0fs (attempt %d/%d)",
                        response.status_code, backoff, attempts + 1,
                        SERVER_ERROR_MAX_RETRIES,
                    )
                    await asyncio.sleep(backoff)
                    total_slept += backoff
                    attempts += 1
                    continue
                out_headers = {"X-Entroly-Source": "upstream"}
                out_headers.update(extra_headers or {})
                return JSONResponse(
                    {
                        "error": "upstream_error",
                        "status": response.status_code,
                        "detail": f"Upstream returned {response.status_code} after {SERVER_ERROR_MAX_RETRIES} retries",
                        "source": "upstream_api",
                    },
                    status_code=response.status_code,
                    headers=out_headers,
                )

            # Success — break out of retry loop
            break

        resp_headers: dict[str, str] = {"X-Entroly-Optimized": "true"}
        resp_headers.update(extra_headers or {})
        with self._stats_lock:
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
        # If verification rejects an answer produced from compressed context,
        # retry once with exact CCR originals before the answer escapes.
        failed_recovery_sources: list[str] = []
        if (
            isinstance(content, dict)
            and self._witness_enabled
            and self._witness_analyzer
        ):
            try:
                response_text = self._extract_response_text(content)
                if response_text:
                    witness_probe = self._witness_analyzer.analyze(
                        witness_context, response_text
                    )
                    self._record_resolution_feedback(
                        recoverable_fragments or [],
                        success=not bool(witness_probe.flagged()),
                    )
                    retry = self._prepare_auto_recovery_retry(
                        body,
                        provider,
                        witness_context,
                        recoverable_fragments or [],
                        witness_probe,
                        recovery_depth=recovery_depth,
                    )
                    if retry is not None:
                        retry_body, retry_context, recovered_sources = retry
                        logger.info(
                            "Auto-recovery retry: %d exact fragment(s), non-streaming",
                            len(recovered_sources),
                        )
                        recovered = await self._forward_response(
                            url,
                            headers,
                            retry_body,
                            selected_frag_ids=selected_frag_ids,
                            witness_context=retry_context,
                            provider=provider,
                            recoverable_fragments=[],
                            recovery_depth=recovery_depth + 1,
                            request_id=request_id,
                        )
                        recovered_ok = (
                            recovered.status_code < 400
                            and recovered.headers.get("X-Entroly-Witness") == "pass"
                        )
                        self._record_auto_recovery(
                            recovered_sources=recovered_sources,
                            success=recovered_ok,
                            trigger="verification",
                            request_id=request_id,
                        )
                        if recovered_ok:
                            self._mark_auto_recovery_headers(
                                recovered, recovered_sources
                            )
                            return recovered
                        failed_recovery_sources = recovered_sources
                        logger.warning(
                            "Auto-recovery retry failed; serving first response"
                        )
            except Exception as e:
                logger.debug("Auto-recovery skipped: %s", e, exc_info=True)

        if self._enable_passive_feedback and selected_frag_ids and isinstance(content, dict):
            try:
                # Extract assistant text from response JSON
                response_text = ""
                # OpenAI / Anthropic: choices[0].message.content
                for choice in content.get("choices", []):
                    msg = choice.get("message", {})
                    if msg.get("content"):
                        response_text += _content_to_text(msg.get("content"))
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

        # WITNESS output gateway: attach proof certificates and optionally
        # annotate/suppress unsupported factual claims before returning JSON.
        if isinstance(content, dict):
            content, witness_headers = self._apply_witness_gateway(content, witness_context)
            resp_headers.update(witness_headers)

            # EICV auto-suppression — Phase 9. Runs AFTER WITNESS so the
            # EICV pipeline sees the already-cleaned text. Defaults to
            # audit mode (zero output change, observability headers only).
            content, eicv_headers = self._apply_eicv_gateway(content, witness_context)
            resp_headers.update(eicv_headers)

        # Add context waste % header
        with self._stats_lock:
            if self._waste_ratios:
                avg_waste = sum(self._waste_ratios) / len(self._waste_ratios)
                resp_headers["X-Entroly-Context-Waste-Pct"] = f"{avg_waste * 100:.1f}"
        if failed_recovery_sources:
            self._mark_auto_recovery_failed_headers(
                resp_headers, failed_recovery_sources
            )

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
        """Build headers for the forwarded request without losing provider metadata."""
        capability = provider_capability(provider)
        forward: dict[str, str] = {"Content-Type": "application/json"}
        for name, value in original.items():
            lower = name.lower()
            if lower in _HOP_BY_HOP_HEADERS:
                continue
            if lower in _COMMON_PROVIDER_HEADERS:
                if lower == "content-type":
                    continue
                forward[name] = value
                continue
            if lower in capability.auth_headers:
                forward[name] = value
                continue
            if any(lower.startswith(prefix) for prefix in capability.header_prefixes):
                forward[name] = value
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
            "# HELP entroly_requests_subscription_blocked Subscription-auth requests blocked (OAuth bearer to public API)",
            "# TYPE entroly_requests_subscription_blocked counter",
            f"entroly_requests_subscription_blocked {proxy._requests_subscription_blocked}",
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

    # ── System 1 → System 2 outcome attribution ──
    # Bayesian-update vault beliefs that were injected into this context.
    # On failure, also enqueue them for reverification by FlowOrchestrator.
    belief_updates: list[dict] = []
    reverify_count = 0
    if proxy._vault is not None and proxy._last_injected_claim_ids:
        try:
            from . import coupling
            belief_updates = coupling.attribute_outcome(
                proxy._last_injected_claim_ids, success, proxy._vault,
            )
            if not success:
                reverify_count = coupling.enqueue_reverification(
                    proxy._last_injected_claim_ids, proxy._vault,
                )
        except Exception as e:
            logger.debug("Belief attribution skipped: %s", e)

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
        "belief_updates": belief_updates,
        "beliefs_reverify_queued": reverify_count,
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

    proxy = request.app.state.proxy
    fragment_lookup = getattr(proxy.engine, "_get_fragment", None)
    entry = store.retrieve_or_materialize(source, fragment_lookup)
    if entry is None:
        return JSONResponse(
            {"error": f"Source '{source}' not found in CCR store", "hint": "GET /retrieve to list available"},
            status_code=404,
        )

    return JSONResponse({
        "source": entry["source"],
        "retrieval_handle": entry["retrieval_handle"],
        "content_sha256": entry["content_sha256"],
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

    if isinstance(body, dict):
        body, _redaction_headers = proxy._apply_outbound_redaction(body)
    else:
        _redaction_headers = {}

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
            return await proxy._stream_response(target_url, forward_headers, body, provider=provider)
        response = await client.request(
            request.method, target_url, json=body, headers=forward_headers
        )
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return JSONResponse(
                content=response.json(),
                status_code=response.status_code,
                headers=_redaction_headers,
            )
        return JSONResponse(
            content={"data": response.text},
            status_code=response.status_code,
            headers=_redaction_headers,
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
            "witness": {
                "enabled": proxy._witness_enabled,
                "mode": proxy._witness_mode,
                "profile": proxy._witness_profile,
                "nli": proxy._witness_use_nli,
                "checked": proxy._witness_total,
                "flagged_claims": proxy._witness_flagged,
                "rewritten_responses": proxy._witness_rewritten,
                "feedback": dict(proxy._witness_feedback),
                "nli_usage": (
                    proxy._witness_analyzer.nli_usage()
                    if proxy._witness_analyzer and hasattr(proxy._witness_analyzer, "nli_usage")
                    else {}
                ),
                "last": proxy._witness_last,
            },
            "auto_recovery": {
                "enabled": proxy._auto_recovery_enabled,
                "attempted": proxy._auto_recovery_attempted,
                "succeeded": proxy._auto_recovery_succeeded,
                "failed": proxy._auto_recovery_failed,
                "max_fragments": proxy._auto_recovery_max_fragments,
                "max_candidates": proxy._auto_recovery_max_candidates,
                "max_tokens": proxy._auto_recovery_max_tokens,
                "last": proxy._auto_recovery_last,
            },
        }
        # Passive RL feedback stats
        stats["implicit_feedback"] = proxy._feedback_tracker.stats()
        # ECE: Epistemic Cascade Engine stats (V5)
        if proxy._ece:
            try:
                stats["ece"] = proxy._ece.stats()
            except Exception:
                stats["ece"] = {"error": "stats_unavailable"}
        # P2: Conformal cascade stats (WITNESS → ECE → escalation rule ★)
        if proxy._cascade_total > 0:
            stats["conformal_cascade"] = {
                "total": proxy._cascade_total,
                "escalations": proxy._cascade_escalations,
                "escalation_rate": round(
                    proxy._cascade_escalations / max(proxy._cascade_total, 1), 4
                ),
                "last": proxy._cascade_last,
            }
        # Active escalation stats
        stats["active_escalation"] = {
            "mode": proxy._escalation_mode,
            "total_decisions": proxy._escalation_total,
            "actual_escalations": proxy._escalation_actual,
            "last": proxy._escalation_last,
            "ladder_models": len(proxy._escalation_ladder),
        }
    return JSONResponse(stats)


async def _witness_certificate(request: Request) -> JSONResponse:
    proxy = request.app.state.proxy
    witness_id = request.path_params.get("witness_id", "")
    with proxy._stats_lock:
        payload = proxy._witness_certificates.get(witness_id)
    if payload is None:
        return JSONResponse({"error": "witness_certificate_not_found"}, status_code=404)
    return JSONResponse(payload)


async def _witness_list(request: Request) -> JSONResponse:
    proxy = request.app.state.proxy
    limit = int(request.query_params.get("limit", "25"))
    limit = max(1, min(limit, 100))
    with proxy._stats_lock:
        items = list(proxy._witness_certificates.values())[-limit:]
    compact = [
        {
            "id": item.get("id"),
            "summary_score": item.get("summary_score"),
            "n_claims": item.get("n_claims"),
            "n_contradicted": item.get("n_contradicted"),
            "n_unsupported": item.get("n_unsupported"),
            "n_unknown": item.get("n_unknown"),
            "policy": item.get("policy"),
            "flagged_claims": [
                {
                    "label": cert.get("label"),
                    "claim_text": cert.get("claim_text"),
                    "risk": cert.get("risk"),
                    "proof_path": cert.get("proof_path", [])[:3],
                }
                for cert in item.get("certificates", [])
                if isinstance(cert, dict) and cert.get("label") != "grounded"
            ][:5],
        }
        for item in reversed(items)
    ]
    return JSONResponse({
        "count": len(compact),
        "items": compact,
        "feedback": dict(proxy._witness_feedback),
    })


async def _witness_feedback_route(request: Request) -> JSONResponse:
    proxy = request.app.state.proxy
    witness_id = request.path_params.get("witness_id", "")
    try:
        body = await request.json()
    except Exception:
        body = {}
    verdict = str(body.get("verdict", "")).strip().lower()
    note = str(body.get("note", "")).strip()[:1000]
    if verdict not in {"false_positive", "correct", "false_negative", "ignored"}:
        return JSONResponse(
            {"error": "invalid_verdict", "allowed": ["false_positive", "correct", "false_negative", "ignored"]},
            status_code=400,
        )
    with proxy._stats_lock:
        payload = proxy._witness_certificates.get(witness_id)
        if payload is None:
            return JSONResponse({"error": "witness_certificate_not_found"}, status_code=404)
        feedback = payload.setdefault("feedback", [])
        if isinstance(feedback, list):
            feedback.append({"verdict": verdict, "note": note, "ts": time.time()})
        proxy._witness_feedback[verdict] += 1
    if body.get("train") and body.get("context") and body.get("output"):
        try:
            from .witness_training import WitnessTrainingStore

            store = getattr(proxy, "_witness_training_store", None)
            if store is None:
                store = WitnessTrainingStore()
                setattr(proxy, "_witness_training_store", store)
            train_label = {
                "false_positive": "false_positive",
                "false_negative": "false_negative",
                "correct": "witness_correct",
                "ignored": "",
            }.get(verdict, "")
            if train_label:
                store.record(
                    context=str(body.get("context")),
                    output=str(body.get("output")),
                    label=train_label,
                    profile=str(body.get("profile") or proxy._witness_profile),
                    source="witness_feedback",
                )
        except Exception as e:
            logger.debug("witness feedback training skipped: %s", e)
    return JSONResponse({"ok": True, "id": witness_id, "verdict": verdict})


async def _witness_train_route(request: Request) -> JSONResponse:
    proxy = request.app.state.proxy
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        from .witness_training import WitnessTrainingStore

        store = getattr(proxy, "_witness_training_store", None)
        if store is None:
            store = WitnessTrainingStore()
            setattr(proxy, "_witness_training_store", store)

        ravs_event = body.get("ravs_event")
        if isinstance(ravs_event, dict):
            record = store.record_ravs_outcome(ravs_event)
            if record is None:
                return JSONResponse({"ok": False, "reason": "no_trainable_ravs_signal"}, status_code=422)
        else:
            context = str(body.get("context") or "")
            output = str(body.get("output") or "")
            label = str(body.get("label") or "")
            if not context or not output or not label:
                return JSONResponse(
                    {"error": "context_output_label_required"},
                    status_code=400,
                )
            record = store.record(
                context=context,
                output=output,
                label=label,
                profile=str(body.get("profile") or proxy._witness_profile),
                source=str(body.get("source") or "proxy_api"),
            )
        return JSONResponse({"ok": True, "training": record.as_dict()})
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        logger.debug("witness training route failed: %s", e, exc_info=True)
        return JSONResponse({"error": "witness_training_failed"}, status_code=500)


# Need os for ENTROLY_RATE_LIMIT env var


def create_proxy_app(
    engine: Any, config: ProxyConfig | None = None,
    start_dashboard: bool = True,
    start_autotune: bool | None = None,
) -> Starlette:
    """Create the Starlette ASGI app for the prompt compiler proxy."""
    proxy = PromptCompilerProxy(engine, config)

    # Auto-start the live value dashboard alongside the proxy
    # (skipped when called from the daemon, which starts its own dashboard)
    if start_dashboard:
        try:
            from .dashboard import start_dashboard as _start_dash
            _start_dash(engine=engine, port=9378, daemon=True, proxy_runtime=proxy)
            logger.info("Value dashboard live at http://localhost:9378")
        except Exception as e:
            logger.warning(f"Dashboard failed to start: {e}")

    # Benchmark autotune is useful, but it is CPU-heavy in source checkouts.
    # Keep proxy startup responsive by making it opt-in for this path.
    if start_autotune is None:
        start_autotune = os.environ.get("ENTROLY_AUTOTUNE_DAEMON", "0").lower() in {
            "1",
            "true",
            "yes",
        }
    if start_autotune:
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
            # OpenAI Responses API (used by Codex CLI, newer OpenAI SDK)
            Route("/v1/responses", proxy.handle_proxy, methods=["POST"]),
            # Gemini: model name is embedded in the URL path
            Route("/v1beta/models/{model_id:path}", proxy.handle_proxy, methods=["POST"]),
            Route("/health", _health),
            Route("/stats", _sidecar_guard(_proxy_stats)),
            Route("/context", _sidecar_guard(_context_inspect)),          # Gap #29
            Route("/metrics", _sidecar_guard(_metrics_prometheus)),        # Gap #34
            Route("/outcome", _sidecar_guard(_record_outcome), methods=["POST"]),  # Gap #37
            Route("/bypass", _sidecar_guard(_toggle_bypass), methods=["POST"]),    # Gap #28
            Route("/feedback", _sidecar_guard(_fragment_feedback), methods=["POST"]),  # Gap #42
            Route("/explain", _sidecar_guard(_context_explain)),                       # Gap #43
            Route("/confidence", _sidecar_guard(_confidence)),                         # IDE widget API
            Route("/trends", _sidecar_guard(_value_trends)),                           # Dashboard trends
            Route("/retrieve", _sidecar_guard(_context_retrieve)),                     # CCR: lossless retrieval
            Route("/witness", _sidecar_guard(_witness_list)),                           # WITNESS certificate index
            Route("/witness/train", _sidecar_guard(_witness_train_route), methods=["POST"]),
            Route("/witness/{witness_id}/feedback", _sidecar_guard(_witness_feedback_route), methods=["POST"]),
            Route("/witness/{witness_id}", _sidecar_guard(_witness_certificate)),       # WITNESS sidecar certificates
            # Catch-all: forward any unmatched path to upstream API
            # Must be LAST — Starlette matches routes in declaration order
            Route("/{path:path}", _catch_all, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]),
        ],
        lifespan=_lifespan,
    )
    app.state.proxy = proxy
    return app
