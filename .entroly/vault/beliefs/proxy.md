---
claim_id: 664d2923-213a-4896-a474-953f5a16e021
entity: proxy
status: inferred
confidence: 0.75
sources:
  - entroly\proxy.py:104
  - entroly\proxy.py:148
  - entroly\proxy.py:173
  - entroly\proxy.py:320
  - entroly\proxy.py:490
  - entroly\proxy.py:629
  - entroly\proxy.py:111
  - entroly\proxy.py:119
  - entroly\proxy.py:130
  - entroly\proxy.py:135
last_checked: 2026-04-23T03:07:07.814368+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: proxy

**Language:** python
**Lines of code:** 2260

## Types
- `class _CircuitBreaker()` — SIRS-inspired circuit breaker: open after N consecutive failures, half-open after cooldown period, close on success. Inspired by the refractory period in agentOS/scheduler.rs SIRS routing.
- `class _TokenBucket()` — Token bucket rate limiter. Ported from agentOS/compliance.rs TokenBucket.
- `class _WelfordStats()` — Welford's online algorithm for streaming mean/variance. Ported from agentOS/persona_manifold.rs Welford tracker. Tracks pipeline latency without storing all samples.
- `class ImplicitFeedbackTracker()` — Extract implicit RL feedback from proxy traffic. Thread-safe. Per-client state tracks query trajectories for rephrase detection. Response text is scanned for confusion indicators to infer success/fail
- `class _CusumEmaDriftDetector()` — Dual online quality drift detector: CUSUM + EMA. Combines two complementary algorithms from the change-point detection literature (Online Kernel CUSUM, arXiv 2025; RL drift detection, NeurIPS 2025): 1
- `class PromptCompilerProxy()` — HTTP reverse proxy that optimizes every LLM request with entroly.

## Functions
- `def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0)`
- `def allow_request(self) -> bool`
- `def record_success(self) -> None`
- `def record_failure(self) -> None`
- `def state(self) -> str`
- `def __init__(self, capacity: float, refill_per_second: float)`
- `def try_consume(self, cost: float = 1.0) -> bool`
- `def __init__(self)`
- `def add(self, x: float) -> None`
- `def variance(self) -> float`
- `def stddev(self) -> float`
- `def to_dict(self) -> dict`
- `def compress_conversation_messages(
    messages: list[dict],
    context_window: int = 128_000,
) -> list[dict]`
- `def __init__(self)`
- `def assess_response(self, response_text: str) -> float` — Assess an LLM response for confusion vs confidence. Returns a reward signal: -1.0  = strong confusion detected (multiple indicators) -0.5  = mild confusion detected (one indicator) 0.0  = ambiguous / 
- `def detect_rephrase(
        self, client_key: str, query_text: str, selected_ids: list
    ) -> tuple | None` — Check if this query is a rephrase of the previous one. Returns: ("rephrase", prev_selected_ids) if rephrase detected -> failure signal ("topic_change", prev_selected_ids) if topic changed -> success s
- `def record_assessment(self, reward: float) -> None` — Track assessment stats and feed the drift detector.
- `def quality_trend(self) -> str` — Return current quality trend: 'stable', 'declining', or 'improving'.
- `def stats(self) -> dict[str, Any]` — Return feedback tracker statistics.
- `def __init__(self)`
- `def update(self, reward: float) -> None` — Feed a new reward observation.
- `def trend(self) -> str` — Return current quality trend.
- `def reset(self) -> None` — Reset detector state (e.g., on session restart).
- `def to_dict(self) -> dict[str, Any]`
- `def __init__(self, engine: Any, config: ProxyConfig | None = None)`
- `def startup(self) -> None`
- `def shutdown(self) -> None`
- `def handle_proxy(self, request: Request) -> StreamingResponse | JSONResponse` — Main proxy handler — intercept, optimize, forward. Uses pipelined async architecture: 1. Parse request + start HTTP connection warmup concurrently 2. Run Rust pipeline in thread pool (off event loop) 
- `def create_proxy_app(
    engine: Any, config: ProxyConfig | None = None
) -> Starlette` — Create the Starlette ASGI app for the prompt compiler proxy.

## Dependencies
- `.proxy_config`
- `.proxy_transform`
- `.value_tracker`
- `__future__`
- `asyncio`
- `collections`
- `hashlib`
- `httpx`
- `json`
- `logging`
- `math`
- `os`
- `re`
- `starlette.applications`
- `starlette.requests`
- `starlette.responses`
- `starlette.routing`
- `sys`
- `threading`
- `time`
