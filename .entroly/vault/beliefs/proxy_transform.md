---
claim_id: d6660f52-6dad-4289-8e3b-b0b5ff02a84a
entity: proxy_transform
status: inferred
confidence: 0.75
sources:
  - entroly\proxy_transform.py:27
  - entroly\proxy_transform.py:64
  - entroly\proxy_transform.py:105
  - entroly\proxy_transform.py:123
  - entroly\proxy_transform.py:165
  - entroly\proxy_transform.py:247
  - entroly\proxy_transform.py:354
  - entroly\proxy_transform.py:444
  - entroly\proxy_transform.py:560
  - entroly\proxy_transform.py:662
last_checked: 2026-04-23T03:07:07.816239+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: proxy_transform

**Language:** python
**Lines of code:** 1910


## Functions
- `def detect_provider(
    path: str,
    headers: dict[str, str],
    body: dict[str, Any] | None = None,
) -> str`
- `def extract_user_message(body: dict[str, Any], provider: str) -> str` — Extract the latest user message text from the request body.
- `def extract_model(body: dict[str, Any], path: str = "") -> str` — Extract the model name from the request body or URL path. Gemini embeds the model in the URL path rather than the body: /v1beta/models/gemini-2.0-flash:generateContent
- `def compute_token_budget(model: str, config: ProxyConfig) -> int` — Compute the token budget for context injection. When ECDB (Entropy-Calibrated Dynamic Budget) is disabled, uses the static context_fraction.  When enabled, use compute_dynamic_budget() instead — this 
- `def compute_dynamic_budget(
    model: str,
    config: ProxyConfig,
    vagueness: float = 0.5,
    total_fragments: int = 0,
) -> int`
- `def calibrated_token_count(content: str, source: str = "") -> int` — Estimate token count using per-language char/token ratios. More accurate than len/4 — saves 15-25% of wasted context budget.
- `def build_context_report(
    fragments: list[dict[str, Any]],
    hcc_result: dict[str, Any] | None = None,
) -> str`
- `def format_context_block(
    fragments: list[dict[str, Any]],
    security_issues: list[str],
    ltm_memories: list[dict[str, Any]],
    refinement_info: dict[str, Any] | None,
    *,
    task_type: str = "Unknown",
    vagueness: float = 0.0,
) -> str`
- `def format_hierarchical_context(
    hcc_result: dict[str, Any],
    security_issues: list[str],
    ltm_memories: list[dict[str, Any]],
    refinement_info: dict[str, Any] | None,
    *,
    task_type: str = "Unknown",
    vagueness: float = 0.0,
) -> str`
- `def inject_context_openai(
    body: dict[str, Any], context_text: str
) -> dict[str, Any]` — Inject optimized context into an OpenAI chat completion request. Inserts a system message at the beginning of the messages array. If there's already a system message at position 0, the context is prep
- `def inject_context_anthropic(
    body: dict[str, Any], context_text: str
) -> dict[str, Any]` — Inject optimized context into an Anthropic messages request. Uses the top-level "system" field. Appends to existing system content.
- `def inject_context_gemini(
    body: dict[str, Any], context_text: str
) -> dict[str, Any]` — Inject optimized context into a Google Gemini generateContent request. Uses the top-level "systemInstruction" field with a parts array. Prepends to existing system instruction if present.
- `def compute_optimal_temperature(
    vagueness: float,
    fragment_entropies: list[float],
    sufficiency: float,
    task_type: str,
    *,
    fisher_scale: float = _FISHER_SCALE,
    alpha: float = _ALPHA,
    gamma: float = _GAMMA,
    eps_d: float = _EPS_D,
) -> float`
- `def apply_trajectory_convergence(
    tau: float,
    turn_count: int,
    c_min: float = _TRAJECTORY_C_MIN,
    lam: float = _TRAJECTORY_LAMBDA,
) -> float`
- `def apply_temperature(
    body: dict[str, Any],
    tau: float,
    provider: str = "openai",
) -> dict[str, Any]`
- `def compress_tool_output(content: str) -> tuple[str, str, float]` — Compress a tool/MCP call result using pattern-based rules. Returns: (compressed_content, compression_type, savings_ratio) If no compression applied: (content, "none", 0.0)
- `def compress_tool_messages(messages: list[dict]) -> tuple[list[dict], int]` — Compress tool/MCP call results in a conversation message list. Processes messages with role="tool" or role="function" or content that looks like tool output, applying pattern-based compression. Return
- `def entropic_conversation_prune(
    messages: list[dict],
    injected_context: str = "",
    provider: str = "openai",
) -> tuple[list[dict], dict]`
- `def distill_response(
    text: str,
    mode: str = "full",
) -> tuple[str, int, int]`
- `def distill_response_sse_chunk(
    chunk_text: str,
    mode: str = "full",
) -> str`

## Dependencies
- `.proxy_config`
- `__future__`
- `copy`
- `hashlib`
- `math`
- `os`
- `re`
- `typing`
