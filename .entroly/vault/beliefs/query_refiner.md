---
claim_id: 102e0edd-782a-4135-89a2-c6ea87e82dcc
entity: query_refiner
status: inferred
confidence: 0.75
sources:
  - entroly\query_refiner.py:43
  - entroly\query_refiner.py:27
  - entroly\query_refiner.py:34
  - entroly\query_refiner.py:54
  - entroly\query_refiner.py:63
  - entroly\query_refiner.py:83
  - entroly\query_refiner.py:113
  - entroly\query_refiner.py:146
last_checked: 2026-04-23T03:07:07.817119+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: query_refiner

**Language:** python
**Lines of code:** 173

## Types
- `class QueryRefiner()` — Refines vague developer queries into precise context-selection prompts. Rust handles all compute. Python only does LLM I/O (optional). Usage: refiner = QueryRefiner() result = refiner.refine("fix the 

## Functions
- `def py_analyze_query(query: str, summaries: list) -> tuple` — Pure-Python fallback for query analysis.
- `def py_refine_heuristic(query: str, summaries: list) -> str` — Pure-Python fallback — returns query unchanged.
- `def __init__(self, llm_fn: Callable[[str], str] | None = None)` — Args: llm_fn: Optional async/sync function that takes a query string and returns a refined query string using an LLM. If None, only the Rust heuristic path is used.
- `def analyze(self, query: str, fragment_summaries: list[str] | None = None) -> dict` — Analyze a query for vagueness and key terms. Returns: dict with: - vagueness_score (float, 0–1) - key_terms (list[str]) - needs_refinement (bool) - reason (str)
- `def refine(self, query: str, fragment_summaries: list[str] | None = None) -> str` — Refine a query using: 1. Rust heuristic (always runs, zero latency) 2. LLM refinement (only if query needs_refinement AND llm_fn is set) Returns the refined query string (original if no refinement app
- `def make_openai_refine_fn(api_key: str, model: str = "gpt-4o-mini") -> Callable[[str], str]` — Return a function that calls OpenAI to refine a query. Requires: openai package (`pip install openai`). The function is synchronous and makes one API call per invocation. Usage: refiner = QueryRefiner
- `def make_anthropic_refine_fn(
    api_key: str, model: str = "claude-haiku-20240307"
) -> Callable[[str], str]` — Return a function that calls Anthropic Claude to refine a query. Requires: anthropic package (`pip install anthropic`). Usage: refiner = QueryRefiner(llm_fn=make_anthropic_refine_fn(os.environ["ANTHRO

## Dependencies
- `__future__`
- `collections.abc`
- `logging`
