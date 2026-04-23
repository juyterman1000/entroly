---
claim_id: f232700c-77b5-47b9-90a1-9cae2b832d53
entity: quality_eval
status: inferred
confidence: 0.75
sources:
  - bench\quality_eval.py:109
  - bench\quality_eval.py:114
  - bench\quality_eval.py:156
  - bench\quality_eval.py:189
  - bench\quality_eval.py:193
  - bench\quality_eval.py:209
  - bench\quality_eval.py:232
  - bench\quality_eval.py:29
  - bench\quality_eval.py:30
  - bench\quality_eval.py:31
last_checked: 2026-04-23T03:07:07.784234+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: quality_eval

**Language:** python
**Lines of code:** 326


## Functions
- `def tokenize_query(q: str) -> list[str]`
- `def gather_baseline_context(query: str, budget_tokens: int) -> str` — Rank repo files by query-term match count; concat top files until budget.
- `def gather_entroly_context(query: str) -> str`
- `def est_tokens(s: str) -> int`
- `def answer(client: OpenAI, query: str, context: str) -> str`
- `def judge(client: OpenAI, query: str, ground_truth: str, ans: str) -> dict`
- `def main() -> int`

## Dependencies
- `__future__`
- `collections`
- `entroly.universal_compress`
- `json`
- `openai`
- `os`
- `pathlib`
- `re`
- `subprocess`
- `sys`

## Linked Beliefs
- [[universal_compress]]
