---
claim_id: a37dfc3e-dc76-4933-a780-0d3d7ce37f73
entity: compare
status: inferred
confidence: 0.75
sources:
  - bench\compare.py:153
  - bench\compare.py:168
  - bench\compare.py:189
  - bench\compare.py:263
  - bench\compare.py:303
  - bench\compare.py:330
  - bench\compare.py:333
  - bench\compare.py:360
  - bench\compare.py:371
  - bench\compare.py:35
last_checked: 2026-04-23T03:07:07.780878+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: compare

**Language:** python
**Lines of code:** 485


## Functions
- `def strategy_raw(corpus: list[dict], query: str, budget: int) -> list[dict]` — Stuff tokens in insertion order until budget exhausted.
- `def strategy_topk(corpus: list[dict], query: str, budget: int) -> list[dict]` — Rank by query term overlap (simulated cosine similarity), take top-K.
- `def strategy_entroly(corpus: list[dict], query: str, budget: int) -> list[dict]` — Knapsack-optimal selection with: - Information density scoring (entropy × (1 - boilerplate)) - Query relevance weighting - Submodular diversity (diminishing returns per module) - Near-duplicate detect
- `def evaluate(strategy_name: str, selected: list[dict], query: str, budget: int) -> dict` — Compute metrics for a context selection.
- `def render_markdown(all_results: list) -> str` — Emit the Results + Summary section as markdown for BENCHMARKS.md.
- `def avg(idx: int, key: str) -> float`
- `def total(idx: int, key: str) -> int`
- `def regression_check(all_results: list) -> tuple[bool, str]` — Return (ok, message). Fails if Entroly regresses below guardrails.
- `def main()`

## Dependencies
- `__future__`
- `argparse`
- `hashlib`
- `math`
- `pathlib`
- `re`
- `sys`
- `time`
- `typing`
