---
claim_id: 98ae237b-5b7a-4bd6-9531-ed50dc0ea69a
entity: dopt_selector
status: inferred
confidence: 0.75
sources:
  - entroly\dopt_selector.py:93
  - entroly\dopt_selector.py:291
last_checked: 2026-04-23T03:07:07.803143+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: dopt_selector

**Language:** python
**Lines of code:** 380

## Types
- `class DOptFragment()`

## Functions
- `def select(
    fragments: list[dict],
    token_budget: int,
    query: str = "",
) -> list[dict]`

## Dependencies
- `__future__`
- `collections`
- `dataclasses`
- `math`
- `numpy`
- `re`
