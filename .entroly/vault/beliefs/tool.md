---
claim_id: 75462b9f-d59a-4a54-877f-32f52bb38d50
entity: tool
status: inferred
confidence: 0.75
sources:
  - .entroly\vault\evolution\skills\ddb2e2969bb0\tool.py:11
  - .entroly\vault\evolution\skills\ddb2e2969bb0\tool.py:16
  - .entroly\vault\evolution\skills\ddb2e2969bb0\tool.py:8
last_checked: 2026-04-23T03:07:07.777359+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: tool

**Language:** python
**Lines of code:** 24


## Functions
- `def matches(query: str) -> bool` — Check if this skill should handle the query.
- `def execute(query: str, context: dict) -> dict` — Execute the skill logic.

## Dependencies
- `re`
