---
claim_id: 4efa7765-3881-4b3a-a0a3-021d88c79b84
entity: sdk
status: inferred
confidence: 0.75
sources:
  - entroly\sdk.py:36
  - entroly\sdk.py:88
last_checked: 2026-04-23T03:07:07.818792+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: sdk

**Language:** python
**Lines of code:** 248


## Functions
- `def compress(
    content: str,
    budget: int | None = None,
    content_type: str | None = None,
    target_ratio: float = 0.3,
) -> str`
- `def compress_messages(
    messages: list[dict[str, Any]],
    budget: int = 50_000,
    preserve_last_n: int = 4,
) -> list[dict[str, Any]]`

## Dependencies
- `.universal_compress`
- `__future__`
- `typing`

## Linked Beliefs
- [[universal_compress]]
