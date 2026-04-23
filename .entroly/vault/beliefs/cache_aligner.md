---
claim_id: b0f95b68-8069-4fd1-a4bd-205a86a2f09e
entity: cache_aligner
status: inferred
confidence: 0.75
sources:
  - entroly\cache_aligner.py:30
  - entroly\cache_aligner.py:44
  - entroly\cache_aligner.py:56
  - entroly\cache_aligner.py:100
  - entroly\cache_aligner.py:105
last_checked: 2026-04-23T03:07:07.793512+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: cache_aligner

**Language:** python
**Lines of code:** 116

## Types
- `class CacheAligner()` — Stabilize context prefixes for LLM provider KV cache optimization. Tracks per-client context injections. When a new injection is sufficiently similar to the previous one (>similarity_threshold), the p

## Functions
- `def __init__(
        self,
        similarity_threshold: float = 0.90,
        max_clients: int = 100,
    )`
- `def align(self, client_key: str, context: str) -> tuple[str, bool]` — Align context for cache stability. Args: client_key: Client identifier (hashed API key) context: The new context block to inject Returns: (aligned_context, cache_hit): The context to use and whether t
- `def invalidate(self, client_key: str) -> None` — Force-invalidate a client's cached context.
- `def stats(self) -> dict[str, Any]` — Return cache alignment statistics.

## Dependencies
- `__future__`
- `collections`
- `hashlib`
- `threading`
- `typing`
