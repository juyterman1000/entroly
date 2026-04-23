---
claim_id: 02020930-3e2c-4d68-b23f-3159f76b9df1
entity: ccr
status: inferred
confidence: 0.75
sources:
  - entroly\ccr.py:24
  - entroly\ccr.py:43
  - entroly\ccr.py:50
  - entroly\ccr.py:86
  - entroly\ccr.py:100
  - entroly\ccr.py:118
  - entroly\ccr.py:123
  - entroly\ccr.py:144
last_checked: 2026-04-23T03:07:07.794488+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: ccr

**Language:** python
**Lines of code:** 151

## Types
- `class CompressedContextStore()` — Lossless compressed context store with LRU eviction. When fragments are compressed (skeleton/reference), their originals are stored here. The LLM can retrieve them on demand via the `/retrieve` endpoi

## Functions
- `def __init__(self, max_entries: int = 500)`
- `def store(
        self,
        source: str,
        original_content: str,
        compressed_content: str,
        resolution: str = "skeleton",
        original_tokens: int = 0,
        compressed_tokens: int = 0,
    ) -> None`
- `def retrieve(self, source: str) -> dict[str, Any] | None` — Retrieve the original content for a compressed fragment. Returns dict with 'original', 'compressed', 'resolution', etc. Returns None if source not found in store.
- `def list_available(self) -> list[dict[str, Any]]` — List all stored fragments with metadata (no content). Returns list of dicts with source, resolution, token counts. Used by the LLM to decide WHICH fragments to retrieve.
- `def clear(self) -> None` — Clear all stored originals.
- `def stats(self) -> dict[str, Any]` — Return store statistics.
- `def get_ccr_store(max_entries: int = 500) -> CompressedContextStore` — Get or create the global CCR store singleton.

## Dependencies
- `__future__`
- `collections`
- `hashlib`
- `threading`
- `typing`
