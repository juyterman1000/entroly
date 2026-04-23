---
claim_id: f6b909d1-93f8-4ef5-bb6a-e8ddc4cb36ff
entity: langchain
status: inferred
confidence: 0.75
sources:
  - entroly\integrations\langchain.py:30
  - entroly\integrations\langchain.py:40
  - entroly\integrations\langchain.py:57
  - entroly\integrations\langchain.py:80
  - entroly\integrations\langchain.py:84
  - entroly\integrations\langchain.py:88
last_checked: 2026-04-23T03:07:07.829161+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: langchain

**Language:** python
**Lines of code:** 132

## Types
- `class EntrolyCompressor()` — LangChain-compatible Runnable that compresses message content. Integrates with LangChain's LCEL (LangChain Expression Language) as a transparent middleware that compresses messages before they reach t

## Functions
- `def __init__(
        self,
        budget: int = 50_000,
        preserve_last_n: int = 4,
        content_type: str | None = None,
    )`
- `def invoke(self, input: Any, config: dict | None = None) -> Any` — Compress input messages or text. Handles: - str → compressed str - list[dict] → compressed message list - LangChain BaseMessage list → compressed messages
- `def batch(self, inputs: list[Any], config: dict | None = None) -> list[Any]` — Batch compress multiple inputs.
- `def ainvoke(self, input: Any, config: dict | None = None) -> Any` — Async version of invoke.
- `def abatch(self, inputs: list[Any], config: dict | None = None) -> list[Any]` — Async batch.

## Dependencies
- `__future__`
- `typing`
