---
claim_id: bd9baec4-5b0b-46e6-b508-02fa3473f4d1
entity: long_term_memory
status: inferred
confidence: 0.75
sources:
  - entroly\long_term_memory.py:79
  - entroly\long_term_memory.py:108
  - entroly\long_term_memory.py:71
  - entroly\long_term_memory.py:87
  - entroly\long_term_memory.py:123
  - entroly\long_term_memory.py:153
  - entroly\long_term_memory.py:156
  - entroly\long_term_memory.py:164
  - entroly\long_term_memory.py:228
  - entroly\long_term_memory.py:278
last_checked: 2026-04-23T03:07:07.810958+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: long_term_memory

**Language:** python
**Lines of code:** 318

## Types
- `class SalienceProfile()` — Maps entroly fragment properties to hippocampus salience values.
- `class LongTermMemory()` — Adapter between entroly's session-level context engine and hippocampus' cross-session memory. Lifecycle: 1. Created once when EntrolyEngine starts (in server.py) 2. On each optimize_context(): recall 

## Functions
- `def is_available() -> bool` — Check if hippocampus-sharp-memory is installed and available.
- `def compute(
        self,
        is_pinned: bool = False,
        entropy_score: float = 0.0,
        was_selected: bool = False,
        relevance: float = 0.0,
    ) -> float`
- `def __init__(
        self,
        capacity: int = 10_000,
        consolidation_interval: int = 50,
        recall_reinforcement: float = 1.3,
    )`
- `def active(self) -> bool`
- `def tick(self) -> None` — Advance the hippocampus clock by 1 tick (called on each advance_turn).
- `def remember_fragments(
        self,
        fragments: list[dict],
        selected_ids: set[str] | None = None,
    ) -> int`
- `def recall_relevant(
        self,
        query: str,
        top_k: int = 5,
        min_retention: float = 0.2,
    ) -> list[dict]`
- `def stats(self) -> dict` — Get long-term memory statistics for the dashboard.
- `def consolidate(self) -> str` — Force a consolidation cycle (sleep-replay).

## Dependencies
- `__future__`
- `dataclasses`
- `logging`
