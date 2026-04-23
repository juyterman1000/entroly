---
claim_id: 5e9f6015-652b-4dfc-b851-49b51946f0f2
entity: adaptive_pruner
status: inferred
confidence: 0.75
sources:
  - entroly\adaptive_pruner.py:39
  - entroly\adaptive_pruner.py:128
  - entroly\adaptive_pruner.py:50
  - entroly\adaptive_pruner.py:59
  - entroly\adaptive_pruner.py:62
  - entroly\adaptive_pruner.py:82
  - entroly\adaptive_pruner.py:112
  - entroly\adaptive_pruner.py:142
  - entroly\adaptive_pruner.py:148
  - entroly\adaptive_pruner.py:151
last_checked: 2026-04-23T03:07:07.786079+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: adaptive_pruner

**Language:** python
**Lines of code:** 163

## Types
- `class EntrolyPruner()` — Adaptive RL pruner backed by ebbiforge_core.AdaptivePruner. Extends entroly's Wilson-score feedback with a `historical_success` dimension: fragments that previously helped get boosted, those that didn
- `class FragmentGuard()` — Code quality scanner backed by ebbiforge_core.CodeQualityGuard. Scans each ingested fragment for: - Hardcoded API secrets  (sk-..., API_KEY = "...") - unsafe Rust blocks - TODO comments - Console spam

## Functions
- `def __init__(self)`
- `def available(self) -> bool`
- `def record_fragment_features(
        self,
        fragment_id: str,
        recency: float,
        relevance: float,
        complexity: float,
        was_selected: bool,
    ) -> None`
- `def apply_feedback(self, fragment_id: str, feedback: float) -> bool` — Apply user feedback to update RL weights for this fragment's features. Args: fragment_id: The fragment that received feedback. feedback:    +1.0 = helpful, -1.0 = not helpful, 0.0 = neutral. Returns: 
- `def score_fragment(
        self,
        recency: float,
        relevance: float,
        historical_success: float,
        complexity: float,
    ) -> float | None`
- `def __init__(self)`
- `def available(self) -> bool`
- `def scan(self, content: str, source: str = "") -> list[str]` — Scan fragment content for code quality issues. Returns list of issue strings (empty = clean).

## Dependencies
- `__future__`
- `logging`
