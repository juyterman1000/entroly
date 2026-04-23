---
claim_id: c3756320-c8a1-4854-a0f2-b1234ed9ed62
entity: provenance
status: inferred
confidence: 0.75
sources:
  - entroly\provenance.py:25
  - entroly\provenance.py:46
  - entroly\provenance.py:36
  - entroly\provenance.py:63
  - entroly\provenance.py:69
  - entroly\provenance.py:75
  - entroly\provenance.py:80
  - entroly\provenance.py:85
  - entroly\provenance.py:97
  - entroly\provenance.py:127
last_checked: 2026-04-23T03:07:07.813420+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: provenance

**Language:** python
**Lines of code:** 185

## Types
- `class FragmentProvenance()` — Provenance record for a single selected context fragment.
- `class ContextProvenance()` — Full provenance record for one optimize_context call. The hallucination_risk is computed from: 1. Fraction of selected fragments with verified sources 2. Average confidence of selection 3. Whether any

## Functions
- `def risk_contribution(self) -> str` — Contribution to hallucination risk.
- `def verified_fraction(self) -> float`
- `def avg_confidence(self) -> float`
- `def source_set(self) -> set` — Set of verified source files — use to check LLM citations.
- `def quality_flagged_sources(self) -> list[str]` — Sources with code quality issues.
- `def hallucination_risk(self) -> str` — low    — all fragments file-backed, high confidence medium — some low-confidence fragments, or 1-2 unverified high   — significant unverified content or very low confidence
- `def to_dict(self) -> dict[str, Any]`
- `def build_provenance(
    optimize_result: dict[str, Any],
    query: str,
    refined_query: str | None,
    turn: int,
    token_budget: int,
    quality_scan_fn=None,  # Optional: FragmentGuard.scan
) -> ContextProvenance`

## Dependencies
- `__future__`
- `dataclasses`
- `typing`
