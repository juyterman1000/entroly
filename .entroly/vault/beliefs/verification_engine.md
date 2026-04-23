---
claim_id: d2b42e82-e3a9-4f0c-9414-ceb51da8c3d2
entity: verification_engine
status: inferred
confidence: 0.75
sources:
  - entroly\verification_engine.py:35
  - entroly\verification_engine.py:46
  - entroly\verification_engine.py:57
  - entroly\verification_engine.py:65
  - entroly\verification_engine.py:75
  - entroly\verification_engine.py:118
  - entroly\verification_engine.py:87
  - entroly\verification_engine.py:126
  - entroly\verification_engine.py:136
  - entroly\verification_engine.py:180
last_checked: 2026-04-23T03:07:07.825799+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: verification_engine

**Language:** python
**Lines of code:** 500

## Types
- `class Contradiction()` — Two beliefs that disagree.
- `class StaleReport()` — A belief past its freshness window.
- `class CoverageGap()` — A file or module with no corresponding belief.
- `class BlastRadius()` — Impact analysis result for a change.
- `class VerificationReport()` — Complete verification pass result.
- `class VerificationEngine()` — Runs automated verification passes against the belief vault. This is the Belief CI engine — it challenges every belief in the vault and produces verification artifacts.

## Functions
- `def to_dict(self) -> dict[str, Any]`
- `def __init__(
        self,
        vault: VaultManager,
        freshness_hours: float = 24.0,
        min_confidence: float = 0.5,
    )`
- `def full_verification_pass(self) -> VerificationReport` — Run a complete verification pass on all beliefs.
- `def check_belief(self, claim_id: str) -> dict[str, Any]` — Verify a single belief by claim_id.
- `def blast_radius(
        self,
        changed_files: list[str],
    ) -> BlastRadius`
- `def coverage_gaps(self, source_dir: str) -> list[CoverageGap]` — Find source files with no corresponding belief.

## Dependencies
- `.vault`
- `__future__`
- `dataclasses`
- `datetime`
- `logging`
- `pathlib`
- `re`
- `typing`
