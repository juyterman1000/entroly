---
claim_id: 4290f107-b0ef-483e-81b2-0b08007599b1
entity: change_pipeline
status: inferred
confidence: 0.75
sources:
  - entroly\change_pipeline.py:35
  - entroly\change_pipeline.py:49
  - entroly\change_pipeline.py:59
  - entroly\change_pipeline.py:70
  - entroly\change_pipeline.py:269
  - entroly\change_pipeline.py:81
  - entroly\change_pipeline.py:128
  - entroly\change_pipeline.py:207
  - entroly\change_pipeline.py:276
  - entroly\change_pipeline.py:280
last_checked: 2026-04-23T03:07:07.796450+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: change_pipeline

**Language:** python
**Lines of code:** 425

## Types
- `class ChangeSet()` — A structured representation of a code change.
- `class BeliefDiff()` — How a code change affects existing beliefs.
- `class ReviewFinding()` — A code review finding.
- `class PRBrief()` — An auto-generated PR brief.
- `class ChangePipeline()` — Processes code changes through the Change-Driven flow (④). Pipeline: Diff → ChangeSet → BeliefDiff → PR Brief → Vault

## Functions
- `def to_markdown(self) -> str`
- `def parse_diff(diff_text: str, commit_message: str = "") -> ChangeSet` — Parse a unified diff into a structured ChangeSet.
- `def review_diff(diff_text: str) -> list[ReviewFinding]` — Review a diff for common issues.
- `def __init__(self, vault: VaultManager, verification: VerificationEngine)`
- `def process_diff(
        self,
        diff_text: str,
        commit_message: str = "",
        pr_title: str = "",
    ) -> PRBrief`
- `def refresh_docs(self, changed_files: list[str]) -> dict[str, Any]` — Trigger belief refresh for changed files.

## Dependencies
- `.vault`
- `.verification_engine`
- `__future__`
- `dataclasses`
- `logging`
- `pathlib`
- `re`
- `typing`
