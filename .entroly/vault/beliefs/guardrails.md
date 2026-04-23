---
claim_id: 4e239328-293d-4bc6-a815-730621944f8a
entity: guardrails
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\guardrails.rs:314
  - entroly-wasm\src\guardrails.rs:24
  - entroly-wasm\src\guardrails.rs:214
  - entroly-wasm\src\guardrails.rs:36
  - entroly-wasm\src\guardrails.rs:142
  - entroly-wasm\src\guardrails.rs:282
last_checked: 2026-04-23T03:07:07.897799+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: guardrails

**Language:** rust
**Lines of code:** 561

## Types
- `pub struct FeedbackTracker` — Feedback loop: record which fragments influenced a successful output.  Extended with per-fragment Welford variance tracking for RAVEN-UCB adaptive exploration (arXiv:2506.02933, June 2025).
- `pub enum Criticality` — Criticality level — overrides entropy and relevance scoring.
- `pub enum TaskType` — Returns a multiplier [1.0, 10.0] for the relevance score. Adaptive budget allocation based on task type.  Different tasks need different context volumes: - Bug tracing: LARGE budget (need call chains,

## Functions
- `fn file_criticality(path: &str) -> Criticality` — Check if a file path matches critical file patterns.
- `fn has_safety_signal(content: &str) -> bool` — Check content for safety signals that must never be stripped.
- `fn compute_ordering_priority(
    relevance: f64,
    criticality: Criticality,
    is_pinned: bool,
    dep_count: usize,
) -> f64` — Context ordering strategy.  LLMs are order-sensitive. Fragment ordering affects reasoning quality. We order by: pinned first → critical → high relevance → imports → rest

## Dependencies
- `serde::`
- `std::collections::HashMap`

## Key Invariants
- has_safety_signal: Check content for safety signals that must never be stripped.
