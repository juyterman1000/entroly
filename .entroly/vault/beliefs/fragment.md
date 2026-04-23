---
claim_id: 3fde34e2-64bb-4660-bfb8-4939bef58e83
entity: fragment
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\fragment.rs:14
  - entroly-wasm\src\fragment.rs:84
  - entroly-wasm\src\fragment.rs:128
  - entroly-wasm\src\fragment.rs:141
last_checked: 2026-04-23T03:07:07.896920+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: fragment

**Language:** rust
**Lines of code:** 217

## Types
- `pub struct ContextFragment` — A single piece of context (code snippet, file content, tool result, etc.)

## Functions
- `fn compute_relevance(
    frag: &ContextFragment,
    w_recency: f64,
    w_frequency: f64,
    w_semantic: f64,
    w_entropy: f64,
    feedback_multiplier: f64,
) -> f64` —  Direct port of ebbiforge-core ContextScorer::score() but with entropy replacing emotion as the fourth dimension.  `feedback_multiplier` comes from FeedbackTracker::learned_value(): - > 1.0 = historic
- `fn softcap(x: f64, cap: f64) -> f64` — Logit softcap: `c · tanh(x / c)`.  Gemini-style bounded scoring. When `cap ≤ 0`, falls back to `min(x, 1)`.
- `fn apply_ebbinghaus_decay(
    fragments: &mut [ContextFragment],
    current_turn: u32,
    half_life: u32,
)` — Apply Ebbinghaus forgetting curve decay to all fragments.  recency(t) = exp(-λ · Δt) where λ = ln(2) / half_life  Same math as ebbiforge-core HippocampusEngine.

## Dependencies
- `serde::`
