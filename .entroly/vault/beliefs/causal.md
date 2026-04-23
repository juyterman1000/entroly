---
claim_id: 9b3fb0bb-49fc-4a46-bcdf-042403861e4d
entity: causal
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\causal.rs:154
  - entroly-wasm\src\causal.rs:175
last_checked: 2026-04-23T03:07:07.889833+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: causal

**Language:** rust
**Lines of code:** 1066

## Types
- `pub struct CausalStats` — Statistics for observability.
- `pub struct CausalContextGraph` — Causal Context Graph — learns fragment causal effects via natural experiments.  Uses the exploration mechanism as an instrumental variable to separate true causal effects from selection bias, discover

## Dependencies
- `serde::`
- `std::collections::`
