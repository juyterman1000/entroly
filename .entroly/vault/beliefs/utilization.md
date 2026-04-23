---
claim_id: 58a42577-f190-4759-88a9-386e52808fce
entity: utilization
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\utilization.rs:28
  - entroly-wasm\src\utilization.rs:43
  - entroly-wasm\src\utilization.rs:55
  - entroly-wasm\src\utilization.rs:73
  - entroly-wasm\src\utilization.rs:82
last_checked: 2026-04-23T03:07:07.911906+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: utilization

**Language:** rust
**Lines of code:** 237

## Types
- `pub struct FragmentUtilization` — Utilization score for a single injected fragment.
- `pub struct UtilizationReport` — Session-level utilization report.

## Functions
- `fn trigrams(text: &str) -> HashSet<Vec<String>>` — Extract word trigrams from text as a HashSet.
- `fn identifier_set(text: &str) -> HashSet<String>` — Extract identifiers from text as a HashSet.
- `fn score_utilization(
    fragments: &[&ContextFragment],
    response: &str,
) -> UtilizationReport` — Score how much of each injected fragment the LLM actually used in its response.  Call this after receiving the LLM response, passing in the fragments that were injected into the prompt context.

## Dependencies
- `crate::depgraph::extract_identifiers`
- `crate::fragment::ContextFragment`
- `serde::`
- `std::collections::HashSet`

## Linked Beliefs
- [[extract_identifiers]]
- [[ContextFragment]]
