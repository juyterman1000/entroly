---
claim_id: 31a2d5e0-8160-44ed-b115-1810891ec2b3
entity: semantic_dedup
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\semantic_dedup.rs:133
  - entroly-wasm\src\semantic_dedup.rs:43
  - entroly-wasm\src\semantic_dedup.rs:50
  - entroly-wasm\src\semantic_dedup.rs:72
  - entroly-wasm\src\semantic_dedup.rs:99
  - entroly-wasm\src\semantic_dedup.rs:142
last_checked: 2026-04-23T03:07:07.910087+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: semantic_dedup

**Language:** rust
**Lines of code:** 264

## Types
- `pub struct DeduplicationResult` — Convenience: deduplicate and return count stats.

## Functions
- `fn content_overlap(a: &str, b: &str) -> f64` — Compute content overlap between two fragments using: 50% n-gram overlap (word trigrams) 50% identifier overlap  Returns [0, 1] where 1.0 = identical information.
- `fn trigram_jaccard(a: &str, b: &str) -> f64` — Symmetric trigram Jaccard similarity.
- `fn identifier_jaccard(a: &str, b: &str) -> f64` — Symmetric identifier Jaccard similarity.
- `fn semantic_deduplicate(
    fragments: &[ContextFragment],
    sorted_indices: &[usize],
    threshold: Option<f64>,
) -> Vec<usize>` — contribute at least `threshold` marginal information.  # Arguments * `fragments` - all fragments in the engine * `sorted_indices` - indices into `fragments`, sorted by relevance descending * `threshol
- `fn semantic_deduplicate_with_stats(
    fragments: &[ContextFragment],
    sorted_indices: &[usize],
    threshold: Option<f64>,
) -> DeduplicationResult`

## Dependencies
- `crate::depgraph::extract_identifiers`
- `crate::fragment::ContextFragment`
- `std::collections::HashSet`

## Linked Beliefs
- [[extract_identifiers]]
- [[ContextFragment]]
