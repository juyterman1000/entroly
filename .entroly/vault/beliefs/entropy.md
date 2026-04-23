---
claim_id: 0c52ce42-dd11-442f-a824-424b796bb5c2
entity: entropy
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\entropy.rs:26
  - entroly-wasm\src\entropy.rs:54
  - entroly-wasm\src\entropy.rs:80
  - entroly-wasm\src\entropy.rs:122
  - entroly-wasm\src\entropy.rs:170
  - entroly-wasm\src\entropy.rs:186
  - entroly-wasm\src\entropy.rs:215
  - entroly-wasm\src\entropy.rs:239
  - entroly-wasm\src\entropy.rs:253
  - entroly-wasm\src\entropy.rs:277
last_checked: 2026-04-23T03:07:07.895893+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: entropy

**Language:** rust
**Lines of code:** 580


## Functions
- `fn shannon_entropy(text: &str) -> f64` — Character-level Shannon entropy in bits per character.  Uses a 256-element byte histogram for O(n) computation with virtually zero allocation overhead.
- `fn normalized_entropy(text: &str) -> f64` — Normalize Shannon entropy to [0, 1]. Max entropy for source code is empirically ~6.0 bits/char.
- `fn renyi_entropy_2(text: &str) -> f64` — Computational advantage: requires only Σ p² (no per-symbol log), making it ~30% faster than Shannon on large fragments.  Used as a secondary signal in the IOS knapsack: fragments with high Shannon but
- `fn renyi_entropy_alpha(scores: &[f64], alpha: f64) -> f64` — normalizes them to a probability distribution internally.  Used by EGSC's admission gate: given per-fragment entropy scores s₁, ..., sₖ, we form pᵢ = sᵢ/Σsⱼ and compute H₂(p) to measure the *diversity
- `fn renyi_max(n: usize) -> f64` — Maximum possible Rényi entropy for n elements: H₂_max = log₂(n).  When all pᵢ = 1/n (uniform distribution), H₂ = log₂(n). Used to normalize EGSC admission threshold to [0, 1] scale.
- `fn entropy_divergence(text: &str) -> f64` — but collision entropy is low, the fragment has many unique-but-rare symbols (e.g., binary-encoded data, UUID strings, minified code).  High divergence → likely noise or encoded data, not useful contex
- `fn bits_per_byte(text: &str) -> f64` — Config / boilerplate:     BPB ≈ 0.30–0.45 Minified / compressed:    BPB ≈ 0.85–0.95  Used in the autotune composite score to reward configs that select high-information fragments. ════════════════════
- `fn bpb_quality(text: &str, redundancy: f64) -> f64` — BPB-weighted quality score: 60% density + 40% uniqueness.
- `fn boilerplate_ratio(text: &str) -> f64` — Boilerplate pattern matcher. Returns the fraction of non-empty lines matching common boilerplate.  Hardcoded patterns for speed (no regex dependency): - import/from imports - pass/... - dunder methods
- `fn is_boilerplate(trimmed: &str) -> bool` — Fast boilerplate check without regex.
- `fn cross_fragment_redundancy(
    fragment: &str,
    others: &[&str],
) -> f64` — - 4-grams (n=4) catch near-verbatim duplication: almost identical code blocks. Discriminative for long files where n=3 is too permissive.  **Adaptive weights by word count:** < 20 words  → (0.55, 0.35
- `fn ngram_redundancy(
    words: &[&str],
    others: &[&str],
    ngram_size: usize,
) -> f64` — Compute single-scale n-gram overlap ratio against a set of other fragments. Parallelises over others when len > 10 (Rayon).
- `fn information_score(
    text: &str,
    other_fragments: &[&str],
) -> f64` — Compute the final information density score.  Combines: 40% Shannon entropy (normalized) 30% Boilerplate penalty (1 - ratio) 30% Uniqueness (1 - adaptive multi-scale redundancy)

## Dependencies
- `std::collections::HashSet`
