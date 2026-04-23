---
claim_id: 398df1d6-93d4-42a8-9520-30330b425534
entity: channel
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\channel.rs:393
  - entroly-wasm\src\channel.rs:41
  - entroly-wasm\src\channel.rs:59
  - entroly-wasm\src\channel.rs:76
  - entroly-wasm\src\channel.rs:100
  - entroly-wasm\src\channel.rs:182
  - entroly-wasm\src\channel.rs:202
  - entroly-wasm\src\channel.rs:318
  - entroly-wasm\src\channel.rs:358
  - entroly-wasm\src\channel.rs:411
last_checked: 2026-04-23T03:07:07.890709+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: channel

**Language:** rust
**Lines of code:** 1330

## Types
- `pub struct ContradictionReport` — Result of contradiction scan.

## Functions
- `fn trigram_hashes(text: &str) -> Vec<u64>` — Extract character 3-gram hashes from text. Uses FNV-1a-style mixing: fast, zero-alloc per gram, good distribution.
- `fn build_trigram_set(frags: &[ContextFragment], indices: &[usize]) -> HashSet<u64>` — Build the set of trigram hashes from selected fragments' content. This is the "information already in the channel."
- `fn marginal_gain(
    candidate: &ContextFragment,
    selected_trigrams: &HashSet<u64>,
    dep_bonus: f64,
) -> f64` — Marginal information gain of adding candidate x to selection S.  ΔI(x | S) = entropy(x) × (1 − overlap(x, S)) × dep_bonus  Where overlap = fraction of x's trigrams already in S's trigram set. Submodul
- `fn channel_trailing_pass(
    frags: &[ContextFragment],
    selected_indices: &[usize],
    token_gap: u32,
    dep_graph: &DepGraph,
) -> Vec<usize>` — information gain instead of greedy density.  After IOS/knapsack selects fragments, a token gap Δ = budget − used remains. This function selects additional fragments to fill that gap, prioritizing frag
- `fn attention_weight(position: usize, total: usize) -> f64` — LLM attention weight at position p in context of length L.  α(p, L) = 0.4·exp(−p/(0.15·L)) + 0.4·exp(−(L−p−1)/(0.15·L)) + 0.2  U-shaped: peaks at start (primacy) and end (recency), valley in the middl
- `fn semantic_interleave(
    frags: &[ContextFragment],
    selected_indices: &[usize],
    relevances: &[f64],
    dep_graph: &DepGraph,
) -> Vec<usize>` —  Algorithm: 1. Topological sort for causal ordering (defs before refs) 2. Within each level, sort by importance descending 3. Place level-0 (defs) at the front, leaf level at the back 4. Most importan
- `fn information_reward(
    fragment_utils: &[(f64, f64, usize)` — Attention-weighted utilization reward.  R = η × (1 + sufficiency_bonus)  η = Σᵢ(util_i × entropy_i × α_i) / Σᵢ(entropy_i × α_i)  This is "information received by decoder" / "information sent by encode
- `fn modulated_reward(success: bool, sufficiency: f64) -> f64` — Success: R ∈ [0.5, 1.0] — higher when sufficiency is high Failure: R ∈ [−1.5, −0.5] — penalizes more when sufficiency is low  Better credit assignment than flat ±1: "failed with bad context" is a stro
- `fn contradiction_guard(
    frags: &[ContextFragment],
    selected_indices: &[usize],
    relevances: &[f64],
    sdr_threshold: f64,       // Default: 0.25
    structural_threshold: f64, // Default: 0.60 — source path similarity
) -> (Vec<usize>, ContradictionReport)` — Two fragments contradict if: 1. Their source paths are structurally similar (SimHash of source ≥ threshold) 2. Their content is divergent (content SimHash distance > threshold)  The fragment with lowe
- `fn bookend_calibrate(
    ordered_indices: &[usize],
    frags: &[ContextFragment],
    relevances_map: &HashMap<usize, f64>,
    dep_graph: &DepGraph,
) -> Vec<usize>` — Within each causal level, reorders fragments so that the most important ones occupy the attention-peak positions (start and end of that level's span within the full sequence).  `ordered_indices`: outp

## Dependencies
- `crate::depgraph::DepGraph`
- `crate::fragment::ContextFragment`
- `crate::guardrails::`
- `std::collections::`

## Linked Beliefs
- [[DepGraph]]
- [[ContextFragment]]
