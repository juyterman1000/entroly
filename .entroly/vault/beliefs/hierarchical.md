---
claim_id: f045edbd-b2ea-4a31-83fc-293b71e38c9f
entity: hierarchical
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\hierarchical.rs:29
  - entroly-wasm\src\hierarchical.rs:54
  - entroly-wasm\src\hierarchical.rs:82
  - entroly-wasm\src\hierarchical.rs:175
  - entroly-wasm\src\hierarchical.rs:224
  - entroly-wasm\src\hierarchical.rs:326
  - entroly-wasm\src\hierarchical.rs:372
  - entroly-wasm\src\hierarchical.rs:392
  - entroly-wasm\src\hierarchical.rs:416
last_checked: 2026-04-23T03:07:07.899900+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: hierarchical

**Language:** rust
**Lines of code:** 750

## Types
- `pub struct HccResult` — Result of hierarchical compression.

## Functions
- `fn compress_level1(fragments: &[ContextFragment]) -> (String, u32)` — Generate a one-line summary for each fragment from its skeleton.  Format: "file:auth.py → AuthService, login(), verify_token()" If no skeleton, falls back to source name only.  Target: ~3-8 tokens per
- `fn extract_oneliner_from_skeleton(skeleton: &str) -> String` — Extract a compact one-liner from a skeleton string. Pulls out function/class/struct/trait names.
- `fn identify_cluster(
    dep_graph: &DepGraph,
    query_relevant_ids: &[String],
    max_depth: usize,
) -> Vec<String>` — via the dependency graph.  This is our "symbol-reachability slice" — an approximation of backward+forward program slicing using the dep graph.  Steps: 1. Find fragments with highest semantic scores (q
- `fn compress_level2(
    fragments: &[ContextFragment],
    cluster_ids: &HashSet<String>,
) -> (String, u32)` — Generate expanded skeleton text for the cluster.  Cluster fragments get their full skeleton. This gives the LLM structural understanding of the query neighborhood.
- `fn allocate_budget(
    total_budget: u32,
    n_files: usize,
    mean_entropy: f64,
) -> (u32, u32, u32)` — Allocate token budget across the three levels.  Strategy: - L1 gets min(budget * 0.05, 500) — always fits - L2 gets min(budget * 0.25, remaining * 0.35) — scales with budget - L3 gets the rest — knaps
- `fn submodular_marginal_gain(
    candidate_source: &str,
    candidate_relevance: f64,
    selected_sources: &[String],
) -> f64` — marginal_gain(f | S) = relevance(f) × diversity_penalty(f, S)  where diversity_penalty decreases as more fragments from the same "cluster" (connected component) are already selected.  This naturally p
- `fn extract_module(source: &str) -> String` — Extract the module/directory path from a source string. "file:src/auth/login.py" → "src/auth"
- `fn hierarchical_compress(
    fragments: &[ContextFragment],
    dep_graph: &DepGraph,
    query_relevant_ids: &[String],
    total_budget: u32,
    mean_entropy: f64,
) -> HccResult` —  Arguments: - fragments: all indexed fragments - dep_graph: the dependency graph - query_relevant_ids: fragment IDs ranked by query relevance (most relevant first) - total_budget: total token budget f

## Dependencies
- `crate::depgraph::DepGraph`
- `crate::fragment::ContextFragment`
- `std::collections::`

## Linked Beliefs
- [[DepGraph]]
- [[ContextFragment]]
