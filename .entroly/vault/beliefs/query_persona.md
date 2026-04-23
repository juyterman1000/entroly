---
claim_id: c2e95971-c58c-41a0-96a1-4067ee899074
entity: query_persona
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\query_persona.rs:73
  - entroly-wasm\src\query_persona.rs:133
  - entroly-wasm\src\query_persona.rs:256
  - entroly-wasm\src\query_persona.rs:339
  - entroly-wasm\src\query_persona.rs:414
  - entroly-wasm\src\query_persona.rs:649
  - entroly-wasm\src\query_persona.rs:660
  - entroly-wasm\src\query_persona.rs:384
last_checked: 2026-04-23T03:07:07.907435+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: query_persona

**Language:** rust
**Lines of code:** 843

## Types
- `pub struct RbfKernel`
- `pub struct ParticleCloud`
- `pub struct QueryArchetype` — A query archetype = a probability measure over query feature vectors + a learned PRISM weight set for context selection.
- `pub struct PitmanYorProcess`
- `pub struct QueryPersonaManifold` — Discovers query archetypes from TF-IDF feature vectors and maintains per-archetype PRISM weight sets for specialized context selection.
- `pub struct ManifoldStats`
- `pub struct ArchetypeInfo`

## Functions
- `fn build_query_features(
    tfidf_scores: &[f64],
    vagueness: f64,
    query_length: usize,
    num_key_terms: usize,
    needs_refinement: bool,
) -> Vec<f32>` — Build a QUERY_DIM-dimensional feature vector from TF-IDF key terms and query metadata. This is the embedding that PSM operates on.  Layout: [0..11]  = top-12 TF-IDF scores (padded with 0.0 if fewer) [

## Dependencies
- `crate::prism::PrismOptimizer`
- `serde::`
