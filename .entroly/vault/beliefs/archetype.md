---
claim_id: c9b7f4d1-e1fd-4fa8-b1b1-988d12a8fc85
entity: archetype
status: inferred
confidence: 0.75
sources:
  - entroly-core\src\archetype.rs:75
  - entroly-core\src\archetype.rs:114
  - entroly-core\src\archetype.rs:135
  - entroly-core\src\archetype.rs:161
  - entroly-core\src\archetype.rs:192
  - entroly-core\src\archetype.rs:630
last_checked: 2026-04-23T03:07:07.832597+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: archetype

**Language:** rust
**Lines of code:** 730

## Types
- `pub struct Fingerprint` — [8]  test_ratio:     fraction of files in test directories [9]  graph_density:  edges / (nodes * (nodes-1)) of dep graph [10] max_depth:      max dependency chain depth (normalized) [11] module_count:
- `pub struct CodebaseStats` — Raw statistics collected from a codebase scan. The fingerprinter converts these into a normalized Fingerprint.
- `pub struct WeightProfile` — A scoring weight profile for a specific archetype.
- `pub struct Archetype` — An archetype cluster: centroid + optimized weights.
- `pub struct ArchetypeEngine` — The core archetype detection and classification engine.  Maintains an online k-means clustering of codebase fingerprints. Each cluster (archetype) has its own optimized weight profile that the Dreamin

## Functions
- `fn log_norm(value: f64, max_ref: f64) -> f64` —  f(x) = ln(1 + x) / ln(1 + max_ref)  Properties: - f(0) = 0 - f(max_ref) ≈ 1 - Compresses heavy tails (file counts, import counts) - Monotonically increasing

## Dependencies
- `pyo3::prelude::`
- `pyo3::types::PyDict`
- `serde::`
- `std::collections::HashMap`
