---
claim_id: 2b6a2b5a-4db6-446c-a97a-c1e57c0c2475
entity: lsh
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\lsh.rs:111
  - entroly-wasm\src\lsh.rs:179
last_checked: 2026-04-23T03:07:07.904093+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: lsh

**Language:** rust
**Lines of code:** 301

## Types
- `pub struct LshIndex` — Multi-Probe LSH Index for sub-linear SimHash similarity search.  Public API: - `insert(fp, idx)` — called on every `ingest()` - `remove(fp, idx)` — called on eviction / dedup replacement - `query(fp) 
- `pub struct ContextScorer` —  Combines similarity (Hamming-based), recency (Ebbinghaus decay), and entropy (information density) into a single relevance score.  Ported from ebbiforge-core/src/memory/lsh.rs::ContextScorer, adapted

## Dependencies
- `std::collections::HashMap`
