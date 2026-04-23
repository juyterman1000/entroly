---
claim_id: 46a920c7-db57-45b9-8b8a-cfe5b46a7645
entity: dedup
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\dedup.rs:105
  - entroly-wasm\src\dedup.rs:23
  - entroly-wasm\src\dedup.rs:34
  - entroly-wasm\src\dedup.rs:46
  - entroly-wasm\src\dedup.rs:92
last_checked: 2026-04-23T03:07:07.893272+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: dedup

**Language:** rust
**Lines of code:** 245

## Types
- `pub struct DedupIndex` — LSH-bucketed deduplication index.  Split 64-bit fingerprints into 4 bands of 16 bits. Two fingerprints sharing any band → candidate pair. Verify with full Hamming distance.  Same approach as ebbiforge

## Functions
- `fn hash_token(token: &str) -> u64` — Hash a token to a 64-bit integer using MD5.
- `fn normalized_content_hash(text: &str) -> u64` — Hash normalized content so duplicate verification can distinguish truly repeated fragments from large files that merely share a stable SimHash despite minor semantic edits.
- `fn simhash(text: &str) -> u64` — Compute the 64-bit SimHash fingerprint of a text.  Algorithm: 1. Extract word trigrams as features 2. Hash each feature → 64-bit 3. For each bit: feature hash bit=1 → +1, bit=0 → -1 4. Final bit = 1 i
- `fn hamming_distance(a: u64, b: u64) -> u32` — Hamming distance between two 64-bit fingerprints.

## Dependencies
- `md5::`
- `serde::`
- `std::collections::`
