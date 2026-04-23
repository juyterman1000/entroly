---
claim_id: 99f3760e-f058-4368-b165-d83fad98920f
entity: anomaly
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\anomaly.rs:58
  - entroly-wasm\src\anomaly.rs:78
  - entroly-wasm\src\anomaly.rs:44
  - entroly-wasm\src\anomaly.rs:86
  - entroly-wasm\src\anomaly.rs:99
  - entroly-wasm\src\anomaly.rs:112
  - entroly-wasm\src\anomaly.rs:257
last_checked: 2026-04-23T03:07:07.887834+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: anomaly

**Language:** rust
**Lines of code:** 372

## Types
- `pub struct EntropyAnomaly` — A single entropy anomaly detected in the codebase.
- `pub struct AnomalyReport` — Full anomaly scan report.
- `pub enum AnomalyType`

## Functions
- `fn median(sorted: &[f64]) -> f64` — Compute the median of a sorted slice.
- `fn directory_of(source: &str) -> String` — Extract directory from a source path. "src/handlers/auth.rs" → "src/handlers" "auth.rs" → ""
- `fn scan_anomalies(fragments: &[&ContextFragment]) -> AnomalyReport` — Scan all fragments for entropy anomalies using robust Z-scores.  Groups fragments by directory, computes MAD-based Z-scores within each group, and flags fragments with |z| > 2.5.
- `fn basename(path: &str) -> &str`

## Dependencies
- `crate::entropy::boilerplate_ratio`
- `crate::fragment::ContextFragment`
- `serde::`
- `std::collections::HashMap`

## Linked Beliefs
- [[boilerplate_ratio]]
- [[ContextFragment]]
