---
claim_id: 5a69a4c0-8007-45d3-834c-d97ca2c414a6
entity: prism
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\prism.rs:33
  - entroly-wasm\src\prism.rs:202
  - entroly-wasm\src\prism.rs:447
last_checked: 2026-04-23T03:07:07.905724+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: prism

**Language:** rust
**Lines of code:** 759

## Types
- `pub struct SymMatrixN` — An NxN symmetric matrix stored as a flat array for tracking gradient covariance.  Uses `N * N` flat layout (row-major) since const generic expressions like `[f64; N * N]` require nightly features. The
- `pub struct PrismOptimizerN` — of their individual contributions. PRISM learns to weight this signal relative to the four individual-fragment dimensions.  Key insight: resonance gradients are inherently noisier than individual scor
- `pub struct ResonanceDiagnostics` — Diagnostics for the resonance dimension in 5D PRISM.

## Dependencies
- `serde::`
- `std::f64`
