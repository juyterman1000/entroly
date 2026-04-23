---
claim_id: 778e96b8-75cf-4203-a3e3-03e07f3092b8
entity: lib
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\lib.rs:150
  - entroly-wasm\src\lib.rs:48
last_checked: 2026-04-23T03:07:07.903088+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: lib

**Language:** rust
**Lines of code:** 1643

## Types
- `pub struct WasmEntrolyEngine`

## Functions
- `fn json_to_js(val: &T) -> JsValue` — Convert any Serialize type → JsValue via JSON string roundtrip. serde_wasm_bindgen::to_value() does NOT handle dynamic serde_json::Value correctly (produces empty objects). This generic helper works f

## Dependencies
- `cache::`
- `causal::CausalContextGraph`
- `dedup::`
- `depgraph::`
- `entropy::`
- `fragment::`
- `guardrails::`
- `knapsack::`
- `knapsack_sds::`
- `prism::PrismOptimizer`
- `prism::PrismOptimizer5D`
- `query_persona::QueryPersonaManifold`
- `resonance::ResonanceMatrix`
- `serde::`
- `std::collections::`
- `wasm_bindgen::prelude::`

## Linked Beliefs
- [[CausalContextGraph]]
- [[QueryPersonaManifold]]
- [[ResonanceMatrix]]
