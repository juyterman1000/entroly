---
claim_id: ca6b063a-b461-4b01-abe9-e2560b31f79f
entity: nkbe
status: inferred
confidence: 0.75
sources:
  - entroly-wasm\src\nkbe.rs:31
  - entroly-wasm\src\nkbe.rs:40
  - entroly-wasm\src\nkbe.rs:50
  - entroly-wasm\src\nkbe.rs:335
  - entroly-wasm\src\nkbe.rs:363
  - entroly-wasm\src\nkbe.rs:375
last_checked: 2026-04-23T03:07:07.904919+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: nkbe

**Language:** rust
**Lines of code:** 481

## Types
- `pub struct AgentBudgetState` — Per-agent state for budget allocation.
- `pub struct NkbeFragment` — Fragment descriptor for NKBE allocation.
- `pub struct NkbeAllocator` — NKBE Allocator — multi-agent token budget allocation.  Implements two-phase KKT bisection with Nash Bargaining refinement and REINFORCE gradient for RL weight learning.

## Functions
- `fn reinforce_gradient(
    features: &[[f64; 4]],    // Per-fragment feature vectors
    selections: &[bool],       // Whether each fragment was selected
    reward: f64,               // Outcome quality
    probabilities: &[f64],     // Selection probabilities p*ᵢ
    tau: f64,                  // Temperature
) -> [f64; 4]` — REINFORCE gradient computation for 4D scoring weights.  ∂E[R]/∂wₖ = Σᵢ (aᵢ − p*ᵢ) · R · σ'(zᵢ/τ) · featureᵢₖ  Returns gradient vector [Δw_recency, Δw_frequency, Δw_semantic, Δw_entropy]. REINFORCE pol
- `fn sigmoid(x: f64) -> f64` — Numerically stable sigmoid.
- `fn softplus(x: f64) -> f64` — Numerically stable softplus: log(1 + exp(x)).

## Dependencies
- `std::collections::HashMap`
