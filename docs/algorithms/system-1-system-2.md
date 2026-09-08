# System 1 ↔ System 2 — Dual-Process Coupling

**Implementation:** [`entroly/coupling.py`](../entroly/coupling.py)
**Math reference:** [`../RESEARCH.md`](../RESEARCH.md)
**Commit history:** `4eaa0a5` (v0.19.1 introduction)

## One-paragraph spec

Entroly has two cognitions on the same substrate that, before v0.19.1,
did not exchange information:

- **System 1** (fast, automatic): Rust engine in the proxy. Knapsack +
  entropy + dedup + guardrails + channel + resonance + causal. Fires
  on every request.
- **System 2** (slow, deliberate): Python flow_orchestrator + verifier
  stack + vault. Compiles beliefs, runs BIPT / FORGE / TRIAD / PROVE /
  CAVE, writes vault artifacts.

Coupling closes the loop in both directions:

- **S2 → S1:** verified vault beliefs project into the engine as ranked
  fragment candidates. The engine's knapsack weighs them against
  IDE-supplied fragments under a single objective. `C* = argmax
  optimize(F ∪ π(B; q), T)`.

- **S1 → S2:** outcomes posted to the proxy trigger Bayesian updates
  on every belief that was used. `c' = (c · L_g) / (c · L_g + (1 − c) ·
  L_u)` with `L_g = α` for success, `1 − α` for failure (defaults
  α = 0.85, β = 0.40). Posteriors clamped to [0.01, 0.99]. Failed
  beliefs enqueue for reverification.

## Why this matters

Most context systems have *either* fast retrieval (RAG) *or* deliberate
verification (research-grade hallucination detectors). Entroly is the
only one with both, with **feedback between them**. The result: the
deliberate verifier's output (verified beliefs) becomes input to the
fast selector, and the fast selector's outcomes (success/failure) tune
the deliberate verifier's beliefs over time.

Mathematically, this is **Kahneman's dual-process architecture**
applied to AI coding context. (Daniel Kahneman, *Thinking, Fast and
Slow*, 2011.)

## Feature gate

`ENTROLY_VAULT_COUPLING=1` enables coupling in production. Default off
pending field tests; the architectural seam is in place since v0.19.1.

## References

Kahneman 2011 (dual-process theory); Bayes 1763 (posterior update);
[`coupling.py:attribute_outcome`](../entroly/coupling.py).
