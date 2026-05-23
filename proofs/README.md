# Entroly Proofs

> Machine-checked theorems for the load-bearing claims entroly makes.

This directory holds Lean 4 formal proofs of entroly's core mathematical
guarantees. The goal is not to prove every line of code — that's a research
program of its own. The goal is to give a **handful of load-bearing claims**
a level of trust that no other AI coding context engine offers:

> If a theorem in this directory verifies with `lake build`, it constitutes a
> machine-checkable proof of the corresponding property over the abstract
> mathematical structure that entroly implements.

## Why one proof beats 82 proofs

A common pattern in adjacent projects is to publish dozens of trivial Lean
theorems for marketing surface ("82 theorems!"). Most of them either prove
nothing load-bearing or prove a tautology. We're taking the opposite bet:

> One end-to-end theorem about a property the user actually cares about,
> shipped in CI, is worth more than eighty theatrical theorems shipped in
> a research/ folder nobody reads.

## Planned theorems

| Theorem | Property | Status |
|---|---|---|
| **knapsack** | Entroly's submodular knapsack selector returns a result within (1 − 1/e) of the optimal under the standard cardinality constraint. | scaffold |
| **bipt** | BIPT's IPD score is monotonic in the byte-level provenance of the output (longer matches → lower IPD, no edge cases). | scaffold |
| **provenance-preservation** | Every fragment injected by the proxy carries a claim_id that survives compression, contradiction-guard eviction, and re-ordering. | scaffold |

Each subdirectory ships with a `lakefile.lean`, the proof itself, and a
short README explaining the theorem in plain English.

## How to run

Once a proof is implemented (status ≠ scaffold):

```bash
cd proofs/<theorem>
lake build       # builds & verifies; non-zero exit on proof failure
```

CI integration: when a theorem is ready, it's wired into `.github/workflows/`
so the build fails if anyone breaks the abstract property the proof targets.

## Contributing

Lean 4 experience welcome. If you spot a theorem we should add or an
issue with an existing proof, open a PR with a `proofs/<name>/` directory
containing `lakefile.lean`, the proof, and a README.

---

*Last updated: v0.19.1.*
