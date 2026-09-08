# Resonance Matrix — Supermodular Pairwise Learning

**Implementation:** [`entroly-core/src/resonance.rs`](../entroly-core/src/resonance.rs)
**Math reference:** [`../RESEARCH.md`](../RESEARCH.md)

## One-paragraph spec

Standard RAG scores fragments independently and sums. But codebases
have **compositional context value**: a struct definition is useless
without the function that uses it; an interface is useless without an
implementation. The Resonance Matrix tracks the **pairwise interaction
strength** between fragments — learned online from outcomes — and adds
a **supermodular bonus** to fragments whose partners are already
selected:

> `resonance_bonus(c | S) = Σ_{s ∈ S} R[c][s] / |S|`

## Why it's novel

Submodular knapsack is the textbook solution to context selection. It
assumes **diminishing returns** — every additional fragment is worth
less than the last. That assumption is wrong for compositional code,
where the second-half of a dependency pair is *worth more* than the
first half alone. Resonance models this by breaking submodularity
explicitly and tracking learned positive co-occurrence.

## How it learns

On `record_success`: for every co-selected pair `(i, j)`,
`R[i][j] += η · (reward − baseline)`.
On `record_failure`: same update with negative reward — the pair gets
suppressed.

## Complexity management

- Sparse storage: only pairs that have been co-selected are tracked.
- Decay: `R[i][j] *= decay_rate` per turn (prevents stale resonances).
- Bound: O(N · k) for N selected fragments and k average resonant
  partners per fragment — bounded by the eviction policy.

## What it solves

Fragments that "explain each other" (struct + ctor, interface + impl,
caller + callee, schema + migration) become preferred selections,
without us hand-coding the dependency graph.
