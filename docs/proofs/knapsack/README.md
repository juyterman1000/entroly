# Knapsack — modular 1/2 approximation theorem

> **Theorem target.** For non-negative modular item values and a 0/1 knapsack
> budget, let \(G\) be the feasible set produced by descending value-per-token
> density and let \(M\) be the highest-value feasible singleton. Then
>
> \[
> \max\{v(G), v(M)\} \ge \tfrac{1}{2} v(OPT).
> \]
>
> Entroly's large-set hard-selection fallback in
> [`entroly-engine/src/knapsack.rs`](../../../entroly-engine/src/knapsack.rs)
> now implements exactly this better-of-two rule.

The earlier scaffold incorrectly targeted a \((1-1/e)\) submodular guarantee.
That theorem does not describe this selector: the shipped fallback objective is
modular, and the selector does not implement the partial-enumeration algorithm
needed for the stronger submodular-knapsack result.

## Why this proof matters

The production fallback is used when the hard-path candidate set is too large
for the bounded dynamic-programming path. A density-only implementation can be
arbitrarily worse than one half of optimum: a tiny high-density item can make a
whole-budget, much higher-value item infeasible. Comparing the density result
with the best feasible singleton is the small algorithmic step that restores
the classical constant-factor bound.

The theorem is deliberately scoped to the **internal non-negative modular
selection objective**. It does not prove answer correctness, evidence recall,
semantic sufficiency, or model quality.

## Status

Scaffold. The Rust implementation and adversarial regression test are present;
a machine-checked Lean 4 proof has not yet been added.

## References

- Dantzig, G. B. (1957). *Discrete-variable extremum problems.*
  Operations Research 5(2).
- Standard LP-rounding analysis for 0/1 knapsack: density-greedy together with
  the best feasible singleton yields the 1/2 approximation bound.
