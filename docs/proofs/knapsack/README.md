# Knapsack — (1 − 1/e) approximation theorem

> **Theorem.** For a non-negative monotone submodular value function `v`
> and a cardinality constraint `k`, entroly's greedy knapsack selector
> returns a subset `S` such that `v(S) ≥ (1 − 1/e) · v(OPT)`.

This is the classic Nemhauser-Wolsey-Fisher result (1978). Entroly's
implementation in [`entroly-core/src/knapsack.rs`](../../entroly-core/src/knapsack.rs)
is a faithful execution of greedy submodular maximization, so the
theorem applies.

The proof targets the *abstract algorithm*, not the bit-level Rust
implementation. The argument is that any concrete implementation that
satisfies the spec inherits the guarantee.

## Why this proof matters

Every benchmark we publish (Accuracy Retention, LooGLE, CodeSearchNet)
is sensitive to selection quality. The (1 − 1/e) ≈ 0.632 bound is the
mathematical floor that justifies trusting the selector below the
empirical numbers we report.

## Status

Scaffold. Lean 4 implementation in progress.

## References

- Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). *An analysis
  of approximations for maximizing submodular set functions—I.*
  Mathematical Programming 14(1).
