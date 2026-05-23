# Causal Context Graph — Do-Calculus on RAG

**Implementation:** [`entroly-core/src/causal.rs`](../entroly-core/src/causal.rs)
**Math reference:** [`../RESEARCH.md`](../RESEARCH.md)

## One-paragraph spec

Standard RAG conflates *fragment was selected* with *fragment caused
success*. The Causal Context Graph uses entroly's RAVEN-UCB exploration
mechanism as a **natural instrument variable** (Pearl, *Causality* 2009)
to distinguish observational from interventional fragment value:

> `confounding_bias(f) = E[Y | observe(f)] − E[Y | do(f)]`

Fragments with bias > 0 are coattail riders (look good because their
partners are good); bias < 0 are hidden gems (suppressor variables);
bias ≈ 0 are honestly valuable.

## Why it's novel

Every other RAG/context engine treats fragment selection as an
observational study. Their RL signal is correlation. Entroly's signal is
**de-confounded causal value** — which means fragment scores converge
to *what actually helps*, not *what's around when things go well*.

## How it's used

Each time a fragment is selected, the Causal Context Graph records
whether the selection was policy-driven (observational) or
exploration-driven (interventional). Over time, the gap between the two
estimates emerges, and the confounding bias is subtracted from the
fragment's effective score.

## Why it works

The RAVEN-UCB exploration introduces *random* fragment swaps that act
as a natural experiment. Those swaps satisfy the standard
exogeneity condition for instrument variables, so the resulting
estimate of `P(success | do(f))` is unbiased.

## References

Pearl 2009; Hernán & Robins 2020; Auer, Cesa-Bianchi & Fischer 2002
(UCB).
