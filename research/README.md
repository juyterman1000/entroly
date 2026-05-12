# Entroly Research

This directory will host long-form research artifacts — extended papers,
proofs, empirical write-ups, and reproducibility scripts — for the
algorithms that power entroly.

For the canonical mathematical overview, see [`../RESEARCH.md`](../RESEARCH.md).
For the standalone BIPT paper, see [`../BIPT.md`](../BIPT.md).

## Planned papers

| Paper | Status | Algorithm |
|---|---|---|
| **BIPT** — Byte-level Information Provenance Tracing via Suffix Automaton on Kolmogorov-Bounded Output | drafted ([`../BIPT.md`](../BIPT.md)) | [`entroly/verifiers/provenance_tracer.py`](../entroly/verifiers/provenance_tracer.py) |
| **NKBE** — Nash-KKT Budgetary Equilibrium for Multi-Agent Context Allocation | outline | [`entroly-core/src/nkbe.rs`](../entroly-core/src/nkbe.rs) |
| **Causal Context Graph** — Do-Calculus on Retrieval-Augmented Generation via Exploration as Instrument Variable | outline | [`entroly-core/src/causal.rs`](../entroly-core/src/causal.rs) |
| **Cognitive Bus** — ISA Event Routing with Welford Online Spike Detection and a Hippocampus Bridge | outline | [`entroly-core/src/cognitive_bus.rs`](../entroly-core/src/cognitive_bus.rs) |
| **Resonance** — Supermodular Pairwise Fragment Learning Beyond Knapsack | outline | [`entroly-core/src/resonance.rs`](../entroly-core/src/resonance.rs) |
| **System 1 ↔ System 2** — A Dual-Process Architecture for AI Coding Context | drafted (commit log: `4eaa0a5`) | [`entroly/coupling.py`](../entroly/coupling.py) |

## Reproducibility

Each paper will ship with a reproducibility script that regenerates its
numbers from a fixed-seed corpus. Targets:

- `research/bipt/reproduce.py` — IPD scores on the SWE-bench Lite held-out set
- `research/nkbe/reproduce.py` — Nash welfare vs. flat allocation on a 4-agent benchmark
- `research/causal_cg/reproduce.py` — observational-vs-interventional fragment value deltas

## Contributing

If you've reproduced a result or extended one of the algorithms, open a
PR with the script under `research/<algorithm>/contrib/`. Real evidence
of independent reproduction is welcome and credited.

---

*Last updated: v0.18.0 (System 1 ↔ System 2 coupling shipped).*
