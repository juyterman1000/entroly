# Entroly Algorithms — index

> One-page explanations of every original algorithm in entroly. For the
> long-form mathematical exposition with citations, see
> [`../RESEARCH.md`](../RESEARCH.md). For each algorithm's
> implementation, follow the file link.

| Algorithm | One-line | Implementation | Doc |
|---|---|---|---|
| **BIPT** | Byte-level hallucination provenance via suffix automaton | [`provenance_tracer.py`](../entroly/verifiers/provenance_tracer.py) | [bipt.md](bipt.md) |
| **NKBE** | Nash-KKT multi-agent token budget allocation | [`nkbe.rs`](../entroly-core/src/nkbe.rs) | [nkbe.md](nkbe.md) |
| **Causal Context Graph** | Pearl-grade do-calculus on RAG | [`causal.rs`](../entroly-core/src/causal.rs) | [causal-cg.md](causal-cg.md) |
| **Cognitive Bus** | ISA inter-agent event routing | [`cognitive_bus.rs`](../entroly-core/src/cognitive_bus.rs) | [cognitive-bus.md](cognitive-bus.md) |
| **Resonance Matrix** | Supermodular pairwise fragment learning | [`resonance.rs`](../entroly-core/src/resonance.rs) | [resonance.md](resonance.md) |
| **System 1 ↔ System 2** | Dual-process proxy ↔ vault coupling | [`coupling.py`](../entroly/coupling.py) | [system-1-system-2.md](system-1-system-2.md) |

## Why split out from RESEARCH.md

[`RESEARCH.md`](../RESEARCH.md) is one long document for serious evaluators
who want the full picture in one read. The files in *this* directory are
short, deep-linkable, one-per-algorithm. You can paste any of them into
a Slack/Reddit/Hacker News thread without dumping 400 lines on someone.
