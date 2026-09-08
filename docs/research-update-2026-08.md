# Research update: stronger evidence-preserving compression

Entroly's latest research cycle produced a practical result: the compression path is becoming better at reducing context **without sacrificing the identifiers, failures, and evidence users need to solve the task**.

## What improved

Recent production changes strengthened several difficult cases that previously forced a trade-off between compression and evidence retention:

- **Identifier-bearing JSON:** identifier columns are now detected from the data rather than only from familiar key names. In measured 200-record cases, previously unrecognised identifiers such as VIN-like and policy-number fields went from being mostly discarded to **200/200 preserved**, while still achieving meaningful compression.
- **Templated logs:** variable fields can now be discovered from disagreement across repeated log structure instead of relying only on numeric/hex-looking tokens. Measured reductions improved from **29% to 82% for hostname-heavy logs**, **30% to 73% for user-ID-heavy logs**, and **0% to 76% for opaque request-ID-heavy logs**, with distinct values preserved.
- **Test/build output:** failure evidence is now treated as critical. In the measured pytest corpus, compression improved while **all 8/8 FAILED lines were retained** instead of only 2/8.
- **Tight token budgets:** evidence-aware ordering now keeps critical failures before repetitive bulk data. Across measured tight-budget log cases, **all 8/8 ERROR lines and both tested status codes remained present** while large token reductions were still achieved.
- **Public SDK path:** `compress()` now reaches the specialized codec registry before falling back to the generic compressor, bringing structural compression closer to the default user experience.

These results are backed by the focused codec/compression/SDK test suites recorded in the corresponding commits. They are not claims of universal superiority; they are concrete improvements on measured workloads.

## Why the research result is encouraging

The latest theoretical pass also clarified the architecture rather than weakening it.

For an unknown future task, there is no single minimal representation that can be guaranteed to remain sufficient for every possible downstream question. That makes a purely irreversible, query-agnostic compressor fundamentally risky: what looks unimportant before the question is known may become essential later.

Entroly's design addresses that constraint directly:

1. preserve a task-invariant evidence core where possible;
2. refuse unsafe derived representations when preservation checks fail;
3. keep the original content addressable and byte-exactly recoverable;
4. recover additional evidence when a later task needs it.

This is theoretical grounding for the architecture, **not a novelty claim**. The useful outcome is that the research gives a clearer reason for why recoverability and preservation gates belong at the center of the system rather than being optional safety features.

## What this means for users

The direction is straightforward:

> **Compress aggressively where Entroly can prove important evidence survives; preserve or recover the original when it cannot.**

The goal is not the highest compression percentage at any cost. The goal is a smaller, more useful working context that still carries the identifiers, errors, structure, and provenance needed by local or cloud models.

## Next engineering focus

The next high-value work is to carry the strongest codec behavior consistently across the remaining production surfaces, strengthen repository/context intelligence, and validate the gains on real task-level benchmarks with null controls and equal-budget baselines.

That keeps the project moving toward a stronger objective than token reduction alone: **better model outcomes from smaller, safer, recoverable context.**
