# BIPT — Byte-level Information Provenance Tracing

**Implementation:** [`entroly/verifiers/provenance_tracer.py`](../entroly/verifiers/provenance_tracer.py)
**Standalone exposition:** [`../BIPT.md`](../BIPT.md)
**Math reference:** [`../RESEARCH.md`](../RESEARCH.md)

## One-paragraph spec

Given input context **C** and LLM output **O**, BIPT computes the
fraction of identifier bytes in O that cannot be matched to any
substring of C. This fraction — the **Identifier Provenance Deficit
(IPD)** — is a Kolmogorov-bounded measure of how invented O is. IPD = 0
means every identifier is a quote of something in C; IPD = 1 means
every identifier is invented.

## Algorithm

1. Build a Suffix Automaton (DAWG) over C — `O(|C|)` time and space.
2. For each position `j` in O, find the longest substring of O starting
   at `j` that occurs in C — `O(|O|)` total via automaton walk.
3. AST-parse O; identify byte ranges of identifiers (function names,
   types, constants, imports).
4. For each identifier span, count bytes not covered by any C-substring.
5. **IPD = Σ(novel_bytes) / Σ(identifier_bytes)** ∈ [0, 1].

## Why it's novel

Every other hallucination detector asks *is this output correct?* —
which requires an oracle for the world. BIPT asks *can every byte be
explained by the input?* — which is decidable in linear time. The
mathematical foundation is **conditional Kolmogorov complexity**
(uncomputable in the limit, but tightly upper-bounded by Lempel-Ziv
factorization).

## What you get

A receipt per output: every identifier tagged with `GROUNDED` (matched
in C at byte range X..Y) or `INVENTED` (no match). The receipt is
machine-parseable, log-able, audit-friendly, and citation-quality.

## References

Kolmogorov 1965; Solomonoff 1964; Lempel-Ziv 1976; Blumer et al. 1985.
