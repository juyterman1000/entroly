# BIPT — Provenance Monotonicity Theorem

> **Theorem.** For any context C and output O, the Identifier Provenance
> Deficit (IPD) of O against C is monotonically non-increasing in the
> length of identifier substrings that match in C. Formally:
>
>   ∀ identifier span `s` in O, if every substring of `s` of length ≥ ℓ
>   has at least one occurrence in C, then `novel_bytes(s) ≤ |s| − ℓ`.

In plain English: the more of an identifier the AI's output quotes from
your context, the lower the BIPT-reported hallucination rate. The theorem
rules out a class of pathological edge cases where a long match would
paradoxically increase the reported novelty.

This is the property that lets BIPT scores be used as monotone alarms —
if grounding goes up, IPD goes down, full stop.

## Why this proof matters

If BIPT's reported IPD wasn't monotone, threshold-based alerting on
hallucinations could trigger false positives in counterintuitive ways
(e.g., a perfectly-quoted identifier reading as more hallucinated than
a partially-quoted one). The proof certifies the metric is sound.

## Status

Scaffold. Lean 4 implementation in progress. The Suffix Automaton
construction (which BIPT relies on) has existing formalizations in
Mathlib that we can build on.

## References

- Blumer, A., Blumer, J., Haussler, D., Ehrenfeucht, A., Chen, M. T., &
  Seiferas, J. (1985). *The smallest automaton recognizing the subwords
  of a text.* Theoretical Computer Science 40.
- [BIPT.md](../../BIPT.md) — non-formal exposition of the algorithm.
