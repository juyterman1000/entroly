# Evidence operations: research basis and claim boundaries

Entroly's evidence-operations loop is an original product design. It combines
four existing Entroly invariants—local-first processing, exact recovery,
receipts, and fail-closed verification—with research findings that argue
against treating token reduction as a sufficient metric.

## Design consequences

1. **Extractive before generative.** LLMLingua-2 frames faithful prompt
   compression as token classification, while LongLLMLingua shows that key
   information density and position matter in long contexts. Entroly's new
   operational codecs therefore select verbatim evidence and retain the exact
   source instead of synthesizing an untraceable summary.
2. **Coverage is a gate.** Conformal context-engineering work motivates
   coverage-controlled filtering rather than uncalibrated confidence. The
   browser envelope passes through when every query term present in the source
   cannot fit in the active budget. This is a deterministic gate, not a claim
   that the current implementation has conformal coverage guarantees.
3. **Position is part of quality.** Lost-in-the-middle evaluations show that a
   model can underuse relevant evidence merely because of placement. Receipts
   preserve source order, and trials measure downstream task outcomes rather
   than assuming shorter input is better.
4. **Continuous inspection must not create significance.** Anytime-valid A/B
   testing research documents how repeated peeking invalidates ordinary
   fixed-horizon tests. Entroly currently labels three matched runs as
   directional only and makes no significance claim. A future statistical
   decision layer must use an explicitly preregistered anytime-valid method.
5. **Accessibility trees are useful but incomplete.** WorkArena/BrowserGym and
   OSWorld use accessibility-tree observations for agents, but browser task
   success remains difficult and visual state can matter. Entroly therefore
   states that an ARIA receipt is not proof of visual equivalence or task
   completion.

## Primary sources

- [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968)
- [LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios](https://arxiv.org/abs/2310.06839)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction](https://arxiv.org/abs/2511.17908)
- [Anytime-Valid Confidence Sequences in an Enterprise A/B Testing Platform](https://arxiv.org/abs/2302.10108)
- [WorkArena and BrowserGym](https://arxiv.org/abs/2403.07718)
- [OSWorld](https://arxiv.org/abs/2404.07972)

These papers motivate design and evaluation choices. Their results are not
Entroly benchmark results, and Entroly does not inherit their guarantees.
