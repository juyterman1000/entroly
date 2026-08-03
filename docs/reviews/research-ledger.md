# Research ledger

One row per source that a mechanism in this repository actually rests on.

**Verification status is recorded per row and is not uniform.** Rows marked
`verified` were checked against a primary source or official repository during
this work. Rows marked `internal` come from pretrained knowledge used as
*hypothesis generation* — they motivated a design, and the design is justified
by measurements in this repo rather than by the citation. An `internal` row is
not evidence and must not be cited as such in user-facing material.

A source appears here only if something in the codebase depends on it. Papers
that are merely interesting are omitted.

---

## Log template mining

| field | value |
|---|---|
| citation | He, Zhu, Zheng, Lyu. *Drain: An Online Log Parsing Approach with Fixed Depth Tree.* IEEE ICWS 2017 |
| verification | **verified** — [author page](https://pinjiahe.github.io/publication/2017-ICWS), [dblp](https://dblp.dagstuhl.de/rec/conf/icws/HeZZL17.html), [logpai/logparser](https://github.com/logpai/logparser/blob/main/logparser/Drain/README.md) |
| research question | can log messages be grouped into event templates online, in one streaming pass? |
| method | fixed-depth parse tree; first layer keyed on message length, then on leading tokens, then similarity against each group's template |
| main result | streaming parsing without offline model training, which batch parsers require |
| assumptions | leading tokens are stable within an event type; message length is informative |
| relevance | Entroly's `_log_template` collapses repeats that differ only by an instance identifier |
| mechanism taken | template-before-key: reduce a line to the event it instantiates, then deduplicate on that |
| mechanism NOT taken | Drain's parse tree. Entroly substitutes variable *shapes* by regex, which is weaker (no learned grouping) and simpler (no state, no tree to persist) |
| claim Entroly must not make | that this is Drain, or that it matches Drain's parsing accuracy. It is the same idea applied far more conservatively |
| measured here | 203-line fixture: 16,311 → 308 chars, 200 repeats collapsed, root cause retained. `tests/test_log_codec_collapses_repeats.py` |
| known failure mode | over-templating merges distinct events. Entroly refuses to normalise status codes, exit codes and bare integers for exactly this reason — see the `never_merges_values_that_change_the_meaning` test |

## Similarity estimation from random hyperplanes

| field | value |
|---|---|
| citation | Charikar. *Similarity Estimation Techniques from Rounding Algorithms.* STOC 2002 |
| verification | **internal** — not re-fetched during this work |
| result relied on | for random hyperplanes, P(bit differs) = θ/π, so cos θ = cos(π·d/B) |
| relevance | `simhash_wide.rs::cosine` and the redundancy penalty in `knapsack_sds` |
| mechanism taken | the cosine estimator in that exact form |
| what this replaced | `1 - d/B`, which is linear in the ANGLE and not in the cosine. Measured MAE 0.502 against 0.080 for the correct form |
| claim Entroly must not make | that a 64-bit fingerprint resolves fine similarity differences. The sampling standard error is π/(2√B) ≈ 0.196 at B=64 |
| measured here | at 64 bits the near-duplicate and unrelated populations OVERLAP (min_dup − max_stranger = −0.0419); at 256 they separate (+0.0123). `simhash_wide.rs` tests |

## Selection bias in maximisation

| field | value |
|---|---|
| citation | Smith & Winkler. *The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis.* Management Science 2006 |
| verification | **internal** — not re-fetched during this work |
| result relied on | taking the maximum over k noisy estimates inflates the winner by roughly σ√(2 ln k) |
| relevance | the redundancy penalty selects the *maximum* similarity against already-chosen fragments, so it is exactly this shape |
| mechanism taken | a union-bounded lower confidence bound instead of the point estimate |
| claim Entroly must not make | that widening the fingerprint fixes this. It shrinks σ; the bias term remains, so the confidence bound stays regardless of width |
| measured here | residual redundancy inflation 88% → 0.000 after applying the bound |

## Submodular maximisation bounds

| field | value |
|---|---|
| citations | Nemhauser, Wolsey, Fisher 1978 (greedy ≥ 1−1/e, monotone submodular, cardinality); Feige 1998 (tightness); Sviridenko 2004 (knapsack, partial enumeration); Khuller, Moss, Naor 1999 (budgeted coverage) |
| verification | **internal** — not re-fetched during this work |
| relevance | this cluster is cited here to record what Entroly may NOT claim |
| finding | Entroly's selection objective is a plain sum of independently-computed per-fragment scores, i.e. **modular**. The (1−1/e) bound requires monotone submodular plus an exact marginal-gain oracle, and under a knapsack additionally needs partial enumeration |
| action taken | twelve unearned guarantees removed from code, docs and user-facing output; `tests/test_no_unsupported_theory_claims.py` prevents their return |
| claim Entroly must not make | any named approximation ratio for the shipped selector. For a modular knapsack, better-of-{density-greedy, best singleton} is ½ |

## Long-context degradation

| field | value |
|---|---|
| citation | Chroma, *Context Rot* (technical report, 18 models) |
| verification | **internal** — the specific figures were not re-fetched during this work |
| result relied on | focused prompts outperform full prompts; a single distractor measurably reduces performance |
| relevance | motivates treating the token budget as a ceiling rather than a target |
| claim Entroly must not make | any quantitative transfer of their numbers to Entroly's workloads. Entroly has run **no** model-in-the-loop evaluation, so the premise remains borrowed |
| status | the premise was tested indirectly and did **not** hold as assumed: with an 806-fragment pool, selection returned only the needle rather than filling the budget. `benchmarks/sufficiency_baseline.py` |

## Query expansion drift

| field | value |
|---|---|
| citation | Rocchio 1971; Salton & Buckley 1990 (relevance feedback weighting) |
| verification | **internal** — not re-fetched during this work |
| result relied on | expansion terms should be weighted below original query terms |
| relevance | `entroly-qccr` injects a whole intent-cluster vocabulary on expansion |
| mechanism taken | separate accumulation of query-term and expansion-term scores, with the expansion aggregate **saturating** rather than summing |
| deviation from the source | a flat α discount was tried first and measured insufficient — six discounted matches still outvoted one full-weight match. The saturating form is justified by correlation among cluster terms, not by Rocchio |
| measured here | evidence recall 66.7% → 100%; the answer's rank 23 of 66 → 1 of 66 |

---

## Sources deliberately not adopted

| area | why not |
|---|---|
| LLMLingua family | requires a model at compression time. Entroly's contract is training-free and local-first; adding an inference dependency would change the product's deployment story. Recorded as a deliberate trade, not an oversight |
| Dense / late-interaction retrieval | needs an embedding model and an index. Same trade as above |
| DPP / log-determinant selection | the redundancy signal is already a confidence-bounded estimate over a noisy fingerprint; a determinant over the same noisy similarities would inherit the noise with more machinery |
| Conformal prediction | the correct eventual home for calibrated sufficiency, but it requires a held-out calibration set, which does not exist here. Current thresholds are in-sample on three fixtures and are labelled as such |

## What this ledger does not contain

No row here justifies a competitive claim. Every measurement cited is on
synthetic frozen fixtures in this repository with no model in the loop, so
"evidence retained" means substring survival and not answer accuracy.
