# Stage 0 closure + Stage 1/2 frontier findings

Baseline `e31f082f`. Measured 2026-10-02.

---

## F-01 — Underfill root cause: a hardcoded 12-file ceiling (CLOSED)

`entroly-qccr/src/lib.rs:67`

```rust
const MAX_FILES_CONSIDERED: usize = 12;
```

applied at `:1107` and `:1111` (`.take(MAX_FILES_CONSIDERED)`). **No dependence on
token budget.**

Sweep (`research/experiments/underfill_rootcause.py`), 300 files / ~1,340,276 est
tokens, 3 queries per budget, `_PREFILTER_FILE_FLOOR = 128`:

| budget | files selected | mean fill | tokens/file |
|---|---|---|---|
| 2,000 | 8–9 / 300 | ~100% | 222 |
| 4,000 | 12 / 300 | 99.1% | 330 |
| 8,000 | 12 / 300 | 95.2% | 635 |
| 16,000 | 12 / 300 | 86.8% | 1,157 |
| 32,000 | 12 / 300 | 71.0% | 1,894 |
| 64,000 | 12 / 300 | **50.0%** | 2,666 |

Hypotheses **excluded** by this data:
- candidate cap — `cap` is 128→1000 across the sweep and never binds at 12
- corpus exhaustion — 300 files / 1.34M tokens available, 288 files untouched
- relevance floor — file count is *constant*, not decaying
- token estimation — per-file tokens scale cleanly 222→3,355 (15×)

Confirmed mechanism: breadth is fixed at 12 files; extra budget buys **more of the
same 12 files**, never more files. Once those 12 are exhausted the window cannot
be filled.

### Strategic consequence (this is the important part)

At a 64K budget on a 300-file repo Entroly admits **12 files (4% of corpus)** and
fills **30–63%** of the allowed window. No ranking improvement can lift this —
the recall ceiling is structural, set by one constant.

It also **worsens with every model generation**: as windows grow to 200K/1M, the
gap between budget available and evidence admitted widens. A context control
plane should get *more* useful as windows grow; this constant makes it less so.

**Answers the user's hypothesis directly: yes — allocation policy currently
matters more than ranking algorithm.** `/goal`-level finding.

Do NOT simply raise the constant. 12 may be protecting precision (README reports
the frozen evidence benchmark selecting "1.02 of 16 passages"). The correct
experiment is a budget-vs-quality curve with the ceiling as a free parameter —
quality may be flat or may improve; both outcomes are informative.

---

## F-02 — Competitor frontier: decision-aware compression is ALREADY PUBLISHED

This is the single most important external finding, and it contradicts an earlier
prejudgment of mine (that the gap would be on the "neural compressor" axis).

### FOCUS — the direct threat to "decision-sufficient context"

| field | value |
|---|---|
| TITLE | FOCUS: Training-Free Decision-Preserving Context Compression for LLM Agents |
| AUTHORS | Shantanu Dixit, Anson Bastos, Xuchao Zhang, Chetan Bansal, Saravan Rajmohan (Microsoft M365 Research) |
| SOURCE | arXiv:2609.37590v1 |
| SOURCE TYPE | **arXiv preprint — no venue acceptance stated** |
| DATE | 2026-09-29 (three days before this review) |
| CONFIDENCE | HIGH on content (abstract + body read); status is preprint, not peer-reviewed |

Criterion (their Definition 1) — **counterfactual future utility**:

```
U(sᵢ) = D( P(Y | Hₜ, g) ‖ P(Y | Hₜ⁻ⁱ, g) )
```

KL divergence between future-trajectory distributions with and without span `sᵢ`,
tied to an Information Bottleneck objective. **This is materially the same
formulation proposed as Entroly's Bet 1/Bet 6** (`Δ(e) = D(π(a|C), π(a|C\e))`).

Training-free, test-time, architecture-agnostic, attaches to any closed-API model.
Beats no-compression, FIFO, Retrieval, LLMLingua, Prompting, **and ACON**:

| benchmark | FOCUS | no-compression | peak tokens |
|---|---|---|---|
| AppWorld | 64.9% | 56.0% | −35% |
| OfficeBench | 78.9% | 76.8% | −47% |
| WebVoyager | +4.5% | — | −27% |
| τ²-Bench | +5.5% | — | — |

**Verdict on Bet 1 / Bet 6: NOT OPEN. Classify as PARTIALLY SOLVED → effectively
captured.** Building "decision-sufficient context" as a novel Entroly moat would
reproduce a Microsoft preprint. Do not claim novelty.

### ACON — the direct threat to "recovery-supervised learning"

| field | value |
|---|---|
| TITLE | ACON: Optimizing Context Compression for Long-horizon LLM Agents |
| SOURCE | arXiv:2510.00615; ICML 2026 poster (icml.cc/virtual/2026/poster/66270) |
| SOURCE TYPE | **accepted conference paper (ICML 2026 poster)** |
| CONTRIBUTION | iteratively refines a natural-language compression guideline from **failure analysis** of the agent; distills into smaller LMs |
| RESULT | 26–54% peak token reduction with improved task success; up to 46% gain enabling small LMs as long-horizon agents |

ACON learns from failures in natural-language space without fine-tuning. **This
occupies much of Bet 2 (recovery-supervised context learning).** Classify Bet 2 as
**PARTIALLY SOLVED**, not open.

### Adjacent 2026 work found (not yet fully characterized)

| paper | arXiv | relevance |
|---|---|---|
| Decision-Aware Memory Cards: Counterfactual-Inspired Context Selection and Compression for Tool-Using LLM Agents | 2606.08151 | counterfactual selection — Bet 1/6 again |
| CoACT: Action-Preserving Observation Compression for Coding Agents | 2607.02911 | action-invariance for **coding agents** — Entroly's core use case |
| What Does Context Compression Cost an Agent? Interaction Costs Unrevealed by Task-Completion Metrics | 2608.16370 | critique of task-completion-only metrics — relevant to benchmark design |
| Toward Reliable Context Compression for Long-Horizon Agents: An Empirical Study of Execution Instability | 2608.06503 | multi-turn compounding degradation (Track E) |
| SWE-Pruner Pro: The Coder LLM Already Knows What to Prune | 2607.18213 | coding-agent pruning |
| LLMLingua-2 | 2403.12968 | **Findings of ACL 2024** (verified). **Task-agnostic — does NOT condition on query.** Requires trained XLM-RoBERTa-large/mBERT at inference. 2–5× compression |

### Correction to my own earlier analysis

I previously wrote that the neural/learned-compressor axis was where the gap would
"bite hardest". **That was prejudgment and is partly wrong.** LLMLingua-2 is
*task-agnostic*: it discards query conditioning for generalizability. Entroly's
QCCR is query-conditioned and needs **no model at inference**. On the query-
awareness and inference-cost axes Entroly is not behind LLMLingua-2.

The real gap is **decision-awareness** (FOCUS/ACON/CoACT), which I had not
identified at all.

---

## F-03 — The white space that survives

FOCUS states plainly (confirmed from the paper body):

> "The method permanently removes spans deemed low-utility. Omitted content is not
> preserved or recoverable — the compressed trace Zₜ contains only retained spans
> verbatim. The paper does not address archival or retrospective recovery."

And its own limitations section requires:

> "a defensive verification stage is needed to catch spans whose deletion causes
> repeated mistakes or loss of causal state"

So the strongest published decision-aware compressor is **irreversible** and
**explicitly needs a recovery/verification mechanism it does not have**.

Entroly already ships the missing half (verified earlier in this program):
- content-addressed recovery (`recover_receipt_omission`)
- receipts with exact UTF-8 byte offsets + source/fragment SHA-256
- omission tracking in receipts

**White space (prior-art search incomplete, stated as such):** decision-aware
compression where every omission is *certified and reversible* — i.e. combining
FOCUS-style counterfactual utility with Entroly-style recoverable omission, so
that a wrong omission is detectable and repairable rather than permanent.

That is a **composition** claim, not a new criterion. Honest novelty label:
**MEDIUM — the criterion is taken; the reversible/auditable composition appears
unoccupied.** Requires a dedicated prior-art pass before any novelty claim.

---

## Hypothesis ledger

| ID | HYPOTHESIS | PRIOR ART | CHEAPEST TEST | STATUS |
|---|---|---|---|---|
| H1 | Raising/removing `MAX_FILES_CONSIDERED` improves task quality at ≥16K budgets | none (internal constant) | budget-vs-quality curve, ceiling as free parameter, needle+longbench | **UNTESTED — highest value** |
| H2 | Decision-sufficient context is an open research direction | **FOCUS (2609.37590), ACON (ICML 2026)** | — | **ALREADY KNOWN** |
| H3 | Recovery events yield causal supervision for selection | **ACON** does failure-driven refinement | IPW/doubly-robust on omission→recovery→outcome traces; needs traces that do not yet exist | **PARTIALLY SOLVED / UNTESTED** |
| H4 | Reversible + certified omission under a decision-aware criterion is unoccupied | FOCUS explicitly lacks it | prior-art pass, then matched comparison vs FOCUS on AppWorld | **UNTESTED — candidate moat** |
| H5 | Entroly is behind on neural/learned compression | LLMLingua-2 is task-agnostic; Entroly is query-conditioned | — | **PARTLY FALSIFIED** |

Sources: arXiv:2609.37590, arXiv:2510.00615, icml.cc/virtual/2026/poster/66270,
arXiv:2403.12968, arXiv:2606.08151, arXiv:2607.02911, arXiv:2608.16370,
arXiv:2608.06503, arXiv:2607.18213.
