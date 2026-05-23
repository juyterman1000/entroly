# RFC-001 — Entropic Budget-Tight Submodular Context Selection with Calibrated Coverage Guarantees

**Status:** Proposal · **Author:** Entroly core · **Date:** 2026-04-20

## 1 · Problem

Given a user query `q`, a repository indexed as fragments `F = {f₁, …, fₙ}` with
token costs `c(fᵢ) ∈ ℕ`, and a hard token budget `B ∈ ℕ`, select `S ⊆ F`
maximising the *posterior utility* `U(S ∣ q)` subject to `Σ_{f∈S} c(f) ≤ B`.

Concretely: the caller passes `B` because the downstream model has a fixed
context window. Exceeding `B` by a single token fails the request. This is a
**constrained submodular maximisation under uncertainty**, not a ranking
problem — a distinction the current implementation does not cleanly make.

## 2 · What the current system gets wrong

The v0.8 pipeline is roughly:

```
score = w₁·recency + w₂·frequency + w₃·semantic + w₄·entropy
budget_eff = B × TaskType.multiplier(q)        # (a)
knapsack_greedy(fragments, budget_eff, score)  # (b)
channel_trailing_pass(fill gap)                # (c)
contradiction_guard(evict near-dups)           # (d)
```

Three structural problems:

**P1. Utility is modelled as a linear sum of independent scalars.** Shannon
entropy per fragment does not capture that the 5th `__init__.py` of re-exports
adds ≈ 0 marginal information once the 1st is present. The objective is not
submodular, so greedy has no approximation guarantee.

**P2. The budget multiplier at (a) silently inflated B by up to 1.5×.** Fixed
in [entroly-core/src/lib.rs:1064-1082](../../entroly-core/src/lib.rs) — the
user's `B` is now a hard ceiling, the multiplier is surfaced as
`recommended_budget` advisory. *(Shipped in this PR.)*

**P3. Relevance is query-keyword-only.** Dog-food run on LangGraph:
`"how does the StateGraph compilation work"` selected 57 trivial `__init__.py`
files ahead of `libs/langgraph/langgraph/graph/state.py` — the file that
literally defines `StateGraph.compile`. Cosine-style scoring over bag-of-terms
collapses in the presence of 374-file repos where most files share vocabulary.

## 3 · What the literature converged on (2023–2026)

Five threads, each individually well-established, not yet combined in a
code-context selector:

1. **Monotone submodular maximisation under knapsack.** Sviridenko (2004) and
   the streaming variant of Badanidiyuru et al. (2014) achieve `(1 − 1/e)`
   approximation in polynomial time. Facility-location and coverage objectives
   are submodular, capturing diminishing returns directly.
2. **Graph-guided candidate generation.** RepoCoder (Zhang et al., 2023),
   LongCoder (Guo et al., 2023), Cedar (2024) show that hopping the import /
   call graph from a small seed set dominates dense-only retrieval on code
   tasks by 8–22 MRR points.
3. **Learned sparse and late-interaction retrievers.** SPLADE++ (Formal et al.,
   2022), ColBERTv2/PLAID (Santhanam et al., 2023), GritLM (Muennighoff et
   al., 2024). Out-of-the-box relevance quality that makes `w₁·recency +
   w₂·frequency + …` look like a 2018 baseline.
4. **Conformal risk control.** Angelopoulos et al. (2022, 2024); Bates et al.
   (2023). Turns any black-box scorer into a distribution-free predictor with
   calibrated coverage guarantees — e.g. "with probability ≥ 0.9 this
   selection contains every fragment the ground-truth agent would have needed".
5. **Information-saturation stopping.** Classical sequential hypothesis
   testing (Wald, 1947) re-applied to RAG by Jiang et al. (2024), Asai et
   al. (2024) under the *adaptive retrieval* banner: stop loading context
   when marginal entropy reduction per token falls below threshold τ.
   Replaces the hardcoded `TaskType.multiplier` with a measured signal.

## 4 · Proposed design

Replace the linear-score pipeline with a four-stage procedure. Each stage has
a concrete algorithmic choice, and each choice is drawn from one of the five
threads above.

### 4.1 Seed-and-expand candidate pool

Form candidate pool `C_q ⊆ F` (typically 10²–10³ fragments) rather than
scoring all `n` fragments:

1. **Seed set** `S₀` via hybrid lexical (BM25) + learned-sparse (SPLADE++)
   retrieval, take top-k fragments, `k = 32`.
2. **Graph expansion** `S₀ → C_q` by traversing the static dependency graph
   `G = (F, E)` up to `h = 2` hops. Edges: imports, function calls,
   inheritance, type references. Already built in `entroly-core/src/depgraph.rs`.

Cost: `O(k · Δ^h)` where `Δ` is average out-degree — tiny relative to scoring
all `n`.

Solves P3: `StateGraph.compile` appears in `S₀` via the query term "StateGraph",
and the two-hop expansion brings in `state.py`, `compile.py`, `pregel/loop.py`
— the files an engineer actually needs.

### 4.2 Submodular objective

For `S ⊆ C_q`, define

```
F(S ∣ q) = α · Coverage(S, q) + β · Diversity(S) − γ · Redundancy(S)
```

where each term is monotone submodular:

* **Coverage** (facility-location):
  `Coverage(S, q) = Σ_{f ∈ C_q} max_{s ∈ S} sim(f, s, q)`
  — how well `S`'s embeddings cover the candidate pool, re-weighted by query
  relevance. Captures the "we've seen enough of this sub-topic" intuition.
* **Diversity** = entropy of fragment types / modules covered. Already
  computable from existing metadata.
* **Redundancy** = SimHash-near-duplicate penalty. Already implemented in
  `entroly-core/src/dedup.rs`.

**Approximation guarantee — honest statement.** `Coverage` and `Diversity`
alone are monotone submodular. The `− γ · Redundancy` term, if kept as a
subtractive penalty, breaks monotonicity: `F(S ∪ {f}) < F(S)` is possible
when `f` is a near-duplicate. For non-monotone submodular maximisation under
a knapsack constraint, the best known polynomial-time ratio is the
`0.325`-approximation of Chekuri, Vondrák & Zenklusen (2011) via continuous
greedy + rounding — not `(1 − 1/e)`.

The pragmatic fix is to absorb the redundancy penalty into `Coverage` via a
**saturating** similarity kernel (e.g., `sim(f,s,q) ← min(sim, θ)`), making
`F` monotone submodular by construction. Sviridenko's modified greedy then
achieves `(1 − 1/e − ε) ≈ 0.63 − ε` under the knapsack — but only for this
saturation-based formulation. Claiming `(1 − 1/e)` for the subtractive form
is incorrect and this RFC will not do so. (See §7: this reformulation is an
open design question.)

### 4.3 Budget discipline (shipped)

`B` is a hard ceiling. Task classification contributes:

* **Shrinking multipliers** (CodeGen 0.7×, Docs 0.6×, Testing 0.8×) are
  applied — they *improve* focus for narrow tasks and stay safe.
* **Expanding multipliers** (BugTracing 1.5×, Exploration 1.3×) become
  advisory: result exposes `recommended_budget` so the caller can opt in.
* `recommended_budget` is the *output of saturation stopping* (§4.4), not a
  hardcoded number.

### 4.4 Saturation-based recommended budget

Instead of a fixed `× 1.3` for Exploration, compute the **marginal entropy
curve** of the selection as budget grows:

```
H(B) = E[ − log p(answer | S*(B)) ]   // estimated via the scorer
ΔH(B) = H(B) − H(B + Δ)
```

Recommend the smallest `B*` such that `ΔH(B*) / Δ < τ` for three consecutive
windows. This is Wald's sequential test adapted to token-budget selection.
`τ` is a *tunable* that the user can set per project (default: `τ = 0.005
nats per 100 tokens`).

Concrete and principled: tells the user *exactly* how many tokens before
returns diminish, instead of a multiplier table baked into the binary.

### 4.5 Conformal coverage guarantee

Treat the selection as a set predictor. Use split-conformal calibration on a
held-out set of `(query, ground-truth-needed-fragments)` pairs (e.g. derived
from successful issue-PR closures in `git log`):

For a user-chosen `α ∈ (0, 1)` (default `α = 0.1`), the returned `S` satisfies

```
P( needed_fragments ⊆ S ) ≥ 1 − α
```

marginal over queries drawn from the same distribution. The calibration step
adjusts `γ` in §4.2 to hit the target coverage; no retraining required.

This is what finally lets us say *"92% probability this selection covers
your query"* rather than a made-up "99/100 Context Score".

## 5 · Expected wins vs current system

On the LangGraph dog-food query `"how does the StateGraph compilation work"`,
budget 8 000:

| Metric                         | v0.8 (current) | RFC-001 (projected) |
|-------------------------------:|:--------------:|:-------------------:|
| Budget respected               | ✗ (10 163/8 k) | ✓ (≤ B by design)   |
| `StateGraph.compile` selected  | ✗              | ✓ (via seed)        |
| Marginal return quantified     | no             | yes (`ΔH(B)` curve) |
| Coverage guarantee             | none           | `P ≥ 1 − α`         |
| Approximation ratio            | none           | `1 − 1/e − ε` (saturating form) or `0.325` (non-monotone) |
| Lines of Rust to change        | —              | ≈ 900 (new module)  |

## 6 · Rollout (by PR)

1. **PR-1 (this PR).** Budget as hard ceiling + `recommended_budget`
   disclosure. *Shipped.*
2. **PR-2.** Import-graph seed expansion. Already have `depgraph.rs`;
   wire it into the optimise path ahead of scoring.
3. **PR-3.** Submodular facility-location objective + streaming greedy.
   New file `entroly-core/src/submodular.rs`. Behind feature flag
   `--selector submodular`. Benchmark vs. current.
4. **PR-4.** Replace `TaskType.multiplier` with saturation-stopping
   `recommended_budget`.
5. **PR-5.** Conformal calibration layer over success-signal data
   (`record_success`/`record_failure` history). Ship coverage probability
   in the result.

Each PR ships independently, passes `--check-regression` in
`bench/compare.py`, and has a rollback path.

## 7 · Open questions

* **SPLADE++ vs ColBERT vs GritLM** for the seed retriever — each has
  different latency/accuracy trade-offs on CPU. Decide via head-to-head on
  SWE-bench Lite.
* **Edge weighting in graph expansion.** Imports vs. calls vs. type refs
  should probably carry different hop costs. No published numbers for
  code specifically.
* **Conformal base distribution.** Per-repo calibration sets will be
  small (N < 100) early on. Investigate Mondrian conformal predictors
  (Vovk, Gammerman, Shafer 2005) for group-conditional guarantees.

## References

Angelopoulos et al., *Learning Optimal Conformal Classifiers* (ICLR 2022);
*Conformal Risk Control* (ICLR 2024). Asai et al., *Self-RAG* (ICLR 2024).
Badanidiyuru, Mirzasoleiman, Karbasi, Krause, *Streaming Submodular Maximization*
(KDD 2014). Bates, Angelopoulos, et al., *Distribution-Free, Risk-Controlling
Prediction Sets* (JACM 2023). Formal, Piwowarski, Clinchant, *SPLADE v2/++*
(SIGIR 2022). Guo et al., *LongCoder* (ICML 2023). Jiang et al., *Active
Retrieval Augmented Generation* (EMNLP 2024). Muennighoff et al., *GritLM*
(2024). Santhanam et al., *ColBERTv2 / PLAID* (NAACL 2022, 2023). Sviridenko,
*A Note on Maximizing a Submodular Set Function Subject to a Knapsack
Constraint* (Oper. Res. Lett. 2004). Zhang et al., *RepoCoder* (EMNLP 2023).
Lin & Bilmes, *A Class of Submodular Functions for Document Summarization*
(ACL 2011). Wei, Iyer, Bilmes, *Submodularity in Data Subset Selection*
(ICML 2015).
