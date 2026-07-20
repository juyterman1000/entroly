# PRISM-R: Reversible Neural Evidence Compression

Status: experimental research prototype. It is not enabled in Entroly's
default compression path and is not yet evidence for a general quality or
production-cost claim.

## Research question

Can a model-agnostic agent context layer use a local transformer to improve
query-conditioned evidence selection while preserving readable source text,
exact recovery, deterministic fallback, and auditable omission risk under
future-query shift?

The current answer is promising but incomplete. A frozen SQuAD v2 pilot passes
the repository's evidence-retention gate at tighter budgets, but downstream LLM
answers, long-agent trajectories, domain shift, and stronger learned-compressor
baselines remain untested.

## Scoped 2021-2026 prior-art synthesis

This is a focused map of directly relevant primary work, not a claim to have
read every paper published in five years.

| Direction | Representative primary work | Design constraint for Entroly |
|---|---|---|
| Learned latent prompts | [Gisting, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html) | Latent compression can be efficient but is model-coupled and opaque. |
| Learned summary vectors | [AutoCompressors, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.232/) | Precomputed soft memories are useful, but exact textual provenance is not inherent. |
| Dynamic learned units | [Nugget, ICML 2023](https://openreview.net/forum?id=LhFE049fh5) | Variable-rate neural selection is established prior art. |
| Query-aware hard compression | [LongLLMLingua](https://arxiv.org/abs/2310.06839) | Query conditioning and position control matter; lexical evidence recovery remains essential. |
| Bidirectional token classification | [LLMLingua-2](https://arxiv.org/abs/2403.12968) | A learned compression objective is stronger than generic embedding similarity. |
| Selective augmentation | [RECOMP, ICLR 2024](https://openreview.net/pdf?id=mlJLVigNHp) | Abstaining from irrelevant augmentation is established; Entroly extends abstention to unsafe compression. |
| Memory-slot autoencoding | [ICAE, ICLR 2024](https://openreview.net/forum?id=uREj4ZuGJE) | Soft memory slots can compress aggressively but require a compatible decoder. |
| One-token RAG fusion | [xRAG, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c5cf13bfd3762821ef7607e63ee90075-Abstract-Conference.html) | Extreme latent compression is possible but is retriever/model specific. |
| Rate-distortion limits | [Fundamental Limits of Prompt Compression](https://arxiv.org/abs/2407.15504) | Query-aware variable-rate policies are theoretically necessary; token reduction alone is not the objective. |
| Certified retrieval risk | [C-RAG, ICML 2024](https://proceedings.mlr.press/v235/kang24a.html) | Finite-sample risk claims require explicit assumptions and calibration data. |
| Evaluator-head salience | [EHPC, NeurIPS 2025](https://openreview.net/forum?id=yOs12gdsaL) | Early transformer attention can identify useful tokens without a separate large compressor. |
| Ultra-long chunk compression | [ParallelComp, ICML 2025](https://proceedings.mlr.press/v267/xiong25b.html) | Chunk-level coherence and long-context memory limits must be evaluated, not inferred from short QA. |
| Semantic chunk integrity | [ChunkKV, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2987f911151b39cd3a1761e212319e8e-Abstract-Conference.html) | Token-level pruning can fragment meaning; source-addressable spans are safer units. |
| Real latency break-even | [Prompt Compression in the Wild](https://arxiv.org/abs/2604.02985) | Compression overhead can erase inference gains; production claims require hardware-specific end-to-end measurement. |
| Multiplex graph pruning | [RAGP, 2026 preprint](https://arxiv.org/abs/2607.01241) | Distributed evidence and bridge relations motivate graph-aware future evaluation. |

Hybrid hard/soft compression, learned selection, submodular diversity,
abstention, and provenance each have prior art. The plausible contribution is
their formal and operational coupling under recoverable agent-memory shift,
not any component in isolation.

## Objective

For normalized semantic vectors `e_i`, define:

- `u_i(q)`: current-query transformer relevance;
- `a_i`: exact-evidence value supplied by deterministic detectors or policy;
- `h_i`: future-query utility learned from agent access history;
- `s_i`: confidence that the semantic representation is reliable;
- `c_i`: token cost.

PRISM-R approximately maximizes

```text
F_q(S) = sum_{i in S} [u_i(q) + gamma*a_i + rho*h_i]
         + lambda log det(I + sum_{i in S} w_i(q) e_i e_i^T)

w_i(q) = s_i * max(0, u_i(q) - tau)^p
```

subject to:

```text
L_q subset S
sum_{i in S} c_i <= B
```

`L_q` contains mandatory evidence locks. Separating modular utility from the
determinant weight avoids counting relevance twice. The floor `tau` prevents
an irrelevant but orthogonal span from winning on novelty. With non-negative
weights, the static core remains monotone submodular; the implementation
compares gain-greedy and density-greedy schedules under the token knapsack.

## Acceptance and recovery contract

1. Load only an explicitly supplied local transformer directory. Remote model
   identifiers are rejected.
2. Normalize every embedding and reject zero, malformed, or non-finite output.
3. Preserve mandatory exact evidence even when that exceeds the requested
   budget; expose the budget failure instead of deleting evidence.
4. When lexical and neural champions disagree, either retain both or require a
   threshold tied to a named calibration artifact. Otherwise abstain.
5. Persist every omitted character span atomically with source offsets and a
   content hash. A single stale offset or hash aborts the write.
6. Rehydrate exact source text by receipt/span ID and debit the recovered tokens
   from realized savings. Repeated retrieval IDs are idempotent.

## Measured results

### Frozen answer-passage retrieval

On 600 deterministic SQuAD v2 trials with 15 distractor paragraphs, calibrated
on 300 trials and evaluated on a disjoint 300:

| Selector | Held-out top-1 evidence recall | Notes |
|---|---:|---|
| Deterministic BM25 | 99.0% | Exact lexical evidence remains the strongest primary scorer. |
| Local `all-MiniLM-L6-v2` transformer | 97.7% | Frozen revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; one transformer-only success, five lexical-only successes; no neural-primary claim. |
| Dual-channel disagreement guard | 99.3% | Selects 1.02 of 16 passages on average, a 15.69x passage compression ratio. |

The generic transformer failed the primary-scorer gate. PRISM-R therefore uses
it as a residual semantic channel rather than replacing lexical evidence.

Artifacts:

- [`neural-evidence-frontier.md`](../benchmarks/neural-evidence-frontier.md)
- [`neural_evidence_frontier.json`](../../benchmarks/results/neural_evidence_frontier.json)
- [`neural_evidence_frontier.md`](../../benchmarks/results/neural_evidence_frontier.md)

### Future-query shift and exact rehydration

Each of 200 trials contains two SQuAD questions whose answers occur in
different sentences of the same paragraph. The context is compressed for `q1`;
`q2` is revealed only afterward. Results use the same frozen trials and local
model at every ratio.

| Nominal active ratio | Actual active ratio | Lexical q1 | PRISM-R q1 | Direct unseen q2 | q2 after exact recovery | Active + recovery |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 44.2% | 92.0% | 95.0% | 34.0% | 95.0% | 72.7% |
| 25% | 24.1% | 60.5% | 87.0% | 9.0% | 90.5% | 50.6% |
| 12.5% | 18.2% | 24.0% | 81.0% | 3.5% | 87.0% | 43.5% |

The 25% and 12.5% settings pass the pilot's paired current-evidence and
rehydration gates. The 50% current-query delta does not (`p=0.109375`) and is
not presented as a win. Actual ratios differ from targets because sentence
spans are indivisible and safety guards may retain both channels.

Artifacts:

- [`neural_query_shift_2x.json`](../../benchmarks/results/neural_query_shift_2x.json)
- [`neural_query_shift.json`](../../benchmarks/results/neural_query_shift.json)
- [`neural_query_shift_8x.json`](../../benchmarks/results/neural_query_shift_8x.json)

## What remains before a paper or public breakthrough claim

- Replace the generic MiniLM scorer with a trained evidence-necessity model and
  compare against LLMLingua-2, RECOMP-style extraction, and other current
  extractive compressors.
- Evaluate NoLiMa-style lexical-gap retrieval, RULER multi-needle and tracing,
  multi-hop bridge evidence, LoCoMo-scale conversations, and repeated
  compaction trajectories.
- Measure downstream answers, unsupported claims, exact locked facts,
  calibration under domain/model/compression shift, recovery latency, and net
  tokens after repeated rehydration.
- Run multiple target LLMs and hardware classes. The committed pilots do not
  establish production latency or billing savings.
- Audit emerging work such as EXIT, IterCOMP, SARA, agent-memory action
  policies, and recent provenance graphs against exact papers and released
  code before making a novelty statement.

The defensible current statement is: PRISM-R is an implemented, locally tested
research prototype for relevance-gated neural selection with deterministic
evidence fallback and exact future-query rehydration. It has passed a bounded
evidence-retention pilot, not the full breakthrough gate.
