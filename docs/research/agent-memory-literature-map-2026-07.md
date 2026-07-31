# Agent memory, context control, and world-model literature map

**Cutoff:** 30 July 2026

**Purpose:** define Entroly's prior-art boundary and falsifiable research program

**Claim status:** research map, not a systematic-review completeness claim

## Method

It is not credible to claim that one implementation pass has read “all papers.”
This map instead uses a reproducible scope:

- primary papers or official proceedings from NeurIPS, ICML/PMLR, ICLR/OpenReview,
  and arXiv;
- work directly relevant to agent memory, context compression, provenance,
  retrieval risk, cache economics, workflow memory, or learned world models;
- emphasis on 2024–2026, with older work included only where it establishes a
  necessary conceptual boundary;
- claims are taken from the linked paper, not from social-media summaries.

The map should be refreshed before any novelty or superiority statement.

## 1. Compression and bounded working context

| Work | What it establishes | Entroly consequence |
|---|---|---|
| [LLMLingua](https://arxiv.org/abs/2310.05736) and [LongLLMLingua](https://arxiv.org/abs/2310.06839) | Learned prompt compression and query-aware long-context compression are established research directions. | Neural or token-level compression is not itself novel; it must be evaluated on evidence retention, provider cost, and recovery. |
| [RAPTOR](https://openreview.net/forum?id=GN921JHCRw) (ICLR 2024) | Recursive clustering and abstractive tree summaries support retrieval at multiple levels. | Hierarchical summaries are prior art; Entroly must preserve source linkage and measure update drift. |
| [Context Folding](https://openreview.net/forum?id=lNRgWoGfYg) (ICML 2026) | Agents can branch into sub-trajectories and fold completed work into summaries. | Entroly's task-dream/session-rescue work should preserve explicit fold boundaries, recovery handles, and receipts. |
| [Agentic Memory Should Localize Compression](https://openreview.net/forum?id=ztmwHisqJ4) (ICLR 2026 workshop) | Global repeated compression risks interference; modular retrieval/update overlap is the relevant stability variable. | Compression should be local to receipt-bound modules, and prior supported queries need regression probes after consolidation. |
| [Proteus](https://openreview.net/forum?id=VsBF0AJuej) (ICML 2026 submission) | Static memory capacity can saturate; incrementally activated memory reduces interference in neural sequence models. | Entroly should distinguish storage capacity from the context capacity exposed at each decision. |

## 2. Graph, hierarchy, and dependency structure

| Work | What it establishes | Entroly consequence |
|---|---|---|
| [HippoRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html) (NeurIPS 2024) | Knowledge graphs plus Personalized PageRank can support multi-hop retrieval. | “Uses a graph” is not a contribution. Entroly must show where a graph changes safety or task outcome. |
| [PRISM](https://arxiv.org/abs/2605.12260) | Pareto retrieval over structured intent-aware memory; its LoCoMo ablation found the graph-path and intent-cost modules nearly inert, while the LLM reranker drove compression. | Graph components require task-specific causal evidence; copying PRISM's visible architecture would not establish value. |
| [Structurally Aligned Subtask-Level Memory](https://openreview.net/forum?id=2CoRS45Ucj) (ICML 2026) | Memory granularity aligned with an SWE agent's functional subtask decomposition outperforms episode-level retrieval in the reported protocol. | Code-agent memory should be indexed by current execution slice and symbol/dependency scope, not only issue similarity. |
| [Beyond Semantic Organization](https://arxiv.org/abs/2606.06090) | Execution-state trees can isolate failed branches and reconstruct the active path better than flat semantic retrieval. | Entroly's session graph must distinguish active state, failed branches, completed subgoals, and recovery boundaries. |

## 3. Provenance, exact recovery, and memory contracts

| Work | What it establishes | Entroly consequence |
|---|---|---|
| [TierMem](https://arxiv.org/abs/2602.17913) | A provenance-linked summary tier can escalate to immutable raw logs when evidence is insufficient. | Raw escalation and provenance are prior art. Entroly's differentiator must be independently verifiable spans, explicit budgets, and testable control policy. |
| [Eywa](https://arxiv.org/abs/2605.30771) | Evidence-before-belief, immutable source evidence, deterministic retrieval, and separated answer instructions form a provenance-grounded memory architecture. | Entroly cannot claim provenance-grounded memory as first; it should interoperate and compare under frozen protocols. |
| [MemIR](https://arxiv.org/abs/2605.25869) | Typed memory atoms separate raw evidence, retrieval cues, and truth-bearing claims to reduce provenance-role collapse. | Receipt schemas should keep evidence, derived claims, selection rationale, and policy instructions structurally separate. |
| [AgenticSTS](https://arxiv.org/abs/2607.02255) | Bounded decision context is a contract governing what each future decision may see; typed layers make ablation possible. | Entroly should make per-decision visibility, omissions, and layer ablations explicit and replayable. |
| [Agent-Memory Protocol](https://proceedings.mlr.press/v317/wu26a.html) | Persistent-agent memory needs an explicit privacy and interaction protocol. | Scoped, revocable attachment and local access controls are part of memory correctness, not deployment polish. |

## 4. Retrieval quality, utilization, and adversarial evidence

| Work | What it establishes | Entroly consequence |
|---|---|---|
| [C-RAG](https://proceedings.mlr.press/v235/kang24a.html) (ICML 2024) | Conformal analysis can certify bounds on generation risk under stated assumptions. | A selection certificate is not a generation-risk certificate; answer-risk calibration is a separate evaluation track. |
| [ClashEval](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3aa291abc426d7a29fb08418c1244177-Abstract-Datasets_and_Benchmarks_Track.html) (NeurIPS 2024) | Models may mishandle conflicts between internal priors and retrieved evidence. | Dependency closure alone is insufficient; contradiction and evidence-conflict tracks are mandatory. |
| [RAGuard](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ed25c00ff6900989116d3ba5d607d33d-Abstract-Datasets_and_Benchmarks_Track.html) (NeurIPS 2025) | Misleading retrieval can make evaluated RAG systems worse than zero-shot baselines. | More retrieved evidence is not monotonically safer; Entroly needs poisoned/misleading corpus tests and refusal/escalation policies. |
| [SeCon-RAG](https://proceedings.neurips.cc/paper_files/paper/2025/hash/668563ef18fbfef0b66af491ea334d5f-Abstract-Conference.html) (NeurIPS 2025) | Semantic and conflict-aware filtering can improve robustness to corpus contamination in the reported setup. | Risk-aware selection must model conflicts, not only relevance and dependency. |
| [Diagnosing Retrieval vs. Utilization Bottlenecks](https://openreview.net/pdf/0d54206c2c2ad572983635ed860047d153cca39c.pdf) (ICLR 2026) | Retrieval precision and whether the model uses retrieved memory are separable failure modes. | Entroly must report retrieval, utilization, and answer support separately; a selected span is not proof of use. |

## 5. Cache economics and serving stability

| Work | What it establishes | Entroly consequence |
|---|---|---|
| [Don't Break the Cache](https://arxiv.org/abs/2601.06007) | Provider prompt caches materially change agent cost/latency; dynamic content placement can make naive caching worse. | Token reduction cannot be labeled cost savings without provider-observed cache and billing data. |
| [Cache-Aware Prompt Compression](https://arxiv.org/abs/2607.15516) | Compression has a cache-dependent cost crossover rather than universal economic value. | The selector must preserve stable prefixes and expose the modeled-versus-observed cost source. |
| [Irminsul](https://arxiv.org/abs/2605.05696) and [Leyline](https://arxiv.org/abs/2606.01065) | Position-independent cache reuse and serving-side cache edit directives are active system directions. | Application-layer context control must not pretend to solve serving-kernel cache reuse; adapters should preserve future interoperability. |

## 6. Workflow memory, self-improvement, and world models

| Work | What it establishes | Entroly consequence |
|---|---|---|
| [Agent Workflow Memory](https://proceedings.mlr.press/v267/wang25bx.html) (ICML 2025) | Agents can induce and reuse workflows from prior trajectories, with reported gains on web navigation. | On-the-fly skills are prior art; Entroly must gate promotion on evidence, task similarity, security, and verified outcomes. |
| [Reflexion](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) (NeurIPS 2023) | Verbal feedback stored in episodic memory can improve later trials. | Reflection is not self-improvement proof; failed or unverifiable reflections must not become durable instructions. |
| [General agents need world models](https://proceedings.mlr.press/v267/richens25a.html) (ICML 2025) | Flexible goal-directed behavior implies increasingly accurate world models under the paper's formal analysis. | A world-model claim needs predictive state/action/next-state accuracy, not a renamed memory graph. |
| [DALI](https://proceedings.neurips.cc/paper_files/paper/2025/hash/696996164c52a52b8a162e62574fc9f9-Abstract-Conference.html) (NeurIPS 2025) | Dynamics-aligned latent imagination can support contextual generalization in Dreamer-style RL. | “Dreaming” must be trained and evaluated against real transitions, with uncertainty preventing unsupported simulated experience from promotion. |
| [Zero-shot World Models via Search in Memory](https://proceedings.neurips.cc/paper_files/paper/2025/hash/da7d7341051b4e0444cc96c78f1e4df9-Abstract-Conference.html) (NeurIPS 2025) | Similarity search over remembered transitions can approximate a world model without training. | Entroly can test transition-memory simulation before introducing a costly learned model, but must quantify prediction error and coverage. |
| [RLVR-World](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b63a24a1832bd14fa945c71f535c0095-Abstract-Conference.html) (NeurIPS 2025) | Verifiable rewards can directly optimize world-model behavior. | Simulated trajectories require verifiers tied to real environment outcomes; unverifiable dreams remain hypotheses. |

## What Entroly may claim now

The implemented Context Receipt path combines:

1. exact UTF-8 source spans and recomputable source/fragment digests;
2. hard-budget, transitive dependency-closed selection;
3. bounded exact optimization of a declared internal retrieval objective;
4. deterministic no-regression fallback with a conservative upper bound;
5. a receipt-bound omitted-bundle recovery frontier.

Each ingredient has adjacent prior art. The defensible contribution is the
**combined, executable evidence-control protocol** and its narrow verified
invariants—not a claim that Entroly invented graphs, provenance, compression,
knapsack optimization, memory, skills, or world models.

## Falsifiable next research program

1. **Same-protocol retrieval:** LoCoMo C1–C4 with frozen answer/judge prompts,
   plus ingestion/query call counts, provider tokens, latency, and cost.
2. **Dependency-sensitive retrieval:** MuSiQue and bridge HotpotQA with
   evidence recall, closure integrity, exact-source verification, and answer
   support.
3. **Code-agent trajectories:** predeclared SWE tasks requiring symbols,
   architecture decisions, tool observations, and prior failures; compare full
   history, flat retrieval, certified closure, and closure plus recovery.
4. **Conflict/adversarial:** ClashEval, RAGuard-style misleading retrieval,
   prompt injection, source mutation, stale memory, and contradictory versions.
5. **Cache economics:** provider-observed cached/uncached tokens and billed cost
   under stable-prefix and query-aware compression policies.
6. **Dream-world validity:** predict state/action/next-state and verifier outcome
   on held-out real transitions; allow simulated experience to influence durable
   skills only after calibrated uncertainty and real-world confirmation.

Public language should advance only when a preregistered track passes. A
selection-score result cannot be promoted into an answer-quality, cost, or
world-model claim.
