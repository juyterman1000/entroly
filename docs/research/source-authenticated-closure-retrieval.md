# Source-authenticated dependency-closure retrieval

**Status:** implemented safety contract; answer-quality research is
preregistered, not claimed
**Public claim:** none until a same-protocol task benchmark passes

## Why the PRISM paper changes the research question

[PRISM](https://arxiv.org/abs/2605.12260) is useful because its ablations make
the headline more precise. On its LoCoMo protocol, the graph-path module (N1)
and intent-conditioned edge costs (N2) did not measurably change retrieval or
judge accuracy. The content-only LLM reranker (N3) was the main context
reducer: the reported answer-side context fell from 4,108 to 2,023 tokens,
while the binary judge change was not statistically significant. The paper
also reports that 1,536 of 1,540 questions produced bit-identical pre-rerank
candidate sets with and without N1/N2.

That is a warning against copying the visible architecture. Adding a typed
graph does not by itself establish value. A useful Entroly follow-up must test
where graph structure changes the safety or outcome of an actual agent.

PRISM's stated scope is long-conversation retrieval. Its limitations defer
actions, tool calls, observations, plans, and feedback to future work. Its
ingestion pipeline uses LLM extraction and causal consolidation, and its
answer-side compression uses an LLM reranker. It does not define an independent
source-byte authentication or exact omitted-fragment recovery contract.

## The Entroly hypothesis

Entroly should treat dependency structure as a **selection constraint**, not
just another relevance feature.

Let `D` be a directed graph where `i -> j` means fragment `i` requires
fragment `j`, and let `Cl_D(i)` be the transitively reachable resolved
dependency closure of `i`. A selected set `S` is admissible only when:

```text
sum(tokens(i) for i in S) <= B
and
for every i in S: Cl_D(i) is a subset of S
```

The first invariant is the caller's hard token budget. The second prevents a
referencing fragment from reaching an agent without the definitions or
evidence needed to interpret it.

For each selected or omitted fragment, a receipt additionally binds:

```text
sha256(source_bytes) == source_sha256
sha256(source_bytes[byte_start:byte_end]) == fragment_sha256
```

An omitted fragment remains locally recoverable by its receipt-bound handle.
This separates four questions that are often collapsed into one score:

1. Was the right evidence retrieved?
2. Was a dependency bundle transmitted intact?
3. Can a recipient authenticate the exact source span?
4. Can omitted evidence be recovered when the task reveals a new need?

None of these invariants says that an answer is correct. They make a wrong or
incomplete selection diagnosable and recoverable.

## What was implemented

The Context Receipt selector now:

- computes transitive resolved dependency closure with cycle detection;
- preflights the entire new bundle against the remaining hard budget;
- adds the entire bundle or omits the root atomically;
- emits `dependency_bundle_exceeds_budget` plus an actionable warning when the
  closure cannot fit;
- records `atomic_transitive_dependency_closure` in the receipt controls;
- preserves unresolved-reference warnings instead of calling the closure
  complete;
- applies the same contract in the Python and Rust backends.

The Rust receipt preview was also changed from byte slicing to Unicode-safe
character truncation, preventing a non-ASCII preview from panicking.

## Narrow proof, not a benchmark headline

`benchmarks/dependency_closure_integrity.py` exhaustively enumerates declared
small graph families, token costs, and every budget from one token through the
full graph cost. It checks:

- partial resolved-dependency closures;
- hard-budget violations;
- whether the legacy partial-add regression control detects the original
  defect;
- whether the production atomic selector emits a partial bundle.

This is a safety-invariant proof on synthetic graphs. It does **not** measure
answer quality, retrieval recall, latency, provider cost, or another product.

The committed 6-node matrix contains 47,862 cases across 17 unique chain,
star, diamond, and cycle graphs, token costs `{1, 2, 3}`, and every positive
budget through the full graph cost. The former partial-add control produced a
partial closure in 41,954 cases; the production atomic selector produced zero
partial closures and zero hard-budget violations. See the
[machine-readable result](../../benchmarks/results/dependency_closure_integrity.json)
and its adjacent SHA-256 sidecar. These numbers establish only the declared
integrity invariant.

## Prior-art boundary

The individual ingredients are not presented as first-of-their-kind:

- [Memex(RL)](https://arxiv.org/abs/2603.04257) uses stable indices and exact
  experience recovery.
- [EMBER](https://arxiv.org/abs/2606.05894) studies source-backed evidence
  retention before a query is known.
- [ContextBudget](https://arxiv.org/abs/2604.01664) treats context management
  as a sequential budget-constrained decision problem.
- [Don't Break the Cache](https://arxiv.org/abs/2601.06007) measures provider
  prompt-caching behavior on agentic workloads.
- [Cache-Aware Prompt Compression](https://arxiv.org/abs/2607.15516) studies
  the cost crossover between caching and query-aware compression.

The research question is therefore not “did Entroly invent graphs, recovery,
or caching?” It is:

> Does enforcing source-authenticated, recoverable dependency bundles improve
> task success and failure diagnosis under a hard, cache-aware context budget?

That combined claim remains unproven until the evaluation below is complete.

## Preregistered evaluation

### Track A: same-protocol conversational memory

- LoCoMo categories 1–4, matching the PRISM answer model, judge model, prompts,
  tokenizer, and 10-conversation split.
- Conditions: full context, Entroly flat retrieval, Entroly atomic closure,
  and—only if a runnable author implementation becomes available—the exact
  PRISM configuration.
- Metrics: answer score, evidence recall at fixed `K`, retrieved tokens,
  ingestion calls/tokens, query-time calls/tokens, latency, and total modeled
  provider cost.

No “beats PRISM” statement is permitted from numbers copied out of the paper or
from a different model/protocol.

### Track B: bridge retrieval

- MuSiQue and bridge-style HotpotQA, because PRISM reports LoCoMo is mostly
  anchor-discoverable and explicitly predicts its graph components may matter
  more on bridge questions.
- Add closure-integrity rate and source-span authentication rate to standard
  answer and retrieval metrics.

### Track C: agent trajectories

- Predeclare coding and tool-use tasks with required source symbols, tool
  observations, prior decisions, and failure evidence.
- Compare raw history, flat compressed context, atomic dependency closure, and
  closure plus bounded exact recovery.
- Measure task success, repair turns, unsupported claims, peak context, cache
  reads/writes, provider-observed usage, wall time, and recovery calls.

### Track D: negative and adversarial cases

- unresolved dependencies;
- cycles and diamonds;
- poisoned or contradictory evidence;
- source mutation after receipt creation;
- multibyte UTF-8 spans;
- budgets smaller than every valid closure;
- interruption and recovery-store restart.

## Decision rules

- If atomic closure improves integrity but not task success, publish only the
  integrity result.
- If it increases context or cost without a compensating task or recovery
  benefit, keep it as an auditable strict mode or revert the default.
- If gains disappear under a same-protocol rerun, the public claim is removed.
- No first-in-the-world, superiority, or breakthrough wording is allowed
  without independent reproduction and confidence intervals on the stated
  workload.
