# Preregistration — does graph-aware selection beat chunk ranking on dependency-sensitive tasks?

**Written before any measurement.** Verdict rules are fixed here and do not move
after results are seen. This registers open question **Q-B** from
`BREAKTHROUGH_RESEARCH.md`.

## Motivation — what is actually unmeasured

Three distinct selectors run on three surfaces (**OBSERVED**, by reading source):

| surface | selector | covered by an accuracy benchmark |
|---|---|---|
| SDK / MCP `optimize_context` | QCCR — sentence BM25 + MMR, lexical | **yes**, all of `benchmarks/results/*_accuracy.json` |
| proxy (`proxy.py:3061`, default-on) | Rust HCC — dep-graph slicing, PageRank, submodular diversity | **no** |
| `context_bridge.load_hcc_context` | Python `HCCEngine` — modular rate-distortion greedy | **no** |

`entroly/server.py:679`: *"Python-routed selectors (currently QCCR) bypass the
Rust optimizer."* The dependency-graph lane has **safety** coverage
(`dependency_closure_integrity.py` — closure is never violated) and **fidelity**
coverage (`agent_symbol_delivery.py` — delivered bytes are real), and both
explicitly disclaim measuring quality. **No benchmark compares graph-aware
selection against chunk ranking for quality at matched budget.**

## Hypothesis

**H1.** On tasks whose answer requires a symbol reachable only through a
dependency edge, graph-aware selection delivers that indirect evidence at a
higher rate than lexical chunk ranking, at matched token budget.

H1 is the claim that would justify the graph machinery's existence. It is
currently assumed, not shown.

## Task construction

Corpus: Python files tracked at a pinned git ref in this repository, parseable,
≤ 400 KB.

A task is a pair `(S, T)` where:

- `S` is a top-level function/method defined in file `A`;
- `S`'s body calls `T`, where `T` is defined in file `B ≠ A`;
- the token `T` does **not** appear anywhere in the query text;
- resolution of `T` to `B` is by AST + import resolution against the pinned ref,
  never by the system under test.

**Query text:** `S`'s name and the first line of its docstring only — never its
body. This protocol is inherited from `AGENT_SYMBOL_DELIVERY_PREREGISTRATION.md`,
which already fixed it to prevent body leakage.

**Gold evidence:** `{A}` (direct) and `{B}` (indirect).

**Primary metric: indirect recall** — the fraction of tasks for which `B` appears
in the delivered context. `B` is nameable only by following the call edge, so a
purely lexical ranker has no signal pointing at it. This is precisely the
condition under which "dependency edges demonstrably matter."

Secondary: direct recall (`A`), delivered tokens, wall-clock latency.

## Arms

All at identical token budget. Budgets: 2000 and 8000.

| arm | purpose |
|---|---|
| **null** — empty context | construct validity; must fail |
| **random** — random fragments to budget | floor |
| **RAW-truncated** — corpus truncated to budget | naive baseline |
| **RAW-full** — entire corpus, no budget | ceiling / reference |
| **BM25** — lexical only | strong simple baseline |
| **QCCR** — current primary selector | the incumbent |
| **HCC** — `hierarchical_compress`, graph-aware | the mechanism under test |

## Verdict rules — fixed in advance

Let `r_HCC` be indirect recall for HCC and `r_base = max(r_QCCR, r_BM25)`.

- **Construct-validity gate (checked first).** If **BM25** attains indirect
  recall ≥ 0.90, the tasks are not dependency-sensitive — `B` was findable
  lexically without the edge — and **the whole run is void** for testing H1.

  *Amendment, recorded before any arm was executed.* The gate originally named
  the null arm. That is wrong for this metric: indirect recall measures whether
  the selector *delivered* file `B`, so an empty context scores 0 by
  construction and the gate would be vacuous. A null arm is the right control
  for task-*success* metrics, where the prompt may leak the answer; it is not a
  validity test for evidence delivery. The null arm is still executed and
  reported, but the load-bearing validity check is the BM25 ceiling. No results
  had been observed when this was changed.
- **A / B — supported.** `r_HCC − r_base ≥ 0.10` **and** the Wilson 95% lower
  bound on `r_HCC` exceeds `r_base`.
- **C — inconclusive.** Difference positive but the interval includes `r_base`.
- **D — rejected.** `r_HCC ≤ r_base`. Published as a negative result, and the
  graph lane's cost is then unjustified by this evidence.

A result is reported with median, p90, p95, p99 and worst case over per-task
outcomes, plus the failure count. Mean alone is not reported.

## Isolation — mandatory

Every engine is constructed inside
`benchmarks.engine_isolation.isolated_engine_dir`, with
`assert_engine_isolated()` called before use, and the ingested fragment count
asserted against what was supplied via
`engine.get_stats()["session"]["total_fragments"]`.

Without this the engine warm-starts from this repository's own index and the
benchmark measures foreign fragments while reporting plausible numbers — the
documented failure `engine_isolation.py` exists to prevent.

## Threats this design does not remove

- **Corpus is one repository, one language.** Any result is about Python in this
  repo. Generality across languages and repos is not established by this run.
- **Indirect recall is an evidence-delivery metric, not task success.** It does
  not show a model answers better; that is Q-A and needs the agent-task harness.
- **HCC may win by spending more tokens on `B` and fewer elsewhere.** Direct
  recall is reported alongside so a trade rather than a gain is visible.
- **Pinned ref only.** Results do not transfer across refactors of the call graph.
