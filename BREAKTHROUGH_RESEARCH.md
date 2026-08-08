# Entroly research update — evidence-preserving context assurance

Entroly's research programme is focused on a practical question:

> **How can an AI system reduce context aggressively while preserving the evidence a future task may need — and recover the original exactly when more detail is required?**

The latest work strengthened that direction with measurable improvements in structured-data compression, log/tool-output handling, tight-budget safety, and the theoretical grounding for recoverable context views.

This document separates **measured engineering results**, **established prior art**, and the **open research frontier**. It does not treat architectural differentiation as a novelty claim or substitute design arguments for head-to-head evaluation.

---

## 1. What improved this cycle

### Identifier-bearing structured data

Real structured payloads often contain fields whose names are domain-specific and impossible to enumerate in advance. Protecting only a hand-written list of keys is therefore insufficient.

Entroly now detects identifier-like columns from the **data itself**: a column whose values are near-unique across records is treated as load-bearing even when its name is unfamiliar.

Measured fixture results from the landing work:

| payload | before | after | preservation |
|---|---:|---:|---|
| `vin` column, 200 unique values | ~99% reduction with only 1/200 VINs retained | 49–62% reduction | **200/200 VINs retained** |
| `policy_no`, unlisted identifier name | identifiers lost | ~60% reduction | **200/200 retained** |
| known `sku` identifier | already protected | ~69% reduction | **200/200 retained** |
| no identifier-bearing columns | ~99% schema reduction | unchanged | no regression |

The important result is not maximum compression. It is that Entroly can still remove substantial repetition **without sacrificing the join keys and identifiers a later task may need**.

A second correctness improvement landed with this work: columnar compression now declines when sibling top-level fields would otherwise be omitted, allowing a whole-document representation to handle the payload instead.

---

## 2. Logs now learn variability from the payload

A production log variable does not have to look numeric. Hostnames, request IDs, user IDs and other opaque values are often the most important changing fields.

Entroly's log codec now discovers variable positions by **disagreement across structurally similar lines**, while retaining the existing token-level detector for cases such as values embedded inside a token.

Measured fixture results:

| log shape | before | after |
|---|---:|---:|
| 300 distinct hostnames | 29% reduction | **82% reduction** |
| 300 distinct user IDs | 30% | **73%** |
| 300 opaque request IDs | 0% | **76%** |
| numeric logs | 47% | **47% — unchanged** |
| interleaved errors | 24% | **24% — both 402 and 500 preserved** |

The representation factors repeated structure while keeping distinct values available instead of collapsing them into an approximate summary.

---

## 3. Test and build output keeps the failures

Tool output is valuable because of the exceptional lines, not the repetitive success lines.

The codec now recognizes common test/build output and treats failure markers as critical evidence.

Measured fixtures:

| output | before | after |
|---|---:|---:|
| pytest run with 8 failures | 74% reduction, **2/8** failure lines retained | **80% reduction, 8/8 retained** |
| cargo build output | 0% | **64% reduction** |
| npm install output | 0% | **46% reduction** |
| timestamped logs | 48% | **53% reduction** |

This is the behavior Entroly should optimize for: **remove repetition while retaining the lines a model or engineer needs to diagnose the problem.**

---

## 4. Tight token budgets now preserve critical evidence

A safe structural representation is not useful if a later budget-enforcement step throws it away and blindly truncates the original text.

Entroly now keeps critical log/test evidence at the front of the safe representation and truncates only forms designed to remain meaningful under line-boundary truncation.

On the measured tight-budget fixtures:

- interleaved logs retained **8/8 ERROR lines** and both status codes 402 and 500;
- pytest output retained **8/8 FAILED lines**;
- savings remained high across the tested budget range.

The key engineering principle is:

> **When a budget forces a trade-off, repetition should disappear before failure evidence does.**

---

## 5. The public compression path is converging on one authority

The specialized codec registry provides structural handling for JSON, logs, shell/tool output, schemas, code, documents and tables, with protected evidence and recoverable representations where applicable.

During this cycle, the regular Python `compress()` path was moved onto the same codec authority used by the receipt-aware path, while keeping the existing budget-aware generic fallback for content no codec safely claims.

Current rollout direction:

```text
content
  -> classify
  -> choose specialized representation
  -> verify protected evidence
  -> enforce caller budget/contract
  -> attach recovery information where the surface supports it
  -> fall back safely when no structural form is appropriate
```

The next production goal is **semantic parity across supported surfaces**, rather than multiple public entry points quietly receiving different compression behavior.

Conversation compression remains intentionally query-conditioned; it should not be forced through a structural codec when span selection is the better contract for that content.

---

## 6. Why exact recovery is central, not incidental

Research on information bottlenecks and sufficient representations reinforces an important constraint: **a representation that is minimal and sufficient for one task is not generally guaranteed to be sufficient for an unknown future task.**

That observation supports a practical architecture for context systems that must prepare information before the future question is known:

1. preserve a task-invariant core such as identifiers and structure;
2. remove or summarize lower-value repetition only when preservation checks pass;
3. keep the original content addressable;
4. recover omitted detail on demand when a later task needs it.

Entroly follows this third path.

This is **theoretical grounding, not a claim that information bottleneck theory itself is novel**. The useful conclusion is architectural: exact recovery allows aggressive context views without pretending that a single lossy pre-computed representation can be sufficient for every unknown future task.

---

## 7. Research relationship to prior work

Several adjacent ideas are well established and should be credited as such:

- fixed-depth and disagreement-based log parsing;
- protected versus lossy semantic channels;
- information-preservation metrics for prompt compression;
- information-bottleneck/minimal-sufficiency theory;
- deterministic bounded approximate-query processing;
- integrity constraints and validation in database systems;
- model-assisted semantic compression.

Entroly does **not** need these ideas to be novel in isolation.

The engineering value is in the system-level contract being built around them:

```text
select evidence
   -> produce a smaller view
   -> verify protected evidence before emission
   -> retain byte-exact source recovery
   -> expose provenance/receipt data
   -> let later tasks recover what the smaller view intentionally omitted
```

That combination should be evaluated by its measurable behavior, not by novelty language.

---

## 8. The strongest current design distinction

Many compression systems optimize a fidelity score and report the result afterward.

Entroly's specialized codec path is designed to make key preservation properties **creation-time gates**. A candidate representation that fails its preservation predicate is not selected as a safe representation.

Examples include:

- identifier-column preservation;
- distinct status/error value preservation;
- structural integrity of columnar views;
- recovery references tied to the original source digest.

This does not prove superiority by itself. It establishes a stronger contract to test.

The right question for benchmarks is therefore not only:

> "How many tokens were removed?"

but also:

> **"Did the task-critical evidence survive, can omitted evidence be recovered exactly, and did the downstream task still succeed?"**

---

## 9. Evidence discipline

Entroly's research programme uses four evidence labels:

- **OBSERVED** — measured directly in the implementation or benchmark;
- **READ** — supported by prior work that was actually inspected;
- **INFERRED** — architectural conclusion consistent with current evidence;
- **UNTESTED** — hypothesis requiring an experiment.

A useful result is allowed to be a correction.

Finding that an idea has prior art does not weaken Entroly; it prevents the project from wasting time defending the wrong novelty claim and lets engineering focus on the system properties that customers can actually measure.

Similarly, a benchmark whose null-context arm succeeds is rejected rather than promoted. This is a feature of the research process: **invalid evidence is removed before it can become a product claim.**

---

## 10. Open research frontier

The most valuable remaining questions are practical and measurable.

### Better task-invariant evidence

Identifiers plus structure are a strong conservative core, but are they the best one?

Investigate evidence classes that remain useful across broad future-query distributions without forcing Entroly to retain everything.

### Model-conditioned context

Smaller local models may require tighter, more explicit evidence packages than frontier models.

Test whether model-aware context compilation improves task success while reducing RAM/VRAM pressure, latency and context size.

### Unified context compilation

Repository context, conversation, memory, RAG, tool output and schemas currently represent different evidence sources.

The long-term opportunity is one budgeted evidence compiler that can choose among all of them for the current task.

### Multi-turn degradation

Measure whether repeated compression accumulates error over 20, 50 and 100-turn sessions, and whether exact recovery plus memory can prevent that compounding.

### Real task outcomes

The primary benchmark should increasingly be real work:

- bug localization;
- code repair;
- test localization;
- RAG question answering;
- configuration diagnosis;
- tool-call correctness;
- long-running agent tasks.

Compression percentage remains a secondary metric.

---

## 11. Current research takeaway

This cycle produced a stronger and clearer engineering direction:

- **identifier-bearing JSON can now compress substantially while retaining all measured identifier values;**
- **opaque log variables can now be factored without being discarded;**
- **test/build output preserves the failures users actually care about;**
- **tight budgets preferentially remove repetition instead of critical errors;**
- **the public Python compression path is moving toward a shared structural authority;**
- **exact recovery has clear theoretical motivation when future queries are unknown.**

The goal is not to build the smallest text at any cost.

The goal is:

> **minimum sufficient context for the task in front of the model, backed by evidence preservation, provenance and exact recovery when more context becomes necessary.**

That is the research direction Entroly will continue to test, falsify and strengthen.

---

## 12. Measured follow-ups to the open frontier (§10)

A parallel session ran experiments against the questions §10 leaves open,
principally **better task-invariant evidence**. Results are reported here rather
than folded into the sections above, so the prior narrative stays intact and
these can be checked independently. Every harness is committed and deterministic;
none requires a model or an API key.

### Architecture corrections — **OBSERVED**

Three claims elsewhere in the repository were checked against source and found
wrong. They matter because the sections above reason about "the compression
path" as if it were single.

- **The selection engine is a fourth crate.** `CLAUDE.md` places `knapsack.rs`,
  `bm25.rs`, `depgraph.rs`, `entropy.rs` and `prism.rs` in `entroly-core/src/`.
  They are in **`entroly-engine/`** (31 modules, 29,189 lines), a path dependency
  of `entroly-core`. It also holds `causal.rs`, `hierarchical.rs` and
  `skeleton.rs` — three mechanisms a research programme might otherwise propose
  as new.
- **Three selectors run on three surfaces.** SDK/MCP `optimize_context` uses
  QCCR; the **proxy** uses the Rust hierarchical path (`proxy.py:3061`,
  default-on); `context_bridge.load_hcc_context` uses a *separate pure-Python*
  `HCCEngine`. Only QCCR is covered by any accuracy benchmark, so the
  highest-traffic surface is the least measured.
- **The sufficiency certificate is wired**, not dormant: `qccr.py:244` attaches
  it inside the primary selector. It is *fail-closed* — `sufficient` requires a
  named `CalibrationPolicy`, and none ships — which is a deliberate posture, not
  an omission.

### Graph-aware selection loses to lexical ranking — **OBSERVED**

Preregistered in `benchmarks/GRAPH_LANE_PREREGISTRATION.md` before running.
60 tasks where the evidence is reachable **only** along an import-and-call edge,
so a dependency graph should be decisive.

| arm | indirect recall @2k | @8k |
|---|---:|---:|
| BM25 | 5.0% | 28.3% |
| **QCCR** | **76.7%** | **81.7%** |
| hierarchical (graph-aware) | 3.3% | 6.7% |

The graph-aware path loses to plain BM25 on the tasks the graph exists to serve.
Validity gates passed: BM25 far below the void threshold, and QCCR selects 8–12
files from a 48-file pool rather than passing everything through.

#### REOPENED — the delivery proxy did not survive a model in the loop — **OBSERVED**

This section previously concluded that the result "retires the assumption that
the graph lane is a latent advantage awaiting wiring". **That conclusion is
withdrawn**, because the measurement it rests on has now been tested against a
model and did not hold.

`benchmarks/answer_correctness_bridge.py` preregistered the check and its own
consequence: *REFUTED if qccr does not beat hcc — the proxy would then be
measuring something that does not reach the model, and the Q-B verdict must be
reopened rather than defended.* Run on `entroly-qwen2.5-7b-32k`, 5 probes,
budget 2000, 0 errors:

| arm | correct | ctx tokens |
|---|---:|---:|
| null | 0/5 | 0 |
| **oracle** | **5/5** | 1,678 |
| bm25 | 0/5 | 1,941 |
| qccr | **1/5** | 2,063 |
| hcc | **1/5** | 1,730 |

Both gates pass — null at 0.0 is below the void threshold, and oracle at 5/5
shows the model performs the task exactly when the defining file is present, so
every failure below is a selection failure rather than a model failure.

**qccr's 76.7%-to-3.3% delivery advantage produced no correctness advantage at
all.** Delivery is therefore not a validated proxy for usefulness, and the
D-rejection above cannot be defended on it.

The mechanism is visible in the transcripts: every miss is `UNKNOWN`, never a
wrong signature, so the model reported absent evidence honestly. qccr delivers
the *file* as extracted sentences and the signature is frequently not among
them — it locates well and renders poorly, the same pattern dogfooding showed.

**Power is the honest limit:** n=5 cannot separate qccr from hcc, since 1/5
against 1/5 is a single probe. What it does support is the 5/5-versus-≤1/5
contrast, which is not marginal — selection produced usable evidence in 2 of 20
arm-probes while the oracle produced it 5 times out of 5.

Read this as "the delivery proxy is not validated", not as "the arms are proven
equal". Every ranking claim in this document that rests on delivery rather than
answers now carries that caveat.

### The index/span regime boundary is binary — **OBSERVED**

The strongest result, and the one no prior-art search matched. A signature index
(paths and top-level signatures, no bodies) was compared against span selection
at matched budget across two task families:

| regime | index | QCCR @matched | raw pool |
|---|---:|---:|---:|
| signature-resident (parameter names) | **12/12** | 9/12 | 12/12 @ 234,351 tok |
| body-resident (raised exception) | **0/20** | **20/20** | 20/20 @ 229,292 tok |
| overall | 12/32 | **29/32** | 32/32 |

Each mechanism is near-perfect in one regime and near-useless in the other, and
what separates them is **where the evidence physically lives** — a property of
the question, knowable before retrieval. The index reaching 0/20 is the metric's
own sanity check: a signature map cannot carry body evidence, so anything above
zero would have meant leakage.

Tools that ship signature maps as the default — aider's repo map among them — do
not publish where that default fails. **This boundary is a measurement, not a
mechanism, and should be presented as one.**

### The four-property conjunction, on real inputs — **OBSERVED**

§8 argues preservation gating is the strongest design distinction. Measured on
28 real inputs (captured command output and tracked source, nothing authored):
recovery verified 28/28, determinism 28/28, protected evidence 28/28, **all four
simultaneously 24/28** at a median 78.6% reduction. The four gaps are coverage,
not correctness: no codec claims a plain file listing, a package listing, or
pytest output.

### Directions closed cheaply — **OBSERVED**

- **Cross-fragment factoring.** Everything compresses fragments independently,
  so the mutual information between correlated fragments is unexploited. Measured
  ceiling for an entropy coder: **15.9–23.7%**. A mechanism emitting readable
  text recovers a fraction of that, and cross-*call* redundancy is already served
  by prefix-cache stability. Not worth building.
- **Sound abstraction** (compressed context as abstract interpretation, so the
  guarantee is universal rather than probabilistic). Over 1,959 functions:
  params 99.3% decidable, calls 98.5%, mutates 93.8%, **raises 6.5%**. The last
  is not a missing builtin model — of 9,844 unresolved call sites only 37% are
  builtins, while **63% are method calls on receivers of unknown type**.
  Interprocedural effect reasoning needs whole-program type inference. This
  predicts the approach is far stronger on Rust, Go or TypeScript, which is
  testable against this repository's own Rust crate.

### Prior art found for every mechanism proposed — **READ**

Seven mechanism-level candidates were generated across information theory,
coding theory, databases, IR, program analysis, compilers and control. **All
seven were already published**, each found in one or two searches, extending the
list in §7:

| candidate | prior art |
|---|---|
| cache-aligned selection (optimise billed, not raw, tokens) | CacheWeaver 2606.19667, RAGCache 2404.12457; also vendor-documented practice |
| conformal risk-constrained selection | arXiv 2511.17908, C-RAG 2402.03181 |
| addressability objective (index over content) | aider repo map, SigMap, jCodeMunch |
| evidence-locus routing | Adaptive-RAG, RAGRouter-Bench 2604.03455 |

This supports §7's position rather than complicating it: the defensible claim is
the assembled system and its measured behaviour, not any single mechanism.
**Generating further mechanism candidates has low expected value.**

### What remains genuinely unmeasured

**No model has been in the loop for any of the above.** Every number here is an
evidence-delivery or compression proxy for a quality effect nobody has observed.
`benchmarks/answer_correctness_bridge.py` is committed, preregistered and
budget-matched to close that gap, and is blocked only on a working API key. Until
it runs, §10's "Real task outcomes" remains the binding constraint on every claim
in this document.
