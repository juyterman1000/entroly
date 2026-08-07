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
