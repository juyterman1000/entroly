# Breakthrough research programme — context compression and assurance

**Status: seeded, not complete.** Sections 1, 2, 5 and 8 carry material that was
actually measured or read. Everything else is scaffolding with the questions
stated so the next session does not re-derive them. Claims are tagged
**OBSERVED** (measured here), **READ** (from a paper actually fetched),
**INFERRED**, or **UNTESTED**. Nothing untagged should be treated as evidence.

---

## 1. Current Entroly architecture (partial map, verified by reading)

Traced by reading source, not by grepping for symbol names — an important
distinction, because symbol-absence was misdiagnosed as capability-absence
three times while building this map.

### Compression path, as it actually runs

```
content
  → CodecRegistry.supports()        confidence-scored; highest bid wins
  → codec.representations()         several candidate Representations
  → caller picks                    smallest that satisfies its own contract
  → RecoveryStore                   content-addressed, byte-exact original
```

`Representation` carries `text`, `token_cost`, `protected_evidence`,
`recovery`, `source_sha256`, `distortion_risk`. A lossy form is only *offered*
when its preservation property has been checked — this gating is the unusual
part of the design and is where the leverage has been (§8).

### Surface coverage — **OBSERVED**

| surface | reaches codec registry |
|---|---|
| `compress_with_receipt()` | yes |
| `compress()` | yes (added this cycle) |
| CLI `entroly compress` | yes (`cli_recover.cmd_compress`) |
| `compress_messages()` | inherits via `compress()` |
| proxy | **no** — `proxy_transform`, zero codec imports |
| MCP | **no** — compression MCP serves stored spans only |
| Rust core | **no** — `context_receipts.rs` only, no codec modules |
| WASM / npm | **no** — exports `ingest`/`optimize` |

**3 of 8 reach it.** The proxy is highest-traffic and should move last, behind
cross-surface parity tests.

`compress_messages()` deliberately should *not* be routed through the registry:
its query-conditioned span-selection path (chunk → rank → select) is better for
conversation than structural codecs, and replacing it would be a regression.

### Codec inventory — **OBSERVED**

`json`, `log`, `shell`, `schema`, `code`, `document`, `table`. Selection is by
support confidence, not registration order (`schema` deliberately outbids
`json` at 0.95 vs 0.90 so a schema is not compressed as generic JSON).

### Not yet mapped

Ingestion/chunking internals, repository intelligence, memory/checkpointing,
proxy provider handling, evaluation harnesses, learned/adaptive mechanisms
(PRISM, RAVS, evolution daemon). **Do not assume these are absent.**

---

## 2. Literature landscape (only papers actually fetched)

| work | core mechanism | relevance | **status** |
|---|---|---|---|
| Drain, He et al., ICWS 2017 | fixed-depth parse tree; bucket by token count + leading token, mark a position variable when tokens *disagree across the bucket* | variability is a property of the data, not of spelling | **READ**, applied §8 |
| Information Preservation in Prompt Compression, arXiv 2503.19114 | measures what compressors destroy | "numerical values, named entities, specialized identifiers" are systematically lost; recommends protecting them, category-aware compression, validation mechanisms | **READ**, validates §8 |
| SemanticZip, arXiv 2605.24541 | protected channels (exact numbers, safety facts) vs lossy channels; model as semantic decompressor | **closest prior art**; see §5 | **READ** |
| Text-Preserving Lossy Compression, arXiv 2605.29000 | strategic deletion + LLM reconstruction | contrast: reconstruction needs a model | **READ** (abstract-level) |
| Lossless Token Sequence Compression via Meta-Tokens, arXiv 2506.00307 | guaranteed recovery via meta-tokens | contrast: lossless throughout, different rate regime | **READ** (abstract-level) |
| Semantic Compression With LLMs, arXiv 2304.12512 | LLM-driven semantic compression | baseline framing | **READ** (abstract-level) |

**Not yet covered** — and each is a named gap, not an omission to gloss over:
LLMLingua / LongLLMLingua, Selective Context, RECOMP, AutoCompressors, ICAE,
gist tokens, KV-cache compression, lost-in-the-middle / position bias,
information bottleneck, rate–distortion–perception, conformal risk control,
submodular / value-of-information selection, program slicing, graph
sparsification, MemGPT-style memory.

---

## 5. Novelty analysis — **the protected-channels idea is NOT novel**

| proposed | closest prior work | similarity | critical difference |
|---|---|---|---|
| protect identifiers verbatim, summarise the rest | **SemanticZip (2605.24541)** | very high — it explicitly separates "protected channels (exact numbers, legal/medical facts)" from "lossy channels" | SemanticZip does **not** require byte-identical reconstruction and treats *the model* as the decompressor |
| category-aware protection | arXiv 2503.19114 | high — recommends exactly this | that paper recommends; it does not supply a gating mechanism |
| positional slot discovery in logs | Drain (2017) | very high | none of substance — this is catching up to a 2017 result |

**Conclusion: D — the hybrid protected/lossy framing is published prior art.**
It was claimed as possibly novel earlier in this programme and then disproven by
searching for it. Recorded so it is not re-claimed.

**What may survive as a narrower contribution (UNTESTED as novelty):**

1. Preservation is **verified and gated**, not prioritised — the representation
   is *refused* unless the property provably holds, rather than weighted.
2. Verification is **structural** — column identity, distinct-value presence —
   catching truncation/reordering/coercion a substring scan misses, at
   O(columns) rather than O(values).
3. Recovery is **byte-exact and model-free**, so the compressed form is a view,
   not a replacement.

Prior-art search for the *conjunction* (gated + structural + model-free
recovery) has **not** been done. Do that before claiming anything.

### Threat resolved — the distinction survives, and sharpens — **READ**

**arXiv 2605.17304**, *Compress the Context, Keep the Commitments* (Trukhina,
Vashkelis). Read via the abstract page after the PDF failed to render.

Its "commitments" are *"goals, constraints, decisions, preferences, tool
results, retrieved evidence, artifacts, and safety boundaries that future
responses must preserve"* — semantic obligations, not cryptographic bindings.
It contributes **Critical Atom Recall** and **round-trip recoverability** as
*metrics*, together with a *taxonomy of semantic compression errors*.

That taxonomy is the decisive detail. Cataloguing compression errors is only
necessary in a framework where they occur; the abstract confirms lossless
recovery is not guaranteed, only measured.

| | 2605.17304 | Entroly |
|---|---|---|
| preservation | **measured** — Critical Atom Recall | **gated** — the form is refused unless it holds |
| recovery | **round-trip recoverability, a metric**; lossless not guaranteed | **byte-exact**, content-addressed, model-free |
| errors | taxonomised | prevented at creation |

**The distinction is therefore: that work formalises how to *evaluate*
preservation; Entroly makes preservation a *precondition of emitting the
representation at all*.** A measured system reports its Critical Atom Recall; a
gated system has no path that emits a form failing the property.

This is a genuine and defensible difference in kind rather than degree, and it
is narrower than the claim originally made. It is **not** a quality claim: no
head-to-head has been run, and their framework covers conversational
commitments that Entroly's codecs do not address at all.

### The database framing — concept not novel, application appears to be — **READ**

The gate/score distinction was searched in the database literature, where it
should exist if anywhere. It does, precisely: an integrity constraint in the
**ENABLE** state *"ensures that all data modifications upon a given table
satisfy the conditions of the constraint"* — enforcement refuses the write —
whereas **VALIDATE** concerns whether existing rows conform. Enforcement versus
validation is textbook. **The concept is therefore not novel, and must not be
claimed as such.**

The useful finding is the adjacent one: *"constraints are generally not
supported for view materializations."* A materialized view **is** a lossy
derived representation of a source, and the database world leaves those
unconstrained — integrity is enforced on base tables, while derived
representations are optimisation artefacts, refreshed and trusted rather than
gated.

That yields the right vocabulary for what Entroly does, and a precise statement
of where it sits:

> A codec representation is a **materialized view over the original with an
> ENABLED integrity constraint** — the preservation predicate — such that a view
> violating it is never emitted. The source remains content-addressed and
> byte-exactly recoverable, so the view never becomes the system of record.

Databases enforce constraints on base tables and not on derived views. LLM
context compression produces derived views and, per the two papers above,
*measures* their fidelity. Enforcing a constraint **on the derived
representation itself**, with the base retained for exact recovery, is the
combination neither field applies.

### Third pass: bounded AQP largely pre-empts this too — **READ**

Approximate query processing was checked, as the previous revision said it must
be. **Bounded AQP (BAQ, TKDE'18)** gives *"deterministic approximate results --
where the estimated query results must be within the error bound with 100%
confidence"*, and such systems *"select a subset of data that is guaranteed to
satisfy the error bound"*, declining when they cannot. **BEAS** likewise answers
exactly when feasible and otherwise within a deterministic accuracy lower bound
under a resource budget.

That is structurally the same move: a derived representation emitted only when a
guarantee provably holds, refused otherwise. **The gated-derived-representation
pattern is therefore not novel either**, and the "placement" claimed in the
previous revision is largely pre-empted. Recorded as a third rejection.

What has *not* been pre-empted, stated as narrowly as the evidence supports:

| | bounded AQP | Entroly |
|---|---|---|
| guaranteed quantity | numeric error of an aggregate | **set inclusion** of evidence spans |
| query | **known at selection time** | **unknown** — the representation is built before anyone asks |
| failure mode | answer outside the bound | evidence absent when a future question needs it |

AQP always has the query in hand; the bound is defined against it. Entroly must
commit to a representation *before* the query exists, so its guarantee has to be
**query-agnostic** — identifiers survive whatever is asked later.

**That is the same open problem as §15 (unknown-future-query compression), and
this is where the remaining research value sits.** A deterministic guarantee
over an unknown future query is not something the AQP formulation expresses,
because a bound requires a query to be a bound *on*. Whether a useful
query-agnostic guarantee exists at all — beyond "keep everything that looks like
an identifier" — is unresolved and is the honest frontier here.

### Fourth pass: the open question has a partly negative answer — **READ**

Searched the sufficient-statistics / information-bottleneck treatment of
"sufficient for an unknown downstream task", named above as the likeliest home
for prior art on the surviving question.

The IB principle defines an optimal representation as **minimal and sufficient
for a task**. The relevant result is the task-dependence: *"the optimal views
for one task may not be suitable for another task"* — a representation that is
minimal *and* sufficient for task A is generally **not** sufficient for task B.

**So a minimal query-agnostic sufficient statistic does not exist in general.**
That is a negative answer to the question this programme had narrowed to, and it
is more useful than a positive one, because it says what cannot be built and
therefore what the sensible design is.

Given the impossibility, a system committing to a representation before the
query exists has exactly three options:

1. **keep everything** — sufficient, not compressed;
2. **optimise for a guessed task** — compressed, silently insufficient when the
   guess is wrong, which is the failure mode arXiv 2503.19114 measures;
3. **keep a task-invariant core, recover the rest on demand** — compressed,
   and sufficiency is restored rather than gambled.

Entroly is (3), and this reframes it as the rational response to a proven
impossibility rather than a heuristic. Identifiers are the right core precisely
because they are *not* task-specific: an identifier is the join key for whatever
question arrives later, which is why destroying them is unrecoverable in a way
that dropping prose is not. Exact recovery is not a safety feature bolted on —
it is what makes (3) admissible at all, since without it (3) degenerates into
(2).

**This is a theoretical grounding, not a novelty claim.** IB, minimal
sufficiency and their task-dependence are long established. What the programme
can honestly say is that Entroly's architecture is the correct response to a
known impossibility result, and that the ceiling on any query-agnostic
compressor is set by how well its invariant core is chosen.

**Open and genuinely unresolved:** whether a better task-invariant core than
"identifiers plus structure" exists — an evidence class that is provably
join-critical across query distributions. That is the remaining research
question, and it is a question about *which invariant*, not about whether the
architecture is right.

**Status after four prior-art passes: two claims killed (protected/lossy
framing; enforcement-vs-validation), one largely pre-empted (gated derived
representation). Surviving question is query-agnostic guarantee, UNTESTED.**
Unsearched: VLDB/SIGMOD constrained view maintenance, OSDI/SOSP storage
integrity, and the sufficient-statistics / information-bottleneck treatment of
"sufficient for an unknown downstream task", which is the closest theoretical
framing and the most likely place for prior art on the surviving question.

### Superseded: outstanding threat (kept for the record)

**"Compress the Context, Keep the Commitments: A Formal Framework for Verifiable
LLM Context Compression"**, arXiv 2605.17304 (Trukhina, Vashkelis).

The title alone targets the same ground as the surviving claim above. A fetch
was attempted and the extraction **hedged throughout** ("appears to",
"suggesting") because the PDF streams did not render — so it has **NOT** been
read and nothing from that attempt is cited here. The tentative signal was that
verification may be *post-hoc* rather than a creation-time gate, which would
preserve the differentiator, but that is far too weak to rely on.

Read it properly and answer, specifically:

1. Are "commitments" cryptographic bindings over the original, or semantic
   assertions about meaning?
2. Is verification a **gate that refuses** a compressed form, or a post-hoc
   score? This is the crux — Entroly refuses.
3. Does verification or compression require an LLM?
4. Is byte-level recovery of omitted content provided?
5. What is actually proved?

If it gates at compression time *and* recovers exactly *and* needs no model,
the remaining claim is dead too, and this section becomes a second
**D — REJECTED**. Record that outcome either way.

### Adjacent baseline, now characterised — **READ**

LLMLingua / LongLLMLingua (arXiv 2310.06839): coarse-to-fine with a budget
controller, then **iterative token-level compression driven by per-token
perplexity from a small LM (GPT-2 / LLaMA-7B)**. Reported up to 20x with ~1.5%
loss on reasoning; LongLLMLingua adds question-aware compression and document
reordering, +21.4% on NaturalQuestions at 4x fewer tokens.

Architectural contrast worth stating precisely, because it is structural rather
than a matter of tuning: that family **requires a model to compress** (download,
load, inference per call, and perplexity is tokenizer- and model-dependent).
Entroly's codecs require none — deterministic, offline, no per-call inference.
That is a real difference in operating envelope, and it is *not* a quality
claim: no head-to-head has been run, and their reported ratios are far above
what the codec path achieves on prose.

---

## 8. Experimental results — **OBSERVED**

All measured in-repo, mutation-tested, CI-verified.

| workload | before | after | preservation |
|---|---|---|---|
| identifier-bearing JSON | 0% | 46–75% | identifiers byte-exact |
| templated logs | 0% | 96% | all values kept |
| high-cardinality request-ID logs | **0%** | **76%** | all values kept |
| pytest output | 74% | 80–97% | **failures 2/8 → 8/8** |
| `vin` column (name not in keyword list) | 99% | 62% | **1/200 → 200/200** |

### The unifying defect

Every codec could only **destroy values** or **refuse to compress**, and
defaulted to refusing on the shapes that mattered most. Refusal looked safe and
was: the failure was invisible because 0% compression raises no alarm.

The fix in each case: **factor out the invariant, keep the load-bearing values
verbatim.** Compression and preservation stop trading off — the gain comes
*from* holding identifiers out, not despite it.

### Negative / corrective results (recorded deliberately)

- **Prose needs no codec.** Measured 51–99% via the generic TF-IDF path;
  a prose codec would be a feature-count chase. **OBSERVED.**
- **A benchmark with no construct validity.** A null arm with *zero context*
  solved 4/4 tasks — the tests stated their own fixes. Any comparison built on
  that set was void. Task sets now require a failing null arm.
- **Budget contract silently overrode safety.** When no codec form fit the token
  budget, the blind generic path destroyed what the codec had protected.
  Front-loading critical lines *alone did nothing* — the caller still declined
  oversized forms rather than truncating them.

---

## Open questions with designed killing experiments

Ordered by expected value. Each states what would **kill** it.

**Q-A. Does any of this improve task success?**
Everything above is *mechanism* evidence. Killing experiment: the preregistered
agent-task benchmark (`benchmarks/AGENTIC_TASKS_PREREGISTRATION.md`) with arms
RAW / COMPRESS / CLOSED-LOOP under matched token budgets, plus **null** and
**random** controls. If COMPRESS loses >3pp, publish it.

**Q-B. Is a program slice denser than a chunk ranking for coding tasks?**
Kill: if graph-aware selection cannot beat BM25+MMR at matched budget on tasks
where dependency edges demonstrably matter.

**Q-C. Can risk-constrained selection replace similarity ranking?**
`min T(S) s.t. P(E ⊄ S) ≤ α` instead of maximise-similarity-under-budget.
Entroly already computes a preservation predicate — this is the natural
generalisation. Kill: if the risk estimate is uncalibrated on held-out data.

**Q-D. Can evidence sufficiency be detected without an LLM call?**
`SufficiencyCertificate` exists but is **not wired** to any live path. Kill: if
its verdict does not correlate with actual answer failure.

**Q-E. Does the compressed form carry enough signal to know recovery is needed?**
Small verification signatures over omitted evidence. Kill: if detection recall
is below the cost of just including the evidence.

---

## Rules this programme runs under

- Every benchmark carries **null**, **random**, **RAW** and **token-matched RAW**
  arms. A task the null arm solves is rejected, not reported.
- Report median, p90, p95, p99 and worst case. A compressor that averages 70%
  and destroys evidence on 5% of queries is dangerous, not good.
- Negative results are published. Goalposts do not move after a result.
- No competitor comparison is claimed without pinned versions executed on shared
  workloads. Nothing in this document supports a leadership claim.
