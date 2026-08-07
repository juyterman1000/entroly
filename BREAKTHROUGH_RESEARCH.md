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
