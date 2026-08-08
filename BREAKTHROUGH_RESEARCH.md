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

### Correction: the selection engine is a fourth crate, and the map above missed it — **OBSERVED**

The previous revision mapped only the *codec* path and left "repository
intelligence, learned/adaptive mechanisms" unmapped with a warning not to assume
absence. That warning was correct, and the omission was larger than expected.

`CLAUDE.md` documents the selection modules (`knapsack.rs`, `bm25.rs`,
`depgraph.rs`, `entropy.rs`, `prism.rs`, `sast.rs`) as living in
`entroly-core/src/`. **They do not.** There are four Rust crates, not three:

| crate | role |
|---|---|
| `entroly-core` | PyO3 binding layer; re-exports the engine |
| **`entroly-engine`** | **the actual selection engine — 31 modules, 29,189 lines** |
| `entroly-qccr` | QCCR support |
| `entroly-wasm` | WASM/npm build |

`entroly-engine` is a path dependency of `entroly-core` with
`features = ["python"]`, and `lib.rs` re-exports its modules. It contains, among
others: `causal.rs`, `depgraph.rs`, `hierarchical.rs`, `skeleton.rs`,
`knapsack_sds.rs`, `learning.rs`, `prism.rs`, `rnr.rs`, `simhash_wide.rs`,
`conversation_pruner.rs`, `resonance.rs`, `trajectory.rs`.

### Three of this programme's candidate breakthroughs are already implemented — **OBSERVED**

Read directly from the module headers, not inferred from names:

- **`causal.rs` — "Causal Context Graph — Interventional Estimation."** Uses
  RAVEN-UCB exploration as a natural instrument variable to separate
  `P(success | do(include f))` from `P(success | observe(include f))`, and reports
  `confounding_bias(f)` as the gap. This is *strictly stronger* than the naive
  leave-one-out `Δᵢ = P(Y|C) − P(Y|C∖cᵢ)` this programme was about to propose,
  because leave-one-out inherits exactly the co-selection confounding that
  `causal.rs` was built to remove. Cites Pearl (2009), Hernán & Robins (2020).
- **`hierarchical.rs` — Hierarchical Context Compression.** Three resolutions
  (skeleton map → dep-graph cluster → full content), with **symbol-reachability
  slicing via the dep graph**, submodular diversity, entropy-gated per-fragment
  resolution, and PageRank centrality for budget allocation. That is the
  "query-induced evidence graph" and "multi-resolution representation" ideas,
  built.
- **`skeleton.rs`** — signature-level extraction claiming ~90% of structural
  information at ~10–30% of token cost; the intermediate resolution HCC selects.

`depgraph.rs` exposes `transitive_deps`, `reverse_deps`, `connected_components`,
`compute_dep_boosts` — the program-slicing substrate Q-B assumes must be built.

### The load-bearing finding: the sophisticated lane is bypassed and unmeasured — **OBSERVED**

| check | result |
|---|---|
| `enable_causal` default | `true` (`lib.rs:458`) |
| references to `hierarchical`/`causal`/`skeleton`/`depgraph` in `entroly/qccr.py` | **0** |
| files in `benchmarks/*.py` exercising the hierarchical or causal lanes | **0** |

`entroly/server.py:679` states it in a source comment: *"Python-routed selectors
(currently QCCR) bypass the Rust optimizer."* QCCR is the primary selector and
the one every committed accuracy benchmark runs. The Rust `optimize()` path —
which is where causal, HCC, skeleton and depgraph live — is reached only on the
fallback branch.

So this machinery is compiled in, enabled by default, **bypassed by the primary
path, and covered by no benchmark.** This is the same "production-live but
benchmark-dark" pattern already recorded for `ios_select`/`knapsack_sds`, but it
is much wider than one selector: it covers the causal and hierarchical lanes
entirely.

**This reframes the programme.** The highest-expected-value next step is not
hypothesis #21; it is measuring whether the mechanisms that already exist do
anything. If HCC's multi-resolution slicing works, it bears directly on Q-B, Q-D,
Q-H and Q-I. If it does not, that is a publishable negative result and a large
deletion. Either outcome is worth more than a new untested mechanism, and neither
can be claimed until a benchmark exists for the lane.

### Correction to Q-D — the certificate IS wired — **OBSERVED**

The open-questions section below states `SufficiencyCertificate` is "not wired to
any live path." **That is wrong.** `entroly/qccr.py:244` calls
`_attach_sufficiency`, which calls `sufficiency.certify` and attaches the payload
to fragments — inside the *primary* selector. What is true, and is the real
constraint, is that it is **fail-closed**: `sufficient` requires
`calibrated and calibration_id and verdict == "sufficient"`, and no
`CalibrationPolicy` ships, so production can never receive a `sufficient`
verdict. The mechanism runs; the authorisation to trust it is deliberately
withheld pending held-out calibration.

Its internals are a KKT/LP-duality construction: `shadow_price` is the maximum
utility density among *excluded* candidates (the dual variable of the budget
constraint) and `residual_risk = shadow_price / captured_mass`. `corpus_gap`
independently detects "a discriminative query term appears in no candidate" and
forces `expand_required`. **Q-C's `min T(S) s.t. P(E ⊄ S) ≤ α` is therefore
half-built already**, and Q-D's "detect sufficiency without an LLM call" has a
running implementation awaiting calibration rather than an unbuilt idea.

### Still not mapped

Ingestion/chunking internals, memory/checkpointing, proxy provider handling,
evaluation harnesses, RAVS and the evolution daemon. **Do not assume these are
absent either** — the record above is what happened last time that assumption was
made.

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

## 6. Candidate mechanisms — **HYPOTHESIZED**

Generated against the corrected architecture map (§1), so anything already
implemented in `entroly-engine` is excluded by construction rather than
proposed and later discovered to exist. Nothing below has been measured.

### Excluded before generation — these already exist here

Proposing any of these would repeat the mistake §1 records. Listed so the
exclusion is explicit and auditable:

| idea | where it already lives |
|---|---|
| causal / interventional fragment value | `causal.rs` (do-calculus, instrument variable) |
| program-slice context via call graph | `depgraph.rs` + `hierarchical.rs` — and **measured worse than lexical**, §Q-B |
| multi-resolution representation | `skeleton.rs`, `hierarchical.rs` |
| submodular diversity selection | `knapsack_sds.rs` |
| near-duplicate suppression | `dedup.rs`, `semantic_dedup.rs`, `simhash_wide.rs` |
| BM25 + MMR query conditioning | `qccr.py` |
| gated preservation + byte-exact recovery | codec registry |
| conformal cascade / selective abstention | `conformal_cascade.py`, `escalation.py` |
| bitemporal provenance ledger | `vault_time.py` |
| cross-fragment statistical factoring | **rejected**, 16–24% ceiling measured |

### The 22 candidates

**Information theory / rate–distortion**

1. **Conformal risk-constrained selection.** Solve `min T(S) s.t. P(E ⊄ S) ≤ α`
   with α a *calibrated* guarantee, not a heuristic. `sufficiency.py` already
   computes the KKT dual (`shadow_price`) and refuses to certify without a named
   policy; the missing half is a conformal calibration mapping residual risk to
   observed evidence loss on held-out queries. Turns an uncalibrated diagnostic
   into a distribution-free bound.
2. **Successive-refinement layering.** Emit a coarse layer, then refinement
   layers that are *provably* successively refinable (Equitz–Cover), so an
   expansion never re-sends what layer 1 already carried. HCC's three levels are
   independent renderings, not a refinement chain — expanding re-sends.
3. **Syndrome context (Wyner–Ziv).** The encoder commits before the query
   exists; the decoder has the query. Slepian–Wolf says decoder-side information
   costs nothing asymptotically. Practically: ship a small syndrome over omitted
   evidence; if the retained context plus query implies the omission, it
   reconstructs, and if it does not, decoding *fails loudly* — an insufficiency
   detector that is a theorem rather than a heuristic.
4. **Distortion measured in answer space, not text space.** Define distortion as
   the probability the answer changes, and fit the rate–distortion curve
   empirically per workload, so budgets are chosen on a measured curve instead of
   a guessed ratio.

**Economics of billed tokens**

5. **Cache-aligned selection.** Optimise *billed* cost, not token count. Under
   prompt-prefix caching a cached read costs roughly a tenth of a fresh one, so a
   selection that is slightly less relevant but preserves a long cached prefix can
   be markedly cheaper. Objective becomes
   `min [ c_fresh·|S \ prefix| + c_cached·|S ∩ prefix| ]` subject to a quality
   floor. Entroly already maintains byte-stable prefixes as an invariant and ships
   `cache_aligner.py`; nothing selects *for* cache reuse.
6. **Session-level budget allocation.** Treat total spend across a session as one
   resource allocated over turns — a knapsack in time — rather than an independent
   budget per call. Early exploratory turns get less, the decisive turn gets more.

**Active learning / value of information**

7. **Expected-information-gain allocation.** Spend the next token where it most
   reduces uncertainty *about the answer*, using the retrieval-score distribution
   as a surrogate posterior, instead of taking the next most relevant fragment.
8. **Sequential context as best-arm identification.** Start minimal, expand only
   the region of highest posterior uncertainty, with a formal stopping rule.
   Distinct from 2: this is a stopping problem, not a coding problem.

**Coding theory**

9. **Erasure-coded evidence.** Encode fragments so any *k* of *n* reconstruct,
   letting the selector drop arbitrary members without losing recoverability.
   Recovery is out-of-band via a tool call, since a model cannot decode.
10. **Omission manifest / Bloom filter.** A tiny structure enumerating what was
    dropped, so the model can request exactly what it lacks instead of guessing.
    Weaker than 3 — detection only, no reconstruction — but far cheaper.

**Database systems**

11. **Materialised context views with incremental maintenance.** Cache selected
    context keyed by (repo state, query class) and invalidate by change impact —
    `change_pipeline`/`blast_radius` already computes the impact set. IVM
    semantics for context.
12. **Synopsis-first retrieval.** Per-file sketches answering "could the answer be
    here?" cheaply, pruning the corpus before ranking. Attacks latency, not quality.
13. **Query planning over context operators.** Treat selection as a plan over
    operators with cost estimates, chosen per query, rather than one fixed pipeline.

**Search / IR**

14. **Learned sparse expansion (SPLADE-style).** Fix qccr's vocabulary-mismatch
    ceiling with term expansion that stays sparse and inspectable, unlike dense
    embeddings — but it needs a model, trading the $0/offline property.
15. **Late interaction over lexical units.** Token-level max-similarity scoring
    rather than whole-fragment scores, keeping determinism.

**Program analysis**

16. **Type-and-contract slicing.** Slice by the type signatures and invariants
    needed to answer, not by call reachability — which Q-B measured as a loser.
17. **Test-as-specification context.** For a failing test, treat its assertions as
    the specification and select context by what those assertions reference.
    Concrete, high-signal, and directly matched to coding-agent work.

**Compiler theory**

18. **Liveness analysis for conversation context.** A fact is live if a future turn
    may read it; dead facts are dropped. Replaces recency/relevance heuristics in
    long agent sessions with a dataflow criterion.
19. **Partial evaluation of context.** Resolve configuration and constants directly
    into the code shown, so the model never needs the config file — removes a
    cross-file dependency rather than compressing it.

**Control theory**

20. **Closed-loop budget controller.** Predict answer quality, adjust budget, stop
    when a conformal bound is met. Compression as feedback control instead of a
    one-shot transform.

**Agent memory**

21. **Write-time / read-time asymmetry.** Compress hard on write, keep recent
    material exact on read, consolidating on a schedule. Partially present in
    `memory/consolidation.rs`; the asymmetry itself is not exploited.
22. **Answer-shaped compression.** Compress toward the expected *answer schema*
    rather than the query string — for "which function does X call", retain call
    edges preferentially.

---

## 7. Scoring — **HYPOTHESIZED**

Weights are the mandate's: novelty 15, quality gain 15, token savings 10,
evidence preservation 15, recoverability 10, provider independence 5, latency 5,
determinism 5, feasibility 5, moat 10, publishability 5.

Scored harshly. Anything needing a model to compress loses provider
independence, determinism *and* the offline property at once, which is why the
IR candidates rank low here despite being strong ideas in general.

| # | candidate | score | note |
|---|---|---:|---|
| ~~5~~ | ~~Cache-aligned selection~~ | ~~82~~ → **D** | **REJECTED on prior art** — mechanism published (CacheWeaver, RAGCache) and vendor-documented practice; see below |
| 1 | **Conformal risk-constrained selection** | **78** | now top candidate; half-built; converts a diagnostic into a guarantee |
| 3 | **Syndrome context (Wyner–Ziv)** | **71** | strongest theory; unproven that a useful syndrome is small |
| 17 | Test-as-specification context | 68 | high signal, narrow to one workload |
| 11 | Materialised context views | 64 | large latency/cost win, modest novelty |
| 2 | Successive-refinement layering | 62 | clean theory, needs the closed loop to matter |
| 19 | Partial evaluation of context | 58 | removes dependencies outright; scope unclear |
| 18 | Liveness for conversation context | 57 | needs a future-read model |
| 20 | Closed-loop budget controller | 55 | depends on 1 for its stopping rule |
| 10 | Omission manifest | 54 | cheap, low ceiling |
| 6 | Session-level budget allocation | 52 | real, but an accounting change |
| 7/8 | EIG allocation / best-arm | 50 | surrogate posterior is the weak link |
| 22 | Answer-shaped compression | 48 | risks over-fitting to query taxonomy |
| 13 | Query planning over operators | 46 | large build, unclear payoff |
| 4 | Answer-space distortion | 45 | measurement programme, not a mechanism |
| 12 | Synopsis-first retrieval | 44 | latency only |
| 21 | Write/read asymmetry | 42 | partly present already |
| 9 | Erasure-coded evidence | 38 | out-of-band recovery already exists and is simpler |
| 16 | Type-and-contract slicing | 35 | Q-B result casts doubt on all slicing |
| 14/15 | Learned sparse / late interaction | 30 | forfeits $0, offline and deterministic at once |

### Why **cache-aligned selection** ranks first

It is the only candidate whose primary metric is **the quantity the user is
actually billed for**, and the one place where this programme's blocked
dependency does not bite: cached versus fresh token counts are reported by the
provider, so it can be validated **without a model-quality judgement** — unlike
every proxy measured so far.

It also inverts the field's assumption. Everyone minimises tokens; under prefix
caching, tokens are not fungible, and a 10x price difference between cached and
fresh means the cheapest context is often *not* the smallest. Entroly already
holds the two prerequisites — byte-stable prefixes as an invariant, and
`cache_aligner.py` — and nothing currently selects *for* reuse.

### Prior-art search done: **D — REJECTED** — **READ**

The mechanism is published, and the search took one query. Recorded as the
fourth novelty rejection in this programme.

| work | what it does |
|---|---|
| **CacheWeaver**, arXiv 2606.19667 — *Cache-Aware Evidence Ordering for Efficient Grounded RAG Inference* | reorders retrieved evidence to maximise shared prefix for KV-cache reuse. Inspected: targets **GPU prefill compute** in self-hosted serving (vLLM Automatic Prefix Caching, SGLang), and **preserves the full evidence set** — reordering only, no gating |
| **RAGCache**, arXiv 2404.12457 (ACM TOCS) | cache-aware reordering to raise hit rate; prefix tree over document sequences serving overlapping requests |
| **Cache-Craft**, SIGMOD 2025 | chunk-cache management for RAG |
| **CacheClip**, **ProphetKV**, **Fusion RAG Cache**, **Grounded Cache Routing** | further KV-reuse variants |
| **arXiv 2601.06007** — *Don't Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks* | the multi-turn agentic evaluation this section proposed running |

Worse for the claim: putting static content first and volatile content last, to
maximise the cacheable prefix, is **documented standard practice** in OpenAI's,
Anthropic's and Azure's own prompt-caching guides. The idea is not merely
published, it is vendor-recommended.

**What survives is a deployment difference, not a mechanism.** CacheWeaver
optimises GPU prefill in a stack you control and reorders a *fixed* set;
Entroly's setting is provider-side billed caching where selection also *drops*
evidence under a budget. That is a different objective and a different
constraint, but the algorithm — order/choose evidence to lengthen the shared
prefix — is the published one. Per this programme's standard, a new deployment
context is not a new mechanism, and the 82/100 score above was inflated by a
novelty term that the evidence does not support.

**Consequence for the ranking.** The top candidate is removed. The next is
**#1, conformal risk-constrained selection**, which has the advantage of being
half-built already: `sufficiency.py` computes the KKT dual and deliberately
refuses to certify without a named calibration policy. Its novelty question is
narrower and sharper — not "is risk-constrained selection new" (it is not) but
"is a *distribution-free calibrated* bound on evidence loss, gating the
representation at creation time, new". That must itself be prior-art searched
before any build, against conformal prediction, selective prediction and the
bounded-AQP literature already surveyed in §5.

**Killing experiment.** Measure billed cost across a realistic multi-turn agent
session with arms {current selection, cache-aligned selection} at matched answer
quality. Kill it if cache-aligned selection fails to reduce billed cost by ≥25%,
or if it degrades evidence recall at all — a cheaper context that loses evidence
is the failure mode this programme exists to prevent.

#### First measurement: premise confirmed, opportunity not yet tested — **OBSERVED**

`entroly/cache_aligner.py::align` matches a SHA-256 of the **whole** context:
byte-identical or total miss. Providers cache by *prefix*, so any change in
selection currently bills everything fresh. `qccr.py` contains **zero**
references to cache state. The premise holds and is sharper than stated above —
there is no partial-reuse path at all.

`benchmarks/cache_reuse_opportunity.py` over 19 consecutive query pairs at a
2000-token budget:

| metric | median | p90 | max |
|---|---:|---:|---:|
| shared prefix (what is billed today) | 0.1% | 0.1% | 0.2% |
| shared content (recoverable by reordering) | 0.0% | 14.3% | 22.2% |

**This does not test the hypothesis, and must not be read as refuting it.** The
20 graph-lane tasks are 20 *unrelated* queries about different symbols. Cache
reuse is a property of successive turns **within one session**, where the task is
fixed and context largely repeats; independent queries have no reason to share
fragments, and 0.0% median overlap is the expected result rather than an
informative one.

What it *does* establish, validly and narrowly: **cache-aligned selection cannot
help a cold, diverse query stream** — there is nothing to reuse. Its value, if
any, is confined to multi-turn sessions on a stable task. That bounds the claim
before any build.

**The correct experiment**, still to run: successive turns of a single agent
task, measuring billed cost with cached-vs-fresh accounting from the provider's
own usage numbers. `usage_ledger.py` already separates uncached, cache-read and
cache-write tokens and prices each tier, so the accounting exists.

---

### Candidate #1 also rejected on prior art — **READ**

| work | what it does |
|---|---|
| **arXiv 2511.17908** — *Principled Context Engineering for RAG: Statistical Guarantees via Conformal Prediction* (also Springer) | inspected: conformal **coverage-controlled filtering that removes irrelevant content while preserving recall of supporting evidence**, model-agnostic, reducing retained context **2–3×**, with downstream factual accuracy stable or improved. This is candidate #1 |
| **C-RAG**, arXiv 2402.03181 | certified generation-risk bounds for retrieval-augmented models |
| **Conformal-RAG** | group-conditional coverage guarantees across sub-domains |
| **arXiv 2410.02914** — *Streamlining Conformal Information Retrieval via Score Refinement* | conformal retrieval sets guaranteed to contain relevant information, made smaller by monotone score transforms |

The one residual distinction — that 2511.17908 *reports and respects* a coverage
target rather than **refusing** to emit a context that fails it — is the same
gate-versus-metric line drawn in §5. And §5 already recorded that distinction as
**not novel**: enforcement-versus-validation is textbook database integrity, and
bounded AQP already emits a derived representation only when a guarantee provably
holds. The residual was pre-empted before it was raised.

**Verdict: D — REJECTED.**

---

## The result of this programme: **B, not A** — and the evidence is consistent

Five novelty claims have now been generated and searched. **All five died.**

| claim | killed by |
|---|---|
| protected / lossy channel separation | SemanticZip (2605.24541) |
| preservation as enforcement, not validation | textbook DB integrity constraints (ENABLE) |
| gated derived representation | bounded AQP (BAQ, TKDE'18), BEAS |
| cache-aligned selection | CacheWeaver, RAGCache; and vendor-documented practice |
| conformal risk-constrained selection | arXiv 2511.17908, C-RAG, Conformal-RAG |

Five for five is no longer a run of bad luck; it is a measurement of the field.
**Context compression and context assurance are saturated at the mechanism
level.** Any single mechanism reachable by reasoning from first principles here
has been published, usually within the last two years, and often by several
groups at once.

The honest consequence, stated against the mandate's decision standard:

> **B — STRONG ENGINEERING INNOVATION.** Not fundamentally new scientifically.
> No individual mechanism in Entroly is novel, and this programme should stop
> looking for one. What is uncommon is the **conjunction** actually shipped:
> preservation *gated* at creation rather than scored; recovery that is
> **byte-exact and model-free**, so the compressed form is a view rather than a
> replacement; **deterministic and offline**, with no model required to
> compress; and receipts that make the whole selection inspectable.
>
> Every element of that conjunction is individually published. The combination,
> operated as one auditable system, is not — but that is an engineering and
> product claim, and it must be defended with measured evidence on real
> workloads, **not** with a novelty claim that five searches have now refuted.

**What this changes about where effort should go.** Chasing mechanism novelty
has produced five rejections and zero validated mechanisms. The unmeasured
things are worth far more: no model has ever been in the loop here, so every
number in this document — 76.7%, 24%→100%, all of it — is a proxy for a quality
effect nobody has observed. Q-A remains the binding constraint, and
`benchmarks/answer_correctness_bridge.py` is committed and waiting on a working
API key.

---

## 9. Candidate #23: context compression as sound abstraction — **HYPOTHESIZED**

Generated after the five rejections, deliberately outside the frame they share.

**The frame all five assumed.** Compression is lossy approximation, and the
guarantee is *probabilistic over a query distribution*: conformal gives
`P(evidence loss) ≤ α`. §5 separately established, from the information-
bottleneck task-dependence result, that a minimal query-agnostic *sufficient*
statistic cannot exist. Both facts point the same way — every surviving approach
must gamble on an unknown future query.

**The move: stop requiring sufficiency, require soundness.** Treat the
compressed context as an **abstract interpretation** of the corpus — concrete
domain the full source, abstract domain the compressed form, joined by a Galois
connection. The guarantee becomes universally quantified rather than
probabilistic:

> For every query in the supported fragment, the answer computed on the
> compressed form is a **sound over-approximation** of the answer on the
> original: never wrong, possibly `unknown`.

This escapes the IB impossibility because soundness is strictly weaker than
sufficiency. An abstraction may lose precision without lying. The failure mode
changes from *silently wrong* to *explicitly unknown, with the exact span to
recover* — which is the fail-closed posture the rest of this system already
takes, given a formal footing instead of a heuristic one.

Concretely for code: abstract a function to signature plus effects (raises,
mutates, calls, returns). "Does `f` touch global state?" is answered soundly
from the abstraction. Anything the domain cannot decide returns
`insufficient → recover span X` rather than a guess.

### Prior-art status — survived two searches, **not** cleared

| search | closest hits | verdict |
|---|---|---|
| abstract interpretation × LLM context compression | **SAIL** (POPL/PACMPL 3808308), **AbsInt-AI**, *Cost-Driven Synthesis of Sound Abstract Interpreters* (2511.13663) | all the **inverse** direction — LLMs used to *synthesise* abstract interpreters. Not abstract interpretation used to compress context |
| soundness certificate over compressed context | arXiv 2605.17304 | already read in §5: *measures* preservation (Critical Atom Recall), does not gate, no soundness certificate |

Five predecessors each died on their **first** search. This one has survived two.
That is a genuine signal and it is **not** clearance: POPL, PLDI and OOPSLA were
not searched properly, and program-abstraction-for-LLM work is the likeliest
place it already exists.

### The weakness, stated before anyone else finds it

Abstract interpretation needs a **formal semantics** for both the query language
and the answer domain. Natural-language questions over code have neither. So the
soundness guarantee applies **only to a restricted, formalisable query
fragment** — call/effect/type/reachability predicates — and says nothing about
open-ended questions, which are most of what an agent actually asks.

That is a real scope limit, not a detail. The honest claim is not "sound context
compression" but "a sound *core* for the decidable fragment, with everything
else explicitly outside it". Whether that fragment covers enough real queries to
matter is an empirical question and is the first thing to measure.

### Why this candidate is worth the next unit of effort

Uniquely among everything here, **it is partially validatable without a model.**
Soundness is a proof obligation discharged once per abstract domain, not
measured. Completeness — the fraction of real queries the abstraction answers
rather than declaring `unknown` — is deterministic and countable offline. With
the API key blocked, this is the only candidate whose primary evidence does not
require inference.

**Killing experiment.** Define the abstract domain (signature + effects), mine
real questions from this repository's own issue and commit history, and measure
the `unknown` rate. Kill it if the sound fragment answers **< 20%** of real
questions — a formally beautiful guarantee covering almost nothing is a paper,
not a product.

**Verdict: C — INTERESTING BUT UNPROVEN.** Not A. There is no implementation, no
abstract domain defined, no completeness measurement, and prior art is not
cleared. Recorded as the strongest surviving direction, not as a result.

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

### The proxy destroys evidence at no compression benefit — **OBSERVED**

`benchmarks/codec_ablation.py` gained a `proxy` arm calling the real
`proxy_transform.compress_tool_output`, measured on the same five fixtures and
required-evidence lists the codec work already used.

| arm | reduction | evidence retained |
|---|---:|---:|
| truncate (blind) | 76.9% | 75.3% |
| generic | 49.7% | 67.3% |
| **proxy (today)** | **75.1%** | **24.0%** |
| **specialized (codecs)** | **76.9%** | **100.0%** |

The proxy compresses no harder than the codec path while retaining a quarter of
the load-bearing evidence — **worse than blind truncation**. Per fixture, and
worst on the shapes that genuinely are tool output:

| fixture | proxy | specialized |
|---|---|---|
| `shell_failing_test_run` | 98.7% reduction, **1/6** evidence | 85.8%, **6/6** |
| `log_root_cause_flood` | 78.8%, **1/5** | 98.1%, **5/5** |
| `table_orders_export` | 51.3%, **0/4** | 96.8%, **4/4** |
| `code_python_module` | 58.8%, **0/5** | 61.4%, **5/5** |
| `json_payment_error` | 88.0%, 5/6 | 42.3%, 6/6 |

This reframes the surface-coverage gap. It was recorded above as the proxy
merely *not reaching* the registry — an optimisation left undone. It is not: the
keyword-pattern compressors hit their ratio by deleting exactly the failures,
identifiers and values the codecs exist to protect. That is a **trust regression
on the highest-traffic surface**, and it is the largest measured compression
defect found in this programme.

**Caveat, stated because it bounds the claim:** these are the five ablation
fixtures, not captured production traffic. They carry hand-written
required-evidence lists and were authored for the codec work, so they are not
neutral third-party samples.

#### That caveat was checked, and it bites — **OBSERVED**

Six real tool outputs were captured (`pytest -v`, `git log --stat`, `pip list`,
`ruff --output-format json`, `git ls-files`, a Python source file) and run
through the repaired `compress_tool_output`:

| sample | in | route taken | savings |
|---|---:|---|---:|
| pytest output | 1.9 KB | `test_output` (pattern) | 24.4% |
| `git log --stat` | 43.3 KB | `git_log` (pattern) | 93.8% |
| `pip list` | 32.2 KB | `esc_universal` | 97.5% |
| `git ls-files` | 41.5 KB | pattern → now ESC | 92.0% |
| Python source | 16.4 KB | **`codec`** | 73.7% |

**The codec registry claims only 1 of 5 real samples.** The 24% → 100% evidence
figure above is therefore an accurate statement about *codec-claimable* content
and a **misleading** one about proxy traffic overall, because the fixtures were
codec-shaped by construction. The wiring is still correct — when a codec claims
content it now preserves everything instead of a quarter — but its blast radius
on real traffic is far smaller than the fixture table implies. Recorded rather
than quietly dropped.

#### A worse defect found in the pattern path — **OBSERVED**

Testing on real output surfaced a live fabrication bug that the codec wiring
does **not** fix, because codecs decline on a bare file listing.

`_compress_build_errors` detected build output with
`any(kw in content for kw in [... "ruff", "tsc", "eslint", "ERROR" ...])` — a
bare substring scan over the whole blob. On a real 1,316-line `git ls-files`
listing, the single substring `"ruff"` in a path triggered it. It then kept the
two filenames containing "error", dropped 1,314 lines, and emitted:

```
[entroly: 2 errors, 0 warnings - 1315 lines compressed]
benchmarks/fever_error_analysis.py
tests/test_control_learning_snapshot_errors.py
```

99.7% of the evidence destroyed **and a fabricated error count asserted about
content containing no errors**. An agent asking "what files are in this repo"
received two filenames and a false diagnostic summary.

Fixed by anchoring detection to diagnostic *shape* (`^error[E…]`,
`^error:`/`warning:`/`note:`, `file:line:col: error`, `…Error:`) instead of
substrings anywhere. Verified: the listing no longer matches, while rustc
diagnostics, `file:line:col:` errors and Python tracebacks still do, and prose
still does not.

**The general lesson, which is the transferable part:** substring detection over
a whole blob decides *content type* on evidence that has nothing to do with
structure, and every keyword-pattern compressor in `proxy_transform` uses that
shape. This one was caught because a fabricated summary is visible; a silent
99.7% drop is not. Auditing the remaining pattern detectors for the same defect
class is open work.

### Q-B answered: graph-aware selection loses to lexical ranking — **OBSERVED**

Preregistered in `benchmarks/GRAPH_LANE_PREREGISTRATION.md`, run by
`benchmarks/graph_lane_quality.py` at pinned ref `16934bf`. 60 tasks, pool of 48
files each, 0 errors. A task is a caller `S` in file `A` importing and calling
`T` defined in file `B`; the query names only `S` and its docstring first line,
so `B` is reachable only along the call edge. Primary metric is indirect
recall — was `B` delivered.

| arm | indirect @2k | indirect @8k | direct @2k |
|---|---:|---:|---:|
| null | 0.0% | 0.0% | 0/60 |
| random | 0.0% | 6.7% | 4/60 |
| raw_truncated | 0.0% | 3.3% | 3/60 |
| bm25 | 5.0% | 28.3% | 19/60 |
| **qccr** (incumbent, lexical) | **76.7%** | **81.7%** | **60/60** |
| **hcc** (graph-aware) | **3.3%** | **6.7%** | **0/60** |
| raw_full (ceiling) | 100% | 100% | 60/60 |

**Verdict: D — REJECTED.** The preregistered rule required
`r_HCC − max(r_QCCR, r_BM25) ≥ 0.10`. The observed gap is **−73.4 pp** at 2k.
Graph-aware selection loses to plain BM25 on the very tasks where dependency
edges are the only route to the evidence.

Validity checks, both passed: BM25 at 5.0% is far below the 0.90 void
threshold, so the tasks are genuinely dependency-sensitive; and QCCR delivers
8–12 files out of a 48-file pool (not ~48), so its score is real selection, not
an unexpandable-query passthrough. Budgets are matched — HCC self-reports
utilisation 0.89–0.97, charging skeletons at their compressed cost.

Of HCC's 6 indirect hits, **5 were skeleton-only** (signatures, no bodies).

**Threat to this verdict, stated plainly:** level 1 of HCC is a one-line-per-file
map, and it is excluded from "delivered" on the grounds that a table of contents
is not evidence. If a consumer treats the level-1 map as actionable, HCC's
effective recall is higher than measured here. The verdict is therefore precise:
*HCC does not deliver the content of dependency-reachable files at these
budgets.* It is not a claim that HCC never names them.

**Why the incumbent wins, INFERRED:** QCCR extracts sentences per file, so a
2000-token budget buys partial coverage of 8–12 files. HCC assigns whole
fragments one of three resolutions, so a large caller file that will not fit
drops to level 1 entirely. Under tight budgets, partial coverage of many files
beats full coverage of few.

This retires Q-B and removes the assumption that the graph lane is a latent
advantage waiting to be wired in. Any future graph-selection proposal must beat
76.7% on this harness before it is worth building.

### Cross-fragment factoring — ceiling measured, direction rejected — **OBSERVED**

Every mechanism in Entroly compresses each fragment **independently** (HCC's own
docstring: the objective is modular, "a fragment's value depends only on its own
assigned level"). `dedup`/SimHash only *drops* near-duplicates; nothing *factors*
partial redundancy. Fragments in one pack are correlated — shared imports,
headers, idioms — so independent coding spends `Σ H(Xᵢ)` where joint coding needs
`H(X₁..Xₙ)`. The gap is unexploited by construction.

`benchmarks/cross_fragment_redundancy.py` measures an **upper bound** on that gap:
`1 − |lzma(concat)| / Σ|lzma(fᵢ)|`, over 8 random 24-file packs per workload.
lzma rather than zlib because zlib's 32 KB window cannot see across files, which
is exactly the redundancy under test.

| workload | median | min | max |
|---|---:|---:|---:|
| Python | 19.8% | 14.8% | 21.1% |
| Rust | 15.9% | 14.0% | 18.3% |
| Markdown | 23.7% | 21.2% | 25.5% |

**Verdict: D — rejected as a headline mechanism.** ~16–24% is the ceiling an
*entropy coder* reaches. A realizable mechanism must emit text the model can
read, so textual template factoring recovers only a fraction of that, against the
cost of new cross-fragment template machinery and a new byte-exact recovery path.
Additionally, redundancy *across calls* — far larger than within-pack — is
already addressed by prompt-prefix cache stability, which the project maintains
as an invariant.

Recorded so the next session does not re-derive it. The measurement is cheap
(stdlib only, no engine) and can be re-run if the workload mix changes.

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
