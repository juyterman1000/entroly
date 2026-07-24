# Experiment 1 — Entroly Selection Reproducibility Matrix

**Status:** complete · **Reference commit:** `77067bc` · **Baseline (pre-fix):** `1a7c6c0`

## Question

For the Reproducible Context Protocol (RCP) thesis, verification is
`Verify(R) = [f(C,q,P,B) = S] ∧ [h(S) = R.h(S)]`. This experiment falsifies the
first conjunct's precondition: **given a fixed corpus `C`, query `q`, policy
`P`, budget `B`, engine build, and environment, does Entroly reproduce the same
ordered selection `S` byte-for-byte?**

Nondeterminism was treated as a hypothesis to test, not an assumed fact.

## Result (one line)

- **Selection `h(C) → S`: strictly deterministic** on every tested axis (byte-identical).
- **Corpus construction `repo → h(C)`: was nondeterministic; fixed in `77067bc`.**

---

## Two-layer decomposition

RCP reproducibility requires *both* layers to be reproducible. They were tested
separately.

### Layer A — Selection (frozen corpus → ordered selection)

**Corpus construction procedure.** The live index (918 fragments over this
repository) was exported once to a frozen JSON via
[`freeze_corpus.py`](exp1/freeze_corpus.py) (`engine._rust.export_fragments()` →
`{source, content, fragment_id, feedback_multiplier}`). Every run reads that
identical frozen corpus, so index state is not a variable.

**Selector under test.** `entroly.qccr.select(fragments, token_budget, query)` —
the validated query-conditioned selector (Rust BM25F + log-linear rerank +
edit-target localizer + budget-bounded sentence extraction).

**Digest serialization (exact).** For an ordered selection `S`:

```python
content_sha_i = sha256(fragment_i.content.encode("utf-8")).hexdigest()
blob = json.dumps([(rank_i, source_i, content_sha_i) for i in order],
                  separators=(",", ":"))
h(S) = sha256(blob.encode("utf-8")).hexdigest()
```

Byte identity `I(S_i, S_j) = 1[h(S_i) == h(S_j)]`.

**Set overlap (Jaccard)** over the *set* of selected sources:
`J(A, B) = |A ∩ B| / |A ∪ B|`.

**Rank agreement (Kendall's τ)** over the ranks of sources common to both
selections: `τ = (concordant − discordant) / (concordant + discordant)`, where a
pair `(s_i, s_j)` is concordant iff its relative order agrees in both rankings.

**Matrix** — query `"where does the proxy inject compressed context into
requests"`, budget `2000`, each condition a **fresh subprocess**. All produced
digest `711a0e35…`, `n = 8`:

| axis / condition | Jaccard | τ | byte-identical |
|---|---|---|---|
| repeat, same process (×3) | 1.000 | +1.000 | ✅ |
| fresh process (baseline-2) | 1.000 | +1.000 | ✅ |
| `PYTHONHASHSEED ∈ {0, 1, 42, random}` | 1.000 | +1.000 | ✅ |
| `RAYON_NUM_THREADS / OMP ∈ {1, 2, 8}` | 1.000 | +1.000 | ✅ |
| input order: 3 seeded shuffles | 1.000 | +1.000 | ✅ |
| input order: full reversal | 1.000 | +1.000 | ✅ |

**Query/budget breadth** — 5 distinct tasks, each byte-identical across
`PYTHONHASHSEED=0` vs `PYTHONHASHSEED=random + RAYON_NUM_THREADS=4`:

| query (truncated) | budget | digest | identical |
|---|---|---|---|
| where does the proxy inject compressed context | 1000 | `ed306e85c99c` | ✅ |
| how are vault beliefs verified and staleness tracked | 3000 | `caf6bd70d322` | ✅ |
| knapsack token budget solver entropy scoring | 500 | `69444100bb4f` | ✅ |
| RAVS routes tasks to cheapest capable model | 2000 | `5486419df381` | ✅ |
| reconcile index content addressed dedup | 4000 | `8be419e13544` | ✅ |

The full-reversal invariance (`τ = +1.000`) is the substantive finding: the
selector's tie-breaks form a **total order** that does not depend on input
enumeration order.

### Layer B — Corpus construction (repo → fragment set)

**Test.** Five fresh cold ingests (`auto_index(force=True)`) of two byte-identical
files (`a_original.py`, `b_twin.py`), each in a fresh temp dir. SimHash keeps one
fragment and dedups the twin; which one survives is the observable.

| | winner distribution |
|---|---|
| **before (`1a7c6c0`)** | `a_original` ×4, `b_twin` ×1 — **nondeterministic (4:1)** |
| **after (`77067bc`)** | `a_original` ×5 — **stable & canonical** |

**Root cause (confirmed).** `_auto_index` assembled the ingest batch in
thread-**completion** order (`concurrent.futures.as_completed`); SimHash dedup
runs in batch order, so a read race decided the winner. The reconcile path had
the same latent bug via `sorted(set(...))`, whose equal-priority ties fell back
to nondeterministic set-iteration order.

**Fix (`77067bc`).** Total-order the ingest (priority DESC, canonical path ASC)
and emit the batch in that order instead of completion order; parallel reads are
unchanged. Regression tests:
[`tests/test_index_determinism.py`](../../tests/test_index_determinism.py).

---

## Environment

| | |
|---|---|
| OS | Windows 10.0.26200 (Windows 11) |
| CPU / arch | AMD64, Intel Family 6 Model 170 (Core Ultra / Meteor Lake) |
| Python | 3.10.0 |
| entroly | 1.0.66 |
| entroly-core (native) | present; **does not expose `__version__`** — `MIN_ENTROLY_CORE_VERSION = 1.0.66` |

> **Gap for RCP:** the native core exposes no version string. A re-derivable
> receipt's `V` field needs a precise `(entroly, entroly-core, schema)` triple.
> Exposing `entroly_core.__version__` (and a build hash) is a prerequisite for
> the protocol.

## Explicitly untested axes

Byte-identity is established **only** for: single machine, single architecture
(x86-64), single OS (Windows), single native build. **Not** tested here:

- **Windows ↔ Linux, x86-64 ↔ ARM** — the primary open risk. BM25/entropy use
  float scoring; floating-point non-associativity across architectures/BLAS
  could shift tie margins. Untested.
- **Cross-machine, same architecture.**
- **Pure-Python vs native selector** — `qccr.select` requires the native core;
  there is no pure-Python selection path to compare (worth confirming no silent
  fallback exists).
- **Full-pipeline warm vs cold index selection** — Layer A froze the corpus;
  the warm/cold *index* path feeding selection was not diffed end-to-end.

If cross-architecture byte-identity fails, the honest guarantee narrows to
**"deterministic within engine version + architecture"**, with the architecture
pinned in the receipt's `V` field. That is still a verifiable claim.

## What this does and does not establish

**Establishes:** given a fixed corpus representation, query, policy, budget,
engine build, and *tested* environment, Entroly reproduces the same ordered
selection byte-for-byte; and corpus construction is now deterministic on the
tested axes.

**Does not establish:** equivalent output across OSes/architectures; semantic
correctness of the selected evidence; resistance to malicious-but-valid corpus
content; independent-implementation agreement; task-level superiority; or market
need for replay. Those are later experiments.

> Note on external comparison: published reports of low retrieved-document
> overlap on repeated identical RAG queries come from ANN/embedding pipelines
> and are **not directly comparable** to this frozen-corpus, deterministic-CPU
> setup. They are context for *why* determinism is hard in mainstream stacks,
> not a benchmarked head-to-head.

## Reproduce

```bash
# from the repo root, at commit 77067bc
export ENTROLY_SOURCE="$(pwd)"

# 1. freeze the corpus from the live index
python docs/research/exp1/freeze_corpus.py docs/research/exp1/frozen_corpus.json

# 2. run the full reproducibility matrix (fresh subprocess per axis)
python docs/research/exp1/repro_harness.py

# 3. dedup-winner determinism (before/after) is guarded by the regression suite
python -m pytest tests/test_index_determinism.py -q
```

`repro_harness.py` writes permuted corpus copies next to itself
(`frozen_corpus*.json`); these are generated artifacts, not source.
