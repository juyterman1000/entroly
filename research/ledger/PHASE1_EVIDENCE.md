# Phase 1 Research Ledger — Entroly Forensic Baseline

**Baseline commit:** `e31f082f` (origin/main)
**Measured:** 2026-10-02
**Rule:** every row is a measurement or a file:line read. Interpretations are
labelled as such and separated from evidence.

---

## E-00 — Integration status

| field | value |
|---|---|
| CLAIM | The Entroly UserPromptSubmit hook failed with `UnicodeEncodeError` on one turn of this session |
| SOURCE | host hook output |
| SOURCE TYPE | runtime observation |
| EVIDENCE | `Entroly activation failed before parsing the host event: UnicodeEncodeError` |
| CONFIDENCE | HIGH (direct observation, single occurrence) |
| COMPONENT | agent_activation / hook integration |
| IMPLICATION | The dogfooding path has a non-ASCII input crash. Not yet root-caused. |
| REPRODUCTION REQUIRED | YES — not investigated |

---

## E-01 — True Rust architecture (contradicts CLAUDE.md)

| crate | files | lines | measured role |
|---|---|---|---|
| `entroly-engine` | 40 | **46,750** | the algorithms |
| `entroly-core` | 22 | 20,790 | mostly PyO3 bindings (7 `*_bindings.rs`) |
| `entroly-qccr` | 2 | 2,332 | query-conditioned ranking + selection |
| `entroly-wasm` | — | — | wasm-bindgen surface |
| `ui/desktop` | — | — | desktop shell |

Dependency direction (from `Cargo.toml`):
`entroly-core → entroly-engine (features=["python"]) + entroly-qccr (regex-full)`
`entroly-wasm → entroly-engine + entroly-qccr (regex-lite)`
`entroly-engine → neither sibling` (base layer)
`entroly-qccr → neither sibling` (pure logic)

**CLAUDE.md defect:** its "Rust Core Modules (`entroly-core/src/`)" table is wrong
for **8 of 10** entries. `knapsack`, `knapsack_sds`, `entropy`, `semantic_dedup`,
`bm25`, `depgraph`, `prism`, `sast` are all in `entroly-engine`. Only `cogops`
and `archetype` are in `entroly-core`. CLAUDE.md also describes `entroly-core` as
"the Rust computation engine" doing "all compute-heavy work" — the larger
algorithm crate is `entroly-engine`, which CLAUDE.md never names.

CONFIDENCE: HIGH. Classification: **DOCUMENTED BUT NOT VERIFIED → now falsified.**

---

## E-02 — Two independent ranking/selection stacks

| path | ranking | selection |
|---|---|---|
| `entroly-qccr/src/lib.rs` | own `bm25_corpus`, `bm25_score` | `mmr_select` (MMR) |
| `entroly-engine` | `BM25Index` + `tokenize_code`/`tokenize_path`/`split_identifier` | `knapsack.rs` (KKT-dual + 0/1 DP), `knapsack_sds.rs` (submodular IOS) |

`entroly-qccr`'s own manifest description claims it is "the single source of truth
for query-conditioned retrieval ranking + selection." **Not accurate at repo
level** — `entroly-engine` carries a complete parallel BM25+selection stack.

CONFIDENCE: HIGH (both implementations read directly).

---

## E-03 — Three selectors, and which one each surface ships

`entroly/engine.py:1542` gates the selector:

```python
if self._use_rust and refined_query.strip():   # -> QCCR
```

`optimize_context` docstring: *"Exact for the modular objective on the Rust
0/1-DP path; the Python fallback is density-greedy with a singleton champion,
provably within ½ of optimal."*

`entroly/qccr.py:7`: **"Rust mandatory — no pure-Python fallback"** (raises via
`_rust_unavailable` at lines 33–41 when symbols are missing).

### Behavior-equivalence matrix

| Surface | Code path | Rust + query | Rust, no query | No Rust |
|---|---|---|---|---|
| `sdk.optimize` | sdk.py:1460 → `engine.optimize_context` | **QCCR** | 0/1-DP knapsack | Python density-greedy (½) |
| MCP `optimize_context` | server.py:1346 → same | **QCCR** | 0/1-DP knapsack | Python density-greedy (½) |
| proxy | proxy.py:3547 → same | **QCCR** | 0/1-DP knapsack | Python density-greedy (½) |
| CLI | cli.py:4864 → `qccr.select` direct | **QCCR** | n/a | raises (Rust mandatory) |
| `sdk.compress` | sdk.py:412, codec path | query-**agnostic** codec — never QCCR | same | same |

**Strategically important:** `engine.py:1542`'s own comment states QCCR "is the
compressor validated by EVERY committed accuracy benchmark (needle/longbench/
squad/bfcl/mmlu/gsm8k/truthfulqa)". Therefore the published accuracy numbers
describe **only** the Rust+query path. Two *shipped* configurations run different
algorithms with different approximation guarantees:

1. **no query** → 0/1-DP knapsack
2. **no Rust** → Python ½-approximation (CI explicitly ships this: job
   "Pure-Python Fallback (base `pip install entroly`, no Rust engine)")

Convergence is nonetheless **materially better** than a prior note claiming
"3 selectors on 3 surfaces, only QCCR benchmarked": 4 of 5 surfaces now funnel to
QCCR under the normal configuration. Classification: **PARTIAL**.

---

## E-04 — The 99-module cycle: FALSIFIED as a research bottleneck

Prior interpretation under test: *"a 99-module SCC containing every entry point
directly constrains the model-neutral control-plane vision."*

Experiment: `research/experiments/cycle_coupling_probe.py` — import each probe in
a **fresh interpreter**, count loaded `entroly*` modules. Controls are hubs
**outside** the SCC.

| group | median `entroly` modules loaded |
|---|---|
| SCC members (codec, sdk, server, proxy, cli, auto_index, cache_aligner, context_receipts) | **146** |
| Controls outside SCC (tokens, path_safety, vault, config) | **146** |

**Identical.** SCC membership makes no difference. Mechanism:

| import | modules | time |
|---|---|---|
| `import entroly` (bare) | **146** of 368 (39.7%) | 1.013s |
| `import entroly_core` (Rust ext) | **2** | **0.003s** |

`entroly_core` is an extension module with **87 exported symbols, zero Python
dependencies, 340× faster import**.

### Classification (as requested, kept separate)

- **ARCHITECTURAL DEBT:** the 99-module SCC, and `entroly/__init__.py` eagerly
  importing 146 modules on any import. Real, costs startup and test isolation.
- **NOT a RESEARCH-LIMITING BOTTLENECK:** the moat-relevant primitives (BM25,
  knapsack/SDS, QCCR, SimHash, receipts, WITNESS) live in Rust behind a clean
  PyO3 boundary that is already extractable in isolation.

My earlier claim was wrong and is withdrawn. A context-state interface **can** be
extracted today without untangling the Python cycle.

---

## E-05 — Force-pin budget domination: FALSIFIED on current main

`entroly-engine/src/guardrails.rs:232` records the historical failure:
*"Broad patterns like 'copyright' or '⚠️' caused hundreds of files to be pinned
in real codebases, destroying budget enforcement entirely (langfuse: 90 pinned
files = 167K tokens on 8K budget)."* The predicate is now narrowed to credential
material (`-----BEGIN PRIVATE KEY-----`, `aws_secret_access_key`, non-comment
`secret_key`/`api_key`/`private_key` assignments).

`knapsack.rs:300–319`: pins are added first, `remaining_budget = token_budget −
pinned_tokens`, and selection **short-circuits returning only pins** if they fill
the budget.

Experiment: `research/experiments/budget_domination.py`, 120 real `entroly/*.py`
files, 5 queries × 3 budgets, fresh `ENTROLY_DIR` per engine.

### FIRST RUN WAS INVALID — recorded so the error is not repeated

The first run reported 0.0% pinned share at every budget. **That measurement was
broken.** It keyed fragments as `{f.get("id"): ...}`, but `export_fragments()`
emits **`fragment_id`**; every entry therefore collapsed to the key `None` and
the dict had length 1 (`fragments_indexed: 1` after ingesting 120 files — the
tell that exposed it). Every `fid in frags` test was asking about `None`.

The conclusion happens to survive, but **not for the reason the run appeared to
show**. The real reason is architectural, read directly from source:

`entroly-core/src/lib.rs:755–764`:

```rust
// Force-pin safety and critical files
// Pinning is operator policy; protection is a storage guarantee.
// Deriving pins from criticality force-included every manifest and
// security file in EVERY query: measured 56 fragments / 223,288
// tokens pinned, consuming 50.6% of delivered tokens at an
// 8,000-token budget, allocated identically regardless of query.
let effective_pinned = is_pinned;   // caller-supplied ONLY
```

So Entroly now separates two concepts:

| flag | source | meaning |
|---|---|---|
| `is_pinned` | **caller-supplied only** | operator policy; force-included in selection |
| `is_protected` | derived (`file_criticality`, `has_safety_signal`) | storage guarantee; never evicted |

`migrate_pin_semantics` (lib.rs:112–128) actively **converts** legacy
content-derived pins to `is_protected=true, is_pinned=false`.

Directly observed: a fragment containing `aws_secret_access_key = 'y'` ingests
with `is_pinned=False`. That is **correct by design**, not a defect — the safety
signal routes to protection, not to budget consumption.

**Therefore budget domination is structurally impossible from content alone**: no
corpus can auto-pin itself, regardless of how many secrets it holds. Pins only
appear when a caller asks for them. This is a stronger and more durable result
than the (broken) empirical run suggested.

**Two distinct historical blowouts are now on record**, both fixed:
1. `guardrails.rs:232` — langfuse: 90 pinned files = 167K tokens on 8K budget
2. `lib.rs:759` — 56 fragments / 223,288 tokens = **50.6% of delivered tokens at
   8K budget, allocated identically regardless of query**

CONFIDENCE: HIGH on the architecture (source-read).

### Corrected measurement (valid `fragment_id` key)

Corpus: 120 files, **2,644,826 chars (~661K tokens)** — 20× the largest budget,
so corpus exhaustion is excluded as an explanation for anything below.

| budget | mean pinned share | mean ranker share | selected tokens (range) |
|---|---|---|---|
| 2,000 | **0.0%** | **99.86%** | 1,994 – 2,000 |
| 8,000 | **0.0%** | **99.00%** | 7,837 – 7,983 |
| 32,000 | **0.0%** | **72.57%** | 21,482 – 26,738 |

**B-answer:** the ranker controls effectively the entire budget at 2K and 8K.
Ranking improvements are therefore **not** neutralised by pinning — the prior
concern is falsified, and selector quality remains a live lever.

### E-05b — Budget underfill: reclassified, likely BY DESIGN

Observed: at a 32K budget the selector returned 21,482–26,738 tokens. A separate
probe (3 fragments totalling 542 tokens, 8K budget) selected **1** fragment.

This is probably **intended precision behaviour, not a bug**: QCCR emits one
synthetic fragment per source file (`qccr.py:255`) and the README describes the
frozen evidence-selection benchmark as "selecting an average of **1.02 of 16
passages**". Under-filling the budget is what a precision-oriented selector does.

Do not treat underfill as waste without a task-quality measurement showing the
omitted budget would have helped. Reclassified from "finding" to **OPEN
QUESTION**: is there a recall cost to underfill at large budgets? Requires a
matched-budget accuracy experiment, not a token count.

---

## E-06 — Semantic/embedding retrieval is opt-in, not default

`sentence-transformers>=5.6.1,<7` appears only in extras (`neural`, `full`);
**0 occurrences in base `dependencies`**. Embedding libraries are referenced by
exactly three modules:

- `entroly/context_receipts/embedding_scorer.py`
- `entroly/neural_evidence_selector.py`
- `entroly/verifiers/local_nli.py`

**The default install ranks lexically** (BM25 + MMR). Classification:
**EXPERIMENTAL / opt-in**, not default. No gap claim is made here — competitor
characterisation is Phase 2 work and deliberately not prejudged.

---

## E-07 — Benchmark inventory, per-family classification

**91 JSON result files.** Provenance quality is **high** — each accuracy record
carries model, seed, budget, `git_sha`, CI95, sample count, audit JSONL path,
scoring function, and the compressor identity.

The `compressor` field independently confirms E-03:
`entroly.qccr.select (via bench.accuracy._compress_messages_modal mode='entroly')`
— every accuracy benchmark exercises **QCCR only**.

| family | n | model | budget | baseline acc | entroly acc | token savings | classification |
|---|---|---|---|---|---|---|---|
| needle | 20 | gpt-4o-mini | 2,000 | 1.00 [0.839,1.0] | 1.00 [0.839,1.0] | **99.5%** | USEFUL BUT LIMITED — accuracy at ceiling (both 1.0), so retention is uninformative; the savings figure is the real result |
| longbench | 50 | gpt-4o-mini | 2,000 | 0.64 [0.501,0.759] | 0.66 [0.522,0.776] | **85.3%** | USEFUL BUT LIMITED — CIs overlap heavily; +0.02 is **not** a significant gain |
| bfcl | 50 | gpt-4o-mini | 500 | 1.00 [0.929,1.0] | 1.00 [0.929,1.0] | **79.3%** | USEFUL BUT LIMITED — ceiling effect again |
| squad | 50 | gpt-4o-mini | 100 | 0.80 [0.670,0.888] | **0.72** [0.583,0.825] | 43.8% | **ACCURACY COST** — point estimate drops 8 points at a 100-token budget. CIs overlap so it is not significant at n=50, but this is the one family where the direction is negative. Needs a powered re-run, not a dismissal |
| gsm8k | 20 | gpt-4o-mini | 50,000 | 0.85 [0.640,0.948] | 0.85 [0.640,0.948] | **−4.0%** | **NOT COMPARABLE** — budget (50,000) vastly exceeds the 283-token input, so there is nothing to compress; the −4% is pure overhead |
| mmlu | 20 | gpt-4o-mini | 50,000 | 0.85 [0.640,0.948] | 0.85 [0.640,0.948] | **0.0%** | **NOT COMPARABLE** — same defect as gsm8k; identical CIs and 0% savings confirm nothing was compressed |

**Scorecard across 6 accuracy families:** 3 informative and positive (needle,
longbench, bfcl), 1 negative in direction (squad), 2 structurally not comparable
(gsm8k, mmlu). Two of six published accuracy benchmarks therefore measure
nothing, and one points the wrong way.

### What these benchmarks do and do not establish

**Do establish:** large token reductions at preserved accuracy on needle (99.5%),
longbench (85.3%), bfcl (79.3%). That is a real and useful result.

**Do NOT establish:**
1. **Accuracy superiority.** Three of four families sit at or near the accuracy
   ceiling, and longbench's CIs overlap. These are **non-inferiority** results.
2. **Model independence.** Every run is `gpt-4o-mini`, seed 42. No second model
   appears, so no cross-model claim is supported.
3. **Current-code validity.** Runs are stamped `git_sha` `0a54982698ee` and
   `ea7b7b2d43b4` (May 2026). Baseline is now `e31f082f`. Re-validation on
   current main is **required** before quoting these for present behaviour.
4. **Behaviour on shipped non-QCCR paths.** Per E-03, the no-query and no-Rust
   configurations run different algorithms that **no accuracy benchmark covers**.

The gsm8k row is the same structural defect fixed earlier this session in
`tests/test_readme_features_langfuse.py`: asserting a compression result when the
budget already exceeds the input. It should be re-run with a budget below the
input or marked not-applicable.

---

## E-08 — Unreachable selection guard

`entroly.guarded_selection` imports QCCR but is among the **44 modules
unreachable from any shipped entry point** (13,511 lines, 7.5% of tree).
A selection *guard* exists that no product path reaches.
Classification: **IMPLEMENTED, NOT PRODUCT-REACHABLE.**

---

## Open items carried into the rest of Phase 1

- [ ] Full capability inventory table (retrieval / selection / compression /
      assurance / memory / learning / surfaces) with per-row classification
- [ ] Benchmark per-family classification (STRONG / USEFUL BUT LIMITED / LEGACY /
      NOT COMPARABLE / INVALID)
- [ ] B1–B7 bottleneck answers
- [ ] Root-cause E-05b underfill
- [ ] Root-cause E-00 UnicodeEncodeError
