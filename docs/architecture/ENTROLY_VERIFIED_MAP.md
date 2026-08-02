# Entroly — Verified Map (Installment 1)

**Purpose:** a map of what is *actually true* in this repo, established by running
things, not by summarising other docs. Where this contradicts `CLAUDE.md` or
`README`, this file is the one that was checked.

**Method:** every claim below is either (a) produced by a command recorded next to
it, or (b) read directly out of the cited `file:line`. Nothing here is inferred
from documentation.

**Status:** Installment 1 of a planned series. Covers scope, the two-engine
problem, entry points, execution lanes, and known doc drift. Later installments
go module-by-module.

Verified 2026-08-01 at commit `5018691` (branch `dogfood/optimize-payload`).

---

## 1. True scope

```bash
find entroly-core/src -name "*.rs" -exec cat {} + | wc -l    # 51,214
find entroly -name "*.py" -exec cat {} + | wc -l             # 114,184
find tests -name "*.py" -exec cat {} + | wc -l               #  61,544
```

| area | files | lines | note |
|---|---:|---:|---|
| Rust core (`entroly-core`) | 52 | 51,214 | the real compute engine |
| Python package (`entroly`) | 216 | 114,184 | orchestration, MCP, proxy, CLI |
| Tests | 285 | 61,544 | |
| `entroly-wasm` | 40 | 73,004 | **41k of this is build artifacts**, see §2 |
| `entroly-qccr` | 3 | 1,293 | |

**~301k lines total; ~165k of unique product logic.**

The `entroly-wasm` figure is inflated: 41,124 of its lines are two copies of a
generated `typenum` test file under `target/`. Real wasm source is ~32k.

---

## 2. THE TWO-ENGINE PROBLEM (most important fact in this file)

`entroly-wasm/src/` is a **port of `entroly-core/src/`**, file for file, and the
two copies have drifted.

```bash
diff entroly-wasm/src/entropy.rs entroly-core/src/entropy.rs | grep -c '^[<>]'   # 596
diff entroly-wasm/src/cache.rs   entroly-core/src/cache.rs   | grep -c '^[<>]'   #  24
diff entroly-wasm/src/skeleton.rs entroly-core/src/skeleton.rs | grep -c '^[<>]' #   2
```

**Live consequence:** `entroly-wasm/src/knapsack_sds.rs:125` still contains

```rust
(1.0 - dist as f64 / 64.0).max(0.0)
```

which is the similarity-estimator defect fixed in the core on 2026-08-01
(commit `ce28f65`). It reports two *unrelated* fragments as 0.5 similar instead
of 0; measured MAE against ground truth is 0.502 versus 0.080 for the correct
form. **The npm/wasm distribution currently ships the broken estimator.**

**What an expert must internalise:** any fix to `entroly-core` is not shipped
until it is also applied to `entroly-wasm`. There is no shared crate and no
compile-time link between them — nothing will fail if you forget.

---

## 3. Entry points (narrower than they look)

`[project.scripts]` declares three console commands, and **none of them is
`entroly/cli.py`**:

| console command | module |
|---|---|
| `entroly` | `entroly.docker_launcher_safe:launch` |
| `entroly-memory` | `entroly.memory_cli:main` |
| `entroly-compression-mcp` | `entroly.compression_mcp:main` |

Plus `python -m entroly` (`entroly.__main__`), and `import entroly` /
`entroly.sdk`.

Reachability is computed from those six, not from `cli.py`:

```bash
python scripts/codebase_graph.py
# modules/edges: 216/526 ; reachable 181/216 ; UNREACHABLE 35 modules, 11,449 lines
```

**36 Python modules (11,785 lines) are not reachable from any shipped entry
point.**

**Do not read that as "dead code" — it is not, and that inference was checked
and refuted.** "Unreachable" here means only *not reachable from entroly's own
six entry points*, which is the expected and correct state for large parts of a
library. Verified breakdown:

| category | count | can it be deleted? |
|---|---:|---|
| `integrations.*` (langchain, slack, discord, telegram, hermes, …) | 11 | **No — public API.** Users import these directly, so they can never appear in entroly's internal graph. Deleting them is a breaking change. |
| exercised by CI (`hermes_context_engine` via `tests/test_hermes_context_engine.py`, referenced in two workflows) | 3 | No |
| imported by tests only | ~19 | Judgement call, not obviously dead |
| genuinely unreferenced anywhere | 3 | `_rust_launcher` (54), `promotion_gate` (126), `integrate_entroly_mcp` (48) — and two of those have `__main__` blocks and are described as examples |

So the deletable surface is roughly **228 lines, not 11,785** — about 2%.

Two traps this exercise exposed, worth repeating because both produced
confidently wrong answers first time:

1. **Grepping bare module leaf names is useless.** Searching for `engine`,
   `retry`, `collectors` reported 73/26/17 "production references" that were
   almost entirely false positives. Match real import forms
   (`from x import y`, `import x.y`, the full dotted path) instead.
2. **A hit is not a reference.** `promotion_gate` appeared in a test file only
   as part of a *test function name*; `integrate_entroly_mcp` appeared in
   `repo_map.py` only as a *string in a description dict*.

The useful direction of the inference still holds, and is the one CLAUDE.md
states: a test importing a module does not prove a *user* can reach it, so
don't promote anything in this set to a README claim without a real product
path. The reverse inference — unreachable therefore dead — does not hold.

---

## 4. Execution lanes — which selector actually runs

This is the single most misunderstood part of the system, and the reason
several defects survived for a long time.

### The MCP `optimize_context` path (`entroly/server.py`)

```
optimize_context (server.py:1021)
  ├─ PRIMARY : qccr.select()            -> result["selector"] == "qccr"
  ├─ FALLBACK: self._rust.optimize()    -> inside `except Exception:` (~line 1214)
  └─ BRANCH  : self._rust.optimize()    -> `elif self._use_rust:` (~line 1222)
```

- **`qccr` is the primary selector.** `entroly/qccr.py` is the validated
  query-conditioned compressor, and it powers the accuracy benchmarks.
- **`ios_select` / `knapsack_sds` is the FALLBACK.** It runs when qccr raises,
  or via the `elif` branch.

### Consequences you must know

1. **ContextBench does not test `ios_select`.**
   `benchmarks/contextbench_determinism_tax.py::entroly_select` calls
   `qccr.select()`. `entroly/qccr.py` contains zero references to `ios_select`,
   `diversity`, `simhash`, or `knapsack`. Running ContextBench to validate a
   `knapsack_sds` change measures nothing.

2. **No benchmark covers the IOS lane at all.** Nothing under `benchmarks/`
   references `.optimize(`, `enable_ios`, or `ios_diversity`.

3. That lane is **production-live but benchmark-dark**, which is exactly how it
   accumulated: a near-constant diversity term, an inverted duplicate detector,
   and a redundancy penalty that grew with iteration count — all silent, with a
   green test suite, because the unit tests were also vacuous (the helper
   `make_frag` set `simhash` but never `has_simhash`).

**Rule of thumb:** before validating a change, confirm which lane your code is
in and whether any benchmark exercises it. Usually the answer is no.

---

## 5. The Rust↔Python contract

The Rust engine exposes **60 methods** to Python via PyO3
(`grep -c "    pub fn " entroly-core/src/lib.rs`). Roughly grouped:

- **ingest / index:** `ingest`, `batch_ingest`, `ingest_paths_stubs`,
  `remove_sources`, `load_index`, `persist_index`, `rebuild_dependencies`
- **selection:** `optimize`, `recall`, `recall_auto`, `recall_bm25`,
  `hierarchical_compress`, `explain_selection`
- **beliefs:** `set_belief`, `load_vault_beliefs`, `apply_belief_conditioning`,
  `get_belief_info_factor`, `update_belief_utilization`
- **learning:** `record_success`, `record_failure`, `record_reward`,
  `record_resolution_outcome`, `score_utilization`
- **analysis:** `analyze_health`, `security_report`, `scan_fragment`,
  `entropy_anomalies`, `compute_pagerank`, `dep_graph_stats`
- **determinism:** `set_rng_seed`, `set_benchmark_seed`

16 Python modules import the native core; each is required to keep a pure-Python
fallback (`python scripts/codebase_graph.py --json` lists them under
`native_boundary`).

---

## 5b. The MCP surface — 59 tools inside one 3,361-line function

`create_mcp_server()` occupies `entroly/server.py:2708-6069` and registers the
**entire** agent-facing API as nested `@mcp.tool()` functions. Extract it with:

```python
import ast
t = ast.parse(open('entroly/server.py', encoding='utf-8').read())
for node in ast.walk(t):
    if isinstance(node, ast.FunctionDef) and node.name == 'create_mcp_server':
        print([n.name for n in node.body
               if isinstance(n, ast.FunctionDef)
               and any(getattr(d.func, 'attr', '') == 'tool' for d in n.decorator_list
                       if isinstance(d, ast.Call))])
```

**59 tools**, grouped by what they are for:

- **context:** `optimize_context`, `smart_read`, `entroly_retrieve`,
  `recall_relevant`, `remember_fragment`, `explain_context`, `prefetch_related`
- **receipts:** `create_context_receipt`, `create_context_receipt_from_path`,
  `render_context_receipt`, `explain_receipt_omission`,
  `recover_receipt_omission`
- **proof-guided:** `prepare_/advance_/inspect_proof_guided_context`
- **outcome capture:** `record_outcome`, `record_test_result`,
  `record_command_exit`, `record_ci_result`, `record_edit_outcome`
- **vault:** `vault_status`, `vault_query`, `vault_search`, `vault_write_belief`,
  `vault_write_action`, `vault_time_travel`, `vault_hygiene_scan`,
  `compile_beliefs`, `verify_beliefs`, `refresh_beliefs`
- **verification:** `verify_provenance`, `verify_and_repair`, `verify_response`,
  `eicv_verify_claim`, `eicv_suppress_hallucinations`
- **analysis:** `analyze_codebase_health`, `security_scan`, `security_report`,
  `scan_for_vulnerabilities`, `blast_radius`, `coverage_gaps`, `repo_file_map`
- **flow / skills:** `epistemic_route`, `execute_flow`, `process_change`,
  `create_skill`, `manage_skills`, `prepare_task_dream`
- **ingest:** `ingest_diagram`, `ingest_voice`, `ingest_diff`
- **session:** `checkpoint_state`, `resume_state`, `sync_workspace_changes`,
  `start_workspace_listener`
- **misc:** `get_stats`, `entroly_dashboard`, `compile_docs`,
  `export_training_data`

### `optimize_context` is defined twice, and both are live

This looks like a bug and is not one — worth knowing before you "fix" it:

| line | scope | role |
|---|---|---|
| 1021 | `EntrolyEngine` (class, lines 609–2490) | the **engine method** |
| 3033 | `create_mcp_server()` (function, 2708–6069) | the **MCP tool** agents call |

They are in different scopes, so neither shadows the other. Verify with AST
rather than by eye — `grep` makes them look like a duplicate definition in one
class, which would silently override.

---

## 6. Architectural hubs and cycles

PageRank over the import graph — highest blast radius first:

```
0.03254  entroly.path_safety
0.03188  entroly.context_receipts.models
0.01864  entroly.esg
0.01747  entroly.server
0.01596  entroly.vault
0.01587  entroly.models.registry
0.01574  entroly.compression_retrieval_store
0.01372  entroly.ravs.events
0.01364  entroly.ccr
0.01363  entroly.privacy
```

`server.py`, `proxy.py`, `cli.py` have the widest fan-*out* but are near-leaves —
little imports them. Changing `path_safety` or `context_receipts.models` touches
everything.

**5 import cycles.** The largest spans 18 modules around
`entroly/__init__` ↔ `auto_index` ↔ `cache_aligner` ↔ `compression_mcp` ↔
`compression_proxy_live`. Import order there is load-bearing — prefer a
function-local import over a new module-level one.

Smaller cycles: `vault ↔ vault_time`, `atomic_decomposition ↔ esg`,
`context_fixed_point ↔ verified_efficiency`, `qccr ↔ sufficiency`.

---

## 7. Known documentation drift

Corrections to `CLAUDE.md`, each verified:

| `CLAUDE.md` says | Reality |
|---|---|
| `semantic_dedup.rs` — "SimHash O(1) duplicate detection" | It is **Jaccard**, **O(n²)**, and **not in the selection path**. Its only non-test caller is the `semantic_dedup_report()` diagnostic (`lib.rs:5165`). SimHash lives in `dedup.rs`. |
| 215 modules, 524 edges, 113,382 lines | Now 216 / 526 / 114,184 — drifts with every commit; re-run `scripts/codebase_graph.py`. |
| Architecture implies IOS is the selection pipeline | IOS is the **fallback**; qccr is primary (§4). |

---

## 8. Measured defects in the selection path

Established 2026-08-01 with `entroly-core/src/simhash_probe.rs`
(`cfg(test)`-only, 5 arms, runs over 1500 real repo fragments):

```bash
cargo test --manifest-path entroly-core/Cargo.toml --lib simhash_probe \
    -- --ignored --nocapture --test-threads=1
```

**Fixed** (commits `ce28f65`, `693c922`, `5018691`):
1. Similarity estimated as `1 - hamming/64` (linear in angle, not cosine).
   MAE 0.502 → 0.080, which is the sampling-noise floor at 64 bits.
2. `has_simhash` ignored at 7 sites, so fingerprint-less fragments were mutual
   "duplicates" taking a 10× penalty on no evidence.
3. Redundancy inflated by selected-set size (optimizer's curse). At k=256, 88%
   of perceived redundancy did not exist. Fixed with a union-bounded lower
   confidence bound.

**Still open** — all require a versioned fingerprint change, since they
invalidate persisted `DedupIndex` state, LSH bands, and prompt-prefix cache
stability (see the warning at `dedup.rs:16`):

4. **Tie-to-zero bias.** `simhash` sets a bit on `sum > 0`, so ties resolve to 0.
   Measured popcounts for 7–9 trigram fragments: 22, 28, 30, 23 (balanced = 32).
   Four *unrelated* one-liners report `diversity_score` 0.46 instead of ~1.0.
5. **Band-gate recall ceiling.** Candidates must share an exact 16-bit band
   across 4 bands, capping near-duplicate recall *independently of the
   threshold*: overlapping windows 17.1%, one-line edit 36.6%. Actual merge
   rates 0.9% / 9.3%.
6. **64 bits is too few** for graded similarity: `sd = π/(2√B)` → 0.196 at 64,
   0.098 at 256, 0.049 at 1024. Entroly already owns unused 1024-bit machinery
   in `memory/episode.rs:152` and `memory/lsh.rs:79`.

---

## 9. Suggested study order

Read in this order; each stage is a prerequisite for the next.

1. **The contract** — `entroly-core/src/lib.rs` PyO3 surface (60 methods). This
   defines everything Python can do.
2. **The engine** — `cogops.rs` (3195), then the primitives it composes:
   `knapsack.rs`, `knapsack_sds.rs`, `entropy.rs`, `bm25.rs`, `dedup.rs`,
   `depgraph.rs`.
3. **The primary lane** — `entroly/qccr.py`, since it is what actually runs.
4. **Trust surfaces** — `context_receipts.rs`, `witness.rs`, `eicv.rs`,
   `guardrails.rs`. These carry the product's differentiator.
5. **Orchestration** — `server.py`, `proxy.py`, `epistemic_router.py`,
   `flow_orchestrator.py`.
6. **Hubs last but know them cold** — `path_safety`, `context_receipts.models`.

Skip on a first pass: the 35 unreachable modules (§3), `sast.rs` (3131 lines of
rules, reference material), and `entroly-wasm` (a drifted copy of §2).

---

## Open questions for later installments

- What gates the `elif self._use_rust` branch in `EntrolyEngine.optimize_context`,
  i.e. how often does the IOS lane actually execute in production?
- Are the 35 unreachable modules dead, or reachable through a path
  `codebase_graph.py` cannot see (dynamic import, entry-point plugin)?
- Does `entroly-wasm` have its own tests, and do they encode the *old* behaviour?
  If so, porting the core fixes will fail those tests — which is the desired
  signal, but plan for it.
- `create_mcp_server()` is 3,361 lines in one function. Is that deliberate
  (closure over shared engine state) or accreted? It is the single hardest file
  region to test, since every tool closes over the same locals.

**Resolved in this installment:**
- ~~Why is `optimize_context` defined twice?~~ Different scopes — engine method
  vs MCP tool. Both live, neither shadows the other. See §5b.
