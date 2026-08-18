# PR #352 — Codebase understanding evidence

> Required by `docs/PR352_DEEP_CODEBASE_AUDIT_GATE.md` §14 and listed in
> `docs/PR352_MASTER_IMPLEMENTATION_PROMPT.md` §24 as a merge-blocking artifact.
>
> **Verdict: the gate is NOT satisfied.** This document records measured
> coverage, not a claim of completion. Per §18 of the gate, PR #352 must remain
> draft while the gaps in section 7 are open. It is currently not a draft.

## 1. Audited revision

```text
branch            integration/workgraph-production-20260817
audited SHA       09a98a5bf72fbac9dd161ac47ee928a85a802cd4
base (main)       51357eca17377669b1c3d4ec4fdf832e51baf406
worktree state    clean (0 modified at audit start)
merge/rebase      none in progress
audit worktree    detached checkout at the exact SHA, not the developer clone
```

The ledger in `PR352_PREVIOUS_SESSION_COMPLETED_WORK.md` names head
`ff4d1a57462432009cc621f0eb72d2ea969be1ce`. That SHA is an ancestor of the
audited head; the 10 commits since are documentation and CI only, including
`ci(integration): allow docs-only movement during dedupe validation`. The ledger
is therefore **not materially stale for code**.

## 2. Stage A — inventory and classification

```text
tracked files                    1,597
classified                       1,597  (100%, no silent remainder)
```

| category | files |
|---|---|
| test (`tests/`) | 422 |
| documentation | 298 |
| python-production (`entroly/`) | 298 |
| benchmark-fixture (`benchmarks/`, `bench/`) | 289 |
| rust-semantic (`*.rs`) | 65 |
| binding-wasm (`entroly-wasm/`) | 54 |
| tooling (`scripts/`) | 41 |
| workflow-ci (`.github/`) | 32 |
| configuration | 31 |
| node-js | 15 |
| resolved remainder (below) | 52 |

The 52-file remainder resolves to: packaging (`Dockerfile`, `Dockerfile.entroly`,
`CITATION.cff`, `LICENSE`, `NOTICE`), git tooling (`.githooks/pre-commit`,
`.githooks/pre-push`, ignore files), deployment
(`deploy/cloudflare-community-savings/`), examples (`examples/*.py`), the
benchmark comparator (`external_adapter/`), research harnesses
(`docs/research/exp1/*.py`), two ad-hoc root scripts (`run_real_mcp_test.py`,
`run_real_proxy_test.py`), and lockfiles.

**Nothing remains `unknown-needs-review`.**

### Finding A1 — a fourth test surface exists

`entroly-core/tests/` contains 4 PyO3-native test files (`conftest.py`,
`test_brutal.py`, `test_integration.py`, `test_native_scripts.py`) which
`pytest tests/` does **not** execute. Any coverage claim scoped to `tests/`
silently excludes the native boundary.

### Finding A2 — an 8.4 MB build artifact is committed

```text
e178.tar.gz      8,380,542 bytes, tracked, not gitignored
```

Every clone pays for it. (`entroly.mcpb` is 545 bytes and harmless.)

## 3. Stage C — structural map

### Python — 100% of the package mapped

Generated with the repository's own `scripts/codebase_graph.py` at the audited
SHA:

```text
modules            290
import edges       776
total lines        149,678
entry points       7
reachable          254 / 290
UNREACHABLE        36 modules, 12,879 lines
import cycles      7   (largest spans 29 modules)
PyO3 boundary      16 modules import the Rust core
```

`CLAUDE.md` records 215 modules / 524 edges / 113,382 lines / 4 cycles (largest
18) at 1.0.70. The architecture has drifted **+35% modules, +48% edges, cycles
4 to 7, largest cycle 18 to 29** since that baseline was written.

The largest cycle is the proxy subsystem plus the package root, `server`, and
`sdk`: `proxy`, `proxy_access_security`, `proxy_control_plane_safe`,
`proxy_gateway_shadow`, `proxy_routing_authority`,
`proxy_routing_official_guard`, `proxy_routing_safety`, `proxy_traffic_receipt`,
`proxy_traffic_receipt_final`, `proxy_traffic_session`, `proxy_traffic_value`,
`proxy_transport_final`, `proxy_transport_safe`, `proxy_value_dashboard`,
`proxy_value_projection`, `dashboard*`, `daemon*`, and 12 more.

The `_safe` / `_final` / `_guard` / `_authority` / `_integrity` suffixes indicate
growth by patched sibling rather than by editing the original — the same pattern
commit `bec85812` already diagnosed and removed for codec / sufficiency / qccr.

### Rust — crates and module surface mapped; source NOT read

```text
entroly-engine    33 files   33,327 lines    canonical semantics
entroly-core      24 files   22,843 lines    PyO3 layer
entroly-wasm       6 files    4,965 lines    WASM layer
entroly-qccr       2 files    2,332 lines    QCCR SSOT

binding registrations
  33 #[pyfunction]   11 #[pymethods]   9 #[pyclass]   70 #[wasm_bindgen]
```

### Node / WASM

```text
js modules 26 · .d.ts 5 · npm package entroly-wasm@1.0.78
work_graph JS: work_graph, _repo, _store, _continuity, _content_digest
npm test script runs 7 suites incl. work graph, repo, store, digest,
continuity, root exports
```

`work_graph_content_digest.js` ships without a `.d.ts`, unlike the other four
work-graph modules. It is an internal module consumed by
`work_graph_continuity.js` and covered by its own Node test, and the Python
counterpart is likewise internal (used by `work_graph_cli.py` and
`work_graph_mcp.py`), so this is a typing gap, not a delivery gap.

## 4. Stage H — duplicate semantics audit

### Finding H1 (corrected) — one live copy, one dead copy

An earlier revision of this document called these "hand-maintained clones with
drift risk". That was wrong, and the correction matters: `mod cognitive_bus;`
and `mod nkbe;` in `entroly-wasm/src/lib.rs` are the *only* references to those
files in that crate. Neither carries a single `#[wasm_bindgen]`, neither is
exported to npm. They were 1,432 lines of unreachable code, while the live
implementations sat in the PyO3 crate. npm never had the capability at all, so
there was no parity to lose -- only dead weight and a second place to change.

Both are now `entroly-engine::cognitive_bus` and `entroly-engine::nkbe`, with
164- and 76-line bindings over them. Original measurements follow.

### Finding H1a — the original measurement

```text
entroly-core/src/cognitive_bus.rs   966 lines   46 fns
entroly-wasm/src/cognitive_bus.rs   946 lines   46 fns
entroly-core/src/nkbe.rs            506 lines   20 fns
entroly-wasm/src/nkbe.rs            486 lines   20 fns
```

Neither pair imports `entroly_engine`. Function-name comparison shows **42
shared names and zero unique to either side** for `cognitive_bus` (20 and 0 for
`nkbe`) — identical public surfaces with roughly 20 lines of body divergence
each.

This is the failure the handoff §14 names: *"If Python and npm can disagree
without a Rust compile/test failing, ask whether the semantic rule belongs in
Rust."* Today they can. About 1,470 lines of parallel semantics, no shared
owner, no compile-time coupling.

### Finding H2 — `qccr` is structured correctly (control case)

`entroly-core/src/qccr.rs` is 47 lines / 3 fns and delegates through
`use entroly_qccr` (3 occurrences) to the 1,725-line SSOT crate. This is what H1
should look like, and it proves the pattern is achievable here.

### Finding H3 — `entroly-core` is not a thin binding layer

At 22,843 lines it carries modules absent from `entroly-engine`: `archetype`,
`cogops`, `compliance`, `compress`, `context_receipts`, `elc_native`, `nkbe`,
`pollination`, `proxy`, `telemetry`, `witness`. Semantic code lives in the
binding crate, against the stated "one semantic implementation, thin surfaces"
architecture.

## 5. Python architecture findings (partial semantic closure: selection path)

### Finding P1 — competing composition roots

`EntrolyEngine.__init__` leaves three subsystems deliberately unwired for the
caller to complete:

```text
set_fast_path_router          server.py:5406        (inside create_mcp_server)
set_journal_callback          server.py:3226 · daemon.py:530
set_crystallization_callback  server.py:5391, 5414  (inside create_mcp_server)
```

`create_mcp_server` — a **3,509-line function** — is the application's real
composition root. Meanwhile `EntrolyEngine(...)` is constructed in **20 places
across 9 modules** (`cli.py` alone 11 times), none of which complete the wiring.

Measured consequence: capability is determined by entry surface.

| capability | assembled in | reaches |
|---|---|---|
| no-match contract | `create_mcp_server` | MCP only |
| fast path (skill replay) | `set_fast_path_router` in `create_mcp_server` | MCP only |
| session rescue | `proxy.py` | proxy only |

This is one architectural fault, not three defects. It contradicts the target
architecture's "many thin delivery surfaces" goal *inside Python*, before any
Rust migration begins.

### Finding P2 — `server.py` holds two responsibilities

```text
6,748 lines total
2,023  class EntrolyEngine        49 methods, largest optimize_context (419)
3,509  def   create_mcp_server    76 nested functions, 2,982 lines
                                  largest nested optimize_context (540)
```

Two functions named `optimize_context` — an engine method and an MCP handler —
confirmed by `ast` scope resolution and `inspect.getsourcelines`. The 76 handlers
being nested leaves them with no importable identity, which is why
`tests/test_context_receipts.py` must assert tool existence through
`inspect.getsource` string matching.

### Finding P3 — a library constructor disables the garbage collector

`EntrolyEngine.__init__` ends with `gc.collect(); gc.freeze(); gc.disable()`.
Defensible in a long-lived server; a process-wide side effect for the 20 call
sites and for any SDK user who constructs an engine.

## 6. Stage G — baseline at the exact head

### Finding G1 (fixed during this audit) — the suite could not be collected on Python 3.10

```text
tests/test_work_graph_entrypoints.py:3  import tomllib
tests/test_work_graph_packaging.py:3    import tomllib
E  ModuleNotFoundError: No module named 'tomllib'
!!! Interrupted: 2 errors during collection !!!   (exit code 2, 0 tests run)
```

`tomllib` is standard library from Python **3.11**; both manifests declare
`requires-python = ">=3.10"`, and `Build Wheel + Functional Test (Python 3.10)`
is a required check that runs `pytest tests/ -x`. This was not two failing
tests — **no test in the repository executed** on the oldest supported
interpreter.

The repository had already solved this twice: `scripts/codebase_graph.py` guards
the import, and `tests/test_release_surface.py` carried a local parser whose
docstring states that Python 3.10 does not ship `tomllib`.

Fix applied here follows that precedent instead of adding a third copy: the
parser moved to `tests/pyproject_compat.py`, taught to read `[project.scripts]`,
and is now shared by `test_release_surface.py` and both Work Graph packaging
tests. One implementation, three call sites.

```text
before   Interrupted: 2 errors during collection      0 tests
after    4,033 tests collected
```

### Baseline results

```text
command   python -m pytest tests/ -v --timeout=300 -p no:randomly
SHA       09a98a5b + the fixes in this change (audit worktree)
result    3,986 passed, 44 skipped, 3 xfailed, 0 failed  (19m34s)
```

For contrast, the same suite at the unmodified head could not run at all:
`Interrupted: 2 errors during collection`, exit code 2, zero tests executed.

`tests/test_mcp_wire_budget.py` carries 2 pre-existing ruff E402 errors,
untouched by this audit and distinguishable from any regression.

```text
cargo test (entroly-engine)      443 passed, 0 failed
npm test  (entroly-wasm)         6 of 7 suites pass
  remaining: test_work_graph_repo -- POSIX-only harness, see G5
```

### Finding G2 (fixed) — the npm package could not be loaded at all

`entroly-wasm/js/work_graph.js` carried two malformed statements:

```js
return fromJSONText(this._inner.snapshotJSON(Boolean(pretty));    // missing )
return fromJSONText(this._inner.unfinishedJSON(Boolean(pretty));  // missing )
```

`SyntaxError: missing ) after argument list`. The module is listed in
`package.json` `files[]` and is required by `index.js:79`, so
`require('entroly-wasm')` failed outright — the published npm surface was
non-functional. Fixed; `node --check` now passes on all 27 shipped JS files.

### Finding G3 (fixed) — the Python content-digest contract was never verified

`tests/test_work_graph_content_digest.py` declared
`def _observation(*changes: dict)` while all ten call sites pass a single change
as keywords. Every test raised `TypeError: _observation() got an unexpected
keyword argument 'path'` before reaching an assertion. Seven tests, zero
executed assertions. Helper corrected to `**change`; 7 passed.

### Finding G4 (fixed) — content identity was silently dead on Windows

`entroly-wasm/js/work_graph_content_digest.js` guarded against symlink swaps
with `sameFile()`, requiring `lstat.dev === fstat.dev`. Measured on win32:

```text
lstat    dev=0            ino=1688849864894978
fstat    dev=576968922    ino=1688849864894978
```

Node reports dev=0 from lstat and a real device id from fstat for the same file,
so `sameFile()` returned false for every file and `gitBlobDigest` always
returned an empty string. Node users on Windows received **no worktree content
identity** — the binding handoff and resume depend on (handoff section 3.7) —
with no error surfaced.

Fixed by comparing dev only when both sides report one, while still requiring
ino, size, mtimeNs and ctimeNs to match across lstat, open and lstat, so the
anti-swap property is preserved on POSIX.

### Finding G5 (not a defect) — one npm test is POSIX-only

`test_work_graph_repo.js:129` sets `process.env.HOME` and expects the default
checkpoint path to follow. Both runtimes resolve the home directory through the
platform (`os.homedir()` in Node, `Path.home()` in Python), and on Windows both
read `USERPROFILE` and ignore `HOME`. Measured: `os.homedir()` is unchanged
after setting HOME, and equals `USERPROFILE`.

Behaviour is consistent across runtimes; the test harness encodes a POSIX
assumption. Expected to pass on Linux CI. Recorded as an environment limitation,
not a code defect, and deliberately not "fixed" in the product.

## 7. What this audit did NOT do — open gate gaps

Stated precisely, per the gate's non-negotiable honesty rule.

```text
Full-text reads (production)        ~18 of 298 python-production files
Rust source read                    work_graph.rs public surface + apply_event,
                                    from_json, struct fields read; 0 of the other
                                    64 files read in full
Node/JS source read                 work_graph.js and work_graph_content_digest.js
                                    read; 0 of the other 39 read in full
Semantic closure (Work Graph)       PARTIAL -- 7,132-line closure sized and mapped,
                                    integrity/import path and both binding layers
                                    read; Rust materialization internals not read
Public user journeys traced         5 (MCP init, CLI optimize, pip install,
                                    entroly-work CLI, entroly-work-graph-mcp)
Persisted schema review             Work Graph DONE (see section 9);
                                    checkpoints/receipts/recovery-store/cache keys
                                    NOT DONE
Cross-runtime parity verification   NOT DONE (H1 shows it is unenforced)
Rust and WASM baselines             DONE (443 passed; 6/7 npm suites)
entroly-core/tests baseline         DONE (2 passed, 96 assertions behind them)
```

Files read in full during this audit: the four PR352 documents,
`entroly/self_heal.py`, `entroly/qccr.py`, `entroly/runtime_capabilities.py`,
`entroly/repository_intelligence/write_authority.py`,
`tests/test_no_match_honesty.py`, `tests/test_release_surface.py`,
`tests/test_work_graph_entrypoints.py`, `tests/test_work_graph_packaging.py`,
plus the `EntrolyEngine` constructor and `optimize_context`.

**Per gate §18, these merge-blocking conditions are currently true:**

1. No changed subsystem has a completed semantic-closure full read.
2. Public entrypoint traces are incomplete (3 partial of 5 or more journeys).
3. Python/npm semantic parity is unknown, and H1 shows it is unenforced.
4. A shared semantic duplicate (H1) remains unexplained.
5. Persisted schema compatibility is unreviewed.
6. Rust and WASM baselines were not run at this head.

## 8. Standing production risk carried by this branch

`entroly-core>=1.0.78,<2` is a hard dependency here. PyPI publishes **no sdist**
and **no musllinux wheel** for `entroly-core` 1.0.78 — only macOS universal2,
manylinux x86_64 and aarch64, and win_amd64:

```text
sdist: 0
entroly_core-1.0.78-cp310-abi3-macosx_10_12_x86_64...universal2.whl
entroly_core-1.0.78-cp310-abi3-manylinux_2_17_x86_64.whl
entroly_core-1.0.78-cp310-abi3-manylinux_2_28_aarch64.whl
entroly_core-1.0.78-cp310-abi3-win_amd64.whl
```

Merging is safe; **publishing is not**. On release, `pip install entroly` would
hard-fail on musl/Alpine, PyPy, i686 and old glibc, where it previously degraded
to the pure-Python fallback. The musl jobs added to `publish-core-wheels.yml`
take effect only on the next `entroly-core` publish. Handoff §16 classifies this
as a production failure, not a documentation detail.

`docs/DETAILS.md` currently states wheels are published for "macOS universal2,
Linux glibc and musl". That is not true at this SHA and must be corrected, or the
release ordering changed, before publishing.


## 9. Stage I — persisted schema review (Work Graph)

`WorkGraph::from_json` was read in full and enforces, in order:

1. exact `schema_version == WORK_GRAPH_SCHEMA_VERSION` (1); anything else is
   rejected rather than reinterpreted;
2. `MAX_EVENTS` bound checked **before** the events are adopted;
3. every event revalidated;
4. every `event_id` recomputed and compared -- per-event tamper detection;
5. duplicate event ids rejected;
6. full `rebuild()`, then `graph_commitment` recomputed and compared --
   whole-graph tamper detection.

All six fail closed. This substantiates handoff section 3.3.8 ("persisted state
is integrity checked on import") as verified rather than claimed.

**Observation, not a defect:** there is no migration path. Strict equality means
a v2 document cannot be read by v1 code and vice versa. Correct while v1 is the
only version, but gate section 12 requires a deliberate migration before any
field meaning changes. The repository already knows the pattern --
`LEGACY_SCHEMA_VERSION` and `LEGACY_BATCH_SCHEMA_VERSION` exist elsewhere.

## 10. Performance finding — event apply is quadratic

`WorkGraph` stores `events: Vec<WorkEvent>` with no id index, and `apply_event`
deduplicates with a linear scan:

```rust
if self.events.iter().any(|existing| existing.event_id == id) { return Ok(id); }
```

Appending N events is therefore O(N^2) string comparisons against
`MAX_EVENTS = 50_000`. Handoff section 15 names "append/event apply latency" and
"event amplification from passive polling" as key production concerns.

`from_json` already builds a `BTreeSet` of seen ids for exactly this purpose
(work_graph.rs:862), so the structure exists but is not retained as state.
Carrying that set on the struct would make append O(log N) without changing any
semantics or persisted format.

## 11. Additional defects found and fixed

### Finding G6 (fixed) — `work_graph.py` bypassed the shared native gate

`entroly/work_graph.py` used a bare `from entroly_core import WorkGraph`,
tripping `test_native_core_gating::test_ungated_native_importers_do_not_grow`.
The gate exists to stop one half of a process accepting a core that the other
half refuses -- the mixed state that once surfaced as
`ContextFragment.__new__() got an unexpected keyword argument 'recency_score'`.

Routed through `native_status`. Deliberately **no** pure-Python fallback was
added: Work Graph semantics are Rust-owned, and a Python re-implementation would
create a second source of truth for status inference and commitments
(handoff section 17.1). Missing or stale core still fails closed.

### Finding G7 (fixed) — the handoff documented commands that do not exist

`test_docs_code_sync::test_documented_cli_subcommands_exist` asserts that every
`entroly <subcommand>` shown in a code block is real. The handoff proposed
`entroly work state`, `entroly work claim` and four more in a fenced block; the
CLI ships `state`, `claim`, `resume`, `handoff` with no `work` namespace. The
proposal is now written without the `entroly` prefix and labelled as proposed,
so a design note stops reading as a shipped command. The guard was not weakened.

### Finding G8 (gating fixed; delivery still blocked) — the published core has no Work Graph

```text
installed entroly_core version : 1.0.78
has WorkGraph symbol           : False
work/graph-ish exports         : <none>
native_status().ok             : True
```

The published `entroly-core` 1.0.78 predates the Work Graph bindings, yet
satisfies both the `>=1.0.78` pin and `MIN_ENTROLY_CORE_VERSION`. `pip install
entroly` therefore resolves to a core that reports a healthy engine and cannot
run the Work Graph at all. Both new console scripts -- `entroly-work` and
`entroly-work-graph-mcp` -- dead-end at `native_work_graph_unavailable` for
every user installing from PyPI today.

This is the same silent capability loss that QCCR gating exists to prevent, on a
new surface. Fixed at the gate: `WORK_GRAPH_SYMBOLS = ("WorkGraph",)` added to
`native_status`, and `work_graph.py` now resolves through
`native_status(WORK_GRAPH_SYMBOLS)`, so the failure is explicit and the message
is accurate ("found the Rust engine but required symbols are missing" rather
than the previous, false "incompatible engine; install entroly-core>=1.0.78"
when 1.0.78 was already installed).

**The delivery gap itself remains open and is merge-relevant:** shipping Work
Graph requires publishing an `entroly-core` that exports `WorkGraph` and raising
the pin to that version. Until then the feature is undeliverable, exactly as
gate section 13 warns -- "a semantic implementation with no delivered binding is
incomplete."


## 12. Stage H, continued — a third graph of the same repository

Reading `entroly-engine/src/depgraph.rs` (1,591 lines) for the increment-4
closure turned up a graph nobody had counted. The repository is now modelled
three times, in three identity schemes:

| Graph | Identity | Vocabulary |
|---|---|---|
| `engine::work_graph` | `stable_node_id(kind, repo_id, key)` | `NodeKind::{Repository, File, Symbol, …}`, edges `contains/defines/imports/depends_on` |
| `engine::depgraph` | `fragment_id` | `DepType::{Import, FunctionCall, TypeReference, SameModule, TestOf, CrossLanguageFFI}` |
| `repository_intelligence` | `path::qualified::kind` | `Symbol`, `CallEdge`, `FileRecord`, `RepositoryIndex` |

`DepGraph` keeps `outgoing`/`incoming` dependency lists, a `symbol_table`
mapping symbol name to the fragment that defines it, and `cross_lang_exports`
recording PyO3/WASM/JNI/CGo boundaries. It already answers the questions
section 4.1 asks -- transitive and reverse dependencies, connected components,
symbol definitions -- but keyed to fragments, which are a selection-time
concept, not to repository artifacts.

**Consequence for increment 4.** Migrating `repository_intelligence` semantics
into Rust is not "port 2,900 lines into an empty module". It is a reconciliation
between two existing Rust graphs and one Python graph, and the identity question
has to be settled first: does a File node in the work graph and a fragment in
the dep graph denote the same thing, and if so which id wins? That is an
architectural decision for the Work Graph owners, not something an audit should
choose unilaterally, and it is why this increment stops at the identity join and
the projection rather than moving code.

The join built in this branch (`graph_identity`, `graph_projection`) connects
the first and third. The second remains unjoined and is the open question.

## 13. Increment status against master prompt Phase 1

```text
1. cognitive_bus -> engine        DONE      1,912 -> 951 engine + 164 binding
2. nkbe -> engine                 DONE        992 -> 486 engine +  76 binding
3. canonical node identity        DONE      exposed to Python and Node, hash-verified
4. repository intelligence        PARTIAL   identity join + bounded projection done;
                                            semantic migration blocked on the
                                            reconciliation in section 12
5. server.py engine split         DONE      6,748 -> 4,238 + 2,942, full suite clean
                                            apart from two of this branch's own
                                            wiring tests, since fixed
```

Semantic closure coverage, honestly:

```text
work_graph.rs   public surface, apply_event, from_json, coordination_report,
                paths_overlap, validate_event_references_and_capacity, identity
                helpers -- read
depgraph.rs     public surface and data model -- read; internals not
the other 63 .rs files                        -- not read in full
```

One invariant found by that reading and worth stating on its own, because it is
the mechanism behind "completed means completed plus verified":

```rust
if node.status != WorkStatus::Unknown && node.status_trust > strongest {
    return Err(... "status trust exceeds supporting evidence trust")
}
```

A node cannot assert a status at a trust level higher than the evidence
supporting it. The rule is enforced in code, not merely documented.


## 14. Stage D continued — semantic closure read, and what it found

Coverage at the time of writing, stated exactly rather than rounded up:

```text
read in full
  entroly-engine/src/fragment.rs        299
  entroly-engine/src/query.rs           495
  entroly-engine/src/cognitive_bus.rs   951   (read as part of moving it)
  entroly-engine/src/nkbe.rs            486   (read as part of moving it)
  entroly-core/src/cognitive_bus.rs     133   (rewritten)
  entroly-core/src/nkbe.rs               76   (rewritten)
  entroly/engine.py                   2,942   (relocated)

read substantially
  entroly-engine/src/work_graph.rs    3,825   public surface, apply_event,
                                              from_json, coordination_report,
                                              paths_overlap, the referential
                                              validator, identity helpers
  entroly-engine/src/depgraph.rs      1,591   public surface and data model
  entroly-engine/src/entropy.rs       1,154   through the NCD section
  entroly-engine/src/dedup.rs           484   through the LSH index
  entroly-engine/src/utilization.rs     257   scoring path

not read in full: the remaining ~58 .rs files, including cache.rs (3,689) and
skeleton.rs (2,906)
```

### Finding D1 — documentation drift in the semantic source of truth

Four gaps between what a module says and what it does, all found by reading and
none by a failing test:

| module | documented | actual |
|---|---|---|
| `query.rs` | `− specificity_bonus × 0.2` | `* 0.7` -- a factor of 3.5 in how strongly a technical term pulls a query out of refinement, and `needs_refinement` is a threshold on it |
| `query.rs` | `{:.0}%%` intended as one percent | Rust does not escape `%`; users were told "50%% of tokens are generic verbs" |
| `entropy.rs` | "ratio 0.80 → score 1.0" | at 0.80 the score is 0.8235; 1.0 is reached at 0.95. The adjacent comment giving the 0.10–0.95 range was the correct one |
| `utilization.rs` | "Trigram Jaccard" in three places | containment (intersection over fragment trigrams). Opposite behaviour as responses grow, and the score feeds weight learning |

Each was corrected in the comment, never in the computation: the code is the
behaviour, and changing it to match a comment would be the opposite of an audit.

This matters beyond the four fixes. `entroly-engine` is the single semantic
source of truth for the Python and Node runtimes, and the product claim is
auditability. Descriptions drifting from implementations in the crate that
defines the meaning is a systemic finding, not four typos.

### Finding D2 — the strongest code in the crate, for contrast

`dedup.rs` is the counter-example and worth recording so this section is not
read as uniformly negative. `simhash_cosine` derives similarity as
`cos(pi * hamming / 64)` rather than the intuitive `1 - hamming/64`, and
documents why with a measurement: over 1.1M real fragment pairs the linear form
has MAE 0.502 against exact cosine, this form 0.080. `simhash_cosine_lcb` then
applies a union bound across comparisons to correct the optimizer's curse, with
the inflation measured (k=1: 0.078, k=256: 0.525, true value 0.062) and a
correct argument for why clamping a *bound* at zero is sound while clamping an
*estimate* injects +0.079 bias.

That is what a semantic source of truth should look like: the claim, the
alternative rejected, and the number that settles it.

### Finding D3 — a documented conflation, already fixed upstream

`fragment.rs` records that criticality once set `is_pinned`, which forces
inclusion in every selection, when it meant `is_protected`, which only prevents
eviction. A manifest or security file was therefore force-included in every
query regardless of relevance. The fields are now separate and
`tests/test_pin_protection_split.py` guards the split.

Worth carrying forward because it is the same shape as two defects this branch
found independently: one name serving two meanings (`optimize_context` as both
engine method and MCP handler) and one value serving two states (`_evidence_backed`
returning False for both "scored zero" and "never scored").


## 15. Finding D4 — the crate's own charter names the failure this branch closed

`entroly-engine/src/lib.rs` explains why the crate exists:

> `entroly-core` and `entroly-wasm` previously carried *copies* of these 32
> modules. Nothing linked them, so nothing failed when one was fixed and the
> other was not, and they drifted by 4,065 lines — including a similarity
> estimator that was corrected in the core while the WebAssembly build kept
> shipping the broken form.
>
> With the algorithms here, that failure mode is unrepresentable: a change
> reaches every distribution channel or it does not compile.

Two things follow.

**The estimator it refers to is `simhash_cosine`.** `dedup.rs` documents the
correction in detail: the intuitive `1 - hamming/64` is linear in the angle
rather than its cosine, and reports two unrelated fragments -- which sit near
orthogonal at `hamming ≈ 32` -- as 0.5 similar instead of 0. Measured over 1.1M
real pairs, MAE 0.502 against exact cosine versus 0.080 for the correct form.
npm shipped the wrong one while Python shipped the right one. That is the
concrete cost of the duplication, in the product's core ranking signal.

**The claim was still false when this branch started.** At
`09a98a5b`, `entroly-engine/src/lib.rs` declared 31 `pub mod` and neither
`cognitive_bus` nor `nkbe` was among them. Both still lived in the binding
crates -- one live copy in `entroly-core`, one dead copy in `entroly-wasm` --
so for those two modules a change did *not* have to reach every channel, and
nothing would have failed to compile if they diverged. The count is 33 now.

This reframes findings H1 and the two consolidation commits. They are not
general tidying: they close the last two instances of the exact failure mode
`entroly-engine` was created to eliminate, in a crate whose own documentation
already declared that mode impossible.

### Observation — `coordination_index` is test-only

`lib.rs:68` declares `#[cfg(test)] mod coordination_index;`. The module is 308
lines and does not ship. Worth knowing before anything is built on it.

### Finding G9 — repository intelligence already has graphs on `main`, and one of them collides with the Work Graph namespace

The question "isn't repository intelligence already projected into a graph?"
has a real basis: `origin/main` carries seven graph-shaped modules under
`entroly/repository_intelligence/` — `graph.py`, `graph_query.py`,
`program_graph.py`, `adaptive_program_graph.py`, `semantic_ir.py`,
`interprocedural_flow.py`, `universal_flow.py`.

They are graphs. None of them is *the* Work Graph, and they do not share an
identity scheme with it or with each other. Counted on `origin/main`:

| Module | Node id it mints |
|---|---|
| `graph_query.py` | `file:{path}` / `symbol:{symbol_id}` |
| `program_graph.py` | `synthetic:{label}` |
| `semantic_ir.py` | `_node_id(path, kind, name, start, end)` |
| `interprocedural_flow.py` | `flow:{sha256(...)}` |
| `syntax_session.py` | `_shape_id(file, kind, node_type, start, end)` |
| `universal_flow.py` | its own `node_id` field |
| `entroly-engine::work_graph` | `{token}:{sha256("node\|{token}\|{repo}\|{key}")[:24]}` |

Six independent id spaces for one repository, which is the condition section 4.1
exists to end.

The specific hazard is `graph_query.py`. It mints `file:` and `symbol:` — the
**same two namespace tokens** `stable_node_id` emits — with entirely different
content after the colon:

```
repo-intel : file:src/app.py
work graph : file:5ad0da59f0c4533433cded64
```

`_node_path` dispatches on that prefix and slices it off positionally
(`node_id[5:]`, `node_id[7:]`), then looks the remainder up as a path:

```python
if node_id.startswith("file:"):
    path = node_id[5:]
    return path if path in index.files else None
```

Feed it a genuine Work Graph node id and it does not raise. It slices to
`5ad0da59f0c4533433cd`, fails the `in index.files` test, and returns `None` —
**reporting a node that exists as absent**. That is the fabricated-completeness
failure mode the handoff forbids, reached without anyone writing a wrong answer:
two id spaces that are indistinguishable to a reader, to `grep`, and to a prefix
dispatch, and that silently disagree at lookup.

So the answer is not "the projection is missing." It is that a projection exists,
a second one exists in Rust, and the two are namespace-compatible and
value-incompatible. `graph_identity.py` and `graph_projection.py` (added on this
branch) supply the single derived identity; `tests/test_repository_graph_identity.py`
pins that Python never recomputes the hash. This finding adds the guard for the
collision itself.

### Finding G10 — the SimHash similarity bug the crate documents as fixed is still live in four engine modules

`dedup.rs` defines `simhash_cosine` and states the case against the linear form
explicitly: two unrelated fragments are near-orthogonal, so `hamming ≈ BITS/2`,
which `1 - hamming/BITS` reports as **0.5 similar** instead of 0; measured over
1.1M real fragment pairs the linear form has MAE 0.502 against exact cosine
versus 0.080 for `cos(π·hamming/64)`. `entroly-wasm/src/lib.rs:680` carries the
same reasoning in a comment and calls `simhash_cosine`. `cache.rs`,
`knapsack_sds.rs`, `entroly-core/src/lib.rs` and `simhash_probe.rs` all call it.

Four modules in `entroly-engine` still compute the linear form inline:

| Site | Consumer | Effect at `hamming ≈ 32` (unrelated) |
|---|---|---|
| `trajectory.rs:43` | `classify_query_transition` | scores 0.5; with `topic_change_threshold = 0.30` an unrelated query is classified **`ambiguous`, never `topic_change`** |
| `cognitive_bus.rs:318` | subscriber novelty | `max_similarity` floors near 0.5, so `raw_novelty` caps near 0.5 — a genuinely novel event reads as half-novel |
| `lsh.rs:216` | `score()` similarity term | the weighted term contributes `w_similarity · 0.5` for unrelated candidates |
| `channel.rs:481` | SDR contradiction detection | `content_sim` floors near 0.5, so `sdr = structural_sim - content_sim` is depressed by ~0.5 against `sdr_threshold` |

`trajectory.rs` is the clearest: it already imports `hamming_distance` from
`crate::dedup` — the fix is one import away — and its documented purpose is to
tell a rephrase from a topic change. Under the linear form the `topic_change`
branch is reachable only for `hamming > 44.8` (similarity < 0.30), i.e. the
*first* 70% of the orthogonality range is misreported as ambiguity.

Not changed here. All four are behaviour, and three of them
(`lsh.rs`, `channel.rs`, `trajectory.rs`) carry thresholds — `0.75`, `0.30`,
`sdr_threshold`, `w_similarity` — that were tuned against the linear scale.
Substituting `simhash_cosine` without recalibrating those constants would move
every decision boundary at once, which is a change to make deliberately with the
thresholds re-derived, not as a drive-by inside an audit. Recorded as the
highest-value correctness item found by the closure read.

### Finding G11 — LSH multi-probe covers 3 of 10 neighbours, chosen by index rather than by probability

`lsh.rs` builds 10-bit keys (`BITS_PER_KEY = 10`) and probes with:

```rust
for flip in 0..MULTI_PROBE_DEPTH.min(BITS_PER_KEY) {   // 0..3
    let neighbor = key ^ (1u16 << flip);
```

`flip` takes 0, 1, 2 — so only key bits 0–2 are ever flipped. Bits 3–9 are never
probed, and since `bit_positions` is sorted ascending, key bit *i* is simply the
*i*-th lowest sampled fingerprint bit. Nothing makes the three lowest more likely
to have flipped than the other seven.

That is not what multi-probe LSH is. The technique (Lv et al., VLDB 2007) derives
a *probe sequence* ordered by the probability each neighbouring bucket contains a
true near neighbour — typically from how close each projection landed to its
quantisation boundary. Probing a fixed low-index prefix instead gives 3 of the 10
single-bit neighbours selected arbitrarily, so recall is systematically biased:
two fragments differing only in bits mapped to key positions 3–9 fall in
different buckets in every one of the 12 tables and are never returned as
candidates.

The effect is silent — `query` returns a shorter candidate list, and the caller
cannot distinguish "no near neighbour exists" from "the near neighbour hashed to
an unprobed bucket."

### Finding G12 — the LSH test that would catch G11 asserts nothing about the neighbour

`test_similar_fingerprints_found` inserts `fp1` and `fp2 = fp1 ^ 0x7` (3 bits
apart), queries `fp1`, and asserts only:

```rust
assert!(candidates.contains(&0));
// fp2 is extremely close, multi-probe should almost always find it
```

The claim about `fp2` is a comment, not an assertion — index `1` is never
checked. `candidates.contains(&0)` passes on the exact-bucket hit alone and would
still pass with `MULTI_PROBE_DEPTH = 0`. So the multi-probe path, the entire
reason the module exists, has no test that can fail.

Not changed here: fixing G11 changes recall, and this test is how the change
would be measured. Both belong in one deliberate commit that sets the probe
sequence and then asserts the neighbour is actually returned.

### Observation — `LshIndex::remove` leaks empty buckets

`DedupIndex::remove` in `dedup.rs` drops a bucket once its vector empties;
`LshIndex::remove` retains it. Unbounded only in the number of distinct keys
(≤ 1024 per table × 12 tables), so it is bounded and small — noted for symmetry,
not as a defect. The method is `#[allow(dead_code)]` and currently unreachable.

### Finding G13 (severe; fixed) — strict-mode hallucination suppression destroyed code indentation in every response

`eicv_suppressor.rs::apply_strict` ends with:

```rust
// Clean up double spaces / leading spaces from suppressions
while result.contains("  ") {
    result = result.replace("  ", " ");
}
let result = result.trim().to_string();
```

The intent is stated in the comment: tidy the gaps left behind when a
hallucinated sentence is cut out. What it actually does is collapse *every* run
of two or more spaces anywhere in the response, including indentation inside
code blocks. Compiled and run against the real function body:

```
--- BEFORE ---                      --- AFTER ---
fn handler() {                      fn handler() {
    let x = parse()?;                let x = parse()?;
    if x > 0 {                       if x > 0 {
        emit(x);                     emit(x);
    }                                }
}                                   }
```

Every nesting level flattens to one space. For Python output this does not merely
look wrong — it produces a syntactically invalid program, and the tool that
produced it is the one whose stated purpose is handling code context.

The control flow makes it worse. `apply_strict` returns early only when
`claims.is_empty()`. With at least one claim — any sentence of four or more
words — the replacement loop may run zero times while the collapse still
executes unconditionally. So a **fully grounded response with nothing suppressed
at all** still comes back with its indentation destroyed, and `suppress()`, the
convenience entry point, defaults to exactly this mode (`"rag"`, `"strict"`).

Fixed. The cleanup now travels with the removal instead of sweeping the
document: a suppressed claim consumes the horizontal whitespace that trailed it,
so no gap is created and no global pass is needed. Newlines are deliberately not
consumed -- they carry the paragraph and code-block structure. The extended span
can reach at most the first following non-space byte, which is where the next
claim begins, so spans stay disjoint and the existing reverse-order application
remains offset-safe.

Three tests pin it, and two of them were confirmed to fail against the previous
code before the fix was kept:

| Test | Against old code |
|---|---|
| `strict_mode_preserves_code_indentation` | FAILED -- both nesting levels lost |
| `strict_mode_leaves_a_fully_grounded_response_alone` | FAILED -- rewritten with zero suppressions |
| `suppressing_a_claim_consumes_its_own_trailing_space` | passed -- pins the property the collapse also achieved, destructively |

The first test grounds the code block in the context on purpose. Left ungrounded
it is a single unsupported claim, strict mode deletes it wholesale, and the test
would have passed for the wrong reason -- which is what the first draft did until
the failure output showed the whole block missing.

`cargo test --lib`: 467 passed. `cargo clippy --lib`: clean.

### Finding G14 — `annotate` mode does not list what it documents as listing

The module table says `annotate` will "Append warning footer **listing**
unverified claims". `apply_annotate` takes `_certs` — underscore-prefixed,
unused — and emits only a count:

```
[EICV Warning: 3 claims could not be verified against provided context]
```

A reader of the response cannot tell *which* three. The certificates carrying
that detail are computed, returned in `SuppressionResult.certificates`, and
discarded by the annotator. Documentation overclaim, not a logic error.

### Observation — `softcap`'s documented fallback is narrower than its real one

`fragment.rs` documents "When `cap ≤ 0`, falls back to `min(x, 1)`", but the
guard is `if cap <= 0.0 || cap >= 10.0`. The upper cutoff is undocumented and
untested — `test_softcap_properties` covers `cap = 0.0` only.

### Observation — `query.rs` is clean, and says so honestly

`compute_vagueness` carries a comment recording that its specificity coefficient
is 0.7 rather than the 0.2 an earlier comment claimed, and that the code was left
alone because the code is the behaviour. That is the correct disposition for a
documentation/behaviour mismatch found by reading, and it is the pattern the rest
of these findings follow.

### Finding G15 (fixed) — the one suite failure was the test, not the engine

Running the full suite against a locally built engine rather than the published
core turned 33 skips into real executions and produced exactly one failure:

```
FAILED tests/test_work_graph_multiprocess.py::
       test_concurrent_agent_processes_merge_without_lost_work - assert 0 == 1
```

The assertion is `len(report["conflicts"]) == 1`. Zero conflicts materialized
because the two agents claimed scopes that do not overlap:

| Agent | Scope claimed | Exists in fixture |
|---|---|---|
| claude | `src/auth` | no -- fixture created `src/auth.py` only |
| codex | `src/auth.py` | yes |

`paths_overlap` treats scopes as overlapping when they are equal or one is a
parent of the other **at a `/` boundary**. `src/auth` is therefore not a parent
of `src/auth.py`, and the engine is right: they are different paths, and a bare
string-prefix test would also collide `src/auth` with `src/authorization.py`.
The engine's own Rust test uses `src/auth/token.rs`, the case that does overlap.

So the defect was the fixture asserting a conflict it never set up. Fixed by
making `src/auth` a real package and narrowing the second claim to a file inside
it -- claude takes `src/auth`, codex takes `src/auth/tokens.py` -- which is a
genuine containment overlap and keeps the auth-implementation vs auth-tests
story the test is written around. The comment left in place records why, so the
next reader does not "fix" it back toward a substring match.

This is the failure mode the whole exercise is aimed at: the test could not fail
informatively while it was skipping, and it had been skipping because the
published core has no `WorkGraph` (G8).

### Suite status at this SHA

With the engine built from this tree and installed to a scratch prefix:

```
4025 passed, 1 failed, 33 skipped, 3 xfailed   (32m09s)
```

after which the single failure above was fixed. `cargo test --lib`: 467 passed
in `entroly-engine`, 112 in `entroly-core`. `cargo clippy --lib`: clean.

Against the *published* core the same suite cannot exercise the Work Graph at
all -- those tests skip rather than run, which is why this run is the first
evidence that the surface works end to end.

---

## 16. Master prompt section 22 — "every file is mapped", answered

Section 22 poses twelve questions and requires a reviewed report rather than an
assertion. Answers below are derived from the branch, not recalled.

**Method.** Engine modules are the `pub mod` declarations in
`entroly-engine/src/lib.rs`. Binding exposure is every reference to a module
path from `entroly-core/src` (PyO3) and `entroly-wasm/src` (WASM), counted
separately and cross-checked per module rather than from a single grep.

### Q5/Q6 — capabilities missing from a runtime

33 public engine modules, plus `coordination_index` which is declared
`#[cfg(test)] mod` and ships nowhere. Four modules are asymmetric:

| Module | Lines | entroly-core (Python) | entroly-wasm (npm) | Used inside engine |
|---|---:|---|---|---|
| `rnr` | 81 | **absent** | present | yes — `eicv.rs` |
| `cognitive_bus` | 955 | present | **absent -> now present** | no |
| `nkbe` | 486 | present | **absent -> now present** | no |
| `simhash_wide` | 235 | **absent** | **absent** | **no** |

The remaining 29 are reachable from both bindings.

`rnr::rnr_score` is callable from npm but not from Python. Python reaches its
behaviour only indirectly, through `eicv`, which consumes it internally.

`cognitive_bus` and `nkbe` are Python-only. **These are the two modules this
session moved out of `entroly-wasm` into `entroly-engine`.** The move satisfied
section 3's architectural law — shared semantics now live once in Rust — and
simultaneously left a section 14 parity gap, because only the PyO3 binding was
written. That is a real cost of this session's own increment, recorded rather
than left for a reviewer to find.

### `simhash_wide` is dead code, and it is the fix for a limitation the crate documents

`simhash_wide` is referenced by no engine module and by neither binding. It is
235 lines carrying `FINGERPRINT_VERSION = 2`, a domain-separated MD5 byte
contract, widths of 64/256/1024 bits, a pinned golden vector, and five tests.

It is not ordinary dead code. `dedup.rs::test_lcb_power_is_limited_by_fragment_length`
documents a measured weakness — short near-duplicates cannot be penalised because
a 64-bit fingerprint over a handful of trigrams carries too little evidence — and
states the remedy explicitly:

> "that is a bit-width problem, not a bound problem. **Widening the fingerprint
> shrinks the standard error and lifts this directly.**"

`simhash_wide` is that widening. Its own test
`wide_fingerprints_separate_duplicates_from_strangers` asserts that 64-bit
populations overlap while 256- and 1024-bit populations separate, and
`noise_floor` quantifies the gain as `pi / (2*sqrt(bits))` — 0.196 at 64 bits,
0.098 at 256, 0.049 at 1024.

So the engine contains a tested implementation of the remedy for its own
documented limitation, wired to nothing. Not changed here: adopting it is a
persisted-fingerprint format change — `FINGERPRINT_VERSION`, `SUPPORTED_WIDTHS`
and the fail-closed `comparable()` rule exist precisely because stored records
must not be compared across widths — so it needs a migration path for existing
indexes, not a switch flip.

### Q1-Q4, Q7-Q12 — status

| # | Question | Answer at this SHA |
|---|---|---|
| 1 | Python files holding shared semantics that should be Rust | Not enumerated. 301 Python modules; 20 import `entroly_core` directly. No ownership matrix exists. |
| 2 | Which have already been ported | `cognitive_bus`, `nkbe` this session; work-graph identity exposed. Not a complete list. |
| 3 | Which remain intentionally Python | Documented only for `repository_intelligence/graph_identity.py` and `graph_projection.py`. |
| 4 | Node files that are equivalent orchestration | Not enumerated. |
| 5 | Capabilities missing from WASM/npm | Was `cognitive_bus`, `nkbe`; **both now bound** (finding G16). None remaining. |
| 6 | Rust modules with no Python/npm exposure | `simhash_wide` (neither), `rnr` (no Python). |
| 7 | Public imports/CLI/MCP depending on legacy paths | Not traced. |
| 8 | Tests covering each migration | Partial: work-graph identity/projection covered by 29 tests, verified against a real engine. |
| 9 | Packaging manifests include required files | Not re-verified at this SHA. `docs/DETAILS.md` still claims musl wheels are published, which is false here. |
| 10 | Generated artifacts treated as source | Not audited. |
| 11 | Repo map/docs stale after changes | `docs/repo_file_map.md` not refreshed for this session's moves. |
| 12 | Can Python and npm differ on the same observation | **Was yes, now narrowed.** `cognitive_bus` and `nkbe` are bound in both and delegate to the same engine serializers, so they cannot diverge. `rnr` remains npm-only. Parity is proven for those two, not yet for the whole surface. |

## 17. Master prompt section 24 — Definition of Done, honest status

Section 24 lists 34 conditions and says not to mark the stage production-ready
until all are true. They are not all true. This is the accounting.

**Satisfied, with evidence in this document:**

| Condition | Evidence |
|---|---|
| Evidence doc exists, reports coverage honestly | this file; coverage stated as 22/35 engine files, not rounded up |
| Repository artifacts graph-addressable, stable identity | `graph_identity.py`; 8 tests pinning the engine construction, run against a real build |
| Bounded/lazy materialization | `graph_projection.py` caps files/symbols/operations and reports drops; 9 tests |
| Rust canonical owner of shared work-graph identity | `stable_node_id_for_token` / `stable_edge_id_for_token` in the engine; Python derives, never recomputes — pinned by `test_identity_is_derived_not_reimplemented` |
| Parallel conflict detection without hard locking | `test_work_graph_multiprocess` passes after G15; real cross-process race, conflict materialized |
| Multi-process persistence tests pass | same test; atomic replace, no debris |
| Security/path/hostile-input tests pass | included in the 4025 |
| PR to main remains draft | PR #352 `isDraft: true` |

**Not satisfied, and why:**

| Condition | Status |
|---|---|
| Deep-codebase audit gate completed | **No.** 22 of 35 engine files read in full; `cache.rs` (3,689) and `skeleton.rs` (2,906) are the largest unread. Python closure not attempted. |
| Production-relevant files classified; no unknown ownership | **No.** Section 5 asks for a 13-field ownership matrix over every production file. None exists for 301 Python modules. |
| Semantic closures for every *changed* subsystem fully read | **Partial.** Changed modules were read (`cognitive_bus`, `nkbe`, `work_graph`, `eicv_suppressor`). `cache.rs` is shared and unread. |
| Public Python/CLI/MCP/provider/npm journeys traced end-to-end | **No.** Not traced this session. |
| Pre-change baseline recorded | **Partial.** 4025/1/33 recorded at this SHA; no baseline captured at the branch point, so a pre-existing failure elsewhere is not yet distinguishable. |
| Python and npm/WASM semantic parity proven | **Partial.** Was demonstrably false with four asymmetric modules. `cognitive_bus` and `nkbe` are now bound in both runtimes and delegate to identical engine serializers, pinned by tests on both sides (G16). `rnr` (npm-only) and `simhash_wide` (neither) remain. |
| Every production file classified in migration/ownership map | **No.** Same gap as above. |
| Large-repo/incremental performance measured | **No.** |
| Dogfood scenarios A-T (section 19) | **Not run** this session. Section 19 defines twenty scenarios; none were executed as a gauntlet. |
| Exact final SHA CI green | **Not checked.** |

**Conclusion.** The gate is not satisfied and this stage is not production-ready.
The largest remaining blocks are the section 5 ownership matrix, the section 19
dogfood gauntlet, and the parity gap this session's own module moves created.
Nothing in this document should be read as clearing them.

### Observation — panic audit of the engine's library paths

Every `unwrap` / `expect` / `panic!` / `unreachable!` outside `#[cfg(test)]`
across all 35 engine modules, resolved:

| Site | Verdict |
|---|---|
| `learning.rs` x6 | **Safe.** `attract`/`repel` are built from `WEIGHT_KEYS.map(...)` and every `get_mut` iterates that same const array, so the key is always present. |
| `cognitive_bus.rs:611` | **Safe.** The key was cloned out of `self.subscribers.keys()` immediately above with no intervening mutation. |
| `conversation_pruner.rs:677` | **Safe.** `sentences.last()` sits in the `else` of `if sentences.len() <= 3`, so the slice is non-empty. |
| `eicv.rs:398` | **Safe.** Guarded by `if cleaned.is_empty() { continue; }` two lines above. |
| `resonance.rs:397` | **Safe.** `max_by` over a group that is non-empty by construction; already uses `total_cmp`. |
| `sast.rs:681` | **Not code.** It is the `fix:` string of the SAST rule that flags `.unwrap()`. |
| `depgraph.rs:736` | **Latent.** `boosts.values().max_by(\|a, b\| a.partial_cmp(b).unwrap())` panics on NaN. Not reachable from current construction — every `strength` is a finite literal (0.8, 0.95, 1.0) and summing finite values cannot produce NaN — but `DependencyEdge.strength` is a `pub` field, so an external constructor can introduce one. |

The last one is worth a one-word change rather than a finding: `resonance.rs`
already uses `total_cmp` for exactly this comparison, so the total-order idiom is
already in-house and `depgraph.rs` is the outlier. Not changed here because it is
unreachable at present and this document's rule has been to separate "wrong now"
from "fragile later".

No mojibake remains anywhere in the engine after the `bm25.rs` repair, and there
are no `TODO`/`FIXME`/`HACK` markers — the `placeholder` matches are all domain
vocabulary (`skeleton.rs` body elision, `anomaly.rs` stub detection, SAST rule
text for SQL bind parameters).

### Finding G16 (introduced by this session; fixed) — the consolidation broke a public payload shape

Closing the section 14 parity gap surfaced a regression in this session's own
work.

`CognitiveBus` exposes two drain methods with **deliberately different** shapes:

```
drain()                -> id, source_agent, event_type, content,
                          timestamp, emotional_tag, salience, is_spike   (8 keys)
drain_memory_bridge()  -> content, source, salience,
                          emotional_tag, event_type                      (5 keys)
```

The bridge payload is not the full event with fields removed. It renames one:
`source_agent` becomes `source`, because it feeds `hippocampus.remember()`.
Confirmed against `164e36fa^`, the commit before the consolidation, where the
PyO3 binding built that 5-key dict explicitly.

The rewritten binding routed **both** methods through one shared `event_dict`
helper. So after the move, Python's `drain_memory_bridge()` returned the 8-key
full-event shape with `source_agent` — three keys added, one renamed away.

The binding's own module comment said:

> "The key set and ordering are unchanged from the previous implementation:
> renaming or dropping a key here is a public API break for every consumer of
> `drain()` and `drain_memory_bridge()`."

That comment was written in the same commit that violated it for one of the two
methods named in it.

Nothing caught it. No Python caller has adopted `drain_memory_bridge()` yet —
`context_bridge.py:1452` only mentions it in a comment — so the whole suite
stayed green over a broken public shape. It would have become a cross-runtime
divergence the moment the WASM binding landed, since a new binding would
naturally call the engine's canonical serializer and disagree with Python.

Fixed by deleting the shared helper and having both bindings delegate to
`entroly_engine::cognitive_bus`, which already emits the correct canonical JSON
for each method. Neither runtime now builds its own payload, so they cannot
disagree and neither can drift from the engine.

Pinned on both sides:

* `cognitive_bus::tests::drain_shapes_are_pinned_for_every_runtime` — engine
  level, asserts both key sets and that the bridge payload carries `source` and
  **not** `source_agent`.
* `tests/test_cognitive_bus_shape_parity.py` — four tests through the real PyO3
  binding, including that the two shapes are not interchangeable and that `id`,
  `timestamp` and `is_spike` stay out of the bridge payload.

The Python tests discriminate by construction: they assert the exact 5-key set
and the absence of `source_agent`, which is precisely what the broken binding
emitted. The pre-move shape was established from git history rather than by
rebuilding the broken binding.

The lesson is the one section 5 states directly — *do not port a file merely
because a Rust equivalent exists; first prove API parity, behavior parity,
serialization compatibility*. The consolidation was correct in direction and
skipped that proof for one method.

### Section 14 parity gap: closed for `cognitive_bus` and `nkbe`

`entroly-wasm/src/cognitive_bus_bindings.rs` and `nkbe_bindings.rs` now expose
both modules to npm, mirroring the PyO3 surface method for method, with the same
defaults (`memorySalienceThreshold` 50.0; NKBE 128000 / 0.1 / 1e-4 / 30 / 5 /
0.01). Divergent defaults would be a parity break neither side's tests would
notice, so they are stated explicitly in both files.

`cargo check --target wasm32-unknown-unknown` passes. `cargo clippy --lib` on
`entroly-core` is clean; `cargo test --lib` is 112 (core) and 468 (engine).

Remaining asymmetries after this change: `rnr` (npm only) and `simhash_wide`
(neither, and dead inside the engine too — see section 16).


> **Note on section 16.** The `cognitive_bus` and `nkbe` rows record the state
> that motivated the work, with the outcome marked inline. Both are now bound in
> `entroly-wasm` and delegate to the same engine serializers as PyO3; see finding
> G16. `rnr` and `simhash_wide` are unchanged.
