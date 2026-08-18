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

### Finding H1 — `cognitive_bus` and `nkbe` are hand-maintained clones

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
