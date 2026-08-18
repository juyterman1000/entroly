# PR #352 — Mandatory Deep Codebase Understanding Gate

> **Purpose:** Prevent surface-level repository reading from causing architectural regressions during the Work Graph / Context Engine / Trust Engine migration.
>
> **Rule:** No major semantic migration on PR #352 may begin until this gate is completed against the **exact current branch head**. A symbol scan, grep pass, repo summary, or reading only changed files does **not** satisfy this gate.
>
> **Required output:** create/update `docs/PR352_CODEBASE_UNDERSTANDING_EVIDENCE.md`. If that evidence file is missing, vague, stale relative to the migration head, or contains unresolved ownership in an area being changed, the migration is blocked and PR #352 is not merge-ready.

---

# 1. Why this gate exists

Entroly is already a large multi-runtime product with overlapping historical implementations, public entrypoints, packaging surfaces, provider integrations, context algorithms, trust/recovery paths, checkpoints, repository intelligence, MCP, CLI, SDK, PyO3, Rust, WASM, npm, tests, workflows, and release artifacts.

The primary migration risk is not inability to write new code. It is accidentally:

- duplicating an existing capability;
- porting only one of several implementations;
- breaking a public import or CLI/MCP route;
- changing Python behavior but not npm behavior;
- moving semantics into Rust but leaving callers on the legacy path;
- deleting a "duplicate" that still has unique behavior;
- breaking receipts/recovery/cache/provider semantics;
- missing packaging files or TypeScript declarations;
- validating against the wrong commit;
- confusing stale documentation with current code;
- assuming a file was understood because its symbols were indexed.

The solution is **measurable understanding before migration**.

---

# 2. Non-negotiable honesty rule

Never say:

> "I read the full codebase"

unless every production-relevant source/config/packaging file was actually read in full.

If using AST maps, symbol indexes, search, dependency graphs, summaries, or selective reading, state the coverage precisely, for example:

```text
Inventory: 100% of tracked files classified
AST/symbol map: 100% of supported source files
Full-text reads: 147/612 production files
Semantic closure for subsystem X: 100%
Public entrypoint traces: 9/9
```

A machine-generated map is excellent for discovery. It is **not equivalent to reading implementation details**.

---

# 3. Freeze the exact audit revision

Before audit work:

```bash
git rev-parse HEAD
git status --short
git branch --show-current
```

Record:

- exact SHA;
- branch;
- dirty/clean state;
- base SHA against `main`;
- whether merge/rebase is in progress.

All audit evidence must identify the SHA it describes.

If the branch advances materially during the audit, refresh affected evidence before coding.

---

# 4. Stage A — complete repository inventory

Build a machine-generated inventory from the actual branch, not only `docs/repo_file_map.md`.

Use tracked files as the canonical starting set.

Every tracked file must be classified into one of these categories or an equally explicit equivalent:

```text
production-semantic
production-orchestration
binding-pyo3
binding-wasm
public-api
cli
mcp
provider-proxy
repo-intelligence
persistence
security-trust
memory-checkpoint
packaging-release
workflow-ci
test
benchmark-fixture
documentation
configuration
generated
build-artifact
vendored
example
legacy-compatibility
unknown-needs-review
```

### Required invariant

```text
number of classified tracked files == number of tracked files
```

No silent remainder is allowed.

Files marked `unknown-needs-review` block migration of adjacent systems until resolved.

---

# 5. Stage B — ownership and migration matrix

For **every production-relevant file/module**, record fields equivalent to:

```text
path
language/runtime
current responsibility
semantic vs orchestration
canonical owner today
intended canonical owner
Rust engine mapping
PyO3 exposure
Python public callers
WASM exposure
Node/npm callers
CLI/MCP/provider entrypoints
tests protecting behavior
packaging/release dependency
migration status
compatibility risk
full-read status
notes
```

The matrix must explicitly distinguish:

- already canonical Rust semantics;
- duplicated semantics awaiting migration;
- Python-only host orchestration that should stay Python;
- Node-only host orchestration that should stay JS;
- compatibility shims;
- dead code candidates;
- generated/build artifacts that should not be migrated.

### Required invariant

No production-relevant file may be "unowned" or "forgotten."

---

# 6. Stage C — structural graph of the entire codebase

Build a structural map using language-aware parsing where practical.

## Python

Map at minimum:

- modules/packages;
- imports;
- public exports;
- classes/functions;
- direct callers where statically recoverable;
- CLI registration;
- MCP tool registration;
- provider/proxy registration;
- configuration/environment dependencies.

## Rust

Map at minimum:

- crates;
- modules;
- public structs/enums/traits/functions;
- feature gates;
- crate dependencies;
- PyO3 registrations;
- wasm-bindgen registrations;
- semantic modules shared through `entroly-engine`;
- any remaining duplicated semantic implementation.

## Node/JS/TypeScript

Map at minimum:

- modules/requires/imports;
- root exports;
- `.d.ts` declarations;
- npm CLI/server entrypoints;
- WASM calls;
- repository/persistence wrappers;
- package `files` inclusion.

## Packaging/config/workflows

Map:

- Python package manifests;
- Rust Cargo manifests/features;
- npm package manifests/files/exports;
- Docker/Homebrew paths where applicable;
- release workflows;
- wheel build matrix;
- version/minimum-native-version contracts.

### Important

This structural map is used to find what must be read. It does not substitute for the full-read gate below.

---

# 7. Stage D — semantic closure full-read gate

Before modifying a subsystem, compute its **semantic closure** and read that closure in full.

The closure includes:

1. the canonical implementation;
2. all duplicate/legacy implementations of the same behavior;
3. direct public wrappers/bindings;
4. direct callers;
5. important callees whose behavior defines correctness;
6. configuration used by the path;
7. persistence/schema formats used by the path;
8. tests protecting the path;
9. packaging/export declarations needed to deliver the path;
10. docs that make public behavioral promises about the path.

### Example: changing Work Graph semantics

At minimum fully read the relevant portions/files across:

```text
entroly-engine/src/work_graph.rs
entroly-engine/src/lib.rs
entroly-core/src/work_graph_bindings.rs
entroly-core/src/lib.rs
entroly/work_graph.py
entroly/work_graph_repo.py
entroly/work_graph_store.py
entroly/work_graph_content_digest.py
entroly/work_graph_cli.py
entroly/work_graph_mcp.py
entroly/work_graph_mcp_server.py
entroly/server.py
entroly-wasm/src/work_graph_bindings.rs
entroly-wasm/src/lib.rs
entroly-wasm/js/work_graph*.js
entroly-wasm/js/work_graph*.d.ts
entroly-wasm/index.js
entroly-wasm/index.d.ts
entroly-wasm/package.json
relevant tests
relevant packaging/release files
```

Then expand further wherever imports/callers show additional semantic dependencies.

### Example: changing Context Engine selection

Do not read only `knapsack.rs` or one Python wrapper. Read the full selection path:

```text
ingestion -> fragment creation -> query analysis -> ranking -> dedup -> guardrails -> budget -> selection -> receipt -> recovery -> public caller
```

and its PyO3/WASM/public surfaces.

### Required evidence

For each major subsystem changed, record the list of files fully read.

---

# 8. Stage E — trace real user journeys end-to-end

Static module knowledge is insufficient. Trace how actual users reach the code.

At minimum trace these user journeys if present/applicable:

## Python SDK

```text
import/public API
 -> wrapper
 -> PyO3/native call
 -> entroly-engine semantic implementation
 -> result/receipt/recovery
```

## Python CLI

```text
CLI parser
 -> command dispatch
 -> orchestration
 -> native semantic path
 -> output/error behavior
```

## MCP

```text
server registration
 -> tool input validation
 -> orchestration
 -> Work Graph/Context/Trust engine
 -> sanitization
 -> MCP response
```

## Provider/proxy

```text
provider request
 -> transform/compiler
 -> context selection/compression
 -> cache/stream/tool semantics
 -> provider forwarding
 -> receipt/recovery
```

## npm/WASM

```text
root npm export / CLI / MCP
 -> JS wrapper
 -> WASM binding
 -> entroly-engine semantic implementation
 -> JS result/types
```

## Package installation

```text
published artifact
 -> dependency resolution
 -> native/WASM availability
 -> imports/entrypoints
 -> runtime behavior
```

### Required artifact

For each journey, identify concrete files/functions/modules, not only conceptual boxes.

---

# 9. Stage F — capture pre-change behavioral contracts

Before migration, capture what must not regress.

Record or test at least the applicable contracts for:

- Python root exports;
- npm root exports;
- TypeScript declarations;
- CLI command names/options/exit behavior;
- MCP tool names/schemas/error shapes;
- public SDK signatures;
- Work Graph persisted schema;
- canonical commitments/hashes;
- receipt/recovery formats;
- provider request/tool/stream semantics;
- cache-stable prefix behavior;
- local-first/network behavior;
- package dependency/version rules;
- supported installation surfaces.

Where a behavior is intentionally changed, mark it explicitly rather than allowing accidental drift.

---

# 10. Stage G — run baseline tests BEFORE changing semantics

A migration cannot prove it preserved behavior if the pre-change baseline is unknown.

Run the relevant focused tests before major changes and record exact results.

At minimum for Work Graph-related changes, use the applicable Rust/Python/WASM test sets already present in PR #352.

For Context/Trust/provider/release changes, expand to the corresponding subsystem tests.

Record:

```text
SHA
command
passed
failed
skipped
xfailed
runtime/environment limitations
```

A pre-existing failure must remain distinguishable from a regression introduced by the migration.

---

# 11. Stage H — duplicate semantics audit

Search for semantic duplication deliberately.

For every shared concept being changed, find all implementations/near-implementations across:

```text
entroly-engine
entroly-core
Python package
entroly-wasm Rust
Node/JS
legacy/compatibility modules
```

Examples:

- ranking;
- dedup;
- repository mapping;
- status inference;
- verification;
- receipts;
- memory;
- routing;
- context pruning;
- recovery;
- hashing/content identity.

Do not delete a duplicate until behavioral comparison proves whether it is truly redundant or contains unique behavior.

---

# 12. Stage I — data/schema compatibility audit

Any migration touching persisted/shared data must inspect all readers and writers.

Audit:

- Work Graph JSON;
- checkpoint formats;
- receipt formats;
- recovery-store records;
- cache keys;
- content digests;
- npm/Python shared state;
- version/schema markers;
- migrations/backward compatibility.

### Required rule

Never change a persisted field's meaning silently.

---

# 13. Stage J — packaging and delivery closure

Before saying a capability is "ported," prove users can actually receive it.

Check applicable:

```text
entroly-engine crate
entroly-core/PyO3 build
Python package manifest
wheel workflows/platform matrix
Python package root exports
CLI/MCP registration
entroly-wasm build
npm package files
root JS exports
TypeScript declarations
Docker
Homebrew
release workflows
release-surface tests
```

A semantic implementation with no delivered binding/export is incomplete.

---

# 14. Mandatory audit artifact before implementation

Create/update:

```text
docs/PR352_CODEBASE_UNDERSTANDING_EVIDENCE.md
```

It must contain at least:

```text
Audited SHA
Tracked file count
Classified file count
Production-relevant file count
AST/symbol mapped count
Full-read count
Unknown/unclassified count
Semantic closures completed
Public user journeys traced
Duplicate-semantic findings
Persisted schemas reviewed
Baseline test commands/results
Known uncertainties
Files/subsystems intentionally not fully read
```

### Gate

If `unknown/unclassified count > 0` in an area being migrated, do not migrate that area yet.

If a major changed subsystem does not have a completed semantic closure, do not modify its shared semantics yet.

The evidence file must use concrete counts and paths. Statements such as "reviewed deeply," "checked the architecture," or "read the codebase" are not acceptable substitutes.

---

# 15. During implementation — change-impact protocol

For every substantial commit:

1. list files/functions changed;
2. compute callers/callees/exports impacted;
3. inspect corresponding Python/PyO3/WASM/npm surfaces;
4. inspect persisted data impact;
5. inspect package/release impact;
6. add/update targeted regression tests;
7. run focused tests;
8. update the ownership/migration matrix;
9. update codebase-understanding evidence if the architecture changed.

No commit should rely solely on "tests passed" if the relevant tests do not cover all delivery surfaces.

---

# 16. Review protocol — second-pass falsification

After implementing a chunk, review it as if trying to prove it wrong.

Ask:

- Which caller did I miss?
- Which runtime still has old semantics?
- Can Python and npm disagree?
- Did I break an import/export/type declaration?
- Did I alter persisted JSON compatibility?
- Did I weaken trust or fail-closed behavior?
- Did I introduce network I/O?
- Did I create unbounded graph growth?
- Did I accidentally ingest generated/vendor files?
- Does a clean repository still remain a null control?
- Can stale CI verify current code incorrectly?
- Can prompt-injected memory/handoff become trusted?
- Did a compatibility shim stop being exercised?
- Is a package artifact missing a newly required file?
- Does an editable-source test pass while an installed-package test fails?

Add tests for any plausible failure found.

---

# 17. Deep dogfood must use real execution paths

Do not dogfood by directly constructing internal objects only.

Exercise public user surfaces:

- installed Python package;
- Python SDK;
- CLI;
- MCP server/tool flow;
- native PyO3 path;
- npm/WASM installed package;
- Node public API;
- cross-Python/Node persistence;
- cross-agent handoff/resume;
- provider/proxy path when modified.

Internal unit tests are necessary but not sufficient.

---

# 18. Merge-blocking understanding criteria

PR #352 must remain draft if any of these is true:

- `docs/PR352_CODEBASE_UNDERSTANDING_EVIDENCE.md` is absent or stale for the changed migration head;
- production files remain unclassified in a migrated area;
- a changed subsystem lacks full semantic-closure reading;
- public entrypoint traces are incomplete;
- Python/npm semantic parity is unknown;
- a shared semantic duplicate remains unexplained;
- persisted schema compatibility is unknown;
- package delivery of a changed capability is unverified;
- baseline failures were not distinguished from regressions;
- exact-head CI is missing/failed for required gates;
- the implementation report uses vague language such as "reviewed the repo" without coverage numbers.

---

# 19. Efficient deep reading — do not waste time, do not cut corners

Deep understanding does **not** mean manually reading generated files, binary artifacts, vendored dependencies, or thousands of irrelevant fixtures line-by-line.

Use a layered approach:

```text
100% inventory/classification
        ↓
100% structural mapping where tooling supports it
        ↓
100% public-entrypoint tracing
        ↓
100% full read of semantic closure for changed systems
        ↓
selective full read elsewhere based on graph/call impact
        ↓
baseline tests + adversarial execution
```

This is stronger than either extreme:

- shallow repo summaries; or
- blindly reading every byte without understanding call relationships.

---

# 20. Final instruction to the continuing agent

Do not begin a large migration by saying "I scanned the repository and found...".

First produce evidence that you know:

- what files exist;
- what each production file owns;
- how the runtimes connect;
- which implementation is canonical;
- which duplicates remain;
- how users actually reach the code;
- what behavior currently passes;
- which persisted/public contracts must survive;
- exactly which implementation closure you fully read.

Then implement.

**The objective is not maximum reading. The objective is zero silent architectural omissions.**
