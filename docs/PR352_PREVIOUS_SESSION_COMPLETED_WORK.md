# PR #352 — Previous Session Completed Work Ledger

> **Purpose:** Prevent the next agent from redoing work that is already present on `integration/workgraph-production-20260817`.
>
> **Verified branch head when this ledger was written:** `ff4d1a57462432009cc621f0eb72d2ea969be1ce`.
>
> This document is an implementation inventory, **not** a claim that every test is green or that PR #352 is ready to merge. Before continuing, re-read the current branch and verify exact-head tests/CI because the branch may have advanced.

## Read order

1. `docs/PR352_CLAUDE_WORKGRAPH_HANDOFF.md` — architecture + continuation context.
2. `docs/PR352_PREVIOUS_SESSION_COMPLETED_WORK.md` — concrete work already present; do not redo blindly.
3. `docs/PR352_DEEP_CODEBASE_AUDIT_GATE.md` — mandatory evidence-based codebase-understanding gate; surface-level scanning does not satisfy it.
4. `docs/PR352_MASTER_IMPLEMENTATION_PROMPT.md` — execution contract and next production work.

Before major semantic migration, create/update `docs/PR352_CODEBASE_UNDERSTANDING_EVIDENCE.md` as required by the deep-audit gate.

---

# 1. Executive summary of what the previous session already built

The previous work did **not** merely write an architecture diagram. It established a real shared Work Graph foundation across Rust, PyO3/Python, WASM/npm, persistence, repository observation, content identity, CLI/MCP integration, and cross-agent recovery tests.

The key architectural decision already implemented is:

```text
entroly-engine = canonical shared semantics
        │
        ├── entroly-core / PyO3 -> Python
        └── entroly-wasm / WASM -> Node/npm
```

Python and JavaScript remain host/orchestration layers. Do not move shared task-state/trust/handoff semantics back into them.

---

# 2. DONE — Rust semantic Work Graph foundation

## `entroly-engine/src/work_graph.rs`

Already implemented on this integration branch:

- evidence-backed temporal Work Graph;
- append-only `WorkEvent` event source of truth;
- deterministic materialized nodes/edges reconstructed from events;
- canonical SHA-256 event IDs;
- canonical graph commitments;
- integrity checking during import;
- repository identity enforcement;
- deterministic graph merge/dedup;
- bounded event/node/edge/evidence/state inputs;
- trust levels;
- work statuses;
- separate status-trust vs node trust;
- evidence types;
- verification state;
- file-change types;
- claim states;
- repository observations;
- branch observations;
- file-change observations;
- commit observations;
- verification observations;
- decision observations;
- claim observations;
- advisory work lease observations;
- model execution observations;
- task hints;
- unfinished-work views;
- resume views;
- coordination conflict reporting;
- graph-bound handoff receipts;
- handoff integrity verification;
- bounded adjacency/materialized traversal support.

Existing node vocabulary includes:

```text
Repository
File
Symbol
Task
Workstream
Agent
Session
Model
ModelExecution
Change
Commit
PullRequest
Test
CiRun
Decision
Claim
Memory
Evidence
Receipt
Handoff
Failure
WorkLease
```

Existing edge vocabulary includes:

```text
Contains
Defines
Calls
Imports
DependsOn
WorksOn
DelegatedTo
Changed
Touches
Affects
SupportedBy
ContradictedBy
VerifiedBy
ProducedBy
Continues
HandedOffTo
Blocks
ConflictsWith
Supersedes
RoutedTo
PartOf
RecoversTo
References
```

### Do not redo

Do not create another Python or JavaScript Work Graph state machine. Extend this Rust model.

---

# 3. DONE — Rust crate is the semantic center

## `entroly-engine/src/lib.rs`

The engine crate already declares itself as the single source of truth for shared compute/algorithms.

The Work Graph is exported from this crate alongside the existing shared context primitives such as:

- BM25;
- cache;
- causal context;
- conversation pruning;
- dedup;
- dependency graph;
- entropy;
- fragment scoring;
- guardrails;
- hierarchical selection/compression;
- knapsack/budget optimization;
- LSH;
- query/query persona;
- semantic dedup;
- skeleton/structure extraction;
- trajectory/utilization;
- other existing engine algorithms.

### Do not redo

Do not reintroduce duplicated Rust algorithm copies between `entroly-core` and `entroly-wasm`.

---

# 4. DONE — PyO3 binding layer

## `entroly-core/src/work_graph_bindings.rs`

Implemented as a thin PyO3 wrapper over `entroly_engine::work_graph`.

Current responsibilities include boundary conversion/error mapping and exposing native Work Graph operations to Python.

## `entroly-core/src/lib.rs`

Already registers the Work Graph binding into the native Python module.

### Do not redo

Do not implement trust/status/handoff semantics inside PyO3. The binding should stay thin.

---

# 5. DONE — Python public Work Graph wrapper

## `entroly/work_graph.py`

Already provides an ergonomic Python API over the native Work Graph, including:

- construct by repository ID;
- deserialize from JSON;
- create from repository observation;
- refresh repository facts;
- apply event;
- merge graphs;
- export/import state;
- summary;
- snapshot;
- unfinished work;
- resume;
- coordination;
- handoff;
- handoff integrity verification.

The module intentionally contains no duplicated task-state/trust semantics.

### Do not redo

Preserve the thin-wrapper architecture.

---

# 6. DONE — conservative repository autodiscovery

## `entroly/work_graph_repo.py`

Already implemented a repository observation adapter that normalizes durable local state for Rust.

Current safeguards/features include:

- Git worktree resolution;
- stable credential-free repository identity;
- remote normalization without embedding credentials;
- read-only Git observation;
- `core.fsmonitor=false` protection;
- `GIT_OPTIONAL_LOCKS=0`;
- no fetch/push/network repository observation;
- bounded Git stdout/stderr;
- branch name/HEAD/default branch/base ref;
- ahead/behind counts;
- detached HEAD detection;
- merge/rebase-in-progress detection;
- porcelain status parsing;
- changed/untracked/renamed/copied/deleted/conflicted file observation;
- bounded changed-file count;
- bounded branch commit history;
- checkpoint metadata integration;
- checkpoint usage only when meaningful current Git work exists;
- clean-repository null-control behavior.

### Critical invariant already established

A stale checkpoint must not resurrect unfinished work in a clean repository.

### Do not redo

Do not infer task intent from branch names, commit prose, or filenames in Python.

---

# 7. DONE — cross-process durable Work Graph store

## `entroly/work_graph_store.py`

Already implemented shared durable Work Graph storage for local multi-process/agent use.

Current behavior includes:

- repository-keyed storage;
- private state directories;
- bounded state file size;
- safe state loading;
- symlink rejection;
- no-follow state-file opening where supported;
- exclusive-create advisory lock;
- lock timeout/backoff;
- stale-lock handling;
- filesystem-clock sampling for stale-lock decisions;
- atomic temporary write;
- file fsync;
- atomic replace;
- directory fsync on POSIX;
- merge-before-save;
- observation submission;
- repository update;
- explicit work claim;
- advisory leases;
- coordination query;
- resume;
- handoff.

### Do not redo

Do not replace this with an unsafe “just write JSON” persistence path.

---

# 8. DONE — content identity for resume/handoff

## `entroly/work_graph_content_digest.py`

Content-digest enrichment is present so resume/handoff can bind to actual worktree content rather than only filenames.

This supports the product requirement that another agent must be able to detect that the repository content changed after a handoff was sealed.

Cross-language digest parity tests are present in the PR.

### Do not redo

Extend the canonical digest/identity contract if needed; do not invent incompatible Python-vs-Node digest schemes.

---

# 9. DONE — Python CLI Work Graph surface

## `entroly/work_graph_cli.py`

Current Work Graph CLI operations include:

```text
state
claim
resume
handoff
```

Implemented behavior includes:

- explicit input validation/bounds;
- advisory lease creation;
- passive repository refresh before resume/handoff;
- content digest enrichment for explicit recovery/sealing operations;
- structured error mapping;
- JSON output support.

### Do not redo

Add commands only where a stable underlying semantic API exists.

---

# 10. DONE — MCP Work Graph orchestration

## `entroly/work_graph_mcp.py`

Already provides MCP-facing Work Graph operations for:

- state;
- claim;
- resume;
- handoff.

Important existing safety behavior:

- bounded request values;
- project path constrained to configured source root;
- bounded rendered output;
- recovered Work Graph state marked as untrusted recovered data;
- prompt-injection sanitization/fencing before model consumption;
- passive repository/content refresh before resume/handoff;
- structured error codes.

### Critical invariant already established

Recovered agent/user text is **data**, not trusted instructions.

### Do not redo

Preserve this trust boundary when adding memory/decision/handoff context.

---

# 11. DONE — MCP server integration

## `entroly/work_graph_mcp_server.py`

Work Graph MCP server integration exists on the branch and is covered by dedicated tests.

## `entroly/server.py`

The PR includes server integration changes so the Work Graph can coexist with the broader Entroly MCP/product shell.

### Do not redo

Do not build a second disconnected MCP server for the same semantics unless there is a deliberate compatibility reason.

---

# 12. DONE — WASM/npm Work Graph delivery

PR #352 already contains the Node/WASM side of the Work Graph.

Important files include:

- `entroly-wasm/src/work_graph_bindings.rs`;
- `entroly-wasm/src/lib.rs`;
- `entroly-wasm/js/work_graph.js`;
- `entroly-wasm/js/work_graph.d.ts`;
- `entroly-wasm/js/work_graph_repo.js`;
- `entroly-wasm/js/work_graph_repo.d.ts`;
- `entroly-wasm/js/work_graph_store.js`;
- `entroly-wasm/js/work_graph_store.d.ts`;
- `entroly-wasm/js/work_graph_content_digest.js`;
- `entroly-wasm/js/work_graph_continuity.js`;
- `entroly-wasm/js/work_graph_continuity.d.ts`;
- root `entroly-wasm/index.js` exports;
- root `entroly-wasm/index.d.ts` declarations;
- npm package file inclusion/tests.

### Do not redo

The next implementation should maintain semantic parity by extending Rust first and exposing thin WASM/JS wrappers.

---

# 13. DONE — Work Graph test foundation

PR #352 already includes a substantial Work Graph-focused test suite.

Files present in the integration change set include tests covering areas such as:

- native/PyO3 bindings;
- CLI;
- MCP;
- MCP server;
- repository observation;
- durable store;
- packaging;
- identity;
- content digests;
- cross-language digest parity;
- attachment/scope guards;
- multiprocessing;
- interrupted agent recovery;
- interrupted agent E2E;
- cross-agent recovery;
- entrypoints;
- verified handoff;
- session protection visibility;
- no-match honesty;
- release surfaces.

WASM/npm test files include Work Graph core, repository, store, content digest, continuity, and root export coverage.

### Important

Presence of these tests does **not** mean every exact-head run is green. Re-run them after current changes.

---

# 14. DONE — verified-handoff/session-rescue adjacency

PR #352 also includes or integrates existing related product work in:

- `entroly/verified_handoff.py`;
- `entroly/session_rescue.py`;
- runtime capability reporting;
- CLI/server integration;
- tests around verified handoff and session protection visibility.

The new Work Graph should integrate with these capabilities rather than replacing them with a disconnected second concept of recovery/handoff.

### Do not redo

Audit and unify contracts carefully before removing any legacy/adjacent path.

---

# 15. DONE — packaging/native-engine honesty work present on the branch

The integration branch also contains production packaging changes adjacent to Work Graph work.

Examples include:

- native `entroly-core` treated as required for correct query-conditioned Python selection;
- self-heal/install behavior when native engine is missing in user-facing measurement/runtime paths;
- documentation/privacy language for that behavior;
- Linux musl wheel publication jobs for native core;
- release-surface tests;
- runtime capability/self-heal changes.

### Do not accidentally regress

Work Graph/Context/Trust changes must not break native package installation, supported wheels, npm packaging, or release-surface consistency.

---

# 16. DONE — branch documentation and execution contract

The previous session added:

## `docs/PR352_CLAUDE_WORKGRAPH_HANDOFF.md`

Contains:

- architecture;
- existing Work Graph implementation inventory;
- invariants;
- Context/Trust integration direction;
- autodiscovery/handoff rules;
- memory/routing direction;
- security/parity constraints;
- production test strategy.

## `docs/PR352_MASTER_IMPLEMENTATION_PROMPT.md`

Contains the larger execution contract:

- full product goal;
- prior-session reasoning;
- full architecture;
- repository/file/symbol node-edge model;
- every-file ownership/mapping audit;
- lazy/bounded graph materialization rule;
- Context Engine integration;
- Trust Engine integration;
- memory;
- routing;
- Python/npm parity;
- migration strategy;
- adversarial testing;
- deep dogfood scenarios;
- exact-head production merge gate.

## `docs/PR352_DEEP_CODEBASE_AUDIT_GATE.md`

Contains the mandatory anti-surface-reading protocol:

- complete tracked-file classification;
- ownership/migration matrix;
- language-aware structural mapping;
- semantic-closure full reads for changed systems;
- concrete user-journey traces;
- pre-change behavioral contracts;
- baseline tests before migration;
- duplicate semantics audit;
- persisted-schema audit;
- packaging/delivery closure;
- quantitative `PR352_CODEBASE_UNDERSTANDING_EVIDENCE.md` evidence.

### Do not skip these documents

They contain decisions made specifically to prevent a large migration from breaking existing Entroly behavior.

---

# 17. IMPORTANT — what is NOT complete yet

Do not confuse the strong Work Graph foundation with completion of the full architecture.

The following remain goals requiring implementation/proof where not already present:

- full repository-wide ownership/migration matrix for all production-relevant files;
- comprehensive stable file/symbol identity and lazy repository graph integration;
- deeper integration of existing repo-map/skeleton/depgraph/code-intelligence capabilities into the Work Graph;
- Work Graph-aware Context Engine selection/ranking;
- Context Receipt -> Work Graph evidence linkage;
- Trust Engine -> Work Graph verification/claim/integrity linkage;
- exact-head CI semantics integrated into work state;
- provenance-bearing Work Graph memory semantics;
- model routing/execution policy/outcome linkage;
- broader CLI/MCP UX only where stable;
- large-repository performance validation;
- complete Python/npm shared-semantic parity for newly added capabilities;
- final migration/removal of any true duplicated semantics after parity proof;
- full production dogfood and adversarial gauntlet;
- exact final SHA green release evidence.

---

# 18. Previous-session architectural decisions — preserve unless evidence disproves them

## Decision A — Rust owns shared meaning

If Python and npm both need to interpret a semantic rule, implement the canonical rule in Rust wherever technically appropriate.

## Decision B — do not blindly port every file to Rust

"Every file accounted for" means every production-relevant module has a known owner/mapping/migration status.

It does **not** mean rewriting platform orchestration into Rust.

## Decision C — every relevant repository artifact should be graph-addressable

Files/symbols/tests/config/packaging artifacts should have stable identities where useful, but use bounded/lazy materialization rather than eagerly creating a gigantic graph for every repository operation.

## Decision D — explicit handoff > inferred recovery

Autodiscovery recovers evidence-backed unfinished state. It does not prove unstated intent.

## Decision E — clean repository is a null control

No stale memory/checkpoint/agent claim should invent work on a clean current repository.

## Decision F — evidence outranks optimistic prose

A verified failing test/CI result must not be overridden by an unverified statement such as "tests pass".

## Decision G — recovered context is untrusted

Memory/decisions/handoffs can contain prompt injection or hostile text. Render them as data.

## Decision H — preserve existing Entroly capabilities

Do not throw away existing compression, retrieval, receipts, recovery, repository intelligence, provider/proxy, checkpoint, memory, routing, MCP, SDK, npm, and packaging behavior. Map them into the architecture and migrate deliberately.

---

# 19. Recommended first action for the continuing agent

Do this before writing major new code:

1. confirm the current PR #352 head;
2. read `CLAUDE.md`;
3. read the PR #352 continuation documents in the required order;
4. complete `PR352_DEEP_CODEBASE_AUDIT_GATE.md` and create/update `docs/PR352_CODEBASE_UNDERSTANDING_EVIDENCE.md`;
5. inspect current changed files;
6. run the existing focused Work Graph Rust/Python/WASM tests on the exact current branch;
7. distinguish current failures from missing future features;
8. refresh the repository ownership/migration matrix;
9. only then continue the master implementation sequence.

Do not redo a completed component just because the implementation could be styled differently. Change it only for a concrete correctness, parity, security, performance, or product requirement.

---

# 20. Continuation principle

The previous session established the foundation. The next session should build **on top of it**:

```text
ALREADY BUILT
Rust Work Graph semantics
      +
PyO3/Python exposure
      +
WASM/npm exposure
      +
repo observation
      +
durable local store
      +
content identity
      +
CLI/MCP recovery/handoff
      +
initial cross-agent test foundation

              ↓ CONTINUE, DO NOT RESTART ↓

NEXT
mandatory deep codebase understanding evidence
      +
repository file/symbol graph integration
      +
Work Graph-aware Context Engine
      +
Trust/evidence integration
      +
scoped memory
      +
model routing/execution
      +
full migration/parity map
      +
deep dogfood and production release proof
```

The goal is a coherent production system, not a second implementation beside the one already present.
