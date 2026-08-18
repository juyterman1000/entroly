# PR #352 MASTER IMPLEMENTATION PROMPT — Entroly AI Work Graph + Context + Trust

> **Branch:** `integration/workgraph-production-20260817`
>
> **PR:** #352
>
> **Status:** integration / production-hardening branch. **DO NOT MERGE TO `main` until the final exact-head production gate in this document passes.**
>
> **Companion context:** read `docs/PR352_CLAUDE_WORKGRAPH_HANDOFF.md` first, then use this document as the execution contract.

---

# ROLE

You are continuing Entroly as a principal product engineer, principal systems engineer, senior Rust engineer, senior Python engineer, senior Node/WASM engineer, reliability engineer, packaging/release engineer, security engineer, test architect, and agent-systems engineer.

Operate like a top-tier production engineering organization:

- understand the existing architecture before changing it;
- preserve stable user behavior unless there is a deliberate, tested migration;
- make shared semantics live once;
- design for deterministic behavior, recovery, failure isolation, concurrency, hostile inputs, packaging, upgrades, and rollback;
- prove claims with tests and exact-head evidence;
- never hide skipped or failing validation;
- do not stop at a prototype when production hardening is feasible;
- do not rewrite working systems merely to make the architecture diagram prettier.

Your job is not to create another isolated feature. Your job is to make the existing Entroly capabilities converge into a coherent production **AI Work Graph + Context Engine + Trust Engine** with Rust as the semantic source of truth and thin Python/npm delivery surfaces.

---

# 1. PRODUCT GOAL — WHAT WE ARE BUILDING

Entroly should become the local-first evidence-backed continuity and context layer that can enter an existing repository and answer:

1. What work is actually happening?
2. What changed?
3. Which files and symbols are involved?
4. Which tasks/workstreams depend on which artifacts?
5. Which agent/session/model worked on them?
6. What is unfinished, blocked, conflicting, abandoned, or awaiting verification?
7. What evidence supports those conclusions?
8. Which decisions and memories remain relevant?
9. What context can safely be selected/compressed?
10. What context must be pinned because it is failure-critical or verification-critical?
11. What omitted context is exactly recoverable?
12. Which claims are supported, contradicted, unsupported, or unknown?
13. Which tests/CI/receipts actually verify the current state?
14. What did another agent leave unfinished?
15. Can a different vendor/agent resume that work without trusting the previous agent's prose?
16. Are multiple agents touching overlapping paths/symbols and likely to conflict?
17. Which model/provider/runtime handled an execution, and what outcome did it produce?
18. What is the smallest trustworthy context required for the next action?

The end-user experience should feel like this:

```text
Agent arrives in repository
        │
        ▼
Entroly observes durable repository/worktree state
        │
        ▼
AI Work Graph reconstructs active work + evidence
        │
        ├── detects unfinished workstreams
        ├── maps files/symbols/dependencies/changes
        ├── identifies conflicts / parallel agents
        ├── restores decisions / scoped memory
        └── identifies verification gaps
        │
        ▼
Context Engine builds the minimum useful context
        │
        ├── selection
        ├── ranking
        ├── compression
        ├── dedup
        ├── budgeting
        ├── cache-aware stability
        └── exact recovery references
        │
        ▼
Trust Engine verifies what can actually be verified
        │
        ├── receipts
        ├── evidence
        ├── claim support / contradiction
        ├── test / CI state
        ├── commitment integrity
        └── recovery integrity
        │
        ▼
Next agent continues with evidence-backed context
```

The differentiator is not "we have a graph." The differentiator is **cross-agent continuity + context assurance + evidence-backed trust using one coherent work model**.

---

# 2. PRIOR SESSION CONTEXT — WHY WE ARE DOING THIS

This implementation follows several product conclusions already reached in the preceding work sessions.

## 2.1 Entroly already has many capabilities; do not restart from zero

The repository already contains substantial context and agent infrastructure: compression/selection, budgeting, deduplication, ranking, cache behavior, retrieval, repository mapping, dependency/symbol analysis, receipts, recovery, verification, checkpoints, memory, routing/provider integration, proxy/MCP/CLI/SDK surfaces, Rust/PyO3, and npm/WASM.

The problem is not lack of features. The problem is making them behave as **one system** instead of disconnected capabilities with possible semantic drift.

Before implementing anything new, identify what already exists, where it lives, whether it is canonical or duplicated, which tests protect it, and how it should map into this architecture.

## 2.2 AI Work Graph is the continuity spine

We decided the Work Graph should model the state required for another agent to resume work without replaying the entire previous conversation.

The graph must understand or reference:

- repositories;
- files;
- symbols;
- tasks;
- workstreams;
- agents;
- sessions;
- models/model executions;
- changes;
- commits;
- pull requests;
- tests;
- CI runs;
- decisions;
- claims;
- memory;
- evidence;
- receipts;
- failures;
- handoffs;
- work leases;
- dependencies and conflicts.

It must be temporal and evidence-backed rather than a static code graph.

## 2.3 Cross-agent/vendor handoff is a core product goal

A repository may be worked on by one agent, interrupted, and then opened by another agent/vendor. Entroly should recover the durable state and allow continuation.

Examples:

```text
Agent A -> Agent B
```

and parallel work:

```text
Agent A / subagent 1 ─┐
Agent A / subagent 2 ─┼─ same repository/work graph
Agent B              ─┤
Agent C              ─┘
```

No vendor-specific semantic rule belongs in the Rust engine.

## 2.4 Explicit handoff is stronger than inferred handoff

If an agent explicitly seals a handoff, it can bind to a graph revision, content commitments, evidence, and unfinished work.

But Entroly must still be useful when no previous agent installed Entroly or explicitly handed anything off. A new agent should be able to inspect durable Git/worktree/checkpoint facts and recover an **evidence-backed unknown-intent workstream** without inventing intent.

## 2.5 Compression and the Work Graph should reinforce each other

The graph tells the Context Engine what work/evidence is relevant.

The Context Engine tells the graph what context was selected, omitted, compressed, deduplicated, and made recoverable.

This creates a feedback loop:

```text
work state -> context selection -> receipt/evidence -> work state
```

## 2.6 Memory should be scoped and provenance-bearing

Memory is not an unbounded bag of summaries. It must know where it came from, what workstream it belongs to, how trustworthy it is, whether it is stale/superseded, and what evidence supports it.

## 2.7 Routing should be inspectable and evidence-backed

Model/provider routing belongs in the work graph as observable decisions and execution outcomes. Do not create opaque Python-only routing semantics that cannot be reproduced across runtimes.

## 2.8 "Hallucination" must be expressed honestly

Do not implement or market universal hallucination detection. Entroly can robustly detect within its observable scope:

- evidence-supported claims;
- contradictions;
- unsupported claims;
- unknown claims;
- stale verification;
- invalid receipt integrity;
- missing recovery targets;
- source commitment mismatch;
- current-vs-stale evidence.

That is defensible and useful.

---

# 3. FULL TARGET ARCHITECTURE

```text
                              ENTROLY ENGINE
                                   RUST
                                    │
             ┌──────────────────────┼──────────────────────┐
             │                      │                      │
       AI WORK GRAPH          CONTEXT ENGINE          TRUST ENGINE
             │                      │                      │
      repo identity            ingestion              receipts
      files/symbols            fragmentation          provenance
      tasks/workstreams        repo intelligence      evidence
      agents/sessions          query analysis         verification
      changes/commits          ranking                claim states
      PR/tests/CI              selection              integrity
      dependencies             compression            recovery
      conflicts                budgeting              contradiction
      handoffs                 dedup                  stale evidence
      memory                   retrieval              commitments
      routing                  cache stability        fail-closed policy
      model executions         conversation state
      temporal events          RAG/tool outputs
             │                      │                      │
             └──────────────────────┼──────────────────────┘
                                    │
                          SHARED RUST PUBLIC API
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
              entroly-core                     entroly-wasm
                  PyO3                             WASM
                   │                                 │
        Python orchestration                Node/JS orchestration
                   │                                 │
     ┌─────────────┼──────────────┐       ┌──────────┼──────────┐
     │             │              │       │          │          │
    SDK           CLI            MCP     npm CLI    MCP       agent glue
     │                            │                   │
 providers/proxy/frameworks      agent hosts        JS ecosystems
```

### Architectural law

**Shared product semantics live once in Rust.**

Python and JS may own host-specific mechanics:

- filesystem access;
- process execution;
- Git invocation;
- environment/config discovery;
- HTTP/provider transport;
- MCP transport;
- CLI rendering;
- package integration;
- platform-specific persistence mechanics where unavoidable.

But Python/JS must not independently define:

- work-state inference;
- status trust upgrades;
- graph commitment semantics;
- handoff validity;
- claim-support logic;
- shared conflict semantics;
- shared ranking meaning;
- memory trust/supersession meaning;
- routing outcome meaning;
- completion rules;
- verification rules.

If more than one runtime needs to interpret a rule, strongly prefer making the rule a Rust API/type.

---

# 4. THE "EVERY FILE AS NODES AND EDGES" RULE — IMPLEMENT IT CORRECTLY

The product intuition is correct: the entire repository should be **graph-addressable** so work can be located, related, resumed, ranked, and verified.

But do **not** eagerly materialize every file, symbol, test, dependency, historical commit, generated artifact, and vendor directory into a huge graph on every observation.

The production design should be **layered + lazy/materialized-on-demand**.

## 4.1 Required logical graph

```text
Repository
  ├── CONTAINS -> File
  │                 ├── DEFINES -> Symbol
  │                 ├── IMPORTS -> File/Module
  │                 ├── DEPENDS_ON -> File/Package
  │                 ├── REFERENCES -> Symbol
  │                 └── TOUCHED_BY <- Change
  │
  ├── CONTAINS -> Workstream
  │                 ├── PART_OF <- Task
  │                 ├── WORKS_ON <- Agent/Session
  │                 ├── TOUCHES -> File/Symbol
  │                 ├── BLOCKS / DEPENDS_ON -> Task/Workstream
  │                 ├── SUPPORTED_BY -> Evidence
  │                 ├── VERIFIED_BY -> Test/CI/Receipt
  │                 └── HANDED_OFF_TO -> Agent
  │
  ├── CONTAINS -> Commit / PR / CI Run
  └── CONTAINS -> Memory / Decision / Claim / Receipt
```

## 4.2 Addressability vs materialization

Every relevant repository artifact should have a stable identity strategy so it **can become a node** when needed.

Examples:

- repository ID;
- normalized repo-relative file path + repository identity;
- language-aware symbol identity;
- content digest where needed;
- commit SHA;
- PR ID;
- test ID/name + source reference;
- task/workstream stable ID;
- agent/session/model execution ID.

But graph materialization should be bounded:

- eagerly represent active/changed/high-value files;
- lazily expand symbols/dependencies around active work;
- index the wider repo separately where appropriate;
- use compact references/commitments instead of copying source text into graph state;
- avoid generated/vendor/binary noise unless directly relevant;
- allow bounded traversal outward from the active workstream.

This lets Entroly behave as though the whole repository is graph-aware without turning every operation into a whole-repo graph rebuild.

## 4.3 File/symbol graph must be deterministic

For the same repository revision/configuration, file/symbol identities and structural edges must be deterministic across runs and runtimes.

Renames should preserve lineage where evidence allows:

```text
old_file --SUPERSEDES/CONTINUES--> new_file
```

Do not silently treat a rename as unrelated delete+add when Git gives better evidence.

## 4.4 Generated/build/vendor files

Classify, do not blindly ingest.

At minimum distinguish:

- source;
- test;
- config;
- docs;
- packaging;
- generated;
- build output;
- vendored dependency;
- binary/media;
- hidden/state artifact.

Graph policy may vary by category.

---

# 5. REPOSITORY-WIDE PORT/MAPPING AUDIT — REQUIRED BEFORE LARGE MIGRATION

Do not assume the current architecture map is complete. Build/refresh a machine-checkable inventory from the actual branch.

The repo already contains a `docs/repo_file_map.md` and repo mapping machinery. Treat existing maps as useful evidence, not unquestionable truth.

For every production-relevant file/module, create or update an ownership/migration matrix with fields equivalent to:

```text
path
current_role
runtime
semantic_or_orchestration
canonical_owner
rust_module_if_shared
python_surface
wasm_node_surface
tests
public_entrypoints
migration_status
compatibility_risk
notes
```

Classify each item into one of these outcomes:

1. **Rust semantic owner** — shared algorithm/state meaning belongs in `entroly-engine`.
2. **PyO3 binding** — thin conversion/export only.
3. **WASM binding** — thin conversion/export only.
4. **Python host orchestration** — filesystem/provider/MCP/CLI/integration glue that should remain Python.
5. **Node host orchestration** — JS environment/integration glue that should remain JS.
6. **Compatibility shim** — temporary, documented, tested, scheduled for removal only when safe.
7. **Legacy duplicate** — remove only after parity tests prove the canonical replacement.
8. **Tests/fixtures/docs/packaging** — map to the canonical behavior they protect/deliver.
9. **Generated/build artifact** — exclude from semantic migration.

### Critical rule

**Do not delete or port a file merely because a Rust equivalent exists.**

First prove:

- API parity;
- behavior parity;
- packaging parity;
- import/entrypoint compatibility;
- error semantics;
- serialization compatibility;
- performance acceptable;
- targeted and integration tests green.

Then migrate callers.

Then keep or remove the old implementation intentionally.

---

# 6. REQUIRED WORK GRAPH SEMANTICS

Preserve and extend the existing Rust Work Graph rather than replacing it.

Core invariants:

1. append-only events are the source of truth;
2. materialized graph state is reconstructable;
3. canonical commitments are deterministic;
4. duplicate events are idempotent;
5. import/export detects tampering;
6. repository mismatch fails closed;
7. bounded limits protect every attacker-controlled dimension;
8. clean repository remains a null control;
9. agent prose is evidence, not verified truth;
10. completion requires explicit completion + suitable verification;
11. contradictions remain visible;
12. failed verification cannot be averaged away by optimistic statements;
13. advisory leases are not filesystem locks;
14. merge is deterministic;
15. cross-runtime semantics are identical.

### Graph layers

Treat the graph as several related layers:

```text
CODE GRAPH
repo -> files -> symbols -> calls/imports/dependencies

WORK GRAPH
tasks -> workstreams -> agents/sessions -> changes -> files/symbols

EVIDENCE GRAPH
claims -> evidence -> tests/CI/receipts -> verification

CONTINUITY GRAPH
sessions -> decisions -> memory -> handoffs -> recovery

EXECUTION GRAPH
workstream -> route -> model execution -> result -> verification
```

They should share stable node identities and edges rather than becoming separate incompatible databases.

---

# 7. REPOSITORY INTELLIGENCE

A newcomer should not have to ask Entroly to read every file sequentially.

Build/strengthen repository intelligence so Work Graph traversal can answer high-value questions cheaply.

Required capabilities should include, where languages permit:

- file inventory;
- language/type classification;
- symbol extraction;
- imports;
- call relationships;
- package/dependency relationships;
- change impact;
- test relationships/localization;
- ownership/entrypoint surfaces;
- changed-symbol detection;
- config/packaging relationship;
- public API surface detection.

Use existing repo-map, skeleton, dependency, index, provenance, and code-intelligence capabilities where they already exist. Do not recreate them unnecessarily.

### Incremental behavior

Prefer incremental update by content digest/change set over full rebuild.

A single file edit should not require rescanning an entire large repository unless the dependency model genuinely requires it.

---

# 8. CONTEXT ENGINE GOAL

The Context Engine should answer:

> Given current work state, evidence state, query, provider constraints, and a budget, what is the smallest context that preserves the information required to complete the next action safely?

Use existing Rust primitives rather than discarding them.

The engine may combine:

- query analysis;
- task/workstream scope;
- file/symbol graph distance;
- dependency/call relationships;
- changed-file priority;
- BM25/lexical relevance;
- semantic signals where available;
- diversity/dedup;
- information density;
- criticality/guardrails;
- verification/failure pinning;
- conversation pruning;
- cache stability;
- budget optimization;
- exact recovery handles.

### Important policy

Do not simply increase relevance weight for anything mentioned by an agent.

Evidence trust matters.

A failed test/verified contradiction should generally be more important to retain than an unverified agent summary saying everything is fine.

### Context receipts

Every meaningful compression/selection path should be able to produce a bounded receipt/reference that records enough to audit:

- source commitment;
- selected spans/fragments;
- omitted/recoverable spans;
- token/budget information;
- relevant workstream/task;
- evidence IDs;
- recovery handles;
- integrity commitment;
- verification information where available.

Do not store giant raw prompts in the Work Graph.

---

# 9. TRUST ENGINE GOAL

The Trust Engine should make assertions only as strong as its evidence.

Suggested states:

```text
supported
contradicted
unsupported
unknown
```

Verification states:

```text
passed
failed
skipped
unknown
stale
```

Integrity states:

```text
valid
invalid
missing
mismatched
```

### Exact-head rule

CI/test evidence must bind to the relevant content/commit when that distinction matters.

A green older SHA cannot verify a newer head.

### Fail-closed rule

If verification cannot run or evidence is missing, return unknown/skipped/unverified rather than success.

---

# 10. MEMORY

Use the existing Work Graph memory vocabulary rather than creating an unrelated second product.

Memory should be:

- scoped by repository/task/workstream/session where appropriate;
- provenance-bearing;
- content-addressed or commitment-bearing where practical;
- trust-labelled;
- bounded;
- supersedable rather than silently rewritten;
- stale-aware;
- recoverable/reference-based for larger content.

A useful memory is not "everything the previous model said." It is the minimum durable decision/state/evidence needed for future work.

---

# 11. ROUTING

Use Work Graph model/model-execution/routing concepts to record inspectable route decisions and outcomes.

Record bounded facts such as:

- provider/model/runtime;
- workstream/task;
- context budget;
- policy/version;
- reason/features where safe;
- latency;
- cost where known;
- success/failure;
- verification outcome;
- fallback path;
- receipt/evidence links.

Start deterministic/inspectable. Learning can later consume recorded outcomes, but fallback behavior must remain stable and explainable.

---

# 12. CROSS-AGENT AUTODISCOVERY AND HANDOFF

## 12.1 First arrival

When Entroly is invoked in a repository with no prior explicit Work Graph state:

1. identify repository safely;
2. observe Git/worktree facts;
3. inspect bounded existing checkpoints/continuity state if meaningful current work exists;
4. map changed files and relevant symbols;
5. create an evidence-backed workstream with unknown intent if necessary;
6. show what is known vs inferred;
7. do not invent a task description.

## 12.2 Existing shared state

When shared state exists:

1. load and integrity-check;
2. refresh current durable repo facts;
3. merge/deduplicate;
4. detect stale content commitments;
5. expose unfinished workstreams;
6. expose conflicts/leases;
7. select bounded evidence for resume.

## 12.3 Explicit handoff

Explicit handoff should capture/bind:

- repository identity;
- graph revision/commitment;
- workstream;
- source/target agent identities;
- relevant nodes/edges/evidence;
- changed content commitments;
- outstanding work;
- verification state;
- recovery references.

Tampering or changed worktree state must be detectable.

---

# 13. MULTI-AGENT COORDINATION

Support parallel agents without pretending Entroly is a distributed transaction manager.

Use advisory leases and conflict analysis.

Detect overlaps in:

- files;
- symbols;
- workstreams/tasks;
- dependency impact where useful.

Show likely conflicts; do not hard-lock the user's repository.

Expired leases must stop blocking coordination views.

Concurrent state writes must remain atomic and converge deterministically.

---

# 14. PYTHON / NODE / WASM PARITY

A user should not receive a different semantic product merely because they chose pip instead of npm.

For every shared semantic capability:

```text
entroly-engine
   ├── PyO3 binding -> Python public wrapper
   └── WASM binding -> Node public wrapper
```

Required parity checks include:

- exported types/functions;
- canonical JSON/schema;
- commitments/hashes;
- state transitions;
- import/export;
- merge behavior;
- handoff verification;
- content digest behavior;
- conflict detection;
- clean-repo null control;
- malformed/tampered input rejection;
- memory/routing semantics once implemented.

Python/Node host orchestration can differ internally but must normalize observations into the same Rust semantics.

---

# 15. PUBLIC API AND COMPATIBILITY

Do not break existing users accidentally.

Audit all relevant entry surfaces:

- Python package root exports;
- Python SDK;
- CLI;
- MCP tools/server;
- provider/proxy paths;
- agent wrappers/integrations;
- PyO3 native package;
- standalone Rust where applicable;
- WASM package;
- npm root exports;
- TypeScript declarations;
- Node CLI/MCP;
- Docker;
- Homebrew;
- packaging metadata/version constraints;
- release workflows.

For a changed public API, either:

- preserve compatibility; or
- provide an explicit migration/compatibility shim with tests.

Never silently leave npm behind after Python gains a semantic feature.

---

# 16. SECURITY / PRIVACY / HOSTILE-INPUT REQUIREMENTS

Treat all of these as untrusted:

- repository contents;
- Git metadata;
- filenames;
- branch names;
- agent/model text;
- recovered handoffs;
- memory;
- receipts imported from disk;
- MCP inputs;
- persisted shared state.

Requirements:

- bounded strings/arrays/state sizes;
- no unexpected network calls;
- no repository-controlled hooks/helpers when observing Git if avoidable;
- no credential leakage in repository identity;
- symlink/path traversal defenses;
- private local state permissions;
- atomic writes;
- prompt-injection fencing/sanitization for recovered text;
- no execution of recovered instructions as trusted commands;
- content/receipt integrity validation;
- safe failure on malformed persisted state.

---

# 17. PERFORMANCE AND SCALE

Design for real repositories, not only tiny fixtures.

Measure:

- initial repo observation/index cost;
- incremental edit cost;
- graph event apply latency;
- graph rebuild/import latency;
- state size growth;
- resume latency;
- conflict detection latency;
- context selection latency;
- PyO3 serialization overhead;
- WASM serialization overhead;
- lock contention;
- large dirty-repo behavior;
- symbol graph expansion cost.

Use bounded traversal/indexes instead of whole-graph scans.

Do not emit duplicate passive events every time a UI polls `state`.

Cache correctness must key against graph/content revision or another explicit validity commitment.

---

# 18. IMPLEMENTATION STRATEGY — DO NOT BIG-BANG REWRITE

Proceed in reviewable phases.

## Phase 0 — establish exact current truth

- read `CLAUDE.md` and branch-specific instructions;
- read the handoff document;
- inspect PR #352 current head and changed files;
- run current focused tests before changes;
- inventory current architecture;
- refresh the ownership/migration matrix;
- identify duplications and gaps with evidence.

## Phase 1 — repository graph foundation

- ensure stable repo/file/symbol identity;
- integrate existing repo map/skeleton/depgraph/index capabilities;
- add incremental file/symbol graph materialization;
- add changed-file/symbol relationships;
- add deterministic parity tests.

## Phase 2 — Work Graph integration

- link tasks/workstreams/changes/files/symbols;
- preserve temporal event model;
- improve resume traversal using code/work/evidence graph edges;
- test clean/null and dirty/unknown-intent cases.

## Phase 3 — Context Engine bridge

- pass bounded Work Graph scope into selection/ranking;
- pin failure/verification-critical evidence appropriately;
- emit context receipt/evidence events;
- ensure exact recovery survives compression.

## Phase 4 — Trust Engine bridge

- record claims/evidence/verification/integrity outcomes;
- bind exact-head verification;
- preserve contradiction dominance where appropriate;
- test stale evidence and tampering.

## Phase 5 — Memory

- add provenance-bearing scoped memory semantics;
- supersession/staleness;
- retrieval using existing selection infrastructure where practical;
- PyO3/WASM parity.

## Phase 6 — Routing/model execution

- record deterministic routing decisions and outcomes;
- connect model execution to workstream/receipt/verification;
- add public APIs only after semantics stabilize.

## Phase 7 — UX surfaces

- CLI/MCP/SDK/npm surfaces;
- first-arrival workflow;
- explicit handoff verification;
- conflict presentation;
- capability reporting.

## Phase 8 — migration cleanup

Only after parity is proven:

- migrate callers from duplicated semantic implementations;
- remove true dead duplicates;
- retain compatibility shims if public users still depend on them;
- update docs/maps/tests.

---

# 19. DEEP DOGFOOD GAUNTLET — REQUIRED BEFORE MAIN

Do not merge merely because unit tests pass.

Use Entroly against Entroly itself and construct realistic interruption/concurrency scenarios.

## Scenario A — first-time dirty repo

- no existing Work Graph state;
- several modified/untracked files;
- no task hint;
- Entroly must surface changed artifacts and an unknown-intent unfinished workstream;
- must not invent goal/author/plan.

## Scenario B — clean repo null control

- clean default branch;
- stale old checkpoint exists;
- must not resurrect an unfinished task without current evidence.

## Scenario C — explicit cross-agent handoff

- Agent A claims scoped work;
- edits multiple files/symbols;
- records decision + test evidence;
- handoff sealed;
- Agent B resumes;
- verify content commitments, workstream, evidence, outstanding work, recovery.

## Scenario D — interrupted agent without handoff

- Agent A disappears mid-change;
- new agent starts from repo only;
- Entroly reconstructs durable work without pretending to know unstated intent.

## Scenario E — parallel non-overlap

- two agents work on disjoint paths/symbols;
- no false conflict.

## Scenario F — parallel overlap

- two agents claim same file/symbol/dependency region;
- coordination reports conflict;
- no hard lock.

## Scenario G — rename + symbol continuity

- file renamed and symbol changed;
- graph preserves useful lineage;
- context retrieval does not retain only stale path.

## Scenario H — stale CI

- old SHA green;
- current SHA changed;
- verification must not report current head verified.

## Scenario I — contradictory agent claim

- agent says tests pass;
- captured test fails;
- verified failure wins status decision; both evidence items remain inspectable.

## Scenario J — tampered graph state

- modify persisted event/commitment;
- import fails closed.

## Scenario K — tampered handoff

- alter receipt fields;
- verification fails.

## Scenario L — content changed after handoff

- valid handoff created;
- worktree changed before resume;
- mismatch/staleness visible.

## Scenario M — prompt injection in recovered memory/decision

- hostile text stored as recovered data;
- MCP/context renderer fences/sanitizes it;
- never executes/trusts it as instruction.

## Scenario N — large repository

- thousands of files;
- verify bounded/lazy graph behavior;
- no unnecessary whole-repo rebuild for one-file edit.

## Scenario O — generated/vendor directories

- confirm classification/exclusion policy works;
- source graph is not drowned in irrelevant artifacts.

## Scenario P — Python/Node convergence

- Python writes/observes state;
- Node reads/merges;
- Node writes next event;
- Python reads;
- commitments and semantics converge.

## Scenario Q — multiprocessing contention

- concurrent local processes update state;
- atomicity/integrity preserved;
- stale lock recovery works.

## Scenario R — crash during persistence

- terminate during temp-write/replace boundary where fixture allows;
- last committed state remains readable.

## Scenario S — compression/recovery

- large tool/file context compressed;
- critical evidence retained;
- omitted material recoverable;
- receipt hashes/locators valid.

## Scenario T — package/user journey

Test as users actually install/use it, not from an editable source tree only:

- Python wheel install;
- native/PyO3 import;
- Python SDK call;
- CLI Work Graph flow;
- MCP flow;
- npm package install/build;
- Node Work Graph flow;
- root exports/types;
- provider/proxy path if touched;
- supported platform packaging checks where CI permits.

---

# 20. TEST MATRIX

Run targeted tests continuously, then broad gates before final status.

At minimum where applicable:

```bash
cargo fmt --all --check
cargo test -p entroly-engine
cargo clippy -p entroly-engine --all-targets -- -D warnings
```

Run applicable `entroly-core` PyO3 build/tests.

Run Work Graph / handoff / repository / parity Python tests, including relevant `tests/test_work_graph_*.py` and adjacent trust/recovery tests.

Run npm/WASM:

```bash
cd entroly-wasm
npm test
```

Run package/release-surface tests affected by changes.

Run full project gates required by repository instructions before declaring merge readiness.

### Exact evidence rule

In the final report include:

- exact commit SHA;
- exact command;
- pass/fail/skip count;
- known environment limitations;
- CI workflow status for that exact SHA;
- unresolved blockers.

Do not summarize "tests green" if only a subset ran.

---

# 21. FAILURE INJECTION / ADVERSARIAL TESTING

Test more than happy paths.

Inject:

- malformed JSON;
- duplicate event IDs;
- reordered events;
- foreign repo ID;
- oversized strings/arrays;
- huge state file;
- symlink state/lock paths;
- stale lock;
- clock anomalies where practical;
- deleted files during observation;
- Git operation timeout;
- malformed Git status;
- detached HEAD;
- merge conflict;
- rebase state;
- Unicode/path edge cases;
- file rename/copy/delete;
- permission denied;
- corrupt checkpoint;
- missing native engine;
- mismatched Rust/Python/npm versions;
- stale content digest;
- invalid recovery reference;
- truncated persisted write;
- conflicting claims;
- skipped verification;
- stale CI.

Production-grade means behavior remains explicit and recoverable when things go wrong.

---

# 22. MIGRATION / PARITY GATE — "EVERY FILE IS MAPPED"

Before final merge, produce a machine-checkable or reviewed report proving every production-relevant area is accounted for.

Required questions:

1. Which Python files contain shared semantics that should live in Rust?
2. Which have already been ported?
3. Which remain intentionally Python because they are orchestration?
4. Which Node files are equivalent orchestration?
5. Which semantic capabilities are missing from WASM/npm?
6. Which Rust modules have no Python/npm exposure but should?
7. Which public imports/CLI/MCP tools depend on legacy paths?
8. Which tests cover each migration?
9. Which packaging manifests include all required files?
10. Are generated artifacts accidentally treated as source?
11. Are repo map/docs stale after changes?
12. Can Python and npm produce different semantic outcomes for the same normalized observation?

**No production-relevant file should be "forgotten."**

But "accounted for" is the goal—not "rewrite every line in Rust."

---

# 23. DO-NOT-BREAK CONTRACT

Do not merge a change that silently breaks:

- compression behavior;
- retrieval;
- receipts;
- exact recovery;
- cache stability;
- provider request semantics;
- streaming/tool semantics;
- MCP contracts;
- Python SDK imports;
- npm exports/types;
- package installation;
- local-first/privacy behavior;
- checkpoint/session rescue;
- existing repository intelligence;
- deterministic behavior;
- backward-compatible persisted state without migration.

When touching an existing subsystem, add a regression test for the old behavior that must survive.

---

# 24. FINAL DEFINITION OF DONE

Do **not** mark this architecture stage production-ready until all are true:

- [ ] Rust is the canonical owner of all shared Work Graph semantics.
- [ ] Shared Context Engine semantics are Rust-owned or intentionally documented where not yet shared.
- [ ] Shared Trust Engine semantics are Rust-owned or intentionally documented where not yet shared.
- [ ] Repository artifacts are graph-addressable with stable identity strategy.
- [ ] Active files/symbols/dependencies are mapped through bounded/lazy graph materialization.
- [ ] Workstreams connect to files/symbols/changes/evidence.
- [ ] First-time dirty-repo autodiscovery works without fabricated intent.
- [ ] Clean-repo null control is preserved.
- [ ] Explicit handoff is integrity-bound and stronger than inferred recovery.
- [ ] Cross-agent continuation works in realistic interrupted-session dogfood.
- [ ] Parallel agent conflict detection works without hard locking.
- [ ] Context selection uses work/evidence scope where useful.
- [ ] Compression remains recoverable and receipted.
- [ ] Verification is exact-head/evidence aware.
- [ ] Contradictory evidence remains visible and fail-closed.
- [ ] Memory is scoped, provenance-bearing, stale/supersession aware.
- [ ] Model routing/execution is inspectable and linked to outcomes.
- [ ] Python and npm/WASM semantic parity is proven.
- [ ] Every production-relevant file/module is classified in the migration/ownership map.
- [ ] Legacy duplicates are removed only after callers/tests migrate safely.
- [ ] Public Python/npm/MCP/CLI surfaces are regression tested.
- [ ] Security/path/symlink/hostile-input tests pass.
- [ ] Multi-process persistence tests pass.
- [ ] Large-repo/incremental performance is acceptable and measured.
- [ ] Package/install/release-surface tests pass.
- [ ] Exact final SHA CI is green for all required gates, or any intentionally skipped gate has an explicit non-merge waiver from the maintainer.
- [ ] PR remains draft until the above evidence exists.

---

# 25. EXECUTION DISCIPLINE

For each substantial implementation chunk:

1. inspect existing code and tests;
2. state the invariant/gap being addressed in the commit message or engineering notes;
3. prefer changing the canonical Rust semantic owner;
4. update thin PyO3/WASM bindings;
5. update Python/Node orchestration only as necessary;
6. add parity/regression/adversarial tests;
7. run targeted tests immediately;
8. review the diff for accidental duplicate semantics;
9. inspect packaging/public API impacts;
10. commit a cohesive unit;
11. continue until the production gate is satisfied.

Do not repeatedly ask for permission for obvious next engineering steps inside this branch. Make conservative production decisions, document genuine tradeoffs, and keep the PR unmerged until evidence says it is ready.

If a requested architectural idea would reduce correctness, scalability, security, or compatibility, implement the stronger engineering interpretation and explain it in code/docs rather than blindly following a literal phrasing.

---

# 26. FINAL PRINCIPLE

Entroly should make repository work **continuable, compressible, recoverable, and verifiable across agents**.

The target mental model is:

```text
              Repository truth
                    │
                    ▼
             AI Work Graph
          /       /   \        \
      code     work   memory   execution
      graph    state   state     state
          \       \   /        /
                    ▼
              Context Engine
                    │
          smallest useful context
                    │
                    ▼
               Trust Engine
                    │
       evidence / receipt / verification
                    │
                    ▼
             Next agent action
                    │
                    └──── outcomes append back to graph
```

Build this as one production system, not a set of demos.

**Read the branch. Map what exists. Preserve what works. Move shared meaning into Rust. Keep Python/npm thin. Prove parity. Dogfood the real cross-agent flows. Fail closed. Do not merge until the exact final head is production-grade.**
