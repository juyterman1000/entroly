# PR #352 — Claude continuation handoff: Entroly Engine / AI Work Graph

> **Purpose:** This is the continuation contract for PR #352 (`integration/workgraph-production-20260817`). Read it before changing the Work Graph, Context Engine, Trust Engine, bindings, persistence, MCP, CLI, or release surfaces.
>
> **Do not treat this document as proof that a feature exists.** Verify the current branch and tests first. This document distinguishes what is already present from the next implementation work.
>
> **Required continuation read order:** after this architecture handoff, read `docs/PR352_PREVIOUS_SESSION_COMPLETED_WORK.md`, then complete `docs/PR352_DEEP_CODEBASE_AUDIT_GATE.md`, then use `docs/PR352_MASTER_IMPLEMENTATION_PROMPT.md` as the execution contract. Before major semantic migration, create/update `docs/PR352_CODEBASE_UNDERSTANDING_EVIDENCE.md` with quantitative coverage, semantic-closure full reads, user-journey traces, duplicate-semantics findings, schema review, and baseline test evidence. A symbol scan, grep pass, repo summary, or reading only changed files does **not** satisfy the audit gate.

## 1. Product mission

Entroly is becoming one coherent **local-first AI work/context/trust engine** rather than a collection of Python features.

The intended production architecture is:

```text
                         ENTROLY ENGINE
                              RUST
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
     AI Work Graph        Context Engine       Trust Engine
          │                    │                    │
     tasks/events          compression           receipts
     agents                selection             evidence
     sessions              retrieval             verification
     dependencies          budgeting             claim risk
     workstreams           dedup                 recovery
     conflicts             cache                 integrity
     handoffs              ranking
     memory
     routing
          │
          └────────────────────┬────────────────────┘
                               │
                         Rust public API
                               │
                  ┌────────────┴────────────┐
                  │                         │
             entroly-core              entroly-wasm
                 PyO3                      WASM
                  │                         │
               Python                     npm
                  │                         │
        CLI / MCP / providers       Node / JS / agent glue
```

The architectural goal is **one semantic implementation in Rust, many thin delivery surfaces**.

Python and JS are allowed to perform host-specific orchestration such as filesystem observation, process spawning, CLI rendering, MCP transport, provider integration, and Node/Python storage mechanics. They must not independently redefine task-state inference, trust upgrades, graph commitments, handoff validity, conflict semantics, ranking semantics, or other shared product meaning.

## 2. PR coordinates at handoff

- Repository: `juyterman1000/entroly`
- PR: `#352 — Integration/workgraph production 20260817`
- Branch: `integration/workgraph-production-20260817`
- Base: `main`
- Pre-handoff head observed: `91e2c487fb131115ffe2849841f11042aec3d2a9`
- At the time this handoff was written, PR #352 was open and GitHub reported it mergeable.
- CI for that pre-handoff head had been triggered; most workflow runs were still queued, so **do not claim the PR is green until exact-head CI is complete**.

## 3. What is already implemented on PR #352

### 3.1 `entroly-engine` is the semantic center

`entroly-engine/src/lib.rs` explicitly defines the crate as the single source of truth for compute. Existing context algorithms already live there and are shared by PyO3 and WASM. The Work Graph has now joined that crate as `pub mod work_graph`.

The current engine includes context-oriented modules such as:

- `bm25`
- `cache`
- `causal`
- `conversation_pruner`
- `dedup`
- `depgraph`
- `eicv`
- `entropy`
- `fragment`
- `guardrails`
- `hierarchical`
- `knapsack`
- `learning`
- `lsh`
- `prism`
- `query`
- `query_persona`
- `semantic_dedup`
- `skeleton`
- `trajectory`
- `utilization`
- `work_graph`

Do not create a second semantic engine elsewhere.

### 3.2 Rust AI Work Graph exists

`entroly-engine/src/work_graph.rs` currently owns the shared Work Graph semantics.

Important existing concepts include:

- append-only `WorkEvent` source of truth;
- deterministic materialized graph rebuilt from events;
- canonical SHA-256 event IDs and graph commitments;
- repository identity checks;
- bounded inputs/state;
- deterministic merge;
- import integrity verification;
- work status and status trust separated from node trust;
- conservative task/workstream inference;
- evidence references;
- decisions and claims;
- Git branch/change/commit observations;
- verification observations;
- model execution observations;
- advisory work leases;
- coordination conflict reporting;
- resume views;
- graph-bound handoff receipts;
- handoff integrity verification.

Current node vocabulary already includes:

- repository, file, symbol;
- task, workstream;
- agent, session;
- model, model execution;
- change, commit, pull request;
- test, CI run;
- decision, claim, memory, evidence, receipt, handoff, failure, work lease.

Current edge vocabulary already includes useful foundations for the next phase:

- contains / defines / calls / imports / depends-on;
- works-on / delegated-to;
- changed / touches / affects;
- supported-by / contradicted-by / verified-by;
- produced-by / continues / handed-off-to;
- blocks / conflicts-with / supersedes;
- routed-to / part-of / recovers-to / references.

Use this vocabulary before inventing parallel concepts.

### 3.3 Non-negotiable Work Graph invariants already encoded

Preserve these invariants:

1. **Events are the source of truth.** Materialized nodes/edges are derived and rebuildable.
2. **Commitments are deterministic.** Event IDs and graph commitments use canonical payloads.
3. **A clean repository is a null control.** Never invent unfinished work merely because an old checkpoint, branch name, or agent statement exists.
4. **Agent/model text is evidence, not truth.** It must not become `Verified` without independent evidence.
5. **Completed means completed plus verified.** Do not turn prose like “done” into trusted completion.
6. **Contradiction/failure blocks rather than averaging away evidence.**
7. **Leases are advisory coordination, not filesystem locks.**
8. **Persisted state is integrity checked on import.**
9. **Cross-runtime semantics must be identical.** Python/npm differences belong only at the adapter boundary.
10. **Fail closed on malformed/partial observations.** A partial repository scan must not masquerade as complete state.

### 3.4 PyO3 boundary exists

`entroly-core/src/work_graph_bindings.rs` is intentionally a thin PyO3 layer over `entroly_engine::work_graph`.

The Python public wrapper in `entroly/work_graph.py` exposes the native engine without duplicating semantics, including:

- create/from JSON;
- repository observation/refresh;
- event application;
- merge/export;
- summary/snapshot;
- unfinished work;
- resume;
- coordination;
- handoff;
- handoff verification.

Keep this layer thin.

### 3.5 Conservative repository autodiscovery exists

`entroly/work_graph_repo.py` observes local durable facts while refusing to infer task intent in Python.

Current safeguards include:

- read-only Git invocation;
- fsmonitor disabled to avoid repository-config execution surprises;
- optional Git locks disabled;
- no fetch/push/network Git operations;
- bounded Git output;
- bounded changed paths and commit history;
- credential-free repository identity;
- clean-repository null-control behavior;
- branch/default-base/ahead/behind observation;
- merge/rebase detection;
- porcelain status parsing;
- checkpoint use only when Git independently proves that work exists.

Do not weaken these protections for convenience.

### 3.6 Cross-process persistence and coordination exist

`entroly/work_graph_store.py` provides repo-keyed durable state for Python processes using canonical Rust JSON.

Current mechanics include:

- private storage directories;
- bounded state size;
- exclusive-create advisory lock file;
- stale-lock handling;
- symlink checks;
- atomic temp-write + fsync + replace;
- merge-on-save;
- explicit work claims;
- leases;
- resume/coordination/handoff helpers.

Node has corresponding Work Graph store/continuity modules. Cross-language parity tests are present. Preserve the common storage contract even though the host-side filesystem mechanics are implemented in Python/JS.

### 3.7 Content identity is part of handoff/resume

PR #352 includes `work_graph_content_digest` support and cross-language digest parity tests.

Resume and handoff are explicit recovery/sealing operations: they refresh repository facts and bind worktree content identity so another agent cannot silently continue from a different working tree while believing it received the same state.

### 3.8 CLI and MCP entry points exist

The Python CLI currently exposes:

```text
state
claim
resume
handoff
```

The MCP-facing layer currently exposes the equivalent work-state/claim/resume/handoff operations and renders recovered state as fenced **untrusted recovered work state**, with prompt-injection sanitization before an agent consumes it.

This is important: recovered agent/user text is not system instruction.

### 3.9 WASM/npm Work Graph support exists

PR #352 includes:

- Rust WASM bindings;
- Node Work Graph wrapper;
- Node repo observation;
- Node durable store;
- Node content digests;
- continuity helpers;
- root exports and `.d.ts` surfaces;
- npm Work Graph tests.

The npm package test script includes Work Graph, repository, store, digest, continuity, and root-export tests. Maintain semantic parity with Python by making meaning live in Rust.

### 3.10 Existing adjacent Entroly capabilities must be integrated, not replaced

This PR also touches/session-protects existing capabilities such as verified handoff/session rescue. Entroly already has Context Receipts, recovery, ranking/compression infrastructure, guardrails, provider paths, and release surfaces.

Do not build a disconnected “new platform.” The task is to connect these existing systems through the Rust engine and shared evidence model.

## 4. The next implementation goal

The Work Graph foundation is strong enough to stop treating it as a standalone feature. The next phase is to make it the **work-state spine** that the Context Engine and Trust Engine can consume and emit evidence into.

Do this incrementally. Do not rewrite the whole repository.

## 5. Priority P0 — unify Work Graph + Context + Trust in Rust

### 5.1 Introduce explicit engine-level integration contracts

Create narrow Rust types/APIs that allow the three subsystems to exchange facts without cyclic high-level dependencies.

Preferred direction:

```text
Work Graph
  -> supplies task/workstream/repo/agent/session/change/evidence scope
  -> Context Engine selects/compresses/retrieves within that scope
  -> Trust Engine evaluates evidence/receipts/verification/recovery
  -> outcomes are appended back to Work Graph as evidence/events
```

Avoid a giant god object. Favor small immutable request/result structs and append-only outcome events.

A good integration contract should answer:

- What workstream/task is active?
- What files/symbols/changes are relevant?
- What evidence is durable vs inferred vs untrusted?
- What context fragments were selected/omitted?
- Which omitted spans are recoverable and by what handle/hash?
- Which claims are supported, contradicted, unsupported, or unknown?
- Which verifications passed/failed/skipped?
- Which model/provider handled the work and what was the outcome?
- What must the next agent know to resume without rereading the whole repository?

### 5.2 Treat Context Receipts as first-class graph evidence

Do not keep receipts adjacent to the graph forever.

Add a Rust-owned way to record/link a context receipt to:

- repository;
- session;
- agent/model execution;
- task/workstream;
- selected evidence/fragments;
- omitted/recoverable spans;
- verification outcome;
- source/context commitment.

Use existing `Receipt`, `Evidence`, `Claim`, `ModelExecution`, `SupportedBy`, `VerifiedBy`, `ProducedBy`, `RecoversTo`, and `References` concepts where possible.

Do **not** embed giant raw contexts into the Work Graph. Store bounded references, commitments, locators, hashes, and recovery handles.

### 5.3 Connect retrieval/selection to work scope

The Context Engine should be able to consume a bounded Work Graph view as an additional source of relevance/provenance, not as an oracle.

Examples:

- changed files/symbols can become candidate-scope evidence;
- active task/workstream can provide query context;
- failed tests and contradictions can be pinned/boosted rather than compressed away;
- verified decisions can be retained preferentially;
- stale or unrelated prior-agent chatter can be demoted;
- exact recovery references must survive compression.

Never make “the agent said it is important” equivalent to verified importance.

### 5.4 Trust Engine: detect evidence problems, do not promise magical hallucination detection

The product architecture uses the word `hallucination`, but implementation and public claims must stay evidence-bounded.

The Trust Engine may safely compute/report states such as:

- supported;
- contradicted;
- unsupported;
- unknown;
- verification passed/failed/skipped;
- receipt integrity valid/invalid;
- recovery target available/missing;
- source commitment matched/mismatched;
- evidence stale/current.

Do not claim that Entroly can universally determine whether arbitrary model text is a hallucination. It can detect and fail closed on **evidence/verification contradictions and unsupported claims** within its observable scope.

### 5.5 Feed verification back into work state

A model saying “tests pass” is an agent statement. A captured test/CI result is evidence.

Implement the bridge so verifications can update/produce Work Graph evidence and status through Rust rules.

Examples:

- failed exact-head CI -> workstream remains blocked/needs verification;
- passing targeted test -> verified evidence for a claim/task, not automatic global completion;
- stale CI from an older SHA -> must not verify the new head;
- skipped test -> not passed;
- missing evidence -> unknown, not success.

## 6. Priority P0 — first-class cross-agent continuation

The product experience should work even when the first agent/vendor never explicitly created a handoff.

### 6.1 First arrival / autodiscovery

When a new agent attaches to a repo:

1. identify the repository;
2. load the shared graph if present;
3. observe current durable Git/worktree/checkpoint facts;
4. merge the new observation;
5. show unfinished workstreams and conflicts;
6. let the agent choose/resume/claim work;
7. do not fabricate a task when the repository is clean and no durable work evidence exists.

A first-time Entroly user with an already-dirty repo should still get useful **evidence-backed unfinished-work discovery** from Git/worktree state. Intent can remain unknown until the user/agent explicitly supplies it.

### 6.2 Explicit handoff remains stronger than inferred recovery

An explicit handoff receipt should remain the strongest cross-agent continuation artifact because it commits to a specific graph revision/content state.

Autodiscovery is a recovery mechanism, not proof that the previous agent intended a particular plan.

Render that distinction in API output/trust labels.

### 6.3 Multi-vendor parallel agents

Claude, Codex, OpenClaw, Cursor, Aider, or other agents may operate on the same repo concurrently.

The shared rules should be vendor-neutral:

- identify each agent/session explicitly when known;
- claim task/path/symbol scope with advisory lease;
- expose overlapping leases/conflicts;
- never lock the user's filesystem to enforce ownership;
- record handoffs/continuations;
- permit independent workstreams;
- merge event histories deterministically;
- refuse to erase contradictory evidence.

The graph should support both:

```text
Claude -> Codex handoff
```

and:

```text
Claude subagent A ─┐
Claude subagent B ─┼─ shared repo/work graph
Codex agent C    ──┤
Other vendor D   ──┘
```

without introducing vendor-specific semantics into Rust.

## 7. Priority P1 — memory becomes provenance-bearing Work Graph memory

`NodeKind::Memory` already exists. Build the product semantics around it instead of creating another unrelated memory store.

### 7.1 Memory record contract

A memory should have or reference:

- stable ID;
- bounded content or content reference;
- source agent/session/model;
- task/workstream/repository scope;
- provenance/evidence IDs;
- trust level;
- creation/update time;
- optional expiry/valid-until semantics;
- supersession relationship;
- content commitment/hash;
- recovery locator where applicable.

### 7.2 Memory trust rules

- user statement != verified repository fact;
- agent summary != verified fact;
- derived memory is `Inferred` unless independently verified;
- stale/superseded memory must not outrank fresh durable evidence;
- contradictions remain visible;
- do not silently mutate old memory into a different claim; append superseding evidence/events.

### 7.3 Memory retrieval

Memory retrieval should use the same Rust ranking/selection infrastructure where practical and be bounded by workstream/task/repo scope.

The goal is not “remember everything.” The goal is **recover the smallest trustworthy state needed for the next action**.

## 8. Priority P1 — routing becomes evidence-backed model routing

`NodeKind::Model`, `NodeKind::ModelExecution`, and `EdgeKind::RoutedTo` already exist. Build on them.

### 8.1 Routing observation/outcome

Record bounded facts such as:

- provider/model;
- task/workstream;
- route reason/features (bounded and inspectable);
- latency;
- token/context budget where available;
- cost where available;
- success/failure/verification outcome;
- fallback path;
- evidence/receipt IDs.

### 8.2 Routing must be conservative

Do not train hidden magical policy in Python or JS.

Start with deterministic/inspectable Rust policy hooks and record outcomes. If learning is introduced later, preserve:

- reproducible fallback;
- observable policy version;
- bounded inputs;
- no secret provider calls;
- user/provider constraints;
- local-first behavior;
- clear distinction between predicted route quality and verified outcome.

## 9. Priority P1 — make continuation discoverable in the product

The engineering work is not complete if users cannot find it.

### CLI

Keep the existing commands and consider a coherent final surface such as:

```text
work state
work claim ...
work resume ...
work handoff ...
work conflicts
work verify-handoff ...
```

These are **proposed**, not shipped. They are written without the `entroly`
prefix on purpose: `tests/test_docs_code_sync.py` asserts that every
`entroly <subcommand>` appearing in a code block is a real subcommand, so
documenting an unimplemented command as runnable turns a design note into a
false claim about the CLI. Add the prefix when the command exists.

Do not add commands merely for symmetry; add only when backed by a stable API and tests.

### MCP

Expose the minimum safe tool set needed by any agent host:

- inspect shared work state;
- claim work;
- resume unfinished work;
- create handoff;
- inspect conflicts;
- verify/consume a handoff;
- optionally record verification/model execution/memory through explicit bounded tools.

Recovered text must remain fenced/untrusted.

### npm

Node users must receive the same **semantic capability**, not a degraded marketing-only wrapper.

If a shared semantic feature exists in Rust, add/verify the corresponding WASM binding and root/type export where technically possible.

### Python

Python orchestration should remain ergonomic, but no Python-only business rules may redefine Rust state meaning.

## 10. API design rules

### 10.1 Prefer explicit typed observations

Do not add a generic “metadata bag does everything” API when a stable semantic type is warranted.

Good:

```text
VerificationObservation
ModelExecutionObservation
WorkLeaseObservation
DecisionObservation
ClaimObservation
```

Similarly, add explicit receipt/memory/routing observations if needed.

### 10.2 Attributes are extension points, not semantic escape hatches

If multiple runtimes need to interpret a field, promote it to the Rust schema instead of hiding it in `attributes` with duplicated parser logic.

### 10.3 Version schema changes

Any persisted event/document schema change must be versioned and migration/backward-compatibility behavior must be deliberate.

Never silently reinterpret persisted v1 fields under new semantics.

### 10.4 Bound everything attacker-controlled

Maintain explicit limits for:

- events;
- operations/event;
- nodes/edges/evidence;
- strings;
- attributes;
- changed paths;
- scope paths/symbols;
- state bytes;
- rendered MCP output;
- Git output;
- evidence returned by resume.

A repo and an agent transcript are untrusted inputs.

## 11. Security/privacy constraints

Preserve Entroly's trust contract:

- local-first;
- no surprise remote calls;
- no prompt/code telemetry by default;
- no shelling out to repository-controlled hooks/helpers when avoidable;
- no prompt injection from recovered work state;
- no credential material in repo identity;
- no symlink-following for shared state where it creates overwrite/read risk;
- no world-readable persisted sensitive state on POSIX;
- no executable “memory” or handoff instructions treated as trusted commands.

A recovered handoff/memory may contain hostile text. Treat it as data.

## 12. Product correctness constraints

### Never fabricate completeness

Do not mark work completed because:

- there are no unstaged files;
- an agent says “done”;
- a commit message says “fix”;
- a PR exists;
- an old CI run is green;
- the branch is ahead.

Completion needs explicit completion evidence plus valid verification according to Rust rules.

### Never fabricate unfinished work either

A clean default branch must remain a null control. Old checkpoints/memory must not resurrect work without current durable evidence or explicit user/agent intent.

### Never collapse contradictions

If one agent says “tests pass” and exact-head CI fails, retain both observations with different trust and let verified failure dominate the work-state decision.

## 13. Tests required for every semantic addition

For any Rust semantic change, add Rust unit tests first or in the same commit.

Then prove binding/delivery parity.

Minimum relevant gates for this work are expected to include:

```bash
cargo fmt --all --check
cargo test -p entroly-engine
```

Run the appropriate native/PyO3 suite for `entroly-core` changes.

For Python Work Graph changes, run the focused tests including the applicable files under:

```text
tests/test_work_graph_*.py
tests/test_verified_handoff.py
tests/test_no_match_honesty.py
tests/test_release_surface.py
```

For npm/WASM Work Graph changes:

```bash
cd entroly-wasm
npm test
```

Also run any broader repository gates required by `CLAUDE.md` and CI for the files touched.

Do not report a command as green unless you actually ran it against the exact current head/environment. Keep failures and skips visible.

## 14. Cross-language parity tests are mandatory

For shared semantics, add tests that prove Python/PyO3 and Node/WASM agree on:

- canonical commitments/hashes;
- status inference;
- merge behavior;
- graph import/export;
- receipt/handoff integrity;
- content digests;
- memory/routing representation if added;
- conflict detection;
- clean-repo null control;
- malformed/tampered input rejection.

If Python and npm can disagree without a Rust compile/test failing, ask whether the semantic rule belongs in Rust.

## 15. Performance constraints

Do not turn every Work Graph query into an unbounded full-graph scan.

Existing code has an adjacency index/materialized state for bounded traversal. Continue in that direction.

Measure before adding caches. Any cache must be invalidated by graph revision/commitment or otherwise have an explicit correctness key.

Key production concerns:

- append/event apply latency;
- rebuild/import latency;
- resume traversal latency;
- coordination conflict latency;
- state-file size growth;
- event amplification from passive polling;
- Python<->Rust JSON serialization overhead;
- WASM boundary overhead;
- large dirty-repo behavior.

Passive `state` inspection should not create endless duplicate events. Explicit refresh/resume/handoff may append evidence only when observations materially differ/deduplicate deterministically.

## 16. Release/packaging constraints

PR #352 also changes native-core distribution assumptions. Do not break release parity while working on architecture.

Review all applicable surfaces after changes:

- `entroly-engine` Rust crate;
- `entroly-core` PyO3 package/wheels;
- Python package;
- CLI;
- MCP server;
- `entroly-wasm` npm package;
- JS root exports and TypeScript declarations;
- Docker/Homebrew if affected;
- README/docs/capabilities;
- release-surface tests.

The branch currently makes native `entroly-core` required for correct Python query-conditioned selection and adds musl wheel publication. Treat packaging failures as production failures, not documentation details.

## 17. Do not do these things

Do **not**:

1. duplicate Work Graph inference in Python/JS;
2. create a second memory system unrelated to `NodeKind::Memory` without proving why;
3. create a second routing database unrelated to Model/ModelExecution/RoutedTo;
4. mark agent statements verified;
5. infer task intent from filenames/branch names alone;
6. upload repo contents or telemetry to make discovery easier;
7. poll Git/network aggressively in the core semantic engine;
8. make advisory leases hard locks;
9. trust recovered handoff text as executable instructions;
10. hide test failures/skips;
11. claim universal hallucination detection;
12. add a Python fallback that silently changes semantics when the Rust engine is missing;
13. silently change persisted schema meaning;
14. remove bounded-input defenses for performance convenience;
15. rewrite stable existing Context Engine algorithms merely to make the architecture look cleaner.

## 18. Suggested implementation sequence

Work in small reviewable commits. A good order is:

### Phase A — exact state verification

- rebase/compare PR #352 with current `main` if necessary;
- run current Rust/Python/WASM Work Graph tests;
- record exact failures before changing code;
- inspect exact-head CI and do not inherit green status from an older SHA.

### Phase B — Rust integration schema

- add the minimum Rust types/API needed to link context receipts/evidence/verification to workstream/session/model execution;
- add deterministic serialization/commitment tests;
- expose only the required functions through PyO3/WASM.

### Phase C — Context Engine bridge

- make context selection able to consume a bounded Work Graph scope;
- ensure critical failure/verification evidence is not casually compressed away;
- emit receipt/result evidence back into the graph;
- keep compression/recovery exactness intact.

### Phase D — Trust Engine bridge

- connect claim support/contradiction/verification/receipt integrity/recovery integrity to Work Graph evidence;
- add stale-SHA and contradictory-agent tests;
- preserve fail-closed status rules.

### Phase E — memory

- implement provenance-bearing memory observation/query semantics in Rust;
- add supersession/staleness behavior;
- expose thin Python/WASM APIs;
- prove parity.

### Phase F — routing

- implement explicit model-routing decision/outcome representation;
- link model executions to workstreams/receipts/verifications;
- start deterministic/inspectable;
- expose thin APIs and tests.

### Phase G — user experience

- improve autodiscovery/first-arrival flow;
- add missing CLI/MCP operations only when the underlying contract is stable;
- make cross-agent conflict/handoff/resume understandable without exposing graph internals unnecessarily.

### Phase H — production gauntlet

- multi-process Python contention;
- Node + Python same-repo state convergence;
- Claude-like agent -> Codex-like agent handoff fixture;
- two parallel agents with overlapping and non-overlapping scopes;
- dirty repo with no task intent;
- clean repo null control;
- tampered state/receipt;
- stale checkpoint;
- stale CI vs current head;
- interrupted process holding stale lock;
- large changed-file set limits;
- prompt-injection text inside recovered decision/memory;
- exact digest parity for unstaged content;
- package install/build on supported surfaces.

## 19. Definition of done for this architecture stage

This stage is complete only when all of the following are true:

- Work Graph semantics are Rust-owned and cross-runtime consistent;
- Context Engine can consume work scope and emit receipt/evidence linkage;
- Trust Engine can attach verification/claim/integrity outcomes to work state without overclaiming;
- cross-agent autodiscovery works from durable repo evidence even without an explicit previous handoff;
- explicit handoff remains stronger and integrity-bound;
- memory is provenance-bearing and scoped;
- routing outcomes are recorded and inspectable;
- Python and npm expose equivalent semantic capabilities where supported;
- overlapping agent work is surfaced without hard locking;
- clean repos do not hallucinate unfinished tasks;
- recovered text remains untrusted data;
- tampering is detected;
- exact-head tests/CI are green or remaining failures are documented as real blockers;
- public documentation describes what is actually implemented, not the target diagram.

## 20. Working style for Claude

Act as a principal product engineer + principal systems engineer + Rust/Python/Node production engineer.

Before each substantial change:

1. read the relevant Rust semantic module and both bindings;
2. identify the invariant being extended;
3. write/adjust the Rust semantic test;
4. implement the smallest shared semantic change;
5. expose it through thin bindings;
6. add Python/npm parity tests;
7. run targeted tests;
8. run broader gates applicable to the touched surfaces;
9. inspect the diff for duplicated semantics, accidental network I/O, trust upgrades, unbounded input, and packaging drift;
10. commit with a message that states the invariant/outcome, not just the filename changed.

Do not stop at a design document if code can safely be implemented and verified. Conversely, do not manufacture code merely to check a box when the existing implementation already satisfies the invariant.

## 21. Final product principle

The reason this architecture matters is not “graph technology.”

Entroly should let any AI agent enter an existing repository and answer, with evidence:

- **What work is actually in progress?**
- **What changed?**
- **Who/what worked on it?**
- **What is unfinished or blocked?**
- **What evidence supports that?**
- **What was already decided?**
- **What context can be safely compressed?**
- **What must be preserved/recoverable?**
- **What claims are verified, contradicted, unsupported, or unknown?**
- **Where should the next model/agent continue?**
- **Can another vendor resume without trusting the previous vendor's prose?**

That is the product: **an evidence-backed AI Work Graph connected to context assurance and trust, with one Rust semantic engine and thin Python/npm orchestration.**
