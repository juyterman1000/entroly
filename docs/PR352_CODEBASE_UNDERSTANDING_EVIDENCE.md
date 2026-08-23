# PR #352 — Codebase understanding evidence

> Required by `docs/PR352_DEEP_CODEBASE_AUDIT_GATE.md` and the PR #352 merge gate.
> This replaces the historical audit snapshot that predated the Work Graph / Context / Trust consolidation.

## Verdict

**Semantic/codebase understanding gate: SATISFIED for the PR #352 changed product closure.**

This is not permission to merge by itself. PR #352 remains draft until normal GitHub CI,
Deep Dogfood, packaging, user-journey, trust, and exact-head checks are green on the
final connector-authored SHA.

## Audited revisions

```text
branch                 integration/workgraph-production-20260817
release candidate      1.0.79
product parent         4a32fcdd58858abcb43831a4f8e6e0caa9e3d900
finalization run head  9cd527572998b4837fd1e984d6a4ea9375c564f9
clean finalized parent 252d58116427305ea3c068162123d32f353b4a1d
base main              51357eca17377669b1c3d4ec4fdf832e51baf406
```

The clean finalized parent contains the reconciled ownership matrix, refreshed understanding
evidence, 1.0.79 release metadata, and no temporary PR352 writer workflows. This document-only
connector commit intentionally changes no product semantics; its head is the exact SHA on which
normal GitHub CI and dogfood must establish final release readiness.

The 1.0.79 release synchronizer raised the public package and native-core minimum
together, so installs can no longer resolve to the published 1.0.78 core that lacks
Work Graph symbols.

The PR #357 continuation audit began at remote/local base
`281a1496213b451b4c927ce20f94d3d7cf4d0355`. Its verified product tranche is
committed and pushed as `77e7106c438470c63584a88495a70255ffbd284e`.
That product SHA is an implementation anchor; GitHub CI still has to certify the
eventual final branch SHA after this evidence update.

## Repository inventory / ownership

```text
repository files classified 1640
unknown ownership          0
review-required            29
partial Rust parity         0
unexposed engine modules    0
```

The machine-generated source of truth is `docs/OWNERSHIP_MATRIX.md`.
`review-required` means a Python computation module is explicitly classified for a
later ownership decision; it is not an unknown or forgotten file. No changed PR #352
semantic owner is left in that queue.

`entroly-engine/src/simhash_wide.rs` is explicitly classified as an internal primitive.
It is not exposed merely to make a parity number green; there is no public Python/npm
contract for it today.

## Changed semantic closures read and reconciled

The production-relevant closure changed by PR #352 has been read/reviewed across:

- `entroly-engine`: Work Graph, engine contracts, Trust facade/EICV bridge, coordination
  index, repository graph identity/projection seams, cache/LSH lifecycle, SAST privacy,
  Cognitive Bus/NKBE consolidation, and engine module ownership.
- `entroly-core`: PyO3 Work Graph, Context and Trust bindings plus root registration.
- `entroly-wasm`: WASM Work Graph, Context and Trust bindings plus npm runtime wrappers,
  package exports and drift guard.
- Python delivery: `work_graph.py`, repo observation, durable store, content digest, CLI,
  MCP, native capability gating, session/handoff adjacency, and host engine lifecycle.
- Node delivery: Work Graph repo/store/digest/continuity, Trust wrapper, Context scope,
  root exports and package tests.
- Packaging/release: Python 3.10-3.14, native wheel build path, npm/WASM, OpenClaw, MCP,
  release metadata and minimum native-core capability.

## Architecture decisions verified

1. Shared Work Graph / Context / Trust meaning is Rust-owned. Python and Node bindings
   are transport/orchestration surfaces rather than second semantic implementations.
2. Repository artifacts use stable Work Graph identities. Repository intelligence projects
   into those identities; the Rust `depgraph` remains a fragment-level selection graph, a
   different domain, rather than a second repository-work identity authority.
3. Explicit handoff is stronger than inferred recovery. Clean repositories remain the
   null control and do not manufacture unfinished tasks.
4. Completion still requires explicit completion plus passing verification; contradictory
   evidence fails closed.
5. Context integration is bounded/reference-based; raw prompts/conversations are not copied
   into Work Graph state.
6. Trust results are evidence-bounded (`supported` / `unsupported` / `unknown`) with
   cryptographic evidence commitments; no universal hallucination claim is made.
7. Persisted Work Graph document schema is unchanged by the lifetime optimizations. Derived
   event-id and incremental commitment state is rebuilt from the append-only log.
8. Host Python GC policy is untouched by the library.
9. Secret-category SAST findings contain no source bytes from a secret-bearing line.
10. Default selection is deterministic; exploration remains an explicit/tuned behavior.

## Production defects closed during this integration

- npm Work Graph load syntax failure.
- Windows worktree digest same-file false negative.
- Python 3.10 `tomllib` collection break.
- native capability gate accepting a core with no Work Graph symbols.
- duplicate Cognitive Bus/NKBE semantics in binding crates.
- Python/npm Context and Trust delivery gaps.
- WASM Trust runtime trap caused by unsupported wall-clock timing.
- Work Graph coordination O(N^2) candidate generation hot path while preserving exact
  overlap semantics as authority.
- Work Graph append duplicate scan and full-history commitment recomputation; incremental
  state is byte-for-byte equivalent to canonical serde commitments.
- cache semantic-index slots surviving eviction/GC and growing with lifetime admissions.
- process-global Python GC mutation from engine construction/hot paths.
- secret-bearing SAST line prefix leakage.
- stochastic default exploration undermining reproducible receipts.
- raw-JSON-only `entroly-work` human output.
- release candidate allowing a native core that predates the shipped Work Graph.

## Validation evidence already established on product commits

Dedicated guarded slices refused to commit until their targeted gates passed. Recorded
results include:

- Context/Trust: Rust tests/clippy, PyO3, Python, WASM/npm runtime, semantic parity and
  ownership convergence.
- Production safety/lifetime: focused falsification tests, full engine suites in normal and
  Python-feature modes, clippy, wasm32 build, PyO3 host-GC tests, Work Graph Python tests,
  npm parity, and unchanged persisted schema.
- Incremental Work Graph commitment: 512 successive appends matched the previous canonical
  serde SHA-256 exactly; 2,048-event membership/import tests passed.
- Release 1.0.79: allowlisted release synchronizer and `tests/test_release_surface.py` passed
  before the version commit was pushed.

Normal exact-head CI remains authoritative for merge readiness.

## Public user journeys traced

- base pip/source install -> import -> native capability gate
- `entroly-work state` human output and `--json` automation output
- claim -> advisory lease -> coordination report
- interrupted repo -> passive refresh -> resume view
- explicit agent A -> agent B graph-bound handoff
- Work Graph MCP state/claim/resume/handoff with recovered content fenced as untrusted
- Python Work Graph -> Context scope
- Python and npm Trust assessment over the same Rust semantics
- npm Work Graph repo/store/digest/continuity path
- proxy/provider, SDK/receipts/compression, OpenClaw and installed MCP paths through Deep Dogfood

## Persisted-state review

Work Graph import validates schema version, event bounds, event IDs, duplicate IDs,
referential/capacity invariants, rebuild integrity, and whole-graph commitment. Tampered or
mismatched state fails closed. Cache lifetime changes did not alter its serialized snapshot
contract. Handoff/content digest checks remain bound to current repository content.

## Known non-blocking boundaries

- `review-required` ownership entries remain explicit future migration decisions outside the
  PR #352 changed shared-semantic closure; they are not unknown files.
- host-specific services such as MCP FeedbackJournal/SkillEngine vault orchestration may be
  assembled at the host boundary. This does not duplicate Work Graph/Context/Trust semantics.
- `depgraph` uses fragment identities for selection-time relationships; it must not be
  mechanically collapsed into repository artifact identity without a semantic migration.
- `simhash_wide` is internal until a calibrated production consumer adopts it.

## Final merge rule

PR #352 must remain draft until the connector-authored final SHA has all required normal CI
and dogfood checks green. No older-SHA success satisfies that rule.

---

## 22. Finding reconciliation against integration head `f734516a`

PR #356 merged on 2026-08-19 and 77 further commits landed on
`integration/workgraph-production-20260817`. Several of them act directly on
findings recorded in this document. This section states which are closed, which
are not, and — where closed — whether the fix was verified rather than assumed.

**Method.** Each finding re-checked against the shipping code at `f734516a`, not
against commit messages. Where a finding carried a measurement, the measurement
was re-run.

### Closed and verified

**G25 — cache semantic-index and slot-table leak. Fixed.**

The repair matches the shape the finding specified: a `free_slots: Vec<usize>`
free-list, a `slot_by_hash` reverse map so a removal can find its slot, and
`semantic_index.remove(entry.query_simhash, slot)` on the removal path, with
`allocate_semantic_slot` reusing a freed slot before extending the vector.

Re-verified with the identical probe that produced the original measurement —
`max_entries: 16`, 2,000 stores:

| | `live_entries` | `slot_to_hash` |
|---|---:|---:|
| at the time of the finding | 16 | **343** |
| at `f734516a` | 16 | **16** |

Two regression tests were added, covering both paths the finding named:
`semantic_slots_stay_bounded_under_long_eviction_churn` and
`gc_recycles_slots_and_snapshot_import_rebuilds_private_indices` — the latter
also covering the snapshot-import rebuild noted in section 20. An
`assert_semantic_slot_invariants` helper asserts `slot_by_hash.len() ==
entries.len()` and that slots and free-slots partition the table.
`cargo test --lib cache`: 47 passed.

**G29 — secret redaction leaked a line prefix. Fixed, more strongly than proposed.**

The finding suggested delimiter-based truncation preserving the key name. The
repair is unconditional full redaction:

```rust
let safe_content = if rule.category == "Hardcoded Secrets" {
    "[REDACTED — secret-bearing line]".to_string()
```

with the reasoning that rule, category, path and line metadata already carry the
debugging context. That removes the leak class entirely rather than narrowing it,
which is the better call for a security scanner — the delimiter approach would
still have depended on correctly identifying the delimiter in every syntax.

### Partially closed

**G8 — the published core cannot serve the Work Graph.** The tree is now at
`entroly-core` 1.0.79 with `WorkGraph` exported, and a release candidate is
staged. PyPI still serves **1.0.78**, which has no `WorkGraph` symbol. So the
gap is prepared, not closed: until 1.0.79 publishes, the Work Graph surface
remains undeliverable to installed users and the associated tests can only skip.

### Still open at `f734516a`

Verified individually against the shipping code:

| Finding | Check | State |
|---|---|---|
| G19 — cost model adds dollars to seconds; `$0.01/token` literal wins the `max()` | all three sites unchanged (`cache.rs:110`, `:232`, `:888`) | open |
| G21 — "Thompson Sampling" performs no sampling | `rand` still absent from `entroly-engine/Cargo.toml` | open |
| G23 — "lazy greedy" is neither lazy nor faster | `_last_computed_at` still `// (structural, write-only)`; `select_victim_lazy` still dispatched above 64 entries | open |
| G24 — hit predictor trained but never consulted | still exactly one `predict(` call site in production, inside `update()` | open |
| G26 — taint matching is substring, not token | `lower.contains(var.as_str())` unchanged at `sast.rs:2163`, `:2189` | open |
| G27 — Python docstrings scanned as code | `trimmed.len() > 3` gate unchanged at `sast.rs:2226` | open |
| G31 — four K8S rules can never fire | `languages: &["k8s"]` unchanged; `detect_lang` still has no Kubernetes branch | open |

That grouping is not arbitrary. The two closed findings were both mechanical
defects with a single correct answer — a leak and an over-broad slice. The seven
open ones are all cases where this document declined to change behaviour because
the fix required a calibration decision, a labelled corpus, or a scoring policy.
They remain open because they were correctly identified as decisions rather than
repairs, and the decisions have not been made yet.

### `scripts/ownership_matrix.py` was extended

The generator added in this audit gained two things worth recording, because both
correct real limitations in the original:

* `engine_dependency_graph()` and `transitive_engine_references()` — the original
  counted only *direct* references from `entroly-core` and `entroly-wasm`, so an
  engine module reached only through another engine module was reported
  unexposed. Reachability is now closed transitively from each delivery crate's
  roots, which is the correct question.
* `ENGINE_INTERNAL_PRIMITIVES`, carrying `simhash_wide` with an explicit note
  that it is a retained internal primitive with no public contract today. That
  converts the "dead code" reading in section 16 into a stated ownership
  decision, which is the right disposition — the point of flagging it was to
  force exactly that choice.

---

## 23. Master prompt section 24 — pre-change baseline, recorded

Section 24 requires that "pre-change baseline behavior/tests were recorded so
regressions are distinguishable from pre-existing failures". This is that record.

**Method.** A detached worktree at `09a98a5b` — the merge-base of the audit
branch with `integration/workgraph-production-20260817`, i.e. the exact state
before any of this session's work. `entroly-core` was built *from that tree*
rather than reusing a later engine, installed to an isolated prefix, and the full
suite run under it. Anything else would compare a new engine against old tests.

### Result: the baseline suite cannot be collected

```
ERROR tests/test_work_graph_entrypoints.py
ERROR tests/test_work_graph_packaging.py
!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!
1 warning, 2 errors in 109.10s
```

Cause, in both files:

```
import tomllib
E   ModuleNotFoundError: No module named 'tomllib'
```

`tomllib` entered the standard library in Python 3.11. On Python 3.10 — which
this project supports and which this environment runs — the import fails, pytest
aborts collection, and **zero tests execute**. Not "some tests fail": the suite
produces no result at all.

### What this establishes

This is finding **G1** from the beginning of this audit, seen from the other
side. G1 was fixed by adding `tests/pyproject_compat.py`, a tomllib-free
pyproject parser that both files now use. The baseline confirms that fix was
load-bearing rather than cosmetic.

Three things follow, and they are the reason section 24 asks for this:

1. **Every test count reported in this document is downstream of that fix.** The
   4,025-passed figure, the 470 engine tests, the 37 work-graph tests — none of
   them were obtainable at the branch point on this interpreter. They are not
   comparable to a baseline number because there is no baseline number.
2. **No regression from this session can be confused with a pre-existing
   failure**, because the prior state had no runnable test result on Python 3.10
   to regress from. The distinguishability section 24 asks for is achieved
   trivially, in the least satisfying way possible.
3. **The branch point was not testable on the project's own supported minimum.**
   That is a more interesting fact than any pass count. A suite that aborts
   collection on a supported interpreter has been green only on interpreters
   where it happened to import, and CI would have to be running 3.11+
   exclusively for this to have gone unnoticed.

### Honest limitation

A baseline that produces zero results is a weak baseline. It bounds the claim in
one direction only: nothing this session did *broke* a previously-passing test on
Python 3.10, because no test previously ran on Python 3.10. It says nothing about
Python 3.11+, where the baseline would collect and where a true before/after
comparison is still missing. Recording that gap explicitly is better than
implying the item is fully discharged.

### Checked — the 1.0.79 release ordering is enforced, not a hazard

The Verdict section states that raising the minimum "so installs can no longer
resolve to the published 1.0.78 core that lacks Work Graph symbols". That is
accurate, and it raises an obvious question worth answering rather than leaving
implicit, because the situation it creates looks alarming:

```
pyproject.toml (x3)          entroly-core>=1.0.79,<2
entroly/pyproject.toml (x3)  entroly-core>=1.0.79,<2
native_status.py             MIN_ENTROLY_CORE_VERSION = "1.0.79"
PyPI                         entroly-core 1.0.78   (1.0.79 not published)
```

So at this moment no `entroly-core` satisfying the pin exists publicly, and a
publish of `entroly` alone would break every install at dependency resolution.

**It is sequenced correctly.** `entroly-publish.yml` declares:

```yaml
publish-core:
  needs: [release-metadata, quality-gate, release-anchor]
  uses: ./.github/workflows/publish-core-wheels.yml

publish-pypi:
  needs: [release-metadata, build-and-push, quality-gate, publish-core]
```

`publish-pypi` cannot start until `publish-core` completes, so the core wheels
reach PyPI before the package that depends on them. The pre-publish state is
therefore expected rather than broken, and the fail-closed direction is the right
one: an unsatisfiable pin fails loudly at install time, where the alternative —
a floor low enough to admit 1.0.78 — would silently install a core with no
`WorkGraph` and defer the failure to first use.

Recorded as a negative result. The concern was worth checking and is not a
defect; leaving it unstated invites someone to re-raise it later from the same
observation.

### Note on this document's history

The evidence artifact was rewritten during the integration. Sections 1–21 of the
version this audit produced — the detailed G1–G32 findings, the section 19
scenario map, the section 22 parity answers, the closure accounting — were
replaced by the reconciled summary that now opens this file. That history remains
in the branch: the findings are recoverable from the commits on
`audit/pr352-deep-codebase-gate`, which merged in PR #356.

The summary's closed-defects list is consistent with what was independently
verified here — it names the Python 3.10 `tomllib` collection break (G1), the
cache semantic-index slot growth (G25) and the SAST line-prefix leakage (G29),
all three of which section 22 re-checked against the shipping code. It does not
claim the seven findings section 22 lists as open, which is the correct
disposition for them; section 22 supplies the record they would otherwise lack.

---

## 24. PR #357 focused closure against the master plan

Section 22 is intentionally a snapshot of `f734516a`; its findings must not be
read as the state of the later audit branch. PR #357 closes the following
bounded part of the master plan:

| Master-plan item | Exact closure in PR #357 |
|---|---|
| P0 exact-head truth | CI and Deep Dogfood now run for `integration/**` pull requests. Candidate CI builds and installs the exact-head native core. The dependency audit excludes only the two local exact-head project artifacts (`entroly` and `entroly-core`) while continuing to audit the complete pinned third-party environment. |
| Section 8, ContextReceipt | Rust owns canonical construction, bounds, serialization, identity/commitment and verification. PyO3 and WASM are adapters, with Python and npm golden/parity/tamper tests. |
| Section 9, RecoveryHandle | Rust owns recoverability and integrity semantics. Python and npm exercise the same golden contract and fail-closed verification behavior. |
| Section 10, MemoryRecord | Rust owns provenance, evidence/trust/freshness admissibility, supersession/contradiction semantics, canonical identity and verified parsing. Verified memory requires evidence; incomplete producer provenance, empty commitments, invalid replay time and self-recommitted invalid payloads fail closed. Python and npm parity tests use the package delivery surfaces. |
| P1 routing, execution and verification | Rust now owns bounded, canonical `RoutingDecision`, `ModelExecutionOutcome` and `VerificationRecord` contracts. Verified parsing rejects unknown fields and tampering. PyO3/Python and WASM/npm use the Rust builders and permanent golden anchors `route_66d4c04a18b4e70f`, `outcome_a130681ddd63dc84` and `verify_4e1487e3d6e73b36`. Provider HTTP remains host orchestration. |
| P3 temporal Trust | A verification binds the exact outcome commitment and repository head. Freshness is recomputed against the current head, and explicit dependency invalidation propagates to derived verification. Stale or invalidated verification cannot upgrade a workstream; a current failed verification blocks it. |
| P4 closed execution loop | `record_execution_chain` validates WorkScope/task/workstream, ContextReceipt linkage, route/outcome identity, exact repository head and verification time, then appends route, execution and verification materialization in one atomic WorkEvent. Durable Python and npm stores expose the same Rust-owned transition. |
| P6 WorkContinuationProof | Rust owns explicit-handoff and evidence-bounded reconstructed continuation. Proof manifests auto-discover scoped receipt, memory, routing, outcome, verification and recovery commitments; caller-supplied values not evidenced by the selected workstream fail closed. Explicit CLI/MCP handoff returns the proof automatically, while npm provides an additive handoff-plus-proof operation. A reconstructed proof has no invented previous-agent identity or handoff commitment, records `unknown:previous-agent-intent`, is bound to the exact graph commitment/head and refuses completed work. Golden anchor: `continuation_53eba6ee3a52be48`. |
| P9 explainability and replay | The append-only `WorkGraph` export is the deterministic replay bundle: it contains the ordered committed WorkEvents whose evidence operations carry ContextReceipt, memory, routing, execution, verification and policy references. `from_json` revalidates every event ID, bounds, references and the aggregate graph commitment before materialization; Python and npm round-trip/tamper tests exercise that delivery path. No hidden model reasoning is required. |
| P9 verified outcome learning | Existing production RAVS `OutcomeBridge` accepts bounded external test/CI/user outcomes, rejects weak agent self-report, corrects the bounded `OnlinePrism` posterior and that posterior is consulted by context selection. The new canonical execution event makes the exact route/outcome/verification chain auditable without replacing this host-side learning loop. |
| Section 12, G19 | Cache utility is expressed in dollars: tokens times dollars-per-token plus latency milliseconds times an explicit dollars-per-millisecond coefficient. Configured model prices drive eviction; the legacy serialized estimate cannot override them. |
| Section 12, G21 | The production mechanism is documented and tested as deterministic Beta-posterior scoring. The public compatibility name remains, but the false stochastic/Thompson-sampling claim is removed rather than adding ornamental randomness. |
| Section 12, G23 | The write-only generation state and false lazy-heap path are removed. Victim selection is an explicit deterministic direct scan, with the bounded tradeoff named and tested instead of claiming an unmeasured optimization. |
| Section 12, G24 | The trained hit predictor is consulted by production admission, combined with posterior and normalized cost signals. Admission, later hits and unhit capacity eviction now use the same stored feature vector; lookup misses do not fabricate zero-valued entropy/cost labels. Regressions prove that the prediction changes admission and that a genuine non-reuse outcome changes the prediction. |
| Section 13, G26/G27/G31 | Taint propagation uses identifier boundaries (including non-ASCII-safe traversal), Python docstrings are excluded as documentation rather than executable code, and Kubernetes manifests reach the Kubernetes rule set. |
| Section 23, parity | ContextReceipt, RecoveryHandle, MemoryRecord, RoutingDecision, ModelExecutionOutcome, VerificationRecord and WorkContinuationProof have Rust-owned semantics exercised through PyO3/Python and the npm package root/WASM surface, including golden identities, errors, bounds and tampering. |
| Section 24, product UX | `entroly-work resume --to-agent AGENT` and the matching MCP `work_resume(to_agent=...)` operation reconstruct unfinished work and return its continuation proof without asking users to manually assemble graph nodes, commitments or a handoff. Explicit handoff returns the same flagship proof. MCP also records canonical context, memory and execution-chain contracts through bounded host orchestration into Rust-owned validation/state transitions. |
| Clean explicit work continuity | A clean explicitly claimed task remains resumable even when it has no changed paths or failures: Python CLI/MCP and npm use the selected workstream's stable task IDs as the evidence-backed continuation fallback. Exact installed-wheel and installed-tarball journeys both seal a non-empty continuation proof, while hostile or malformed collection shapes remain bounded and fail safe. |
| Large dirty repositories | One observation accepts up to 16,384 complete file changes and atomically splits it into deterministic events of at most 512 changes. Consecutive identical passive polls collapse without history scans, while A-to-B-to-A remains auditable. Context scopes expose bounded inline prefixes plus total counts and commitments to complete path/evidence sets. |
| Scenarios C/K/L, explicit handoff | A permanent production-store journey creates multi-file/multi-symbol work, attaches decision and passing-test evidence plus a recoverable ContextReceipt, seals a Claude-to-Codex handoff and continuation proof, then loads the same durable graph through an independent receiving store. The receiver verifies graph/content integrity and resumes the exact bounded work/evidence/recovery scope. Mutating the receipt fails self- and graph-verification; changing worktree bytes at the same Git head preserves the original artifact's self-integrity but makes graph-bound verification and proof construction fail closed. |
| Scenario G, active symbols and rename lineage | The production Python store enriches exact worktree content identity, incrementally indexes the repository, and atomically projects the active changed files, their symbols, and one-hop import boundary into the Rust-owned Work Graph. `WorkScope` exposes bounded symbol IDs with total counts and full-set commitments. A rename records the new file as superseding the old file, preserves `renamed_to` lineage on the old node, keeps only the active new path in `changed_paths`, and exposes the current symbol rather than a stale predecessor. The persisted repository node carries the projection bounds, and a passive observation plus its scope projection is one idempotent poll group across export/import while A-to-B-to-A remains auditable. Direct Python and npm store updates now derive content digests themselves instead of depending on CLI/MCP wrappers. |
| Scenario N, measured scale | The executable release gates now measure the full 2,000-file index, one-file active incremental index, bounded symbol/dependency projection, Rust event apply/rebuild/resume/context/coordination, PyO3 and WASM serialization, eight-way durable-store contention, passive-poll amplification and state size. Production active projection catalogues dependency targets without reading/parsing unchanged source. On the recorded Windows Python 3.10 run, the one-file path improved from 32,970.3747 ms to 804.6548 ms while cataloguing all 2,000 paths and parsing exactly one changed file; eight writers preserved all eight events in 323.8932 ms. The matching full npm Node/WASM 2,000-file/500-edit/100-poll run measured 0.7495 ms p95 append, 133.4177 ms import, 38.1721 ms resume, 11.9771 ms summary p95, zero poll growth and the same 504-event shape. |
| Scenario O, generated/vendor trees | Repository source discovery excludes VCS/cache/venv, `node_modules`, `vendor`, `dist`, `build`, Rust `target` and bytecode trees before file-budget accounting. A gauntlet fixture places 150 generated Python files across those trees under a two-file limit and still indexes exactly the two first-party source/test files without truncation. |
| Scenario P, Python/Node convergence | One end-to-end test uses the real PyO3 and WASM stores against the same repository/state root: Python writes, Node verifies the exact prior commitment and appends a leased continuation, then Python reloads the Node commitment and byte-identical canonical export. The exact-head WASM CI job builds both runtimes and runs this test. |

### Deep Dogfood A-T certification matrix

The focused executable matrix reports `97 passed, 1 skipped`; the skip is the
Windows symlink-permission fixture, with the non-symlink security and durability
cases passing. The complete exact-native Python suite reports `4059 passed, 50
skipped, 3 xfailed`; Rust reports `564` engine tests, `112` core tests with five
measurement probes ignored, and `12` WASM crate tests. The full npm suite passes
all 42 engine E2E cases, Work Graph store/recovery/root-export contracts, every
cross-runtime parity vector, and its 2,000-file performance gate.

Scenario T's local package/user journey is complete. At exact package candidate
`0edd95ed`, fresh Python/native wheels installed as version 1.0.79 with `pip
check` clean, SDK and real MCP initialize/tools-list dogfood passing, and a clean
claim producing both reconstructed and explicit Claude-to-Codex continuation
proofs. The installed npm tarball proves runtime/root exports/types and the same
clean-task handoff fallback. Later commits change only CI/test/evidence files;
the final exact-SHA cross-platform CI matrix remains the independent merge gate.

| Scenario | Status | Permanent evidence |
|---|---|---|
| A, first-time dirty repo | Pass | `test_replacement_agent_recovers_unclaimed_interrupted_work_from_repo`; Rust `dirty_repo_creates_in_progress_workstream`. Changed artifacts are surfaced with no invented task prose. |
| B, clean null control | Pass | `test_clean_repo_is_null_control` and `test_checkpoint_can_name_existing_git_work_but_not_resurrect_clean_repo`; matching Rust and npm null controls. |
| C, explicit handoff | Pass | `test_receiving_agent_verifies_explicit_handoff_and_detects_later_edit` loads an independent receiver from durable state and verifies multi-file/symbol, decision, test, outstanding-work and recovery scope. |
| D, interruption without handoff | Pass | Bidirectional receiver parameterization in `test_replacement_agent_recovers_work_when_previous_agent_never_used_entroly`, CLI recovery, and Rust `interrupted_agent_gets_evidence_bounded_continuation_without_a_handoff`. |
| E, parallel non-overlap | Pass | Process-level `non-overlapping-vendors` case and Rust `disjoint_parallel_leases_produce_no_conflict`. |
| F, parallel overlap | Pass | Process-level `overlapping-vendors` case and Rust `overlapping_parallel_leases_are_reported_but_not_locked`. |
| G, rename and symbol continuity | Pass | Production-store rename/symbol test plus Rust lineage test; only the active new path enters scope. |
| H, stale CI | Pass | Rust `stale_or_transitively_invalidated_verification_cannot_upgrade_work` and exact-version freshness contract. |
| I, contradictory claim | Pass | Rust `contradicted_claim_never_becomes_trusted_fact` and `failing_verification_blocks_work`; both evidence items remain in the graph. |
| J, tampered graph state | Pass | Rust `persisted_document_detects_tampering`, Python/Node durable-store tamper rejection, duplicate-event rejection and canonical import validation. |
| K, tampered handoff | Pass | The explicit receiver journey plus Rust/Python/npm handoff integrity mutation tests fail self- and graph-verification. |
| L, content changed after handoff | Pass | The explicit receiver journey changes exact worktree bytes without changing Git HEAD; the old sealed artifact remains self-consistent but graph verification and proof construction fail closed. |
| M, recovered prompt injection | Pass | `test_mcp_state_is_fenced_as_untrusted` stores hostile instruction-like work text, returns an explicit data fence, and reports injection matches. |
| N, large repository | Pass | Python/PyO3 and Node/WASM 2,000-file/500-edit/100-poll release gates, one-file active indexing, bounded symbol projection and contended durable writes. |
| O, generated/vendor trees | Pass | `test_generated_and_vendor_trees_do_not_consume_source_graph_budget` proves 150 irrelevant files cannot drown a two-file first-party scope. |
| P, Python/Node convergence | Pass | `test_python_node_same_repo_state_converges` compares the real PyO3/WASM store commitment and byte-identical canonical export across both write directions. |
| Q, multiprocessing contention | Pass | `test_concurrent_agent_processes_merge_without_lost_work`, stale-lock recovery, and repeated eight-writer performance runs preserve all events. |
| R, crash during persistence | Pass | Replace-boundary fault injection preserves the last committed state; failed writes leave no temporary debris and the store remains writable. |
| S, compression/recovery | Pass | CLI cross-process compression recovery and recoverable-receipt tests verify original bytes, digests, locators, corruption rejection and unavailable-material errors. |
| T, installed product journey | Pass locally; final CI gate enforced | Fresh 1.0.79 Python/native wheels and npm tarball pass clean installs from outside the checkout. SDK, actual MCP protocol, CLI claim/resume/handoff, native import, npm Work Graph runtime, root exports and shipped types are exercised. A clean task seals a proof containing its stable task ID instead of failing with empty resumable state. Exact-SHA cross-platform CI remains required before merge. |

The implementation closure is complete for the contracts and product paths
listed above. It is not yet a production-release declaration. Three gates cannot
truthfully be manufactured by source changes in a local checkout:

1. GitHub CI and cross-OS packaging must run against the eventual exact final SHA;
   source-level and local package checks do not substitute for those remote jobs.
2. The real-provider Section 18 A-J certification needs the intended vendor
   credentials and live endpoints. Local Claude-to-Codex and Codex-to-Claude
   interrupted-agent simulations prove product semantics, not vendor availability.
3. Release publication and downstream install checks must use artifacts built by
   that same final SHA. Local clean-wheel and npm-tarball installation prove the
   package shape but do not substitute for publication.

Static package reachability currently reports 302 modules, 834 import edges,
153,730 lines, 264 modules reachable from declared package entry points and 38
direct-opt-in/test/benchmark modules (13,237 lines). Those 38 are not omitted from
ownership: they are classified in the 1,640-file matrix. The static report is kept
strict so a directly tested experimental module is never silently promoted to a
normal-user product claim.

## 25. Local validation evidence for the PR #357 continuation

The product tests below cover the current PR #357 implementation lineage; the
clean-install rows were run against locally built 1.0.79 artifacts:

| Surface | Result |
|---|---|
| Python full source suite | Pre-scale closure baseline: `4085 passed, 33 skipped, 3 xfailed`; post-scale focused Work Graph/performance suites pass and a final exact-head full rerun remains required |
| Rust engine | `564 passed` with all features; Clippy clean with warnings denied |
| PyO3 core | `112 passed, 5 ignored`, plus one doc test; Clippy clean |
| WASM crate | `12 passed`; Clippy clean |
| npm package | Full npm suite passed, including 42 E2E cases, WorkGraph persistence, parity, root exports, interrupted continuity and the measured WASM scale gate |
| Measured scale gate | Current Python source plus installed exact-head PyO3 core, 2,000-file initial observation/index plus 500 edits and 100 timestamp-changing passive polls: 504 events, zero poll growth, 1.5231 ms p95 append, 91.1969 ms import rebuild, 55.2266 ms resume, 16.0257 ms PyO3 summary p95, 27,649.8125 ms cold full index, 804.6548 ms one-file active incremental index, 6.2852 ms symbol projection, 323.8932 ms for eight contended durable writes, and a 5,531,773-byte state. Scope accounted for all 2,000 paths and 3,502 evidence IDs through bounded prefixes and full-set commitments. |
| Python clean install | Built `entroly-core` 1.0.79 and `entroly` 1.0.79 wheels, installed them in an isolated Python 3.10 environment, verified native readiness and WorkGraph construction; `pip check` reported no broken requirements |
| npm clean install | Packed and installed the local npm tarball, loaded `entroly-wasm` 1.0.79, constructed WorkGraph and resolved every new package-root contract helper |
| Ownership | 1,640 tracked/non-ignored repository files classified, zero unknown ownership; local `.codex/` task state excluded |

The release rule remains unchanged: only checks attached to the final head can
certify it; skipped, missing, stale and older-head results do not count as pass.
