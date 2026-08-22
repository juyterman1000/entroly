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

## Repository inventory / ownership

```text
tracked and classified     1631
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
