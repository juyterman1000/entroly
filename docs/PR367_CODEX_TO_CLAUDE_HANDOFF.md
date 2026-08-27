# PR #367 — Codex → Claude handoff

## Scope

Continue production hardening for PR #367 (`feat: add receipt-bound code context page faults`).

Do **not** merge to `main` or publish/release without explicit user approval after a final exact-head production gate is green.

## Branches / exact state

- PR #367 branch: `fix/codebase-intelligence-integrity`
- Last fully certified PR candidate: `6966739dec47904c6bff816241f0aae1d20c0924`
- Isolated parity/hardening branch: `pr367-context-snapshot-parity`
- Last product hardening commit before this handoff doc: `768e802e8d692eb7394a3084c70df489c3162920`

The parity branch is intentionally ahead of PR #367. Do not fast-forward #367 until the parity branch is cleaned of temporary one-shot workflows and its final combined tree passes the permanent gates.

## What is already verified on PR candidate `6966739d…`

The exact PR candidate was green across the recorded production matrix, including:

- Python CI / supported interpreter matrix
- Rust tests + clippy
- PyO3/native exact-head installation
- WASM/npm Work Graph and Context Receipt parity
- Commit Identity Guard
- Onboarding Self-Dogfood
- Deep Dogfood Runtime
- User Journey Trust Gate
- QCCR Dependency Evidence Gate
- Code Intelligence Optional Extra
- competitive context-index benchmark

The previously intermittent onboarding mismatch (`simulate total_tokens_saved = 85436` vs `perf = 85437`) was traced to cold/warm native reconstruction drift. The fix rebuilds selection-relevant derived state deterministically after `load_index`, preserves persisted slot identity, excludes dependency stubs from SimHash/LSH reconstruction, rebuilds dependency state, and clears stale caches. Onboarding was rerun after the fix and passed again with the strict equality assertion unchanged.

Do not weaken that equality assertion.

## PR #367 product hardening already landed on the PR branch

Key changes include:

1. Public code-context page faults are bound to the canonical Rust `RecoveryHandle` path.
2. Weaker repository MCP/CLI fault surfaces that bypassed the handle were removed.
3. Low-level recovery helpers remain internal and are no longer advertised as package-root product APIs.
4. `context_ref` has its own bounded reference contract rather than being treated as a short ID.
5. The production-shape MCP compile→fault test no longer monkeypatches the renderer.
6. Snapshot persistence is repository-scoped and content-addressed (`wctx1.<context_sha256>`), not an inline base64 duplicate of the full context.
7. `WorkContextSnapshotStore` reuses Work Graph storage/locking mechanics and is bounded, atomic, symlink-aware, tamper-detecting, and fail-closed.
8. Snapshot stable bytes omit volatile `generation` / `command`, matching the existing context commitment scope.

## Parity branch work (`pr367-context-snapshot-parity`)

The parity branch extends this so Python and Node use the same durable snapshot format.

Implemented:

- Node/npm `WorkContextSnapshotStore`
- npm root runtime export
- TypeScript declaration surface
- package file inclusion
- real Python → Node → Python exact-byte snapshot interoperability test
- Unicode + Python numeric lexeme (`1.0`) preservation test
- tamper/noncanonical rewrite refusal
- stale generated-WASM capability detection
- permanent cross-runtime snapshot parity workflow
- shared Rust verified-context snapshot verifier in `entroly-engine`
- thin PyO3 projection
- thin WASM projection
- Python snapshot store delegates semantic commitment validity to Rust
- Node snapshot store delegates semantic commitment validity to Rust

Architectural invariant:

> Rust owns snapshot commitment validity. Python/Node own filesystem/storage mechanics only.

Do not reintroduce a JavaScript or Python implementation of the semantic commitment rule.

## Important subtlety fixed at `768e802e…`

`context_sha256` is excluded from its own hash. A raw verifier could otherwise accept a snapshot where only that field was moved to another position because the preimage remains unchanged.

Python's canonical-byte check already rejected this layout. The Rust verifier was hardened to enforce the canonical placement of `context_sha256`, with a regression test. Rust tests + clippy passed before the product commit was pushed.

## Current known blocker: npm tarball gate harness

The combined parity run already proved:

- exact-head Python/native install: PASS
- exact-head WASM build: PASS
- Python↔Node snapshot parity: PASS
- full npm package tests: PASS
- Work Graph npm performance gate: PASS
- root exports: PASS
- Context Receipt / Recovery Handle / Memory / routing/freshness/continuation parity: PASS

The only failure in that run was the tarball assertion harness.

Cause: the workflow used:

```bash
npm --prefix entroly-wasm pack --dry-run --json
```

but npm still tried to open repository-root `package.json`, producing ENOENT.

Do **not** weaken the tarball assertion. Fix the harness to execute from inside `entroly-wasm`, for example:

```bash
(
  cd entroly-wasm
  npm pack --dry-run --json
)
```

Then keep checking that the publishable tarball contains at least:

- `js/work_context_snapshot_store.js`
- `js/work_context_snapshot_store.d.ts`
- `index.js`
- `index.d.ts`

## Remaining production work

1. Fix and rerun the npm tarball gate on the exact parity head.
2. Run the final combined PyO3↔WASM/raw verifier + Python→Node→Python snapshot parity test after the canonical-placement hardening.
3. Add/verify unusual filesystem/source-byte handling (especially Python surrogate-escape edge cases) rather than assuming UTF-8 behavior matches across runtimes.
4. Remove all temporary one-shot PR367 wiring/hardening workflows. Keep only the permanent production parity workflow.
5. Audit the final parity-branch diff and commit identities; no CI-bot-authored product commits should remain in PR history.
6. Fast-forward or otherwise bring the clean, proven parity tree into `fix/codebase-intelligence-integrity`.
7. Run the entire PR #367 exact-head production matrix on that **same final SHA**.
8. Require all required gates green before marking the PR ready for review.
9. Keep it draft / do not merge until explicit user approval.

## Production-readiness standard

Do not claim the full product is production-ready merely because a subsystem is green. Final certification needs the same SHA to prove Python, Rust, PyO3, WASM/npm, MCP, packaging, exact recovery, tamper/stale-state handling, concurrency, dogfood/user journeys, and cross-runtime snapshot recovery.

Repository reality and exact-head test evidence override this document if they disagree.
