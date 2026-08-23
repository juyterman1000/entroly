#!/usr/bin/env node
'use strict';

// One receipt contract, identical in every runtime — the Node half.
//
// The Python half is tests/test_context_receipt_parity.py and the Rust anchor is
// engine_contracts::tests::GOLDEN_RECEIPT_COMMITMENT. All three assert the same
// constant because all three call the same engine function, so byte-equality is
// checkable in each runtime on its own: drift in one breaks its own test rather
// than silently diverging from the others.
//
// This exists because receipts were Python-only. A Node caller could join a
// workstream through the Work Graph and still not prove what evidence it
// received — the Python receipt's reproducibility_hash covers selected text, so
// no other runtime can reproduce it. That hash is fine for a host presentation
// record; it just cannot be the shared contract.

const assert = require('assert');
const wasm = require('./');
const { contextReceiptBuildJSON,
        contextReceiptVerifyJSON,
        contextReceiptCommitment,
        contextReceiptGraphRefJSON,
        contextReceiptSchemaVersion } = wasm;

// Must equal engine_contracts::tests::GOLDEN_RECEIPT_COMMITMENT.
const GOLDEN_COMMITMENT =
  '672457349ba403bc885ea2104162fe212fb8e9bddf51a884df27d33c37a77c84';
const GOLDEN_RECEIPT_ID = 'cr_672457349ba403bc';

function goldenReceipt(budgetTokens = 4096, selectedRefs = ['ref:alpha', 'ref:beta']) {
  return contextReceiptBuildJSON(
    'repo:golden',
    'sha256:repo-golden',
    'sha256:graph-golden',
    'workstream:golden',
    'sha256:source-golden',
    JSON.stringify(selectedRefs),
    JSON.stringify(['ref:omitted']),
    JSON.stringify(['ref:pinned']),
    JSON.stringify(['ref:recoverable']),
    JSON.stringify(['handle:alpha']),
    JSON.stringify(['evidence:alpha']),
    budgetTokens,
    'knapsack/v1',
    'exec:golden',
    1700000000000,
  );
}

// The parity anchor.
{
  const receipt = JSON.parse(goldenReceipt());
  assert.strictEqual(receipt.receipt_commitment, GOLDEN_COMMITMENT,
    'node commitment diverged from the cross-runtime golden vector');
  assert.strictEqual(receipt.receipt_id, GOLDEN_RECEIPT_ID);
  assert.strictEqual(receipt.schema_version, contextReceiptSchemaVersion());
}

// Canonicalisation is what lets two runtimes enumerate differently. The
// commitment attests to which evidence was involved; ranking order is
// presentation and stays in the host receipt.
{
  const shuffled = JSON.parse(goldenReceipt(4096, ['ref:beta', 'ref:alpha', 'ref:beta']));
  assert.strictEqual(shuffled.receipt_commitment, GOLDEN_COMMITMENT);
  assert.deepStrictEqual(shuffled.selected_refs, ['ref:alpha', 'ref:beta']);
}

// The other half of determinism: equivalence must not be too generous.
{
  const different = JSON.parse(goldenReceipt(8192));
  assert.notStrictEqual(different.receipt_commitment, GOLDEN_COMMITMENT);
}

// Verification round-trips.
{
  const receipt = goldenReceipt();
  assert.strictEqual(contextReceiptVerifyJSON(receipt), receipt);
  assert.strictEqual(contextReceiptCommitment(receipt), GOLDEN_COMMITMENT);
}

// Fail closed: a tampered receipt must raise, not come back unverified.
{
  const tampered = goldenReceipt().replace('"budget_tokens":4096', '"budget_tokens":999999');
  assert.throws(() => contextReceiptVerifyJSON(tampered), /receipt_commitment/);
}

// A newer receipt cannot be interpreted under today's rules.
{
  const future = goldenReceipt().replace('"schema_version":1', '"schema_version":99');
  assert.throws(() => contextReceiptVerifyJSON(future), /schema_version/);
}

// Section 8's rule enforced at the boundary rather than trusted to callers.
{
  const graphRef = JSON.parse(
    contextReceiptGraphRefJSON(goldenReceipt(), 'workstream:golden', 'agent:codex', 'session:1'),
  );
  assert.strictEqual(graphRef.receipt_id, GOLDEN_RECEIPT_ID);
  assert.strictEqual(graphRef.reproducibility_hash, GOLDEN_COMMITMENT);
  for (const bodyField of ['selected_refs', 'omitted_refs', 'selection_policy', 'budget_tokens']) {
    assert.ok(!(bodyField in graphRef),
      `graph reference leaked receipt body field ${bodyField}`);
  }
}

// JavaScript numbers are f64. Truncating a fractional millisecond would quietly
// produce a different commitment than the caller asked for.
{
  assert.throws(
    () => contextReceiptBuildJSON(
      'repo:golden', 'sha256:repo-golden', 'sha256:graph-golden', 'workstream:golden',
      null, null, null, null, null, null, null, 0, null, null, 1.5,
    ),
    /JavaScript-safe integer/,
  );
}

console.log('context receipt parity: node matches the cross-runtime golden vector');

// ── Recovery handles ────────────────────────────────────────────────────────
//
// Section 9's two rules — "never call destructive omission recoverable" and
// "verify the expected commitment before returning material" — are enforced in
// the engine, so both runtimes must refuse the same claims as well as produce
// the same ids. A contract that accepted different inputs in different runtimes
// would not be one contract.

const crypto = require('crypto');
const {
  recoveryHandleBuildJSON,
  recoveryHandleVerifyJSON,
  recoveryHandleVerifyBytes,
} = wasm;

// Must equal engine_contracts::tests::GOLDEN_RECOVERY_HANDLE_ID.
const GOLDEN_HANDLE_ID = 'rh_61e976bc425ad0de';
const FIXTURE_BODY = Buffer.from('recoverable bytes');
const fixtureCommitment = crypto.createHash('sha256').update(FIXTURE_BODY).digest('hex');

function recoverableHandle() {
  return recoveryHandleBuildJSON(
    'repo:demo',
    'cr_672457349ba403bc',
    'omitted_but_recoverable',
    'src/auth.py',
    'sha256:source',
    fixtureCommitment,
    0,
    17,
    'commit:abc123',
    null,
    1700000000000,
  );
}

// The recovery parity anchor.
{
  assert.strictEqual(JSON.parse(recoverableHandle()).handle_id, GOLDEN_HANDLE_ID,
    'node handle id diverged from the cross-runtime golden vector');
}

// The refusals are the contract.
{
  assert.throws(
    () => recoveryHandleBuildJSON('repo:demo', 'cr_x', 'omitted_but_recoverable',
      'src/auth.py', 'sha256:source'),
    /recover/i,
    'a promise without a fragment commitment must be refused',
  );
  assert.throws(
    () => recoveryHandleBuildJSON('repo:demo', 'cr_x', 'omitted_but_recoverable',
      null, null, fixtureCommitment),
    /recover/i,
    'a promise without a way back must be refused',
  );
}

// The honest state must always be reachable, or callers overclaim to get a handle.
{
  const gone = JSON.parse(recoveryHandleBuildJSON('repo:demo', 'cr_x', 'omitted_and_unavailable'));
  assert.strictEqual(gone.disposition, 'omitted_and_unavailable');
  assert.strictEqual(
    recoveryHandleVerifyBytes(JSON.stringify(gone), Buffer.from('anything')),
    'not_recoverable',
  );
}

// Verification hashes the bytes rather than trusting the handle.
{
  const handle = recoverableHandle();
  assert.strictEqual(recoveryHandleVerifyBytes(handle, FIXTURE_BODY), 'verified');
  assert.strictEqual(
    recoveryHandleVerifyBytes(handle, Buffer.from('different bytes')),
    'commitment_mismatch',
  );
}

// An edited handle fails closed.
{
  const edited = recoverableHandle().replace('src/auth.py', 'src/other.py');
  assert.throws(() => recoveryHandleVerifyJSON(edited), /handle_id/);
}

// An unknown disposition is refused rather than defaulted.
{
  assert.throws(
    () => recoveryHandleBuildJSON('repo:demo', 'cr_x', 'probably_fine'),
    /disposition/,
  );
}

// A fractional byte offset addresses different bytes than the caller meant.
{
  assert.throws(
    () => recoveryHandleBuildJSON('repo:demo', 'cr_x', 'omitted_and_unavailable',
      null, null, null, 1.5),
    /JavaScript-safe integer/,
  );
}

console.log('recovery handle parity: node matches the cross-runtime golden vector');

// ── Provenance-bearing memory ───────────────────────────────────────────────

const {
  memoryRecordBuildJSON,
  memoryRecordAdmissibility,
  memoryRecordVerifyJSON,
  memoryRecordSchemaVersion,
} = wasm;

const GOLDEN_MEMORY_ID = 'mem_a3b337c53411d1a5';

function goldenMemory(evidenceIds = ['evidence:1']) {
  return memoryRecordBuildJSON(
    'repo:demo',
    'vault/beliefs/auth.md',
    'observed',
    'task:auth',
    'workstream:1',
    'agent:claude',
    'session:1',
    'exec:1',
    'sha256:content',
    JSON.stringify(evidenceIds),
    1700000000000,
    1700000000000,
    0,
    JSON.stringify([]),
    JSON.stringify([]),
    null,
  );
}

{
  const memory = JSON.parse(goldenMemory());
  assert.strictEqual(memory.memory_id, GOLDEN_MEMORY_ID,
    'node memory id diverged from the cross-runtime golden vector');
  assert.strictEqual(memory.schema_version, memoryRecordSchemaVersion());
  assert.strictEqual(memoryRecordVerifyJSON(goldenMemory()), goldenMemory());
  assert.strictEqual(memoryRecordAdmissibility(goldenMemory(), 1700000100000), 'admissible');
}

// A caller cannot make an unsupported recollection trustworthy by choosing a
// stronger label, and producer provenance is mandatory at construction.
{
  const unsupported = memoryRecordBuildJSON(
    'repo:demo', 'vault/beliefs/auth.md', 'verified',
    'task:auth', 'workstream:1', 'agent:claude', 'session:1', 'exec:1',
    'sha256:content', JSON.stringify([]), 1700000000000, 1700000000000,
  );
  assert.strictEqual(
    memoryRecordAdmissibility(unsupported, 1700000100000),
    'unsupported',
  );
  assert.throws(
    () => memoryRecordBuildJSON(
      'repo:demo', 'vault/beliefs/auth.md', 'observed',
      'task:auth', 'workstream:1', null, 'session:1', 'exec:1',
      'sha256:content', JSON.stringify(['evidence:1']),
    ),
    /source_agent/,
  );
}

// Invalid replay time and transport tampering both fail closed.
{
  assert.strictEqual(memoryRecordAdmissibility(goldenMemory(), -1), 'unsupported');
  const edited = goldenMemory().replace('agent:claude', 'agent:someone');
  assert.throws(() => memoryRecordVerifyJSON(edited), /record_commitment/);
}

console.log('memory record parity: node matches the cross-runtime golden vector');
