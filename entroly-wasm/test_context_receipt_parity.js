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
const { contextReceiptBuildJSON,
        contextReceiptVerifyJSON,
        contextReceiptCommitment,
        contextReceiptGraphRefJSON,
        contextReceiptSchemaVersion } = require('./pkg/entroly_wasm.js');

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
