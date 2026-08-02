// Cross-runtime release gate: the packaged Node/WASM adapter must emit the
// exact same audited QCCR receipt as the native PyO3 adapter's frozen golden.
const assert = require('node:assert/strict');
const wasm = require('./pkg/entroly_wasm.js');
const expected = require('../tests/fixtures/audited_qccr_runtime_golden.json');

const first = 'Résumé background. The Dutch name is:\nRhijn.';
const second = 'Authentication tokens rotate. Authentication failures are logged.';
const fragments = [
  {
    fragment_id: 'unicode-1',
    source: 'doc.txt',
    content: first,
    start_byte: 0,
    end_byte: Buffer.byteLength(first),
    token_count: 20,
  },
  {
    fragment_id: 'other-1',
    source: 'other.txt',
    content: second,
    start_byte: 0,
    end_byte: Buffer.byteLength(second),
    token_count: 20,
  },
];

const actual = JSON.parse(
  wasm.qccr_select_with_audit(
    JSON.stringify(fragments),
    12,
    'What is the Dutch name?',
    '{}',
    '[]',
  ),
);

assert.deepStrictEqual(actual, expected);
assert.strictEqual(actual.metrics.source_span_integrity, true);
assert.strictEqual(actual.metrics.verdict, 'sufficient');
assert.strictEqual(actual.emitted_tokens, 7);
console.log('audited_qccr_runtime_parity=ok');
