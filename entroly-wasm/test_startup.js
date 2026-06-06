// Startup regression: the MCP handshake must NOT load the persistent index.
// Construction is instant; the index loads lazily on first tool use. This is the
// npm twin of the pip/MCP instant-start fix (cold start ~6.5s -> instant
// handshake). If this regresses, npm users get a blocking cold start and the
// client's connect window times out -> "feels broken" -> churn.
const assert = require('assert');
const { EntrolyMCPServer } = require('./js/server');

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); pass++; console.log(`  PASS ${name}`); }
  catch (e) { fail++; console.log(`  FAIL ${name}: ${e.message}`); }
}

console.log('Entroly npm - startup (instant-start) regression\n');

// 1. Construction must not load the index (deferred, not eager).
const s = new EntrolyMCPServer();
test('construction does not load index', () => {
  assert.strictEqual(s._indexLoaded, false, 'index loaded at construct - startup will block');
});

// 2. The MCP handshake (initialize + tools/list) must answer WITHOUT warming.
test('handshake answers without loading index', () => {
  const writes = [];
  const orig = process.stdout.write.bind(process.stdout);
  process.stdout.write = (chunk) => { writes.push(chunk.toString()); return true; };
  try {
    s._handleMessage({ jsonrpc: '2.0', id: 1, method: 'initialize', params: {} });
    s._handleMessage({ jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} });
  } finally {
    process.stdout.write = orig;
  }
  const joined = writes.join('');
  assert.ok(joined.includes('"serverInfo"'), 'initialize did not respond');
  assert.ok(joined.includes('"tools"'), 'tools/list did not respond');
  assert.strictEqual(s._indexLoaded, false, 'handshake loaded the index - must stay deferred');
});

// 3. First tool call warms the engine (lazy load happens on first use).
test('first tool call warms the index', () => {
  s.handleTool('get_stats', {});
  assert.strictEqual(s._indexLoaded, true, 'tool call did not warm the index');
});

// 4. _ensureWarm is idempotent (safe to call from both warm-up and tool path).
test('_ensureWarm is idempotent', () => {
  s._ensureWarm();
  s._ensureWarm();
  assert.strictEqual(s._indexLoaded, true);
});

console.log(`\nResults: ${pass} passed, ${fail} failed`);
process.exit(fail > 0 ? 1 : 0);
