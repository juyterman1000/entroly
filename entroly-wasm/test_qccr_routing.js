// Regression: npm's optimize_context routes through the QCCR Rust SSOT, so npm
// ranks/selects identically to pip/MCP/SDK. Repository must beat a high-term-
// frequency UI decoy for a persistence query, via general code conventions.
const assert = require('assert');
const { EntrolyMCPServer } = require('./js/server');

let pass = 0, fail = 0;
function test(name, fn) {
  try { fn(); pass++; console.log(`  PASS ${name}`); }
  catch (e) { fail++; console.log(`  FAIL ${name}: ${e.message}`); }
}

console.log('Entroly npm - optimize_context routes through QCCR\n');

const s = new EntrolyMCPServer();
s._indexLoaded = true; // skip lazy autoIndex so the candidate set is isolated

s.engine.ingest('dataset scores persisted display table. '.repeat(20),
  'file:web/src/components/DatasetTable.tsx', 0, false);
s.engine.ingest(
  'export const upsertScore = async (s) => { await db.insert(s); };\n' +
  'export type ScoreRecordInsertType = {}; the repository upserts scores into clickhouse.',
  'file:packages/server/repositories/scores.ts', 0, false);

const r = s.handleTool('optimize_context', { query: 'how are scores persisted', token_budget: 300 });

test('uses the qccr selector (true routing, not native knapsack)', () => {
  assert.strictEqual(r.selector, 'qccr', `selector=${r.selector}`);
});
test('repository ranks above the UI decoy', () => {
  const top = r.selected && r.selected[0] && r.selected[0].source;
  assert.ok(top && top.includes('repositories/scores.ts'), `top=${top}`);
});
test('respects the token budget', () => {
  assert.ok(r.tokens_used <= 360, `tokens_used=${r.tokens_used}`);
});
test('reports savings for the value tracker', () => {
  assert.ok(typeof r.tokens_saved === 'number' && r.tokens_saved >= 0, `tokens_saved=${r.tokens_saved}`);
});

console.log(`\nResults: ${pass} passed, ${fail} failed`);
process.exit(fail > 0 ? 1 : 0);
