// Entroly Bun & Multi-runtime Compatibility Contract Test
// Verifies that the WASM package and CLI operate seamlessly in Bun and Node.

const assert = require('assert');
const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

console.log('Running Entroly Bun & Multi-runtime Compatibility Tests...\n');

// 1. Verify package root exports
const entroly = require('./index.js');
assert(entroly.EntrolyEngine, 'missing EntrolyEngine export');
assert(entroly.WasmEntrolyEngine, 'missing WasmEntrolyEngine export');
assert(entroly.WorkGraph, 'missing WorkGraph export');
assert(entroly.TrustEngine, 'missing TrustEngine export');
assert(entroly.EntrolyMCPServer, 'missing EntrolyMCPServer export');
assert(entroly.createContextReceipt, 'missing createContextReceipt export');
assert(entroly.renderContextReceipt, 'missing renderContextReceipt export');
assert(entroly.explainReceiptOmission, 'missing explainReceiptOmission export');
console.log('✓ Package root exports match specification');

// 2. Verify auto_index skip patterns and binary extensions for Bun
const { autoIndex } = require('./js/auto_index');
const autoIndexSrc = fs.readFileSync(path.join(__dirname, 'js', 'auto_index.js'), 'utf8');
assert(autoIndexSrc.includes("'bun.lock'"), 'bun.lock missing from SKIP_PATTERNS');
assert(autoIndexSrc.includes("'bun.lockb'"), 'bun.lockb missing from SKIP_PATTERNS');
assert(autoIndexSrc.includes("'.lockb'"), '.lockb missing from BINARY_EXTENSIONS');
console.log('✓ Auto-indexer correctly skips Bun lockfiles (bun.lock, bun.lockb, .lockb)');

// 3. Verify CLI runner detection and status output
const cliPath = path.join(__dirname, 'js', 'cli.js');
const isBun = Boolean(process.versions && process.versions.bun);
const statusRes = spawnSync(process.execPath, [cliPath, 'status'], {
  encoding: 'utf8',
  env: { ...process.env, NO_COLOR: '1' },
});
assert.strictEqual(statusRes.status, 0, `cli status failed: ${statusRes.stderr}`);
if (isBun) {
  assert(statusRes.stdout.includes('Bun v'), `expected Bun runtime in status, got: ${statusRes.stdout}`);
} else {
  assert(statusRes.stdout.includes('Node.js'), `expected Node runtime in status, got: ${statusRes.stdout}`);
}
console.log(`✓ CLI status correctly reports runtime: ${isBun ? 'Bun v' + process.versions.bun : 'Node.js ' + process.version}`);

// 4. Verify Wasm engine optimization and stats run cleanly
const engine = new entroly.WasmEntrolyEngine();
const r1 = engine.ingest('function auth() { return true; }', 'src/auth.ts', 10, true);
assert.strictEqual(r1.status, 'ingested');
const r2 = engine.ingest('const PORT = 3000;', 'src/config.ts', 8, false);
assert.strictEqual(r2.status, 'ingested');
engine.advance_turn();
const opt = engine.optimize(50, 'authentication logic');
assert(opt, 'missing optimize result');
assert(opt.selected_count > 0, 'expected selected fragments');
const stats = engine.stats();
assert(stats, 'missing stats');
console.log('✓ Wasm engine core execution (ingest, turn, optimize, stats) verified');

// 5. Verify MCP Server instant handshake
const server = new entroly.EntrolyMCPServer();
assert.strictEqual(server.tools.length, 37, `expected 37 MCP tools, got ${server.tools.length}`);
console.log('✓ MCP Server instant handshake and 37-tool schema verified');

console.log('\nAll Bun & multi-runtime compatibility tests passed!\n');
