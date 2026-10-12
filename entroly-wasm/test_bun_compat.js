// Entroly Bun & Multi-runtime Compatibility Contract Test
// Verifies that the WASM package and CLI operate seamlessly in Bun and Node.

const assert = require('assert');
const path = require('path');
const fs = require('fs');
const os = require('os');
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

// 2. Verify Bun lockfiles are excluded by actual indexing
const { autoIndex } = require('./js/auto_index');
const lockfileIndexDir = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-bun-index-'));
try {
  fs.writeFileSync(path.join(lockfileIndexDir, 'source.ts'), 'const ready = true;');
  fs.writeFileSync(path.join(lockfileIndexDir, 'bun.lock'), 'lockfile');
  fs.writeFileSync(path.join(lockfileIndexDir, 'bun.lockb'), Buffer.from([0, 1, 2]));
  fs.writeFileSync(path.join(lockfileIndexDir, 'other.lockb'), Buffer.from([0, 1, 2]));
  const indexedEngine = new entroly.WasmEntrolyEngine();
  const summary = autoIndex(indexedEngine, lockfileIndexDir, true);
  assert.strictEqual(summary.files_indexed, 1, 'only source.ts should be indexed');
} finally {
  fs.rmSync(lockfileIndexDir, { recursive: true, force: true });
}
console.log('✓ Auto-indexer excludes Bun lockfiles during ingestion');

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

const lockfileDir = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-runner-'));
try {
  fs.writeFileSync(path.join(lockfileDir, 'bun.lock'), '');
  const helpRes = spawnSync(process.execPath, [cliPath, '--help'], {
    cwd: lockfileDir,
    encoding: 'utf8',
    env: { ...process.env, NO_COLOR: '1' },
  });
  assert.strictEqual(helpRes.status, 0, `cli help failed: ${helpRes.stderr}`);
  const expectedRunner = isBun ? 'bunx' : 'npx';
  assert(helpRes.stdout.includes(`${expectedRunner} entroly-wasm`), 'runner must follow the active runtime, not the project lockfile');
} finally {
  fs.rmSync(lockfileDir, { recursive: true, force: true });
}

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

// 5. Verify MCP server tool schema construction
const server = new entroly.EntrolyMCPServer();
const toolNames = server.tools.map(({ name }) => name);
assert(toolNames.length > 0, 'MCP tool schema is empty');
assert.strictEqual(new Set(toolNames).size, toolNames.length, 'MCP tool names must be unique');
for (const name of ['remember_fragment', 'optimize_context', 'work_record_review']) {
  assert(toolNames.includes(name), `missing ${name} MCP tool`);
}
console.log('✓ MCP server tool schema construction verified');

console.log('\nAll Bun & multi-runtime compatibility tests passed!\n');
