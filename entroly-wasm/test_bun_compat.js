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
  const expectedRunner = isBun ? 'bunx --bun' : 'npx';
  assert(helpRes.stdout.includes(`${expectedRunner} entroly-wasm`), 'runner must follow the active runtime, not the project lockfile');
  fs.mkdirSync(path.join(lockfileDir, '.cursor'));
  const initRes = spawnSync(process.execPath, [cliPath, 'init'], {
    cwd: lockfileDir,
    encoding: 'utf8',
    env: { ...process.env, NO_COLOR: '1' },
  });
  assert.strictEqual(initRes.status, 0, `cli init failed: ${initRes.stderr}`);
  const mcpConfig = JSON.parse(fs.readFileSync(path.join(lockfileDir, '.cursor', 'mcp.json'), 'utf8'));
  assert.deepStrictEqual(mcpConfig.mcpServers.entroly, {
    command: isBun ? 'bunx' : 'npx',
    args: isBun ? ['--bun', 'entroly-wasm', '--serve'] : ['-y', 'entroly-wasm', '--serve'],
  });
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

// 6. Verify the actual stdio MCP handshake in the active runtime
const handshakeDir = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-bun-mcp-'));
try {
  const requests = [
    { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'compat-test', version: '1' } } },
    { jsonrpc: '2.0', id: 2, method: 'tools/list', params: {} },
  ];
  const handshake = spawnSync(process.execPath, [cliPath, '--serve'], {
    cwd: handshakeDir,
    input: `${requests.map((request) => JSON.stringify(request)).join('\n')}\n`,
    timeout: 10000,
    maxBuffer: 1024 * 1024,
    env: { ...process.env, ENTROLY_DIR: path.join(handshakeDir, '.entroly'), ENTROLY_SOURCE: handshakeDir },
  });
  assert.strictEqual(handshake.status, 0, `MCP handshake failed: ${handshake.error?.message || handshake.stderr}`);
  const responses = [];
  const frameSeparator = Buffer.from('\r\n\r\n');
  let offset = 0;
  while (offset < handshake.stdout.length) {
    const headerEnd = handshake.stdout.indexOf(frameSeparator, offset);
    assert(headerEnd >= 0, 'MCP response missing frame separator');
    const header = handshake.stdout.subarray(offset, headerEnd).toString('ascii');
    const contentLength = Number(header.match(/^Content-Length:\s*(\d+)$/i)?.[1]);
    assert(Number.isInteger(contentLength), 'MCP response missing Content-Length');
    const bodyStart = headerEnd + frameSeparator.length;
    const bodyEnd = bodyStart + contentLength;
    assert(bodyEnd <= handshake.stdout.length, 'MCP response body truncated');
    responses.push(JSON.parse(handshake.stdout.subarray(bodyStart, bodyEnd).toString('utf8')));
    offset = bodyEnd;
  }
  assert(responses.find((response) => response.id === 1)?.result?.serverInfo, 'initialize response missing');
  const listedTools = responses.find((response) => response.id === 2)?.result?.tools;
  assert(Array.isArray(listedTools), 'tools/list response missing');
  assert.deepStrictEqual(listedTools.map(({ name }) => name), toolNames, 'stdio tool list differs from server schema');
} finally {
  fs.rmSync(handshakeDir, { recursive: true, force: true });
}
console.log('MCP stdio initialize and tools/list verified');

console.log('\nAll Bun & multi-runtime compatibility tests passed!\n');
