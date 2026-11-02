'use strict';

// scan_for_vulnerabilities takes the code to scan (`content`), as the Python
// server's tool does. The npm server looked up a stored fragment named by
// `source` instead, so through the official MCP client every call answered
// {"error": "Fragment 'app.py' not found"} and nothing was scanned.

const fs = require('fs');
const os = require('os');
const path = require('path');
const assert = require('assert');

const room = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-scan-tool-'));
process.env.ENTROLY_DIR = path.join(room, '.entroly');
process.chdir(room);

const { EntrolyMCPServer } = require('./js/server');

// The server keeps timers running, so the test exits explicitly -- with 1 on
// any failure, never from a finally block that would swallow it.
let failure = null;
try {
  const server = new EntrolyMCPServer();
  const code = 'import os\npassword = "hunter2"\ndef run(cmd):\n    os.system(cmd)\n';
  const fragments = () => server.handleTool('get_stats', {}).dedup.indexed_fragments;
  const before = fragments();

  const report = server.handleTool('scan_for_vulnerabilities', { content: code, source: 'app.py' });

  assert(!report.error, `scan failed: ${JSON.stringify(report)}`);
  const rules = report.findings.map((finding) => finding.rule_id);
  assert(rules.includes('SEC-001'), `hardcoded password not found: ${rules}`);
  assert(report.critical_count >= 2, `expected the password and the command injection: ${rules}`);
  assert.strictEqual(fragments(), before, 'scanning must not add the code to the session context');

  const clean = server.handleTool('scan_for_vulnerabilities', { content: 'def add(a, b):\n    return a + b\n' });
  assert(!clean.error && clean.findings.length === 0, `clean code reported: ${JSON.stringify(clean)}`);

} catch (error) {
  failure = error;
} finally {
  process.chdir(os.tmpdir());
  fs.rmSync(room, { recursive: true, force: true });
}
if (failure) {
  console.error(failure);
  process.exit(1);
}
console.log('MCP scan_for_vulnerabilities contract: PASS');
process.exit(0);
