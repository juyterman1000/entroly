'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const { EntrolyMCPServer } = require('./js/server');

function git(repo, ...args) {
  execFileSync('git', ['-C', repo, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-agent-review-'));
const repo = path.join(root, 'repo');
fs.mkdirSync(repo);
git(repo, 'init', '-b', 'main');
git(repo, 'config', 'user.email', 'test@example.com');
git(repo, 'config', 'user.name', 'Test');
const file = path.join(repo, 'app.py');
fs.writeFileSync(file, 'VALUE = 1\n');
git(repo, 'add', 'app.py');
git(repo, 'commit', '-m', 'initial');
git(repo, 'checkout', '-b', 'feature/review');
fs.writeFileSync(file, 'VALUE = 2\n');
process.env.ENTROLY_DIR = path.join(root, 'state');
process.env.ENTROLY_SOURCE = repo;
process.env.ENTROLY_MCP_PROFILE = 'public';

const server = new EntrolyMCPServer();
const names = new Set(server.tools.map((tool) => tool.name));
assert(names.has('work_record_review'));
assert(names.has('work_handoff'));
assert(!names.has('vault_write_belief'));
const capabilities = server.handleTool('work_capabilities', {});
assert.strictEqual(capabilities.project_root, repo);
const claimed = server.handleTool('work_claim', {
  agent_id: 'claude', task_title: 'Change value', task_id: 'change-value',
});
assert.strictEqual(claimed.status, 'ok');
const unfinished = server.handleTool('work_state', {}).unfinished;
const workstream = unfinished.find((item) => item.task_ids && item.task_ids.length);
assert(workstream);
const sealed = server.handleTool('work_handoff', {
  from_agent: 'claude', to_agent: 'codex', workstream_id: workstream.node_id,
});
assert.strictEqual(sealed.handoff.to_agent, 'codex');
assert.strictEqual(server.handleTool('work_pending_handoffs', { to_agent: 'codex' }).handoffs[0].current, true);
const resumed = server.handleTool('work_resume', {
  workstream_id: workstream.node_id, to_agent: 'codex',
});
assert.strictEqual(resumed.status, 'ok');
assert.strictEqual(resumed.continuation_proof.to_agent, 'codex');
assert(resumed.resume.current_workspace_commitment.startsWith('workspace:'));
assert.throws(() => server.handleTool('work_claim', {
  agent_id: 'codex', task_title: 'Fix review findings', task_id: 'fix-review',
}), /acknowledge/);
assert.strictEqual(server.handleTool('work_acknowledge_recovery', {
  token: resumed.acknowledgement.token,
}).acknowledged, true);
assert.strictEqual(server.handleTool('work_pending_handoffs', { to_agent: 'codex' }).handoffs[0].current, true);
const reviewed = server.handleTool('work_record_review', {
  handoff: sealed.handoff, reviewer_id: 'codex', verdict: 'changes_requested',
  findings: [{ path: 'app.py', line: 1, severity: 'high', message: 'Value changed' }],
  summary: 'Check callers', checks: [{ name: 'node focused', exit_code: 0 }],
});
assert.strictEqual(reviewed.status, 'ok');
assert(reviewed.review.findings[0].source_digest.startsWith('sha256:'));
assert.deepStrictEqual(reviewed.review.checks, [{ name: 'node focused', exit_code: 0 }]);
assert.strictEqual(server.handleTool('work_reviews', {}).reviews[0].current, true);
fs.writeFileSync(file, 'VALUE = 3\n');
assert.strictEqual(server.handleTool('work_reviews', {}).reviews[0].current, false);
assert.throws(() => server.handleTool('work_record_review', {
  handoff: sealed.handoff, reviewer_id: 'codex', verdict: 'approve',
  findings: [], summary: 'Looks good',
}), /stale|integrity/);
const interrupted = server.handleTool('work_resume', { to_agent: 'codex' });
assert.strictEqual(interrupted.continuation_proof.to_agent, 'codex');
fs.writeFileSync(file, 'VALUE = 4\n');
assert.throws(() => server.handleTool('work_acknowledge_recovery', {
  token: interrupted.acknowledgement.token,
}), /repository changed/);
const currentRecovery = server.handleTool('work_resume', { to_agent: 'codex' });
assert.strictEqual(server.handleTool('work_acknowledge_recovery', {
  token: currentRecovery.acknowledgement.token,
}).acknowledged, true);
console.log('work graph review MCP test passed');
