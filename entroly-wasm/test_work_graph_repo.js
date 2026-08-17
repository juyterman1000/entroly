'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const zlib = require('zlib');
const { spawnSync } = require('child_process');
const {
  RepositoryDiscoveryError,
  discoverRepositoryObservation,
} = require('./js/work_graph_repo');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

function git(repo, ...args) {
  const result = spawnSync('git', ['-C', repo, ...args], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr || `git ${args.join(' ')} failed`);
  return String(result.stdout || '').trim();
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-workgraph-'));
const repo = path.join(root, 'repo');
fs.mkdirSync(repo);

try {
  git(repo, 'init', '-b', 'main');
  git(repo, 'config', 'user.email', 'test@example.com');
  git(repo, 'config', 'user.name', 'Test');
  fs.writeFileSync(path.join(repo, 'app.js'), 'console.log("one")\n');
  git(repo, 'add', 'app.js');
  git(repo, 'commit', '-m', 'initial');

  const clean = discoverRepositoryObservation(repo, { observedAtMs: 1000 });
  assert(clean.branch.name === 'main', 'clean branch mismatch');
  assert(clean.branch.ahead_by === 0, 'clean repo must not be ahead');
  assert(clean.changes.length === 0, 'clean repo must have no worktree changes');
  assert(clean.commits.length === 0, 'clean repo must remain a null-control observation');
  assert(clean.task_hint === null, 'observer must not invent task intent');

  const explicit = discoverRepositoryObservation(repo, {
    defaultBranch: 'main', observedAtMs: 1001,
  });
  assert(explicit.branch.default_branch === 'main', 'valid branch override rejected');
  assert(explicit.branch.base_ref === 'refs/heads/main', 'explicit base ref mismatch');
  let invalidRejected = false;
  try { discoverRepositoryObservation(repo, { defaultBranch: '--help' }); }
  catch (error) { invalidRejected = error instanceof RepositoryDiscoveryError; }
  assert(invalidRejected, 'unsafe default branch override was accepted');

  const checkpointDir = path.join(root, 'checkpoints');
  fs.mkdirSync(checkpointDir);
  const checkpoint = {
    checkpoint_id: 'ckpt_test_1',
    metadata: {
      task: 'Fix streaming',
      step: 'finish tests',
      decisions: ['Preserve event order'],
    },
  };
  fs.writeFileSync(
    path.join(checkpointDir, 'ckpt_test_1.json.gz'),
    zlib.gzipSync(Buffer.from(JSON.stringify(checkpoint))),
  );
  const cleanWithCheckpoint = discoverRepositoryObservation(repo, {
    checkpointDir, observedAtMs: 1100,
  });
  assert(cleanWithCheckpoint.task_hint === null, 'stale checkpoint resurrected work in clean repo');
  assert(cleanWithCheckpoint.decisions.length === 0, 'stale decisions leaked into clean repo');

  git(repo, 'checkout', '-b', 'feature/work');
  fs.writeFileSync(path.join(repo, 'app.js'), 'console.log("two")\n');
  git(repo, 'add', 'app.js');
  git(repo, 'commit', '-m', 'feature change');
  fs.writeFileSync(path.join(repo, 'app.js'), 'console.log("three")\n');
  fs.writeFileSync(path.join(repo, 'new.txt'), 'new\n');

  const active = discoverRepositoryObservation(repo, {
    agentId: 'codex',
    sessionId: 's1',
    checkpointDir,
    observedAtMs: 2000,
  });
  assert(active.branch.base_ref === 'refs/heads/main', 'base ref mismatch');
  assert(active.branch.ahead_by === 1, `expected one ahead commit, got ${active.branch.ahead_by}`);
  assert(active.commits.length === 1, 'expected one branch commit');
  assert(active.commits[0].subject === 'feature change', 'commit subject mismatch');
  const kinds = Object.fromEntries(active.changes.map(change => [change.path, change.kind]));
  assert(kinds['app.js'] === 'modified', 'modified file missing');
  assert(kinds['new.txt'] === 'untracked', 'untracked file missing');
  assert(active.task_hint.title === 'Fix streaming', 'checkpoint did not annotate real Git work');
  assert(active.task_hint.remaining_work[0] === 'finish tests', 'checkpoint remaining work missing');
  assert(active.decisions[0].text === 'Preserve event order', 'checkpoint decision missing');

  git(repo, 'remote', 'add', 'origin', 'https://alice:secret-one@example.com/org/repo.git');
  const firstId = discoverRepositoryObservation(repo, { observedAtMs: 3000 }).repo_id;
  git(repo, 'remote', 'set-url', 'origin', 'https://bob:secret-two@example.com/org/repo.git');
  const secondId = discoverRepositoryObservation(repo, { observedAtMs: 3001 }).repo_id;
  assert(firstId === secondId, 'credential changes must not change repo identity');
  assert(!firstId.includes('secret') && !firstId.includes('alice'), 'repo identity leaked credentials');

  console.log('Work Graph npm Git discovery contract: PASS');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
