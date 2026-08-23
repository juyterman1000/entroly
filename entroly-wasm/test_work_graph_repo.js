'use strict';

const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const zlib = require('zlib');
const { spawnSync } = require('child_process');
const {
  RepositoryDiscoveryError,
  discoverRepositoryIdentity,
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

  const identity = discoverRepositoryIdentity(repo);
  assert(identity.repo_id === clean.repo_id, 'identity-only repo ID drifted from full observation');
  assert(path.resolve(identity.root) === path.resolve(repo), 'identity root mismatch');

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

  const oldHome = process.env.HOME;
  const oldUserProfile = process.env.USERPROFILE;
  const oldEntrolyDir = process.env.ENTROLY_DIR;
  const fakeHome = path.join(root, 'home');
  fs.mkdirSync(fakeHome);
  process.env.HOME = fakeHome;
  // Node's os.homedir() reads USERPROFILE on Windows and HOME on Unix. Set
  // both so this test controls the same API on every supported platform.
  process.env.USERPROFILE = fakeHome;
  delete process.env.ENTROLY_DIR;
  try {
    const absentRoot = path.join(fakeHome, '.entroly');
    const beforeDefault = discoverRepositoryObservation(repo, {
      includeCheckpoint: true,
      observedAtMs: 2100,
    });
    assert(beforeDefault.task_hint === null, 'missing default checkpoint invented a task');
    assert(!fs.existsSync(absentRoot), 'checkpoint lookup created ~/.entroly as a read side effect');

    const projectHash = crypto.createHash('sha256')
      .update(path.resolve(repo), 'utf8').digest('hex').slice(0, 12);
    const defaultDir = path.join(fakeHome, '.entroly', 'checkpoints', projectHash);
    fs.mkdirSync(defaultDir, { recursive: true });
    fs.copyFileSync(
      path.join(checkpointDir, 'ckpt_test_1.json.gz'),
      path.join(defaultDir, 'ckpt_test_1.json.gz'),
    );
    const defaultCheckpoint = discoverRepositoryObservation(repo, {
      includeCheckpoint: true,
      observedAtMs: 2200,
    });
    assert(defaultCheckpoint.task_hint.title === 'Fix streaming', 'default checkpoint path parity failed');
  } finally {
    if (oldHome === undefined) delete process.env.HOME; else process.env.HOME = oldHome;
    if (oldUserProfile === undefined) delete process.env.USERPROFILE; else process.env.USERPROFILE = oldUserProfile;
    if (oldEntrolyDir === undefined) delete process.env.ENTROLY_DIR; else process.env.ENTROLY_DIR = oldEntrolyDir;
  }

  git(repo, 'remote', 'add', 'origin', 'https://alice:secret-one@example.com/org/repo.git');
  const firstId = discoverRepositoryObservation(repo, { observedAtMs: 3000 }).repo_id;
  git(repo, 'remote', 'set-url', 'origin', 'https://bob:secret-two@example.com/org/repo.git');
  const secondId = discoverRepositoryObservation(repo, { observedAtMs: 3001 }).repo_id;
  assert(firstId === secondId, 'credential changes must not change repo identity');
  assert(!firstId.includes('secret') && !firstId.includes('alice'), 'repo identity leaked credentials');

  const hugeRepo = path.join(root, 'huge');
  fs.mkdirSync(hugeRepo);
  git(hugeRepo, 'init', '-b', 'main');
  git(hugeRepo, 'config', 'user.email', 'test@example.com');
  git(hugeRepo, 'config', 'user.name', 'Test');
  fs.writeFileSync(path.join(hugeRepo, 'base.txt'), 'base\n');
  git(hugeRepo, 'add', 'base.txt');
  git(hugeRepo, 'commit', '-m', 'initial');
  for (let index = 0; index < 513; index += 1) {
    fs.writeFileSync(path.join(hugeRepo, `untracked-${index}.txt`), 'x\n');
  }
  const hugeIdentity = discoverRepositoryIdentity(hugeRepo);
  assert(hugeIdentity.repo_id.startsWith('git-root:'), 'identity-only lookup failed on huge dirty repo');
  const hugeObservation = discoverRepositoryObservation(hugeRepo);
  assert(hugeObservation.changes.length === 513,
    'large dirty observation silently truncated changed paths');

  console.log('Work Graph npm Git discovery contract: PASS');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
