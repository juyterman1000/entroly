'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const { enrichWorktreeContentDigests } = require('./js/work_graph_content_digest');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

function git(repo, args) {
  return execFileSync('git', ['-C', repo, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-work-digest-'));
try {
  git(root, ['init', '-b', 'main']);
  git(root, ['config', 'user.email', 'test@example.com']);
  git(root, ['config', 'user.name', 'Test']);
  fs.writeFileSync(path.join(root, 'app.py'), 'VALUE = 1\n', 'utf8');
  git(root, ['add', 'app.py']);
  git(root, ['commit', '-m', 'initial']);

  fs.writeFileSync(path.join(root, 'app.py'), 'VALUE = 2\n', 'utf8');
  const first = { changes: [{ path: 'app.py', kind: 'modified', staged: false, conflicted: false, content_digest: '' }] };
  const second = { changes: [{ path: 'app.py', kind: 'modified', staged: false, conflicted: false, content_digest: '' }] };
  enrichWorktreeContentDigests(root, first);
  enrichWorktreeContentDigests(root, second);
  assert(first.changes[0].content_digest.startsWith('git-blob:'), 'unstaged digest missing');
  assert(first.changes[0].content_digest === second.changes[0].content_digest, 'same bytes changed digest');

  fs.writeFileSync(path.join(root, 'app.py'), 'VALUE = 3\n', 'utf8');
  const changed = { changes: [{ path: 'app.py', kind: 'modified', staged: false, conflicted: false, content_digest: '' }] };
  enrichWorktreeContentDigests(root, changed);
  assert(changed.changes[0].content_digest !== first.changes[0].content_digest, 'changed bytes kept old digest');

  const staged = { changes: [{ path: 'app.py', kind: 'modified', staged: true, conflicted: false, content_digest: '' }] };
  const conflicted = { changes: [{ path: 'app.py', kind: 'unmerged', staged: false, conflicted: true, content_digest: '' }] };
  enrichWorktreeContentDigests(root, staged);
  enrichWorktreeContentDigests(root, conflicted);
  assert(staged.changes[0].content_digest === '', 'staged change became dedupeable');
  assert(conflicted.changes[0].content_digest === '', 'conflicted change became dedupeable');

  fs.unlinkSync(path.join(root, 'app.py'));
  const deleted = { changes: [{ path: 'app.py', kind: 'deleted', staged: false, conflicted: false, content_digest: '' }] };
  enrichWorktreeContentDigests(root, deleted);
  assert(deleted.changes[0].content_digest === 'worktree:deleted', 'deletion marker mismatch');

  const missing = { changes: [{ path: 'missing.py', kind: 'modified', staged: false, conflicted: false, content_digest: '' }] };
  enrichWorktreeContentDigests(root, missing);
  assert(missing.changes[0].content_digest === '', 'missing file did not fail closed');

  const outsideDir = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-work-digest-outside-'));
  try {
    const outside = path.join(outsideDir, 'secret.txt');
    fs.writeFileSync(outside, 'outside-secret\n', 'utf8');
    const link = path.join(root, 'outside-link.txt');
    try {
      fs.symlinkSync(outside, link);
      const symlinked = { changes: [{ path: 'outside-link.txt', kind: 'untracked', staged: false, conflicted: false, content_digest: '' }] };
      enrichWorktreeContentDigests(root, symlinked);
      assert(symlinked.changes[0].content_digest === '', 'symlink target outside repository was fingerprinted');
    } catch (error) {
      // Some Windows environments disallow symlink creation without developer
      // mode/elevation. Only that setup failure may skip this assertion.
      if (!error || !['EPERM', 'EACCES', 'ENOSYS'].includes(error.code)) throw error;
    }
  } finally {
    fs.rmSync(outsideDir, { recursive: true, force: true });
  }

  const largePath = path.join(root, 'large.bin');
  const largeFd = fs.openSync(largePath, 'w');
  try { fs.ftruncateSync(largeFd, 64 * 1024 * 1024 + 1); }
  finally { fs.closeSync(largeFd); }
  const oversized = { changes: [{ path: 'large.bin', kind: 'untracked', staged: false, conflicted: false, content_digest: '' }] };
  enrichWorktreeContentDigests(root, oversized);
  assert(oversized.changes[0].content_digest === '', 'oversized file was fingerprinted');

  console.log('Work Graph npm content fingerprint contract: PASS');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
