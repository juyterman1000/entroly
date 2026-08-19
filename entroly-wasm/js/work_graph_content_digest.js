'use strict';

// Node orchestration-only content identity for passive Work Graph snapshots.
// Rust decides whether observations are semantically duplicate; this helper
// merely supplies exact Git blob identity where it can do so without guessing.
// It never follows symlinks, reads special files, or reads an unbounded file.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const MAX_HASH_FILE_BYTES = 64 * 1024 * 1024;
const HASH_CHUNK_BYTES = 1024 * 1024;

function gitEnv() {
  return {
    ...process.env,
    GIT_TERMINAL_PROMPT: '0',
    GIT_OPTIONAL_LOCKS: '0',
    GIT_PAGER: 'cat',
    LC_ALL: 'C',
    LANG: 'C',
  };
}

function runGitText(cwd, args) {
  const result = spawnSync(
    'git',
    [
      '-c', 'core.fsmonitor=false',
      '-c', 'core.untrackedCache=false',
      '-c', 'submodule.recurse=false',
      '-C', String(cwd),
      ...args,
    ],
    {
      encoding: 'utf8',
      env: gitEnv(),
      maxBuffer: 1024 * 1024,
      timeout: 5000,
      windowsHide: true,
    },
  );
  if (result.error || result.status !== 0) return '';
  return String(result.stdout || '').trim();
}

function resolveRoot(repoPath) {
  const candidate = path.resolve(String(repoPath || '.'));
  const output = runGitText(candidate, ['rev-parse', '--show-toplevel']);
  if (!output) throw new Error(`not a Git worktree: ${candidate}`);
  return path.resolve(output);
}

function objectHashAlgorithm(root) {
  const value = runGitText(root, ['rev-parse', '--show-object-format']).toLowerCase();
  return value === 'sha1' || value === 'sha256' ? value : '';
}

function relativeRegularPath(root, repoRel) {
  const value = String(repoRel || '').replace(/\\/g, '/');
  if (!value || value.startsWith('/') || /[\0\r\n]/.test(value)) return null;
  const parts = value.split('/');
  if (parts.some(part => !part || part === '.' || part === '..')) return null;
  const candidate = path.join(root, ...parts);
  let before;
  try { before = fs.lstatSync(candidate, { bigint: true }); }
  catch (_) { return null; }
  if (!before.isFile() || before.size > BigInt(MAX_HASH_FILE_BYTES)) return null;
  return { candidate, before };
}

function sameFile(left, right) {
  if (!left.isFile() || !right.isFile()) return false;
  // `dev` is compared only when both sides report one. Node on Windows returns
  // dev=0 from lstat but a real device id from fstat for the same file, so
  // requiring strict equality rejected every legitimate file and made
  // gitBlobDigest return '' for all input -- content identity was silently
  // dead on win32, which is the binding handoff/resume rely on. The anti-swap
  // property is preserved: the file index (ino), size, and both timestamps
  // must still match across lstat -> open -> lstat.
  if (left.dev !== 0n && right.dev !== 0n && left.dev !== right.dev) return false;
  return left.ino === right.ino
    && left.size === right.size
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs;
}

function gitBlobDigest(root, repoRel, algorithm) {
  const safe = relativeRegularPath(root, repoRel);
  if (!safe) return '';

  let fd;
  try {
    let flags = fs.constants.O_RDONLY;
    if (typeof fs.constants.O_NOFOLLOW === 'number') flags |= fs.constants.O_NOFOLLOW;
    fd = fs.openSync(safe.candidate, flags);
    const opened = fs.fstatSync(fd, { bigint: true });
    const pathAfterOpen = fs.lstatSync(safe.candidate, { bigint: true });
    if (!sameFile(safe.before, opened) || !sameFile(opened, pathAfterOpen)) return '';
    if (opened.size > BigInt(MAX_HASH_FILE_BYTES)) return '';

    const hash = crypto.createHash(algorithm);
    hash.update(Buffer.from(`blob ${opened.size.toString()}\0`, 'ascii'));
    const buffer = Buffer.allocUnsafe(HASH_CHUNK_BYTES);
    let offset = 0n;
    while (offset < opened.size) {
      const remaining = opened.size - offset;
      const length = Number(remaining > BigInt(buffer.length) ? BigInt(buffer.length) : remaining);
      const bytes = fs.readSync(fd, buffer, 0, length, Number(offset));
      if (bytes <= 0) return '';
      hash.update(buffer.subarray(0, bytes));
      offset += BigInt(bytes);
    }

    const after = fs.fstatSync(fd, { bigint: true });
    const pathAfterRead = fs.lstatSync(safe.candidate, { bigint: true });
    if (!sameFile(opened, after) || !sameFile(after, pathAfterRead)) return '';
    return `git-blob:${hash.digest('hex')}`;
  } catch (_) {
    return '';
  } finally {
    if (fd !== undefined) {
      try { fs.closeSync(fd); } catch (_) {}
    }
  }
}

function enrichWorktreeContentDigests(repoPath, observation) {
  const changes = observation && observation.changes;
  if (!Array.isArray(changes)) return observation;
  const root = resolveRoot(repoPath);
  const algorithm = objectHashAlgorithm(root);

  for (const change of changes) {
    if (!change || typeof change !== 'object') continue;
    change.content_digest = '';
    if (change.staged || change.conflicted) continue;
    if (String(change.kind || '') === 'deleted') {
      change.content_digest = 'worktree:deleted';
      continue;
    }
    if (!algorithm) continue;
    change.content_digest = gitBlobDigest(root, change.path, algorithm);
  }
  return observation;
}

module.exports = {
  enrichWorktreeContentDigests,
};
