'use strict';

// Node orchestration-only content identity for passive Work Graph snapshots.
// Rust decides whether observations are semantically duplicate; this helper
// merely supplies exact worktree identity where it can do so without guessing.
const { spawnSync } = require('child_process');
const path = require('path');

const MAX_HASH_OUTPUT_BYTES = 4 * 1024 * 1024;

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

function runGit(cwd, args, options = {}) {
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
      input: options.input,
      maxBuffer: MAX_HASH_OUTPUT_BYTES,
      timeout: 5000,
      windowsHide: true,
    },
  );
  if (result.error || result.status !== 0) return null;
  const stdout = String(result.stdout || '');
  if (Buffer.byteLength(stdout, 'utf8') > MAX_HASH_OUTPUT_BYTES) return null;
  return stdout;
}

function resolveRoot(repoPath) {
  const candidate = path.resolve(String(repoPath || '.'));
  const output = runGit(candidate, ['rev-parse', '--show-toplevel']);
  if (!output || !output.trim()) throw new Error(`not a Git worktree: ${candidate}`);
  return path.resolve(output.trim());
}

function enrichWorktreeContentDigests(repoPath, observation) {
  const changes = observation && observation.changes;
  if (!Array.isArray(changes)) return observation;
  const root = resolveRoot(repoPath);
  const pending = [];

  for (const change of changes) {
    if (!change || typeof change !== 'object') continue;
    if (typeof change.content_digest !== 'string') change.content_digest = '';
    if (change.staged || change.conflicted) continue;
    if (String(change.kind || '') === 'deleted') {
      change.content_digest = 'worktree:deleted';
      continue;
    }
    const repoPathValue = String(change.path || '');
    if (!repoPathValue || /[\0\r\n]/.test(repoPathValue)) continue;
    pending.push([change, repoPathValue]);
  }

  if (!pending.length) return observation;
  const output = runGit(
    root,
    ['hash-object', '--no-filters', '--stdin-paths'],
    { input: pending.map(([, repoPathValue]) => `${repoPathValue}\n`).join('') },
  );
  if (output == null) return observation;
  const hashes = output.split(/\r?\n/).filter(Boolean).map((value) => value.trim().toLowerCase());
  if (hashes.length !== pending.length) return observation;
  if (hashes.some((digest) => !/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(digest))) {
    return observation;
  }
  for (let index = 0; index < pending.length; index += 1) {
    pending[index][0].content_digest = `git-blob:${hashes[index]}`;
  }
  return observation;
}

module.exports = {
  enrichWorktreeContentDigests,
};
