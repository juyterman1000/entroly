'use strict';

// Conservative Git observer for the Rust Work Graph. This file is orchestration
// only: it gathers durable repository facts and never infers task intent.

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

class RepositoryDiscoveryError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RepositoryDiscoveryError';
  }
}

function runGit(cwd, args, { timeoutMs = 5000, check = true } = {}) {
  const result = spawnSync('git', ['-C', cwd, ...args], {
    encoding: 'utf8',
    timeout: timeoutMs,
    windowsHide: true,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0', LC_ALL: 'C', LANG: 'C' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (result.error) {
    throw new RepositoryDiscoveryError(`git ${args.join(' ')} failed: ${result.error.message}`);
  }
  if (check && result.status !== 0) {
    const detail = String(result.stderr || '').trim() || `exit ${result.status}`;
    throw new RepositoryDiscoveryError(`git ${args.join(' ')} failed: ${detail}`);
  }
  return String(result.stdout || '');
}

function tryGit(cwd, args, options = {}) {
  try {
    return runGit(cwd, args, options).trim();
  } catch (_) {
    return '';
  }
}

function normalizeRemote(remote) {
  let value = String(remote || '').trim().replace(/\/+$/, '');
  if (!value) return '';
  if (value.endsWith('.git')) value = value.slice(0, -4);
  if (value.includes('://')) {
    try {
      const parsed = new URL(value);
      const host = parsed.port ? `${parsed.hostname.toLowerCase()}:${parsed.port}` : parsed.hostname.toLowerCase();
      const pathname = parsed.pathname.replace(/^\/+|\/+$/g, '');
      return host && pathname ? `${host}/${pathname}` : '';
    } catch (_) {
      return '';
    }
  }
  const colon = value.indexOf(':');
  if (colon > 0 && value.slice(0, colon).includes('@')) {
    const lhs = value.slice(0, colon);
    const rhs = value.slice(colon + 1).replace(/^\/+|\/+$/g, '');
    const host = lhs.slice(lhs.lastIndexOf('@') + 1).toLowerCase();
    return host && rhs ? `${host}/${rhs}` : '';
  }
  return '';
}

function digest32(value) {
  return crypto.createHash('sha256').update(value).digest('hex').slice(0, 32);
}

function repositoryId(root) {
  const remote = normalizeRemote(tryGit(root, ['config', '--get', 'remote.origin.url']));
  if (remote) return `git:${digest32(remote)}`;
  const roots = tryGit(root, ['rev-list', '--max-parents=0', 'HEAD']).split(/\r?\n/).filter(Boolean);
  if (roots.length) return `git-root:${digest32(`${roots[0]}\0${path.basename(root)}`)}`;
  const canonical = process.platform === 'win32' ? root.toLowerCase() : root;
  return `git-local:${digest32(canonical)}`;
}

function resolveRoot(input) {
  let candidate = path.resolve(String(input || '.'));
  try {
    if (fs.statSync(candidate).isFile()) candidate = path.dirname(candidate);
  } catch (error) {
    throw new RepositoryDiscoveryError(`repository path is unavailable: ${candidate}`);
  }
  const root = tryGit(candidate, ['rev-parse', '--show-toplevel']);
  if (!root) throw new RepositoryDiscoveryError(`not a Git worktree: ${candidate}`);
  return path.resolve(root);
}

function defaultBranch(root, override) {
  if (override) return String(override).replace(/^origin\//, '');
  const remoteHead = tryGit(root, ['symbolic-ref', '--quiet', '--short', 'refs/remotes/origin/HEAD']);
  if (remoteHead.startsWith('origin/')) return remoteHead.slice('origin/'.length);
  for (const candidate of ['main', 'master']) {
    if (tryGit(root, ['rev-parse', '--verify', '--quiet', `refs/heads/${candidate}`])) return candidate;
  }
  return '';
}

function baseRef(root, defaultName) {
  if (!defaultName) return '';
  for (const candidate of [`origin/${defaultName}`, defaultName]) {
    if (tryGit(root, ['rev-parse', '--verify', '--quiet', candidate])) return candidate;
  }
  return '';
}

function aheadBehind(root, base) {
  if (!base) return [0, 0];
  const parts = tryGit(root, ['rev-list', '--left-right', '--count', `${base}...HEAD`]).split(/\s+/);
  if (parts.length < 2) return [0, 0];
  const behind = Number(parts[0]);
  const ahead = Number(parts[1]);
  return [Number.isFinite(ahead) ? ahead : 0, Number.isFinite(behind) ? behind : 0];
}

function parseStatus(root) {
  const raw = runGit(root, ['status', '--porcelain=v1', '-z', '--untracked-files=all']);
  const tokens = raw.split('\0');
  const changes = [];
  let index = 0;
  while (index < tokens.length) {
    const entry = tokens[index++];
    if (!entry || entry.length < 3) continue;
    const xy = entry.slice(0, 2);
    const currentPath = entry.slice(3).replace(/\\/g, '/');
    let oldPath = '';
    if ((xy.includes('R') || xy.includes('C')) && index < tokens.length) {
      oldPath = tokens[index++].replace(/\\/g, '/');
    }
    const conflicted = xy.includes('U') || xy === 'DD' || xy === 'AA';
    let kind = 'unknown';
    if (xy === '??') kind = 'untracked';
    else if (conflicted) kind = 'unmerged';
    else if (xy.includes('R')) kind = 'renamed';
    else if (xy.includes('C')) kind = 'copied';
    else if (xy.includes('D')) kind = 'deleted';
    else if (xy.includes('A')) kind = 'added';
    else if (xy.includes('M') || xy.includes('T')) kind = 'modified';
    changes.push({
      path: currentPath,
      kind,
      staged: ![' ', '?', '!'].includes(xy[0]),
      conflicted,
      old_path: oldPath,
    });
  }
  changes.sort((a, b) => `${a.path}\0${a.old_path}\0${a.kind}`.localeCompare(`${b.path}\0${b.old_path}\0${b.kind}`));
  return changes;
}

function branchCommits(root, base, aheadBy, maxCommits) {
  if (aheadBy <= 0 || !base || maxCommits <= 0) return [];
  const limit = Math.min(Number(maxCommits) || 0, 100);
  const raw = runGit(root, [
    'log', '--no-decorate', `-n${limit}`, '--format=%H%x00%ct%x00%P%x00%s%x00', `${base}..HEAD`,
  ]);
  const fields = raw.split('\0');
  const commits = [];
  for (let offset = 0; offset + 3 < fields.length; offset += 4) {
    const sha = fields[offset].trim();
    if (!sha) continue;
    const timestamp = Number(fields[offset + 1].trim());
    commits.push({
      sha,
      subject: fields[offset + 3].trim(),
      timestamp_ms: Number.isFinite(timestamp) ? timestamp * 1000 : 0,
      parent_shas: fields[offset + 2].trim().split(/\s+/).filter(Boolean),
    });
  }
  commits.sort((a, b) => (a.timestamp_ms - b.timestamp_ms) || a.sha.localeCompare(b.sha));
  return commits;
}

function discoverRepositoryObservation(input = '.', options = {}) {
  const root = resolveRoot(input);
  const branchName = tryGit(root, ['symbolic-ref', '--quiet', '--short', 'HEAD']);
  const headSha = tryGit(root, ['rev-parse', '--verify', 'HEAD']);
  const defaultName = defaultBranch(root, options.defaultBranch);
  const base = baseRef(root, defaultName);
  const [ahead, behind] = aheadBehind(root, base);
  const gitDirText = tryGit(root, ['rev-parse', '--absolute-git-dir']);
  const gitDir = gitDirText ? path.resolve(gitDirText) : null;
  const mergeInProgress = Boolean(gitDir && fs.existsSync(path.join(gitDir, 'MERGE_HEAD')));
  const rebaseInProgress = Boolean(
    gitDir && (fs.existsSync(path.join(gitDir, 'rebase-merge')) || fs.existsSync(path.join(gitDir, 'rebase-apply'))),
  );

  return {
    repo_id: repositoryId(root),
    observed_at_ms: options.observedAtMs == null ? Date.now() : Number(options.observedAtMs),
    repository_label: path.basename(root),
    agent_id: String(options.agentId || ''),
    session_id: String(options.sessionId || ''),
    task_hint: options.taskHint || null,
    branch: {
      name: branchName,
      head_sha: headSha,
      base_ref: base,
      default_branch: defaultName,
      ahead_by: ahead,
      behind_by: behind,
      merge_in_progress: mergeInProgress,
      rebase_in_progress: rebaseInProgress,
      detached: !branchName,
    },
    changes: parseStatus(root),
    commits: branchCommits(root, base, ahead, options.maxCommits == null ? 20 : options.maxCommits),
    verifications: [],
    decisions: [],
    claims: [],
    leases: [],
    model_executions: [],
  };
}

module.exports = { RepositoryDiscoveryError, discoverRepositoryObservation };
