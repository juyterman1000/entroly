'use strict';

// Local repository observation only. Work-state inference/trust remains in Rust.
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const zlib = require('zlib');
const { execFileSync } = require('child_process');

const MAX_GIT_OUTPUT_BYTES = 8 * 1024 * 1024;
const MAX_CHANGES = 16384;
const MAX_COMMITS = 20;
const MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024;

class RepositoryDiscoveryError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RepositoryDiscoveryError';
  }
}

function runGit(cwd, args, { timeout = 5000, check = true } = {}) {
  const commandArgs = [
    '-c', 'core.fsmonitor=false',
    '-c', 'core.untrackedCache=false',
    '-c', 'submodule.recurse=false',
    '-C', cwd,
    ...args,
  ];
  const env = {
    ...process.env,
    GIT_TERMINAL_PROMPT: '0',
    GIT_OPTIONAL_LOCKS: '0',
    GIT_PAGER: 'cat',
    LC_ALL: 'C',
    LANG: 'C',
  };
  try {
    return execFileSync('git', commandArgs, {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout,
      maxBuffer: MAX_GIT_OUTPUT_BYTES,
      env,
    });
  } catch (error) {
    if (!check) return '';
    const stderr = error && error.stderr ? String(error.stderr).trim() : '';
    throw new RepositoryDiscoveryError(
      `git ${args.join(' ')} failed: ${stderr || (error && error.message) || 'unknown error'}`,
    );
  }
}

function tryGit(cwd, args, options = {}) {
  try { return runGit(cwd, args, options).trim(); }
  catch (_) { return ''; }
}

function normalizeRemote(remote) {
  let value = String(remote || '').trim().replace(/\/$/, '');
  if (!value) return '';
  if (value.endsWith('.git')) value = value.slice(0, -4);
  if (value.includes('://')) {
    try {
      const parsed = new URL(value);
      const host = parsed.port
        ? `${parsed.hostname.toLowerCase()}:${parsed.port}`
        : parsed.hostname.toLowerCase();
      const repoPath = parsed.pathname.replace(/^\/+|\/+$/g, '');
      return host && repoPath ? `${host}/${repoPath}` : '';
    } catch (_) { return ''; }
  }
  const scp = value.match(/^(?:[^@]+@)?([^:]+):(.+)$/);
  if (scp) return `${scp[1].toLowerCase()}/${scp[2].replace(/^\/+|\/+$/g, '')}`;
  return '';
}

function sha32(value) {
  return crypto.createHash('sha256').update(value).digest('hex').slice(0, 32);
}

function resolveRoot(repoPath) {
  let candidate = path.resolve(String(repoPath || '.'));
  try { if (fs.statSync(candidate).isFile()) candidate = path.dirname(candidate); }
  catch (_) {}
  const root = tryGit(candidate, ['rev-parse', '--show-toplevel']);
  if (!root) throw new RepositoryDiscoveryError(`not a Git worktree: ${candidate}`);
  return path.resolve(root);
}

function repositoryId(root) {
  const remote = normalizeRemote(tryGit(root, ['config', '--get', 'remote.origin.url']));
  if (remote) return `git:${sha32(remote)}`;
  const roots = tryGit(root, ['rev-list', '--max-parents=0', 'HEAD'])
    .split(/\r?\n/).filter(Boolean);
  if (roots.length) return `git-root:${sha32(`${roots[0]}\0${path.basename(root)}`)}`;
  return `git-local:${sha32(process.platform === 'win32' ? root.toLowerCase() : root)}`;
}

function discoverRepositoryIdentity(repoPath = '.') {
  const root = resolveRoot(repoPath);
  return { repo_id: repositoryId(root), root };
}

function validateDefaultBranch(root, override) {
  if (!override) return '';
  const name = String(override).replace(/^origin\//, '').trim();
  if (!name || name.startsWith('-') || /[\0\r\n]/.test(name)) {
    throw new RepositoryDiscoveryError(`invalid default branch override: ${JSON.stringify(override)}`);
  }
  try { runGit(root, ['check-ref-format', `refs/heads/${name}`]); }
  catch (_) {
    throw new RepositoryDiscoveryError(`invalid default branch override: ${JSON.stringify(override)}`);
  }
  return name;
}

function defaultBranch(root, override) {
  const explicit = validateDefaultBranch(root, override);
  if (explicit) return explicit;
  const remoteHead = tryGit(root, ['symbolic-ref', '--quiet', '--short', 'refs/remotes/origin/HEAD']);
  if (remoteHead.startsWith('origin/')) return remoteHead.slice('origin/'.length);
  for (const candidate of ['main', 'master']) {
    if (tryGit(root, ['rev-parse', '--verify', '--quiet', `refs/heads/${candidate}`])) return candidate;
  }
  return '';
}

function baseRef(root, defaultName) {
  if (!defaultName) return '';
  for (const candidate of [`refs/remotes/origin/${defaultName}`, `refs/heads/${defaultName}`]) {
    if (tryGit(root, ['rev-parse', '--verify', '--quiet', candidate])) return candidate;
  }
  return '';
}

function aheadBehind(root, base) {
  if (!base) return [0, 0];
  const parts = tryGit(root, ['rev-list', '--left-right', '--count', `${base}...HEAD`]).split(/\s+/);
  if (parts.length < 2) return [0, 0];
  const behind = Number.parseInt(parts[0], 10);
  const ahead = Number.parseInt(parts[1], 10);
  return [Number.isFinite(ahead) ? ahead : 0, Number.isFinite(behind) ? behind : 0];
}

function parseStatus(root) {
  const raw = runGit(root, [
    'status', '--porcelain=v1', '-z', '--untracked-files=all', '--ignore-submodules=all',
  ]);
  const tokens = raw.split('\0');
  const changes = [];
  for (let index = 0; index < tokens.length;) {
    const entry = tokens[index++];
    if (!entry) continue;
    if (entry.length < 3) throw new RepositoryDiscoveryError('malformed porcelain status record');
    const xy = entry.slice(0, 2);
    const repoPath = entry.slice(3).replace(/\\/g, '/');
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
      path: repoPath,
      kind,
      staged: ![' ', '?', '!'].includes(xy[0]),
      conflicted,
      old_path: oldPath,
    });
    if (changes.length > MAX_CHANGES) {
      throw new RepositoryDiscoveryError(
        `repository has more than ${MAX_CHANGES} changed/untracked paths; ` +
        'refusing a partial Work Graph observation',
      );
    }
  }
  changes.sort((a, b) =>
    `${a.path}\0${a.old_path}\0${a.kind}`.localeCompare(`${b.path}\0${b.old_path}\0${b.kind}`));
  return changes;
}

function branchCommits(root, base, ahead, maxCommits) {
  const limit = Number(maxCommits);
  if (!Number.isSafeInteger(limit) || limit < 0 || limit > MAX_COMMITS) {
    throw new RepositoryDiscoveryError(`maxCommits must be a safe integer between 0 and ${MAX_COMMITS}`);
  }
  if (ahead <= 0 || !base || limit === 0) return [];
  const output = runGit(root, [
    'log', '--no-decorate', `-n${limit}`, '--format=%H%x00%ct%x00%P%x00%s%x00', `${base}..HEAD`,
  ]);
  const fields = output.split('\0');
  const commits = [];
  for (let offset = 0; offset + 3 < fields.length; offset += 4) {
    const sha = fields[offset].trim();
    if (!sha) continue;
    const seconds = Number.parseInt(fields[offset + 1].trim(), 10);
    commits.push({
      sha,
      subject: fields[offset + 3].trim(),
      timestamp_ms: Number.isFinite(seconds) ? seconds * 1000 : 0,
      parent_shas: fields[offset + 2].trim().split(/\s+/).filter(Boolean),
      changed_paths: [],
    });
  }
  commits.sort((a, b) => a.timestamp_ms - b.timestamp_ms || a.sha.localeCompare(b.sha));
  return commits;
}

function projectCheckpointCandidates(root, explicitDir = null) {
  if (explicitDir) return [path.resolve(String(explicitDir))];
  if (process.env.ENTROLY_DIR) return [path.resolve(process.env.ENTROLY_DIR)];
  const projectHash = crypto
    .createHash('sha256')
    .update(path.resolve(root), 'utf8')
    .digest('hex')
    .slice(0, 12);
  return [
    path.join(os.homedir(), '.entroly', 'checkpoints', projectHash),
    path.join(os.tmpdir(), 'entroly', 'checkpoints', projectHash),
  ];
}

function existingCheckpointDir(root, explicitDir = null) {
  for (const candidate of projectCheckpointCandidates(root, explicitDir)) {
    try {
      const stat = fs.lstatSync(candidate);
      if (!stat.isSymbolicLink() && stat.isDirectory()) return candidate;
    } catch (_) {}
  }
  return '';
}

function latestCheckpointMetadata(checkpointDir) {
  if (!checkpointDir) return ['', {}];
  const directory = path.resolve(String(checkpointDir));
  let entries;
  try {
    entries = fs.readdirSync(directory)
      .filter(name => /^(?:ckpt_|checkpoint_).+\.json\.gz$/.test(name))
      .map(name => {
        const full = path.join(directory, name);
        return { name, full, stat: fs.statSync(full) };
      })
      .filter(item => item.stat.isFile() && item.stat.size <= MAX_CHECKPOINT_BYTES)
      .sort((a, b) => b.stat.mtimeMs - a.stat.mtimeMs || b.name.localeCompare(a.name));
  } catch (_) { return ['', {}]; }
  for (const item of entries) {
    try {
      const raw = zlib.gunzipSync(fs.readFileSync(item.full), { maxOutputLength: MAX_CHECKPOINT_BYTES });
      const checkpoint = JSON.parse(raw.toString('utf8'));
      const id = String(checkpoint.checkpoint_id || item.name.replace(/\.json\.gz$/, ''));
      const metadata = checkpoint.metadata && typeof checkpoint.metadata === 'object'
        ? checkpoint.metadata
        : checkpoint;
      return [id, metadata];
    } catch (_) {}
  }
  return ['', {}];
}

function discoverRepositoryObservation(repoPath = '.', options = {}) {
  const root = resolveRoot(repoPath);
  const branchName = tryGit(root, ['symbolic-ref', '--quiet', '--short', 'HEAD']);
  const headSha = tryGit(root, ['rev-parse', '--verify', 'HEAD']);
  const defaultName = defaultBranch(root, options.defaultBranch || null);
  const base = baseRef(root, defaultName);
  const [ahead, behind] = aheadBehind(root, base);
  const gitDirText = tryGit(root, ['rev-parse', '--absolute-git-dir']);
  const gitDir = gitDirText ? path.resolve(gitDirText) : null;
  const mergeInProgress = Boolean(gitDir && fs.existsSync(path.join(gitDir, 'MERGE_HEAD')));
  const rebaseInProgress = Boolean(gitDir && (
    fs.existsSync(path.join(gitDir, 'rebase-merge')) ||
    fs.existsSync(path.join(gitDir, 'rebase-apply'))
  ));
  const changes = parseStatus(root);
  const maxCommits = options.maxCommits == null ? MAX_COMMITS : options.maxCommits;
  const commits = branchCommits(root, base, ahead, maxCommits);
  const meaningfulGit = Boolean(changes.length || ahead || mergeInProgress || rebaseInProgress);

  let taskHint = options.taskHint ? { ...options.taskHint } : null;
  const decisions = [];
  const includeCheckpoint = options.includeCheckpoint !== false;
  const checkpointDir = existingCheckpointDir(root, options.checkpointDir || null);
  if (includeCheckpoint && meaningfulGit && checkpointDir) {
    const [checkpointId, metadata] = latestCheckpointMetadata(checkpointDir);
    if (checkpointId) {
      const sourceRef = `checkpoint:${checkpointId}`;
      if (!taskHint) {
        const task = String(metadata.task || '').trim();
        if (task) {
          const step = String(metadata.step || metadata.current_step || '').trim();
          taskHint = {
            task_id: `checkpoint:${checkpointId}`,
            title: task,
            trust: 'observed',
            explicit_status: 'unknown',
            remaining_work: step ? [step] : [],
            source_kind: 'checkpoint',
            source_ref: sourceRef,
          };
        }
      }
      const rawDecisions = Array.isArray(metadata.decisions) ? metadata.decisions.slice(0, 20) : [];
      rawDecisions.forEach((value, index) => {
        const text = String(value || '').trim();
        if (text) decisions.push({
          decision_id: `${checkpointId}:${index}`,
          text,
          source_ref: sourceRef,
          source_kind: 'checkpoint',
          trust: 'observed',
        });
      });
    }
  }

  const observedAt = options.observedAtMs == null ? Date.now() : Number(options.observedAtMs);
  if (!Number.isSafeInteger(observedAt)) {
    throw new RepositoryDiscoveryError('observedAtMs must be a JavaScript-safe integer');
  }

  return {
    repo_id: repositoryId(root),
    observed_at_ms: observedAt,
    repository_label: path.basename(root),
    agent_id: String(options.agentId || ''),
    session_id: String(options.sessionId || ''),
    task_hint: taskHint,
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
    changes,
    commits,
    verifications: [],
    decisions,
    claims: [],
    leases: [],
    model_executions: [],
  };
}

module.exports = {
  RepositoryDiscoveryError,
  discoverRepositoryIdentity,
  discoverRepositoryObservation,
};
