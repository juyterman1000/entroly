'use strict';

// Persistence mechanics only. Work-state semantics remain in shared Rust.
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const { WorkGraph } = require('./work_graph');
const { discoverRepositoryObservation } = require('./work_graph_repo');

const DEFAULT_LOCK_TIMEOUT_MS = 5000;
const DEFAULT_STALE_LOCK_MS = 120000;
const DEFAULT_LOCK_SETTLE_MS = 1000;
const DEFAULT_MAX_STATE_BYTES = 64 * 1024 * 1024;

class WorkGraphStoreError extends Error {
  constructor(message) { super(message); this.name = 'WorkGraphStoreError'; }
}
class WorkGraphLockTimeout extends WorkGraphStoreError {
  constructor(message) { super(message); this.name = 'WorkGraphLockTimeout'; }
}
class WorkGraphStateError extends WorkGraphStoreError {
  constructor(message) { super(message); this.name = 'WorkGraphStateError'; }
}

function finiteNonnegative(value, name) {
  const number = Number(value);
  if (!Number.isFinite(number) || number < 0) {
    throw new WorkGraphStateError(`${name} must be a finite non-negative number`);
  }
  return number;
}

function positiveSafeInteger(value, name) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < 1) {
    throw new WorkGraphStateError(`${name} must be a positive safe integer`);
  }
  return number;
}

function defaultStoreRoot() {
  const base = process.env.ENTROLY_DIR
    ? path.resolve(process.env.ENTROLY_DIR)
    : path.join(os.homedir(), '.entroly');
  return path.join(base, 'work-graphs');
}

function repoKey(repoId) {
  if (!repoId) throw new WorkGraphStateError('repo_id must not be empty');
  return crypto.createHash('sha256').update(String(repoId), 'utf8').digest('hex').slice(0, 32);
}

function secureDirectory(dir) {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  const stat = fs.lstatSync(dir);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new WorkGraphStateError(`unsafe Work Graph directory: ${dir}`);
  }
  if (process.platform !== 'win32') fs.chmodSync(dir, 0o700);
}

function sleepMs(ms) {
  const wait = new Int32Array(new SharedArrayBuffer(4));
  Atomics.wait(wait, 0, 0, Math.max(0, ms));
}

function fsyncDirectory(dir) {
  if (process.platform === 'win32') return;
  let fd;
  try {
    fd = fs.openSync(dir, fs.constants.O_RDONLY);
    fs.fsyncSync(fd);
  } catch (_) {
    // File fsync + atomic rename remain the primary durability guarantee.
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
  }
}

class WorkGraphStore {
  constructor(repoId, options = {}) {
    this.repoId = String(repoId || '');
    this.root = options.root ? path.resolve(String(options.root)) : defaultStoreRoot();
    this.repoDir = path.join(this.root, repoKey(this.repoId));
    this.statePath = path.join(this.repoDir, 'state.json');
    this.lockPath = path.join(this.repoDir, '.lock');
    this.lockTimeoutMs = finiteNonnegative(
      options.lockTimeoutMs ?? DEFAULT_LOCK_TIMEOUT_MS, 'lockTimeoutMs',
    );
    this.staleLockMs = Math.max(1000, finiteNonnegative(
      options.staleLockMs ?? DEFAULT_STALE_LOCK_MS, 'staleLockMs',
    ));
    this.maxStateBytes = Math.max(1024, positiveSafeInteger(
      options.maxStateBytes ?? DEFAULT_MAX_STATE_BYTES, 'maxStateBytes',
    ));
    secureDirectory(this.root);
    secureDirectory(this.repoDir);
  }

  static forRepository(repoPath = '.', options = {}) {
    const observation = discoverRepositoryObservation(repoPath, { observedAtMs: 0 });
    return new WorkGraphStore(observation.repo_id, options);
  }

  readLockToken() {
    try {
      const stat = fs.lstatSync(this.lockPath);
      if (stat.isSymbolicLink() || !stat.isFile()) {
        throw new WorkGraphStateError(`unsafe Work Graph lock path: ${this.lockPath}`);
      }
      return fs.readFileSync(this.lockPath, 'utf8').split('\n', 1)[0];
    } catch (error) {
      if (error instanceof WorkGraphStateError) throw error;
      return '';
    }
  }

  filesystemNowMs() {
    const probe = path.join(this.repoDir, `.clock-${crypto.randomUUID().replace(/-/g, '')}`);
    let fd;
    try {
      fd = fs.openSync(probe, 'wx', 0o600);
      fs.closeSync(fd);
      fd = undefined;
      return fs.statSync(probe).mtimeMs;
    } catch (error) {
      throw new WorkGraphStoreError(`cannot sample Work Graph filesystem clock: ${error.message}`);
    } finally {
      if (fd !== undefined) { try { fs.closeSync(fd); } catch (_) {} }
      try { fs.unlinkSync(probe); } catch (_) {}
    }
  }

  lockIsStale() {
    let stat;
    try { stat = fs.lstatSync(this.lockPath); }
    catch (_) { return false; }
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new WorkGraphStateError(`unsafe Work Graph lock path: ${this.lockPath}`);
    }
    const first = stat.mtimeMs;
    if (this.filesystemNowMs() - first < this.staleLockMs) return false;
    sleepMs(DEFAULT_LOCK_SETTLE_MS);
    let second;
    try { second = fs.statSync(this.lockPath).mtimeMs; }
    catch (_) { return false; }
    return second === first && this.filesystemNowMs() - second >= this.staleLockMs;
  }

  breakStaleLock() {
    if (!this.lockIsStale()) return false;
    try { fs.unlinkSync(this.lockPath); return true; }
    catch (error) { return error && error.code === 'ENOENT'; }
  }

  tryAcquireLock(token) {
    let fd;
    try {
      fd = fs.openSync(this.lockPath, 'wx', 0o600);
      fs.writeFileSync(fd, `${token}\n${(Date.now() / 1000).toFixed(6)}\n`, 'utf8');
      fs.fsyncSync(fd);
      fs.closeSync(fd);
      return true;
    } catch (error) {
      if (fd !== undefined) { try { fs.closeSync(fd); } catch (_) {} }
      if (error && error.code === 'EEXIST') return false;
      throw new WorkGraphStoreError(`cannot acquire Work Graph lock: ${error.message}`);
    }
  }

  acquireLock() {
    const started = process.hrtime.bigint();
    const timeoutNs = BigInt(Math.trunc(this.lockTimeoutMs * 1e6));
    const token = `${os.hostname()}:${process.pid}:${crypto.randomUUID().replace(/-/g, '')}`;
    let delay = 1;
    for (;;) {
      if (this.tryAcquireLock(token)) return token;
      if (this.breakStaleLock()) continue;
      if (process.hrtime.bigint() - started >= timeoutNs) {
        throw new WorkGraphLockTimeout(`timed out acquiring Work Graph lock for ${this.repoId}`);
      }
      const jitter = 0.5 + crypto.randomInt(0, 65536) / 65535;
      sleepMs(Math.max(1, Math.floor(delay * jitter)));
      delay = Math.min(delay * 2, 50);
    }
  }

  releaseLock(token) {
    if (this.readLockToken() !== token) return;
    try { fs.unlinkSync(this.lockPath); } catch (_) {}
  }

  withLock(fn) {
    const token = this.acquireLock();
    try { return fn(); }
    finally { this.releaseLock(token); }
  }

  loadUnlocked() {
    if (!fs.existsSync(this.statePath)) return new WorkGraph(this.repoId);
    const stat = fs.lstatSync(this.statePath);
    if (stat.isSymbolicLink() || !stat.isFile()) {
      throw new WorkGraphStateError(`unsafe Work Graph state: ${this.statePath}`);
    }
    if (stat.size > this.maxStateBytes) {
      throw new WorkGraphStateError(`Work Graph state is ${stat.size} bytes; limit is ${this.maxStateBytes}`);
    }
    let graph;
    try { graph = WorkGraph.fromJSON(fs.readFileSync(this.statePath, 'utf8')); }
    catch (error) { throw new WorkGraphStateError(`cannot load Work Graph state: ${error.message}`); }
    if (graph.repoId !== this.repoId) {
      throw new WorkGraphStateError(
        `stored Work Graph repo mismatch: expected ${this.repoId}, got ${graph.repoId}`,
      );
    }
    return graph;
  }

  saveUnlocked(graph) {
    if (graph.repoId !== this.repoId) {
      throw new WorkGraphStateError(
        `cannot persist foreign Work Graph: expected ${this.repoId}, got ${graph.repoId}`,
      );
    }
    const payload = Buffer.from(graph.exportJSON(false), 'utf8');
    if (payload.length > this.maxStateBytes) {
      throw new WorkGraphStateError(`Work Graph state is ${payload.length} bytes; limit is ${this.maxStateBytes}`);
    }
    secureDirectory(this.repoDir);
    const temp = path.join(this.repoDir, `.state-${process.pid}-${crypto.randomUUID().replace(/-/g, '')}.tmp`);
    let fd;
    try {
      fd = fs.openSync(temp, 'wx', 0o600);
      fs.writeFileSync(fd, payload);
      fs.fsyncSync(fd);
      fs.closeSync(fd);
      fd = undefined;
      fs.renameSync(temp, this.statePath);
      if (process.platform !== 'win32') fs.chmodSync(this.statePath, 0o600);
      fsyncDirectory(this.repoDir);
    } finally {
      if (fd !== undefined) { try { fs.closeSync(fd); } catch (_) {} }
      try { fs.unlinkSync(temp); } catch (_) {}
    }
  }

  load() { return this.withLock(() => this.loadUnlocked()); }

  save(graph) {
    return this.withLock(() => {
      const current = this.loadUnlocked();
      current.merge(graph);
      this.saveUnlocked(current);
      return current;
    });
  }

  submitObservation(observation) {
    if (!observation || observation.repo_id !== this.repoId) {
      throw new WorkGraphStateError(
        `repository identity changed: expected ${this.repoId}, got ${observation && observation.repo_id}`,
      );
    }
    return this.withLock(() => {
      const graph = this.loadUnlocked();
      graph.observeRepository(observation);
      this.saveUnlocked(graph);
      return graph;
    });
  }

  updateRepository(repoPath = '.', options = {}) {
    return this.submitObservation(discoverRepositoryObservation(repoPath, options));
  }

  claimWork(repoPath, options = {}) {
    const agentId = String(options.agentId || '').trim();
    const taskTitle = String(options.taskTitle || '').trim();
    if (!agentId || !taskTitle) {
      throw new WorkGraphStateError('agentId and taskTitle must not be empty');
    }
    const nowMs = options.observedAtMs == null ? Date.now() : Number(options.observedAtMs);
    if (!Number.isSafeInteger(nowMs)) {
      throw new WorkGraphStateError('observedAtMs must be a JavaScript-safe integer');
    }
    const ttlMs = positiveSafeInteger(options.ttlMs ?? 900000, 'ttlMs');
    const leaseId = String(options.leaseId || crypto.randomUUID().replace(/-/g, ''));
    const taskId = String(options.taskId || '');
    const observation = discoverRepositoryObservation(repoPath, {
      agentId,
      sessionId: String(options.sessionId || ''),
      observedAtMs: nowMs,
      taskHint: {
        task_id: taskId,
        title: taskTitle,
        trust: 'observed',
        explicit_status: 'in_progress',
        remaining_work: [],
        source_kind: 'user_statement',
        source_ref: `work-claim:${leaseId}`,
      },
    });
    observation.leases = [{
      lease_id: leaseId,
      agent_id: agentId,
      task_id: taskId,
      scope_paths: [...new Set(options.scopePaths || [])].map(String).sort(),
      scope_symbols: [...new Set(options.scopeSymbols || [])].map(String).sort(),
      expires_at_ms: nowMs + ttlMs,
      source_ref: `work-lease:${leaseId}`,
    }];
    return { graph: this.submitObservation(observation), leaseId };
  }

  coordination(nowMs = Date.now()) { return this.load().coordination(nowMs); }
  resume(workstreamId = null, maxEvidence = 128) {
    return this.load().resume(workstreamId, maxEvidence);
  }
  handoff(workstreamId, fromAgent, toAgent, generatedAtMs = Date.now()) {
    return this.load().handoff(workstreamId, fromAgent, toAgent, generatedAtMs);
  }
}

module.exports = {
  WorkGraphLockTimeout,
  WorkGraphStateError,
  WorkGraphStore,
  WorkGraphStoreError,
};
