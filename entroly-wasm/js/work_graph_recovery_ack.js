'use strict';

// Same local recovery marker contract as entroly/work_graph_recovery_ack.py.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { WorkGraphStateError } = require('./work_graph_store');

const MARKER = 'pending-recovery-ack.json';
const TRUST = 'untrusted_recovered_work_state';

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function token(view) {
  return `recovery:${crypto.createHash('sha256').update(canonical(view), 'utf8').digest('hex').slice(0, 32)}`;
}

function markerPath(store) { return path.join(store.repoDir, MARKER); }

function pending(store) {
  let raw;
  try { raw = fs.readFileSync(markerPath(store), 'utf8'); }
  catch (error) {
    if (error.code === 'ENOENT') return null;
    return { token: '', trust: TRUST, unknowns: ['unknown:gate-unreadable'] };
  }
  try {
    const record = JSON.parse(raw);
    if (record && typeof record === 'object' && typeof record.token === 'string') return record;
  } catch (_) { /* malformed markers remain armed */ }
  return { token: '', trust: TRUST, unknowns: ['unknown:gate-corrupt'] };
}

function requireAcknowledged(store) {
  if (pending(store) !== null) {
    throw new WorkGraphStateError('recovered work state has not been acknowledged; call work_acknowledge_recovery');
  }
}

function arm(store, view) {
  const value = token(view);
  const unknowns = [...new Set([
    'unknown:previous-agent-intent',
    ...(Array.isArray(view.unknowns) ? view.unknowns : []),
    ...(view.current_worktree_changes_truncated ? ['unknown:current-worktree-changes-truncated'] : []),
  ])].sort();
  const record = { token: value, trust: TRUST, unknowns,
    workspace_commitment: view.current_workspace_commitment };
  fs.mkdirSync(store.repoDir, { recursive: true, mode: 0o700 });
  const temporary = path.join(store.repoDir, `.ack-${crypto.randomUUID()}.tmp`);
  try {
    const fd = fs.openSync(temporary, 'wx', 0o600);
    try { fs.writeSync(fd, Buffer.from(JSON.stringify(record))); fs.fsyncSync(fd); }
    finally { fs.closeSync(fd); }
    fs.renameSync(temporary, markerPath(store));
  } finally {
    try { fs.unlinkSync(temporary); } catch (_) {}
  }
  return { ...record, required: true };
}

function acknowledge(store, supplied, workspaceCommitment) {
  const record = pending(store);
  if (record === null) return { acknowledged: true, already_clear: true };
  if (!record.token || typeof supplied !== 'string' || supplied.trim() !== record.token) {
    throw new WorkGraphStateError('acknowledgement token does not match recovered state; call work_resume again');
  }
  if (record.workspace_commitment && workspaceCommitment !== record.workspace_commitment) {
    throw new WorkGraphStateError('repository changed after recovery; call work_resume again');
  }
  fs.unlinkSync(markerPath(store));
  return { acknowledged: true, token: record.token, trust: TRUST, unknowns: record.unknowns };
}

module.exports = { acknowledge, arm, pending, requireAcknowledged };
