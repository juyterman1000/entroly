'use strict';

// Process-local host handoff transport. Work semantics stay in the Rust graph.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { enrichWorktreeContentDigests } = require('./work_graph_content_digest');
const { discoverRepositoryObservation } = require('./work_graph_repo');
const { graphSourceState, handoffGraph, sourceState } = require('./work_graph_review');
const { WorkGraphStateError } = require('./work_graph_store');

const MAX_HANDOFF_BYTES = 256 * 1024;
const MAX_INBOX_ITEMS = 128;

function inbox(store) {
  const folder = path.join(store.repoDir, 'handoffs');
  fs.mkdirSync(folder, { recursive: true, mode: 0o700 });
  const stat = fs.lstatSync(folder);
  if (stat.isSymbolicLink() || !stat.isDirectory()) {
    throw new WorkGraphStateError('unsafe Work Graph handoff inbox');
  }
  if (process.platform !== 'win32') fs.chmodSync(folder, 0o700);
  return folder;
}

function saveHandoff(store, handoff, continuationProof) {
  if (!store.load().verifyHandoff(handoff)) {
    throw new WorkGraphStateError('cannot save a handoff that does not match the current graph');
  }
  const payload = { schema_version: 1, handoff, continuation_proof: continuationProof };
  const encoded = Buffer.from(JSON.stringify(payload));
  if (encoded.length > MAX_HANDOFF_BYTES) {
    throw new WorkGraphStateError('handoff exceeds local inbox size limit');
  }
  const name = crypto.createHash('sha256').update(String(handoff.payload_commitment)).digest('hex');
  const folder = inbox(store);
  const target = path.join(folder, `${name}.json`);
  const temporary = path.join(folder, `.${name}.${crypto.randomUUID().replace(/-/g, '')}.tmp`);
  try {
    const fd = fs.openSync(temporary, 'wx', 0o600);
    try { fs.writeSync(fd, encoded); fs.fsyncSync(fd); }
    finally { fs.closeSync(fd); }
    fs.renameSync(temporary, target);
  } finally {
    try { fs.unlinkSync(temporary); } catch (_) {}
  }
  return `handoff:${name}`;
}

function pendingHandoffs(root, store, toAgent) {
  if (typeof toAgent !== 'string' || !toAgent.trim() || toAgent.length > 512 || toAgent.includes('\0')) {
    throw new WorkGraphStateError('to_agent must be a bounded agent identifier');
  }
  const folder = inbox(store);
  const files = fs.readdirSync(folder).filter((name) => /^[a-f0-9]{64}\.json$/.test(name))
    .map((name) => ({ name, stat: fs.lstatSync(path.join(folder, name)) }))
    .filter(({ stat }) => stat.isFile() && !stat.isSymbolicLink() && stat.size <= MAX_HANDOFF_BYTES)
    .sort((a, b) => b.stat.mtimeMs - a.stat.mtimeMs).slice(0, MAX_INBOX_ITEMS);
  const graph = store.load();
  const current = sourceState(store.repoId,
    enrichWorktreeContentDigests(root, discoverRepositoryObservation(root)));
  const entries = [];
  for (const { name } of files) {
    try {
      const payload = JSON.parse(fs.readFileSync(path.join(folder, name), 'utf8'));
      const handoff = payload.handoff;
      if (payload.schema_version !== 1 || handoff.to_agent !== toAgent) continue;
      let valid = false;
      try { valid = current === graphSourceState(handoffGraph(graph, handoff), store.repoId); }
      catch (_) { /* invalid receipts stay visible as stale */ }
      entries.push({
        ref: `handoff:${name.slice(0, -5)}`,
        current: valid,
        handoff, continuation_proof: payload.continuation_proof,
      });
    } catch (_) { /* unreadable or invalid records are not agent instructions */ }
  }
  return { status: 'ok', kind: 'work_pending_handoffs', to_agent: toAgent, handoffs: entries };
}

module.exports = { pendingHandoffs, saveHandoff };
