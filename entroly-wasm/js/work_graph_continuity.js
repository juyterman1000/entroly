'use strict';

// Thin Node orchestration for interrupted-agent recovery and fresh handoff.
// Repository observation and persistence happen here; all work-state meaning
// remains in shared Rust.
const { enrichWorktreeContentDigests } = require('./work_graph_content_digest');
const crypto = require('crypto');
const { discoverRepositoryObservation } = require('./work_graph_repo');
const { WorkGraphStateError, WorkGraphStore } = require('./work_graph_store');

const MAX_RESUME_EVIDENCE = 4096;
const MAX_WORK_ID_CHARS = 512;

function validateMaxEvidence(value) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < 0 || number > MAX_RESUME_EVIDENCE) {
    throw new WorkGraphStateError(
      `maxEvidence must be a safe integer between 0 and ${MAX_RESUME_EVIDENCE}`,
    );
  }
  return number;
}

function requiredId(value, name) {
  const text = String(value == null ? '' : value).trim();
  if (!text) throw new WorkGraphStateError(`${name} must not be empty`);
  if (text.length > MAX_WORK_ID_CHARS || text.includes('\0')) {
    throw new WorkGraphStateError(
      `${name} may not exceed ${MAX_WORK_ID_CHARS} characters or contain NUL`,
    );
  }
  return text;
}

function optionalWorkstreamId(value) {
  if (value == null || value === '') return null;
  const text = String(value).trim();
  if (text.length > MAX_WORK_ID_CHARS || text.includes('\0')) {
    throw new WorkGraphStateError(
      `workstreamId may not exceed ${MAX_WORK_ID_CHARS} characters or contain NUL`,
    );
  }
  return text || null;
}

function passiveObservation(repoPath, repositoryOptions) {
  const observation = discoverRepositoryObservation(repoPath, repositoryOptions);
  return enrichWorktreeContentDigests(repoPath, observation);
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function workspaceCommitment(observation) {
  const stable = {
    repo_id: observation.repo_id ?? null,
    branch: observation.branch ?? null,
    changes: observation.changes ?? null,
    path_disclosure: observation.path_disclosure ?? null,
  };
  return `workspace:${crypto.createHash('sha256').update(canonical(stable), 'utf8').digest('hex')}`;
}

function currentWorkspaceCommitment(repoPath) {
  return workspaceCommitment(passiveObservation(repoPath, {}));
}

function resumableWorkstreamId(store) {
  const unfinished = store.load().unfinished();
  const content = ['remaining_work', 'decision_ids', 'verification_ids', 'failure_ids', 'agent_ids'];
  const selected = unfinished.find((item) => content.some((field) => Array.isArray(item[field]) && item[field].length))
    || unfinished[0];
  return selected ? selected.node_id : null;
}

function attachCurrentWorktreeChanges(view, observation, maxPaths) {
  const paths = [...new Set((observation.changes || []).map((item) => item.path).filter(Boolean))].sort();
  return {
    ...view,
    current_worktree_changed_paths: paths.slice(0, maxPaths),
    current_worktree_change_count: paths.length,
    current_worktree_changes_truncated: paths.length > maxPaths,
    current_workspace_commitment: workspaceCommitment(observation),
  };
}

function resumeRepository(repoPath = '.', options = {}) {
  // Validate before constructing the store or observing Git. Invalid recovery
  // requests must not mutate shared state.
  const maxEvidence = validateMaxEvidence(options.maxEvidence ?? 128);
  const workstreamId = optionalWorkstreamId(options.workstreamId);
  const storeOptions = options.storeOptions || {};
  const repositoryOptions = options.repositoryOptions || {};

  const store = WorkGraphStore.forRepository(repoPath, storeOptions);
  const observation = passiveObservation(repoPath, repositoryOptions);
  store.submitObservation(observation);
  const selected = workstreamId || resumableWorkstreamId(store);
  return attachCurrentWorktreeChanges(store.resume(selected, maxEvidence), observation, maxEvidence);
}

function handoffRepository(repoPath = '.', options = {}) {
  // A receipt is a state-sealing operation. Validate everything before touching
  // disk, then refresh the bounded durable facts exactly once before Rust seals
  // the handoff against the graph commitment.
  const workstreamId = requiredId(options.workstreamId, 'workstreamId');
  const fromAgent = requiredId(options.fromAgent, 'fromAgent');
  const toAgent = requiredId(options.toAgent, 'toAgent');
  const storeOptions = options.storeOptions || {};
  const repositoryOptions = options.repositoryOptions || {};
  const generatedAtMs = options.generatedAtMs == null ? Date.now() : Number(options.generatedAtMs);
  if (!Number.isSafeInteger(generatedAtMs)) {
    throw new WorkGraphStateError('generatedAtMs must be a JavaScript-safe integer');
  }

  const store = WorkGraphStore.forRepository(repoPath, storeOptions);
  store.submitObservation(passiveObservation(repoPath, repositoryOptions));
  return store.handoff(workstreamId, fromAgent, toAgent, generatedAtMs);
}

function continuationOutstandingRefs(view) {
  const safeView = view && typeof view === 'object' ? view : {};
  const selected = safeView.selected_workstream && typeof safeView.selected_workstream === 'object'
    ? safeView.selected_workstream
    : {};
  const values = [
    ...(Array.isArray(safeView.outstanding_work) ? safeView.outstanding_work : []),
    ...(Array.isArray(safeView.current_worktree_changed_paths) ? safeView.current_worktree_changed_paths : []),
    ...(Array.isArray(safeView.changed_paths) ? safeView.changed_paths : []),
    ...(Array.isArray(safeView.failures) ? safeView.failures : []),
    ...(Array.isArray(selected.task_ids) ? selected.task_ids : []),
  ];
  return [...new Set(values
    .filter((value) => typeof value === 'string' && value.trim())
    .map((value) => value.trim()))];
}

function handoffRepositoryWithProof(repoPath = '.', options = {}) {
  // Preserve the receipt-only API above for compatibility while making the
  // complete continuation artifact available as one low-ceremony operation.
  const workstreamId = requiredId(options.workstreamId, 'workstreamId');
  const fromAgent = requiredId(options.fromAgent, 'fromAgent');
  const toAgent = requiredId(options.toAgent, 'toAgent');
  const storeOptions = options.storeOptions || {};
  const repositoryOptions = options.repositoryOptions || {};
  const generatedAtMs = options.generatedAtMs == null ? Date.now() : Number(options.generatedAtMs);
  if (!Number.isSafeInteger(generatedAtMs)) {
    throw new WorkGraphStateError('generatedAtMs must be a JavaScript-safe integer');
  }

  const store = WorkGraphStore.forRepository(repoPath, storeOptions);
  store.submitObservation(passiveObservation(repoPath, repositoryOptions));
  const handoff = store.handoff(workstreamId, fromAgent, toAgent, generatedAtMs);
  const view = store.resume(workstreamId, 128);
  const outstanding = continuationOutstandingRefs(view);
  const continuationProof = store.continuationProof(handoff, {
    outstanding_work_refs: outstanding,
    created_at_ms: generatedAtMs,
  });
  return { handoff, continuation_proof: continuationProof };
}

module.exports = {
  MAX_RESUME_EVIDENCE,
  MAX_WORK_ID_CHARS,
  handoffRepository,
  handoffRepositoryWithProof,
  resumeRepository,
  currentWorkspaceCommitment,
  continuationOutstandingRefs,
};
