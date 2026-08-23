'use strict';

// Thin Node orchestration for interrupted-agent recovery and fresh handoff.
// Repository observation and persistence happen here; all work-state meaning
// remains in shared Rust.
const { enrichWorktreeContentDigests } = require('./work_graph_content_digest');
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

function resumeRepository(repoPath = '.', options = {}) {
  // Validate before constructing the store or observing Git. Invalid recovery
  // requests must not mutate shared state.
  const maxEvidence = validateMaxEvidence(options.maxEvidence ?? 128);
  const workstreamId = optionalWorkstreamId(options.workstreamId);
  const storeOptions = options.storeOptions || {};
  const repositoryOptions = options.repositoryOptions || {};

  const store = WorkGraphStore.forRepository(repoPath, storeOptions);
  store.submitObservation(passiveObservation(repoPath, repositoryOptions));
  return store.resume(workstreamId, maxEvidence);
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
  const outstanding = [
    ...(Array.isArray(view.changed_paths) ? view.changed_paths : []),
    ...(Array.isArray(view.failures) ? view.failures : []),
  ];
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
};
