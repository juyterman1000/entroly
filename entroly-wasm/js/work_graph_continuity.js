'use strict';

// Thin Node orchestration for interrupted-agent recovery. Repository observation
// and persistence happen here; all work-state meaning remains in shared Rust.
const { WorkGraphStateError, WorkGraphStore } = require('./work_graph_store');

const MAX_RESUME_EVIDENCE = 4096;

function validateMaxEvidence(value) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < 0 || number > MAX_RESUME_EVIDENCE) {
    throw new WorkGraphStateError(
      `maxEvidence must be a safe integer between 0 and ${MAX_RESUME_EVIDENCE}`,
    );
  }
  return number;
}

function resumeRepository(repoPath = '.', options = {}) {
  // Validate before constructing the store or observing Git. Invalid recovery
  // requests must not mutate shared state.
  const maxEvidence = validateMaxEvidence(options.maxEvidence ?? 128);
  const workstreamId = options.workstreamId == null || options.workstreamId === ''
    ? null
    : String(options.workstreamId);
  const storeOptions = options.storeOptions || {};
  const repositoryOptions = options.repositoryOptions || {};

  const store = WorkGraphStore.forRepository(repoPath, storeOptions);
  store.updateRepository(repoPath, repositoryOptions);
  return store.resume(workstreamId, maxEvidence);
}

module.exports = {
  MAX_RESUME_EVIDENCE,
  resumeRepository,
};
