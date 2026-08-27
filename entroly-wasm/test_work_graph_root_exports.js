'use strict';

const pkg = require('./');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

for (const name of [
  'WorkGraph',
  'WorkGraphStore',
  'WorkContextSnapshotStore',
  'WorkContextSnapshotError',
  'verifyCanonicalSnapshotBytes',
  'discoverRepositoryIdentity',
  'discoverRepositoryObservation',
  'resumeRepository',
  'handoffRepository',
  'handoffRepositoryWithProof',
  'createRoutingDecision',
  'createModelExecutionOutcome',
  'createVerificationRecord',
  'verificationFreshness',
  'continuationProofState',
]) {
  assert(typeof pkg[name] === 'function', `missing npm root Work Graph export: ${name}`);
}

assert(Number.isSafeInteger(pkg.MAX_RESUME_EVIDENCE), 'MAX_RESUME_EVIDENCE is not exported');
assert(pkg.CONTEXT_SNAPSHOT_TOKEN_PREFIX === 'wctx1.', 'snapshot token prefix is not exported');
assert(Number.isSafeInteger(pkg.DEFAULT_MAX_CONTEXT_BYTES), 'snapshot max bytes is not exported');
assert(Number.isSafeInteger(pkg.DEFAULT_MAX_SNAPSHOTS), 'snapshot max entries is not exported');
assert(Number.isSafeInteger(pkg.DEFAULT_MAX_TOTAL_BYTES), 'snapshot max total bytes is not exported');
console.log('Work Graph npm root exports: PASS');
