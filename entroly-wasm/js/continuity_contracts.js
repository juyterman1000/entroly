'use strict';

// Ergonomic Node facade over the canonical Rust continuity contracts. This
// module performs transport conversion and JavaScript safe-integer checks only;
// identity, commitments, cross-field validation and freshness remain in Rust.

function native() {
  return require('../pkg/entroly_wasm');
}

function asObject(value) {
  return typeof value === 'string' ? JSON.parse(value) : value;
}

function checkedInput(input, integerFields) {
  const value = { ...(input || {}) };
  for (const field of integerFields) {
    if (value[field] == null) continue;
    const number = Number(value[field]);
    if (!Number.isSafeInteger(number) || number < 0) {
      throw new TypeError(`${field} must be a JavaScript-safe integer >= 0`);
    }
    value[field] = number;
  }
  return JSON.stringify(value);
}

function createRoutingDecision(input) {
  return JSON.parse(native().routingDecisionBuildJSON(checkedInput(input, [
    'context_budget_tokens', 'decided_at_ms',
  ])));
}

function createModelExecutionOutcome(input) {
  return JSON.parse(native().modelExecutionOutcomeBuildJSON(checkedInput(input, [
    'latency_ms', 'input_tokens', 'output_tokens', 'cost_micro_usd', 'completed_at_ms',
  ])));
}

function createVerificationRecord(input) {
  return JSON.parse(native().verificationRecordBuildJSON(checkedInput(input, [
    'observed_at_ms', 'valid_until_ms',
  ])));
}

function verificationFreshness(
  record,
  currentRepositoryCommitment,
  nowMs,
  invalidatedCommitments = [],
) {
  if (!Number.isSafeInteger(nowMs) || nowMs < 0) {
    throw new TypeError('nowMs must be a JavaScript-safe integer >= 0');
  }
  return native().verificationRecordFreshness(
    JSON.stringify(asObject(record)),
    String(currentRepositoryCommitment),
    nowMs,
    JSON.stringify(invalidatedCommitments),
  );
}

function continuationProofState(proof, repositoryId, graphRevision, graphCommitment) {
  if (!Number.isSafeInteger(graphRevision) || graphRevision < 0) {
    throw new TypeError('graphRevision must be a JavaScript-safe integer >= 0');
  }
  return native().workContinuationProofState(
    JSON.stringify(asObject(proof)),
    String(repositoryId),
    graphRevision,
    String(graphCommitment),
  );
}

module.exports = {
  continuationProofState,
  createModelExecutionOutcome,
  createRoutingDecision,
  createVerificationRecord,
  verificationFreshness,
};
