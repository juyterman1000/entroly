'use strict';

const {
  WorkGraph,
  createRoutingDecision,
  createModelExecutionOutcome,
  createVerificationRecord,
  verificationFreshness,
  contextReceiptBuildJSON,
  memoryRecordBuildJSON,
} = require('./index');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

const observation = {
  repo_id: 'repo:test',
  observed_at_ms: 1000,
  repository_label: 'test repo',
  agent_id: 'claude',
  session_id: 'session-1',
  task_hint: {
    task_id: 'task-stream',
    title: 'Fix streaming parity',
    trust: 'observed',
    explicit_status: 'in_progress',
    remaining_work: ['finish Rust parity'],
    source_kind: 'user_statement',
    source_ref: 'user:task',
  },
  branch: {
    name: 'feature/streaming',
    head_sha: 'abc123',
    default_branch: 'main',
    ahead_by: 1,
  },
  changes: [
    { path: 'src/stream.rs', kind: 'modified', staged: false, conflicted: false },
  ],
  decisions: [
    {
      decision_id: 'decision-1',
      text: 'Preserve provider event ordering',
      source_ref: 'checkpoint:1',
      source_kind: 'checkpoint',
      trust: 'observed',
    },
  ],
};

const first = new WorkGraph('repo:test');
const second = new WorkGraph('repo:test');
first.observeRepository(observation);
second.observeRepository(JSON.parse(JSON.stringify(observation)));

assert(first.graphCommitment === second.graphCommitment, 'deterministic commitment drift');
assert(JSON.stringify(first.unfinished()) === JSON.stringify(second.unfinished()), 'unfinished state drift');

const unfinished = first.unfinished();
assert(unfinished.length === 1, `expected one unfinished workstream, got ${unfinished.length}`);
const resume = first.resume(unfinished[0].node_id);
assert(resume.task_labels.includes('Fix streaming parity'), 'resume lost task label');
assert(resume.changed_paths.includes('src/stream.rs'), 'resume lost changed path');

const restored = WorkGraph.fromJSON(first.exportJSON());
assert(restored.graphCommitment === first.graphCommitment, 'round-trip commitment drift');
assert(JSON.stringify(restored.snapshot()) === JSON.stringify(first.snapshot()), 'round-trip snapshot drift');

const workstreamId = unfinished[0].node_id;
const taskId = unfinished[0].task_ids[0];
const contextReceipt = JSON.parse(contextReceiptBuildJSON(
  'repo:test', 'abc123', first.graphCommitment, workstreamId,
  'sha256:sources', JSON.stringify(['src/stream.rs#0:20']),
  JSON.stringify(['src/stream.rs#20:40']), JSON.stringify(['evidence:test']),
  JSON.stringify(['src/stream.rs#20:40']), JSON.stringify(['rh_example']),
  JSON.stringify(['evidence:test']), 512, 'work-scope/v1', 'execution:pending', 1050,
));
first.recordContextReceipt(contextReceipt, 'claude', 'session-1');
const memory = JSON.parse(memoryRecordBuildJSON(
  'repo:test', 'vault/streaming-decision', 'observed', taskId, workstreamId,
  'claude', 'session-1', 'execution:pending', 'sha256:memory',
  JSON.stringify(['evidence:test']), 1060, 1060,
));
first.recordMemory(memory, 1070);
const route = createRoutingDecision({
  repository_id: 'repo:test',
  task_id: taskId,
  workstream_id: workstreamId,
  provider: 'openai',
  model: 'gpt-5',
  runtime: 'responses-api',
  context_budget_tokens: 4096,
  policy_version: 'policy:v1',
  reason_codes: ['capability_match'],
  feature_commitments: ['sha256:features'],
  receipt_id: contextReceipt.receipt_id,
  evidence_ids: ['evidence:route'],
  decided_at_ms: 1100,
});
const outcome = createModelExecutionOutcome({
  routing_id: route.routing_id,
  repository_id: 'repo:test',
  task_id: taskId,
  workstream_id: workstreamId,
  provider: 'openai',
  model: 'gpt-5',
  runtime: 'responses-api',
  receipt_id: contextReceipt.receipt_id,
  request_commitment: 'sha256:request',
  response_commitment: 'sha256:response',
  state: 'succeeded',
  verification_state: 'passed',
  latency_ms: 25,
  input_tokens: 100,
  output_tokens: 20,
  cost_micro_usd: 250,
  evidence_ids: ['evidence:outcome'],
  completed_at_ms: 1200,
});
const verification = createVerificationRecord({
  repository_id: 'repo:test',
  subject_id: outcome.outcome_id,
  subject_commitment: outcome.outcome_commitment,
  verified_repository_commitment: 'abc123',
  verdict: 'passed',
  evidence_ids: ['evidence:test'],
  dependency_commitments: ['sha256:source'],
  observed_at_ms: 1300,
});
assert(
  verificationFreshness(verification, 'abc123', 1300) === 'current',
  'public freshness facade did not preserve Rust verdict',
);
first.recordExecutionChain(route, outcome, verification);
assert(first.snapshot().nodes.some(
  (node) => node.attributes && node.attributes.freshness === 'current',
), 'current verification did not enter the Work Graph');

const receipt = first.handoff(workstreamId, 'claude', 'codex', 2000);
assert(WorkGraph.verifyHandoffIntegrity(receipt), 'handoff integrity failed');
assert(first.verifyHandoff(receipt), 'graph-bound handoff verification failed');
const proof = first.continuationProof(receipt, {
  context_receipt_commitments: [contextReceipt.receipt_commitment],
  routing_commitments: [route.decision_commitment],
  execution_outcome_commitments: [outcome.outcome_commitment],
  verification_commitments: [verification.record_commitment],
  memory_commitments: [memory.record_commitment],
  outstanding_work_refs: ['run Linux CI'],
  recovery_handle_ids: ['rh_example'],
  created_at_ms: 2100,
});
assert(proof.graph_commitment === first.graphCommitment, 'proof is not graph-bound');
const reconstructed = first.reconstructedContinuationProof(workstreamId, 'codex', {
  context_receipt_commitments: [contextReceipt.receipt_commitment],
  verification_commitments: [verification.record_commitment],
  outstanding_work_refs: ['run Windows CI'],
  created_at_ms: 2101,
});
assert(reconstructed.from_agent === '', 'reconstruction invented a source agent');
assert(reconstructed.handoff_commitment === '', 'reconstruction invented a handoff');
assert(reconstructed.outstanding_work_refs.includes('unknown:previous-agent-intent'),
  'reconstruction did not preserve unknown intent');
receipt.to_agent = 'tampered';
assert(!WorkGraph.verifyHandoffIntegrity(receipt), 'tampered receipt passed self-integrity');
assert(!first.verifyHandoff(receipt), 'tampered receipt passed graph-bound verification');

let unsafeTimestampRejected = false;
try { first.coordination(Number.MAX_SAFE_INTEGER + 1); }
catch (_) { unsafeTimestampRejected = true; }
assert(unsafeTimestampRejected, 'unsafe timestamp was accepted by npm/WASM boundary');

let negativeEvidenceRejected = false;
try { first.resume(unfinished[0].node_id, -1); }
catch (_) { negativeEvidenceRejected = true; }
assert(negativeEvidenceRejected, 'negative maxEvidence was accepted');

console.log('Work Graph npm contract: PASS');
