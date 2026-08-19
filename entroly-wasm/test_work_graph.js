'use strict';

const { WorkGraph } = require('./index');

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

const receipt = first.handoff(unfinished[0].node_id, 'claude', 'codex', 2000);
assert(WorkGraph.verifyHandoffIntegrity(receipt), 'handoff integrity failed');
assert(first.verifyHandoff(receipt), 'graph-bound handoff verification failed');
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
