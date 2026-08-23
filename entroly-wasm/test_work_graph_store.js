'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const {
  WorkGraphLockTimeout,
  WorkGraphStateError,
  WorkGraphStore,
} = require('./js/work_graph_store');
const {
  contextReceiptBuildJSON,
  createModelExecutionOutcome,
  createRoutingDecision,
  createVerificationRecord,
  memoryRecordBuildJSON,
} = require('./index');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

function git(repo, ...args) {
  const result = spawnSync('git', ['-C', repo, ...args], { encoding: 'utf8' });
  if (result.status !== 0) throw new Error(result.stderr || `git ${args.join(' ')} failed`);
  return String(result.stdout || '').trim();
}

function evidenceKindForSource(graph, sourceRef) {
  const document = graph.exportState();
  for (const event of document.events || []) {
    for (const operation of event.operations || []) {
      if (operation.op === 'add_evidence' && operation.evidence && operation.evidence.source_ref === sourceRef) {
        return operation.evidence.kind;
      }
    }
  }
  return null;
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-workgraph-store-'));
const repo = path.join(root, 'repo');
const stateRoot = path.join(root, 'state');
fs.mkdirSync(repo);

try {
  git(repo, 'init', '-b', 'main');
  git(repo, 'config', 'user.email', 'test@example.com');
  git(repo, 'config', 'user.name', 'Test');
  fs.writeFileSync(path.join(repo, 'app.js'), 'console.log("one")\n');
  git(repo, 'add', 'app.js');
  git(repo, 'commit', '-m', 'initial');
  git(repo, 'checkout', '-b', 'feature/work');
  fs.writeFileSync(path.join(repo, 'app.js'), 'console.log("two")\n');

  let nonfiniteRejected = false;
  try {
    new WorkGraphStore('repo:test', {
      root: path.join(root, 'nan'),
      lockTimeoutMs: NaN,
    });
  } catch (error) {
    nonfiniteRejected = error instanceof WorkGraphStateError;
  }
  assert(nonfiniteRejected, 'NaN lock timeout was accepted');

  const store = WorkGraphStore.forRepository(repo, {
    root: stateRoot,
    lockTimeoutMs: 50,
    staleLockMs: 1000,
  });
  const graph = store.updateRepository(repo, {
    agentId: 'codex',
    sessionId: 's1',
    observedAtMs: 1000,
    taskHint: {
      task_id: 'work',
      title: 'Finish work',
      trust: 'observed',
      source_kind: 'user_statement',
      source_ref: 'user:task',
    },
  });
  const loaded = store.load();
  assert(loaded.graphCommitment === graph.graphCommitment, 'persisted commitment drift');
  assert(JSON.stringify(loaded.unfinished()) === JSON.stringify(graph.unfinished()), 'persisted state drift');
  if (process.platform !== 'win32') {
    assert((fs.statSync(store.statePath).mode & 0o777) === 0o600, 'state permissions are not private');
  }

  const token = store.acquireLock();
  try {
    let timedOut = false;
    try { store.load(); }
    catch (error) { timedOut = error instanceof WorkGraphLockTimeout; }
    assert(timedOut, 'lock contention did not time out safely');
    assert(fs.existsSync(store.lockPath), 'contender removed owner lock');
  } finally {
    store.releaseLock(token);
  }

  fs.writeFileSync(store.lockPath, 'stale-owner\n0\n', { mode: 0o600 });
  const old = new Date(Date.now() - 5000);
  fs.utimesSync(store.lockPath, old, old);
  const reclaimed = store.acquireLock();
  const owner = fs.readFileSync(store.lockPath, 'utf8').split('\n', 1)[0];
  assert(owner !== 'stale-owner', 'stale lock was not reclaimed');
  store.releaseLock(reclaimed);

  const second = new WorkGraphStore(store.repoId, {
    root: stateRoot,
    lockTimeoutMs: 50,
    staleLockMs: 1000,
  });
  fs.writeFileSync(path.join(repo, 'second.txt'), 'two\n');
  second.updateRepository(repo, { agentId: 'claude', sessionId: 's2', observedAtMs: 2000 });
  assert(store.load().eventCount >= 2, 'latest-state merge lost an observation');

  const claimA = store.claimWork(repo, {
    agentId: 'claude',
    taskTitle: 'Fix auth',
    taskId: 'auth',
    scopePaths: ['src/auth'],
    observedAtMs: 3000,
  });
  assert(
    evidenceKindForSource(claimA.graph, `work-claim:${claimA.leaseId}`) === 'agent_statement',
    'npm agent claim was not recorded as agent_statement',
  );
  const claimB = store.claimWork(repo, {
    agentId: 'codex',
    taskTitle: 'Add auth tests',
    taskId: 'auth-tests',
    scopePaths: ['src/auth/token.js'],
    observedAtMs: 3100,
  });
  assert(claimA.leaseId !== claimB.leaseId, 'leases were not independently identified');
  const report = store.coordination(3200);
  assert(report.active_leases === 2, 'active lease count drift');
  assert(report.conflicts.length === 1, 'overlapping leases were not surfaced');

  const humanClaim = store.claimWork(repo, {
    agentId: 'human-cli',
    taskTitle: 'Review auth',
    taskId: 'auth-review',
    sourceKind: 'user_statement',
    scopePaths: ['docs/auth'],
    ttlMs: 1,
    observedAtMs: 3300,
  });
  assert(
    evidenceKindForSource(humanClaim.graph, `work-claim:${humanClaim.leaseId}`) === 'user_statement',
    'explicit npm human claim was not recorded as user_statement',
  );
  let badSourceRejected = false;
  try {
    store.claimWork(repo, {
      agentId: 'bad', taskTitle: 'Bad provenance', sourceKind: 'verified', observedAtMs: 3400,
    });
  } catch (error) {
    badSourceRejected = error instanceof WorkGraphStateError;
  }
  assert(badSourceRejected, 'npm accepted an unsupported claim provenance');

  // The product chain must be durable, not merely available on an in-memory
  // WorkGraph instance. Each mutation is loaded, commitment-checked, appended,
  // and atomically persisted under the store lock.
  const selected = store.resume();
  const workstreamId = selected.selected_workstream.node_id;
  const taskId = selected.selected_workstream.task_ids[0];
  const beforeReceipt = store.load();
  const head = git(repo, 'rev-parse', 'HEAD');
  const contextReceipt = JSON.parse(contextReceiptBuildJSON(
    store.repoId, head, beforeReceipt.graphCommitment, workstreamId,
    'sha256:sources', JSON.stringify(['app.js#0:20']), JSON.stringify([]),
    JSON.stringify(['evidence:test']), JSON.stringify([]), JSON.stringify([]),
    JSON.stringify(['evidence:test']), 256, 'work-scope/v1', 'execution:pending', 3500,
  ));
  const receiptWrite = store.recordContextReceipt(contextReceipt, {
    agentId: 'codex', sessionId: 's3',
  });
  assert(receiptWrite.graph.eventCount === beforeReceipt.eventCount + 1,
    'context receipt was not durably appended');

  const memory = JSON.parse(memoryRecordBuildJSON(
    store.repoId, 'vault/auth-decision', 'observed', taskId, workstreamId,
    'codex', 's3', 'execution:pending', 'sha256:memory',
    JSON.stringify(['evidence:test']), 3510, 3510,
  ));
  const memoryWrite = store.recordMemory(memory, 3520);
  assert(memoryWrite.graph.eventCount === receiptWrite.graph.eventCount + 1,
    'memory record was not durably appended');

  const route = createRoutingDecision({
    repository_id: store.repoId,
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
    decided_at_ms: 3530,
  });
  const outcome = createModelExecutionOutcome({
    routing_id: route.routing_id,
    repository_id: store.repoId,
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
    completed_at_ms: 3540,
  });
  const verification = createVerificationRecord({
    repository_id: store.repoId,
    subject_id: outcome.outcome_id,
    subject_commitment: outcome.outcome_commitment,
    verified_repository_commitment: head,
    verdict: 'passed',
    evidence_ids: ['evidence:test'],
    dependency_commitments: ['sha256:source'],
    observed_at_ms: 3550,
  });
  const executionWrite = store.recordExecutionChain(route, outcome, verification);
  assert(executionWrite.graph.eventCount === memoryWrite.graph.eventCount + 1,
    'execution chain was not durably appended');
  assert(store.load().graphCommitment === executionWrite.graph.graphCommitment,
    'durable execution chain commitment drifted on reload');

  const reconstructed = store.reconstructedContinuationProof(workstreamId, 'claude', {
    verification_commitments: [verification.record_commitment],
    outstanding_work_refs: ['run package tests'],
    created_at_ms: 3560,
  });
  assert(reconstructed.from_agent === '' && reconstructed.handoff_commitment === '',
    'durable no-handoff reconstruction invented handoff evidence');
  assert(reconstructed.outstanding_work_refs.includes('unknown:previous-agent-intent'),
    'durable no-handoff reconstruction hid unknown intent');

  const document = JSON.parse(fs.readFileSync(store.statePath, 'utf8'));
  document.graph_commitment = '0'.repeat(64);
  fs.writeFileSync(store.statePath, JSON.stringify(document), 'utf8');
  let tamperRejected = false;
  try { store.load(); }
  catch (error) { tamperRejected = error instanceof WorkGraphStateError; }
  assert(tamperRejected, 'tampered persisted graph was accepted');

  if (process.platform !== 'win32') {
    const real = path.join(root, 'real-state');
    const alias = path.join(root, 'alias-state');
    fs.mkdirSync(real);
    fs.symlinkSync(real, alias, 'dir');
    let symlinkRejected = false;
    try { new WorkGraphStore('repo:test', { root: alias }); }
    catch (error) { symlinkRejected = error instanceof WorkGraphStateError; }
    assert(symlinkRejected, 'symlinked Work Graph root was accepted');
  }

  console.log('Work Graph npm shared-store contract: PASS');
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
