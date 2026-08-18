'use strict';

const {
  WorkGraphStateError,
  WorkGraphStore,
} = require('./js/work_graph_store');
const {
  handoffRepository,
  resumeRepository,
} = require('./js/work_graph_continuity');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

const original = WorkGraphStore.forRepository;
try {
  const calls = [];
  const fakeStore = {
    updateRepository(repoPath, options) {
      calls.push(['observe', repoPath, options]);
      return { ignored: true };
    },
    resume(workstreamId, maxEvidence) {
      calls.push(['resume', workstreamId, maxEvidence]);
      return { repo_id: 'repo:test', graph_revision: 1, selected_workstream: { node_id: 'w' } };
    },
    handoff(workstreamId, fromAgent, toAgent, generatedAtMs) {
      calls.push(['handoff', workstreamId, fromAgent, toAgent, generatedAtMs]);
      return {
        repo_id: 'repo:test',
        workstream_id: workstreamId,
        from_agent: fromAgent,
        to_agent: toAgent,
      };
    },
  };
  WorkGraphStore.forRepository = (repoPath, options) => {
    calls.push(['store', repoPath, options]);
    return fakeStore;
  };

  const view = resumeRepository('/repo', {
    workstreamId: 'workstream:1',
    maxEvidence: 17,
    storeOptions: { root: '/state' },
    repositoryOptions: { maxCommits: 3 },
  });
  assert(view.repo_id === 'repo:test', 'resume view was not returned');
  assert(JSON.stringify(calls) === JSON.stringify([
    ['store', '/repo', { root: '/state' }],
    ['observe', '/repo', { maxCommits: 3 }],
    ['resume', 'workstream:1', 17],
  ]), 'npm recovery did not run store -> observe -> resume in order');

  calls.length = 0;
  let rejected = false;
  try { resumeRepository('/repo', { maxEvidence: -1 }); }
  catch (error) { rejected = error instanceof WorkGraphStateError; }
  assert(rejected, 'invalid maxEvidence was accepted');
  assert(calls.length === 0, 'invalid recovery request touched repository/store state');

  calls.length = 0;
  const receipt = handoffRepository('/repo', {
    workstreamId: 'workstream:1',
    fromAgent: 'claude',
    toAgent: 'codex',
    generatedAtMs: 1234,
    storeOptions: { root: '/state' },
    repositoryOptions: { maxCommits: 2 },
  });
  assert(receipt.to_agent === 'codex', 'handoff receipt was not returned');
  assert(JSON.stringify(calls) === JSON.stringify([
    ['store', '/repo', { root: '/state' }],
    ['observe', '/repo', { maxCommits: 2 }],
    ['handoff', 'workstream:1', 'claude', 'codex', 1234],
  ]), 'npm handoff did not run store -> observe -> handoff in order');

  calls.length = 0;
  rejected = false;
  try {
    handoffRepository('/repo', {
      workstreamId: '',
      fromAgent: 'claude',
      toAgent: 'codex',
    });
  } catch (error) {
    rejected = error instanceof WorkGraphStateError;
  }
  assert(rejected, 'invalid handoff workstream was accepted');
  assert(calls.length === 0, 'invalid handoff touched repository/store state');

  calls.length = 0;
  rejected = false;
  try {
    handoffRepository('/repo', {
      workstreamId: 'workstream:1',
      fromAgent: 'claude',
      toAgent: 'codex',
      generatedAtMs: Number.MAX_SAFE_INTEGER + 1,
    });
  } catch (error) {
    rejected = error instanceof WorkGraphStateError;
  }
  assert(rejected, 'unsafe handoff timestamp was accepted');
  assert(calls.length === 0, 'unsafe handoff timestamp touched repository/store state');

  console.log('Work Graph npm interrupted-agent continuity contract: PASS');
} finally {
  WorkGraphStore.forRepository = original;
}
