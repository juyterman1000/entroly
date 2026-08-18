'use strict';

const {
  WorkGraphStateError,
  WorkGraphStore,
} = require('./js/work_graph_store');
const { resumeRepository } = require('./js/work_graph_continuity');

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

  console.log('Work Graph npm interrupted-agent recovery contract: PASS');
} finally {
  WorkGraphStore.forRepository = original;
}
