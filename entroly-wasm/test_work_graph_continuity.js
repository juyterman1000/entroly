'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
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

function git(repo, args) {
  return execFileSync('git', ['-C', repo, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
}

const repo = fs.mkdtempSync(path.join(os.tmpdir(), 'entroly-work-continuity-'));
git(repo, ['init', '-b', 'main']);
git(repo, ['config', 'user.email', 'test@example.com']);
git(repo, ['config', 'user.name', 'Test']);
fs.writeFileSync(path.join(repo, 'app.py'), 'VALUE = 1\n', 'utf8');
git(repo, ['add', 'app.py']);
git(repo, ['commit', '-m', 'initial']);
git(repo, ['checkout', '-b', 'feature/interrupted']);
fs.writeFileSync(path.join(repo, 'app.py'), 'VALUE = 2\n', 'utf8');

const original = WorkGraphStore.forRepository;
try {
  const calls = [];
  let submitted = null;
  const fakeStore = {
    submitObservation(observation) {
      submitted = observation;
      calls.push(['persist', observation.changes[0].content_digest]);
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

  const view = resumeRepository(repo, {
    workstreamId: 'workstream:1',
    maxEvidence: 17,
    storeOptions: { root: '/state' },
    repositoryOptions: { maxCommits: 3 },
  });
  assert(view.repo_id === 'repo:test', 'resume view was not returned');
  assert(submitted && submitted.changes.length === 1, 'passive observation was not persisted');
  assert(submitted.changes[0].path === 'app.py', 'wrong changed path was persisted');
  assert(submitted.changes[0].content_digest.startsWith('git-blob:'), 'resume snapshot lacks content identity');
  assert(calls[0][0] === 'store' && calls[1][0] === 'persist' && calls[2][0] === 'resume',
    'npm recovery did not run store -> fingerprinted persist -> resume in order');

  calls.length = 0;
  submitted = null;
  let rejected = false;
  try { resumeRepository(repo, { maxEvidence: -1 }); }
  catch (error) { rejected = error instanceof WorkGraphStateError; }
  assert(rejected, 'invalid maxEvidence was accepted');
  assert(calls.length === 0, 'invalid recovery request touched repository/store state');

  calls.length = 0;
  submitted = null;
  const receipt = handoffRepository(repo, {
    workstreamId: 'workstream:1',
    fromAgent: 'claude',
    toAgent: 'codex',
    generatedAtMs: 1234,
    storeOptions: { root: '/state' },
    repositoryOptions: { maxCommits: 2 },
  });
  assert(receipt.to_agent === 'codex', 'handoff receipt was not returned');
  assert(submitted && submitted.changes[0].content_digest.startsWith('git-blob:'),
    'handoff snapshot lacks content identity');
  assert(calls[0][0] === 'store' && calls[1][0] === 'persist' && calls[2][0] === 'handoff',
    'npm handoff did not run store -> fingerprinted persist -> handoff in order');

  calls.length = 0;
  rejected = false;
  try {
    handoffRepository(repo, {
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
    handoffRepository(repo, {
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
  fs.rmSync(repo, { recursive: true, force: true });
}
