#!/usr/bin/env node
'use strict';

const assert = require('assert');
const crypto = require('crypto');
const { TrustEngine, WorkGraph } = require('./index');

function observation() {
  return {
    repo_id: 'repo:context-trust-test',
    observed_at_ms: 1234,
    repository_label: 'demo',
    agent_id: 'agent:test',
    session_id: 'session:test',
    task_hint: {
      task_id: 'task:auth',
      title: 'repair auth',
      trust: 'observed',
      explicit_status: 'in_progress',
      remaining_work: ['run tests'],
      source_kind: 'user_statement',
      source_ref: 'test://task',
    },
    branch: {
      name: 'main', head_sha: 'abc', base_ref: 'refs/heads/main', default_branch: 'main',
      ahead_by: 0, behind_by: 0, merge_in_progress: false, rebase_in_progress: false,
      detached: false,
    },
    changes: [{ path: 'src/auth.py', kind: 'modified', staged: false, conflicted: false, old_path: '' }],
    commits: [], verifications: [], decisions: [], claims: [], leases: [], model_executions: [],
  };
}

const graph = new WorkGraph('repo:context-trust-test');
graph.observeRepository(observation());
const scope = graph.contextScope();
assert.strictEqual(scope.repo_id, graph.repoId);
assert.strictEqual(scope.graph_revision, graph.revision);
assert.strictEqual(scope.graph_commitment, graph.graphCommitment);
assert.ok(scope.changed_paths.includes('src/auth.py'));
assert.equal(scope.changed_paths_total, scope.changed_paths.length);
assert.match(scope.changed_paths_commitment, /^sha256:/);
assert.ok(Array.isArray(scope.symbol_ids));
assert.equal(scope.symbol_ids_total, scope.symbol_ids.length);
const scopeText = JSON.stringify(scope);
assert.ok(!scopeText.includes('repair auth'));
assert.ok(!scopeText.includes('run tests'));

const evidence = 'The service retries a request three times before returning an error.';
const claim = 'The service retries a request three times.';
const trust = new TrustEngine('rag');
const assessment = trust.assessClaim(evidence, claim);
const digest = crypto.createHash('sha256').update(evidence, 'utf8').digest('hex');
assert.strictEqual(assessment.evidence_commitment, `sha256:${digest}`);
assert.ok(['supported', 'unsupported', 'unknown'].includes(assessment.status));
assert.strictEqual(trust.fileCriticality('file:SECURITY.md'), 'safety');
assert.strictEqual(trust.hasSafetySignal('AWS_SECRET_ACCESS_KEY=example'), true);
assert.throws(() => new TrustEngine('rga'));
console.log('Context/Trust delivery tests passed');
