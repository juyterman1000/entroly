'use strict';

// Compact review records use the same Rust Work Graph event log as Python.
// Reviewer verdicts remain untrusted agent statements.
const crypto = require('crypto');
const fs = require('fs');
const path = require('path');
const { enrichWorktreeContentDigests } = require('./work_graph_content_digest');
const { discoverRepositoryIdentity, discoverRepositoryObservation } = require('./work_graph_repo');
const { WorkGraphStore, WorkGraphStateError } = require('./work_graph_store');
const { WorkGraph } = require('./work_graph');
const { workGraphNodeId, workGraphEdgeId } = require('../pkg/entroly_wasm');

const MAX_FINDINGS = 32;
const MAX_FILE_BYTES = 4 * 1024 * 1024;
const VERDICTS = new Set(['approve', 'changes_requested', 'comment']);
const SEVERITIES = new Set(['critical', 'high', 'medium', 'low', 'note']);

function bounded(value, name, limit = 1024) {
  if (typeof value !== 'string' || !value.trim() || value.length > limit || value.includes('\0')) {
    throw new WorkGraphStateError(`${name} must be nonempty text of at most ${limit} characters`);
  }
  return value.trim();
}

function observation(root) {
  return enrichWorktreeContentDigests(root, discoverRepositoryObservation(root));
}

function sourceState(repoId, observed) {
  const branch = observed.branch;
  const changes = new Map((observed.changes || []).map((change) => [
    String(change.path || ''), String(change.content_digest || ''),
  ]));
  return sourceStateParts(repoId, branch.name || '', branch.head_sha || '', changes);
}

function sourceStateParts(repoId, branch, head, changes) {
  const lines = [repoId, String(branch), String(head)];
  for (const [file, digest] of [...changes].sort(([a], [b]) => Buffer.compare(Buffer.from(a), Buffer.from(b)))) {
    lines.push(`${file}\t${digest}`);
  }
  return `sha256:${crypto.createHash('sha256').update(lines.join('\n'), 'utf8').digest('hex')}`;
}

function graphSourceState(graph, repoId) {
  for (const event of [...graph.exportState().events].reverse()) {
    const operations = event.operations || [];
    const repository = operations.find((op) => op.op === 'upsert_node' && op.node
      && op.node.kind === 'repository' && Object.hasOwn(op.node.attributes || {}, 'head_sha'));
    if (!repository) continue;
    const changes = new Map();
    for (const op of operations) {
      if (op.op === 'add_evidence' && op.evidence && op.evidence.kind === 'git_status') {
        changes.set(String(op.evidence.locator), String(op.evidence.digest || ''));
      }
    }
    return sourceStateParts(repoId, repository.node.attributes.branch || '',
      repository.node.attributes.head_sha || '', changes);
  }
  throw new WorkGraphStateError('handoff graph contains no repository observation');
}

function handoffGraph(graph, handoff) {
  const revision = handoff && handoff.graph_revision;
  if (!Number.isSafeInteger(revision) || revision < 1 || revision > graph.revision) {
    throw new WorkGraphStateError('handoff is stale or has an invalid graph revision');
  }
  const historical = revision === graph.revision ? graph : new WorkGraph(graph.repoId);
  if (historical !== graph) {
    for (const event of graph.exportState().events.slice(0, revision)) historical.applyEvent(event);
  }
  if (!historical.verifyHandoff(handoff)) {
    throw new WorkGraphStateError('handoff is stale or fails graph integrity verification');
  }
  return historical;
}

function checkedFindings(root, values) {
  if (!Array.isArray(values) || values.length > MAX_FINDINGS) {
    throw new WorkGraphStateError(`findings must be a list of at most ${MAX_FINDINGS} items`);
  }
  return values.map((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new WorkGraphStateError(`findings[${index}] must be an object`);
    }
    const rel = bounded(item.path, 'finding path', 4096).replace(/\\/g, '/');
    const candidate = path.resolve(root, rel);
    const real = fs.realpathSync(candidate);
    const inside = path.relative(root, real);
    if (!inside || inside === '..' || inside.startsWith(`..${path.sep}`) || path.isAbsolute(inside) || fs.lstatSync(candidate).isSymbolicLink()) {
      throw new WorkGraphStateError('finding path must be a regular file inside the repository');
    }
    const stat = fs.statSync(real);
    if (!stat.isFile() || stat.size > MAX_FILE_BYTES) {
      throw new WorkGraphStateError('finding path must be a readable file of at most 4 MiB');
    }
    if (!Number.isSafeInteger(item.line) || item.line < 1) {
      throw new WorkGraphStateError('finding line must be a positive integer');
    }
    const content = fs.readFileSync(real);
    const text = content.toString('utf8');
    const lineCount = text.length ? text.split(/\r\n|\n|\r/).length - (/[\r\n]$/.test(text) ? 1 : 0) : 0;
    if (item.line > lineCount) throw new WorkGraphStateError('finding line is outside the current file');
    const severity = bounded(item.severity, 'finding severity', 16);
    if (!SEVERITIES.has(severity)) throw new WorkGraphStateError('finding severity is invalid');
    return {
      path: inside.split(path.sep).join('/'),
      line: item.line,
      severity,
      message: bounded(item.message, 'finding message'),
      source_digest: `sha256:${crypto.createHash('sha256').update(content).digest('hex')}`,
    };
  });
}

function checkedChecks(values) {
  if (values == null) return [];
  if (!Array.isArray(values) || values.length > 16) {
    throw new WorkGraphStateError('checks must be a list of at most 16 items');
  }
  return values.map((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new WorkGraphStateError(`checks[${index}] must be an object`);
    }
    if (!Number.isSafeInteger(item.exit_code) || item.exit_code < -255 || item.exit_code > 255) {
      throw new WorkGraphStateError('check exit_code must be an integer between -255 and 255');
    }
    return { name: bounded(item.name, 'check name', 256), exit_code: item.exit_code };
  });
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function recordReview(repoPath, { handoff, reviewerId, verdict, findings, summary, checks, storeOptions = {} }) {
  const reviewer = bounded(reviewerId, 'reviewer_id', 512);
  const choice = bounded(verdict, 'verdict', 32);
  if (!VERDICTS.has(choice)) throw new WorkGraphStateError('invalid review verdict');
  if (!handoff || typeof handoff !== 'object' || handoff.to_agent !== reviewer) {
    throw new WorkGraphStateError('handoff must target the reviewer');
  }
  if (handoff.from_agent === reviewer) throw new WorkGraphStateError('writer and reviewer must differ');
  const root = discoverRepositoryIdentity(repoPath).root;
  const store = WorkGraphStore.forRepository(root, storeOptions);
  const summaryText = bounded(summary, 'summary');
  const validatedChecks = checkedChecks(checks);
  const { result } = store.mutate((graph) => {
    const sealed = handoffGraph(graph, handoff);
    const observed = observation(root);
    const state = sourceState(store.repoId, observed);
    if (state !== graphSourceState(sealed, store.repoId)) {
      throw new WorkGraphStateError('handoff is stale against current repository source');
    }
    const checked = checkedFindings(root, findings);
    if (sourceState(store.repoId, observation(root)) !== state) {
      throw new WorkGraphStateError('repository source changed while recording review');
    }
    if (choice === 'approve' && checked.some((finding) => ['critical', 'high'].includes(finding.severity))) {
      throw new WorkGraphStateError('approve conflicts with blocking findings');
    }
    const payload = {
      schema_version: 1,
      record_kind: 'agent_review',
      repo_id: store.repoId,
      workstream_id: handoff.workstream_id,
      handoff_commitment: handoff.payload_commitment,
      writer_id: handoff.from_agent,
      reviewer_id: reviewer,
      source_state: state,
      verdict: choice,
      summary: summaryText,
      findings: checked,
      checks: validatedChecks,
      trust_boundary: 'agent_statement; source digests observed; verdict unverified',
    };
    const reviewKey = `agent-review:${crypto.createHash('sha256').update(canonical(payload), 'utf8').digest('hex')}`;
    const reviewId = workGraphNodeId('decision', store.repoId, reviewKey);
    const now = Date.now();
    graph.applyEvent({
      observed_at_ms: now,
      source_kind: 'agent_statement',
      source_ref: reviewId,
      actor_id: reviewer,
      operations: [{
        op: 'upsert_node',
        node: {
          node_id: reviewId, kind: 'decision', label: 'agent review recorded',
          trust: 'untrusted', status: 'completed', status_trust: 'untrusted',
          attributes: payload, updated_at_ms: now,
        },
      }, {
        op: 'upsert_edge',
        edge: {
          edge_id: workGraphEdgeId(reviewId, 'part_of', handoff.workstream_id),
          from_node: reviewId, to_node: handoff.workstream_id,
          kind: 'part_of', trust: 'untrusted', updated_at_ms: now,
        },
      }],
    });
    return { review_id: reviewId, ...payload };
  });
  return { status: 'ok', kind: 'work_record_review', review: result };
}

function reviews(repoPath, { workstreamId = '', storeOptions = {} } = {}) {
  const root = discoverRepositoryIdentity(repoPath).root;
  const store = WorkGraphStore.forRepository(root, storeOptions);
  const current = sourceState(store.repoId, observation(root));
  const records = store.load().snapshot().nodes
    .filter((node) => node.attributes && node.attributes.record_kind === 'agent_review')
    .filter((node) => !workstreamId || node.attributes.workstream_id === workstreamId)
    .map((node) => ({ review_id: node.node_id, current: node.attributes.source_state === current, ...node.attributes }));
  return { status: 'ok', kind: 'work_reviews', reviews: records };
}

module.exports = { graphSourceState, handoffGraph, recordReview, reviews, sourceState };
