'use strict';

const { WorkGraph } = require('./index');

function assert(condition, message) {
  if (!condition) throw new Error(message || 'assertion failed');
}

function milliseconds(operation) {
  const started = process.hrtime.bigint();
  const value = operation();
  return [value, Number(process.hrtime.bigint() - started) / 1e6];
}

function percentile(values, fraction) {
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.min(ordered.length - 1, Math.max(0, Math.round((ordered.length - 1) * fraction)));
  return ordered[index];
}

function digest(index) {
  return `git-blob:${index.toString(16).padStart(40, '0')}`;
}

function change(index, version) {
  return {
    path: `src/module_${String(index).padStart(5, '0')}.js`,
    kind: 'modified',
    staged: false,
    conflicted: false,
    content_digest: digest(version),
  };
}

function observation(files, observedAtMs, version) {
  return {
    repo_id: 'repo:wasm-work-graph-performance',
    observed_at_ms: observedAtMs,
    repository_label: 'WASM performance fixture',
    branch: {
      name: 'feature/performance',
      head_sha: 'head-performance',
      default_branch: 'main',
      ahead_by: 1,
    },
    changes: Array.from({ length: files }, (_, index) => change(index, version + index)),
  };
}

const files = 2000;
const edits = 500;
const polls = 100;
const thresholds = {
  initial_observation: 10000,
  incremental_append_p95: 250,
  export: 5000,
  import_rebuild: 10000,
  resume: 1000,
  context_scope: 1000,
  coordination: 1000,
  wasm_summary_p95: 100,
};

const graph = new WorkGraph('repo:wasm-work-graph-performance');
const initial = observation(files, 1000, 1);
const [, initialMs] = milliseconds(() => graph.observeRepository(initial));
const workstreamId = graph.unfinished()[0].node_id;

const beforePolls = graph.eventCount;
const [, pollMs] = milliseconds(() => {
  for (let index = 0; index < polls; index += 1) {
    graph.observeRepository({ ...initial, observed_at_ms: 1001 + index });
  }
});
const passiveEventGrowth = graph.eventCount - beforePolls;

const appendMs = [];
for (let index = 0; index < edits; index += 1) {
  const incremental = observation(1, 2000 + index, 10000 + index);
  appendMs.push(milliseconds(() => graph.observeRepository(incremental))[1]);
}

const [exported, exportMs] = milliseconds(() => graph.exportJSON());
const [restored, importMs] = milliseconds(() => WorkGraph.fromJSON(exported));
const [resume, resumeMs] = milliseconds(() => restored.resume(workstreamId));
const [scope, contextMs] = milliseconds(() => restored.contextScope(workstreamId));
const [, coordinationMs] = milliseconds(() => restored.coordination(10000));
const summaryMs = Array.from({ length: 100 }, () => milliseconds(() => restored.summary())[1]);

const measurements = {
  initial_observation: initialMs,
  incremental_append_p95: percentile(appendMs, 0.95),
  incremental_append_max: Math.max(...appendMs),
  passive_poll_total: pollMs,
  export: exportMs,
  import_rebuild: importMs,
  resume: resumeMs,
  context_scope: contextMs,
  coordination: coordinationMs,
  wasm_summary_p95: percentile(summaryMs, 0.95),
};
for (const [name, threshold] of Object.entries(thresholds)) {
  assert(measurements[name] <= threshold, `${name}=${measurements[name].toFixed(3)}ms exceeds ${threshold}ms`);
}
assert(passiveEventGrowth === 0, `passive polling appended ${passiveEventGrowth} events`);
assert(restored.eventCount === Math.ceil(files / 512) + edits, 'WASM event count drifted');
assert(restored.graphCommitment === graph.graphCommitment, 'WASM import commitment drifted');
assert(resume.graph_commitment === graph.graphCommitment, 'WASM resume lost graph binding');
assert(scope.graph_commitment === graph.graphCommitment, 'WASM context scope lost graph binding');
assert(scope.changed_paths_total === files, 'WASM scope omitted changed paths');
assert(scope.changed_paths.length <= 512, 'WASM scope exceeded inline path bound');
assert(Buffer.byteLength(exported, 'utf8') <= 64 * 1024 * 1024, 'WASM state exceeded 64 MiB');

process.stdout.write(`${JSON.stringify({
  schema_version: 'entroly.work-graph-wasm-performance.v1',
  inputs: { files, edits, passive_polls: polls },
  measurements_ms: Object.fromEntries(
    Object.entries(measurements).map(([name, value]) => [name, Number(value.toFixed(6))]),
  ),
  state: {
    bytes: Buffer.byteLength(exported, 'utf8'),
    events: restored.eventCount,
    passive_event_growth: passiveEventGrowth,
    changed_paths_inline: scope.changed_paths.length,
    changed_paths_total: scope.changed_paths_total,
  },
  thresholds_ms: thresholds,
  passed: true,
})}\nWork Graph WASM performance gate: PASS\n`);
