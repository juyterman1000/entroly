'use strict';

// Thin npm orchestration over the shared Rust/WASM AI Work Graph. No task-state
// inference, trust upgrades, coordination logic, or handoff verification lives
// here; those semantics are single-source-of-truth in entroly-engine.

const {
  RepositoryDiscoveryError,
  discoverRepositoryObservation,
} = require('./work_graph_repo');

function wasmWorkGraph() {
  const { WasmWorkGraph } = require('../pkg/entroly_wasm');
  if (!WasmWorkGraph) {
    throw new Error('Rust Work Graph is unavailable; rebuild entroly-wasm');
  }
  return WasmWorkGraph;
}

function toJSONText(value) {
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function fromJSONText(value) {
  return JSON.parse(value);
}

function requireSafeInteger(value, name, { min = Number.MIN_SAFE_INTEGER } = {}) {
  const number = Number(value);
  if (!Number.isSafeInteger(number) || number < min) {
    throw new TypeError(`${name} must be a safe integer${min === 0 ? ' >= 0' : ''}`);
  }
  return number;
}

class WorkGraph {
  constructor(repoId, inner = null) {
    const Native = wasmWorkGraph();
    this._inner = inner || new Native(String(repoId || ''));
  }

  static fromJSON(serialized) {
    const Native = wasmWorkGraph();
    return new WorkGraph('', Native.fromJSON(toJSONText(serialized)));
  }

  static fromRepository(repoPath = '.', options = {}) {
    const observation = discoverRepositoryObservation(repoPath, options);
    const graph = new WorkGraph(observation.repo_id);
    graph.observeRepository(observation);
    return graph;
  }

  static verifyHandoffIntegrity(receipt) {
    return wasmWorkGraph().verifyHandoffIntegrityJSON(toJSONText(receipt));
  }

  get repoId() { return this._inner.repoId; }
  get revision() { return Number(this._inner.revision); }
  get graphCommitment() { return this._inner.graphCommitment; }
  get eventCount() { return Number(this._inner.eventCount); }

  applyEvent(event) {
    return this._inner.applyEventJSON(toJSONText(event));
  }

  observeRepository(observation) {
    return this._inner.observeRepositoryJSON(toJSONText(observation));
  }

  refreshRepository(repoPath = '.', options = {}) {
    const observation = discoverRepositoryObservation(repoPath, options);
    if (observation.repo_id !== this.repoId) {
      throw new Error(
        `repository identity changed: expected ${this.repoId}, got ${observation.repo_id}`,
      );
    }
    return this.observeRepository(observation);
  }

  merge(other) {
    const payload = other instanceof WorkGraph ? other.exportJSON(false) : toJSONText(other);
    return Number(this._inner.mergeJSON(payload));
  }

  exportJSON(pretty = false) {
    return this._inner.exportJSON(Boolean(pretty));
  }

  exportState() {
    return fromJSONText(this.exportJSON(false));
  }

  summary() {
    return fromJSONText(this._inner.summaryJSON());
  }

  snapshot(pretty = false) {
    return fromJSONText(this._inner.snapshotJSON(Boolean(pretty)));
  }

  unfinished(pretty = false) {
    return fromJSONText(this._inner.unfinishedJSON(Boolean(pretty)));
  }

  resume(workstreamId = null, maxEvidence = 128, pretty = false) {
    const id = workstreamId == null ? undefined : String(workstreamId);
    return fromJSONText(this._inner.resumeJSON(
      id,
      requireSafeInteger(maxEvidence, 'maxEvidence', { min: 0 }),
      Boolean(pretty),
    ));
  }

  coordination(nowMs = Date.now(), pretty = false) {
    return fromJSONText(this._inner.coordinationJSON(
      requireSafeInteger(nowMs, 'nowMs'),
      Boolean(pretty),
    ));
  }

  handoff(workstreamId, fromAgent, toAgent, generatedAtMs = Date.now(), pretty = false) {
    return fromJSONText(this._inner.handoffJSON(
      String(workstreamId),
      String(fromAgent),
      String(toAgent),
      requireSafeInteger(generatedAtMs, 'generatedAtMs'),
      Boolean(pretty),
    ));
  }

  verifyHandoff(receipt) {
    return this._inner.verifyHandoffJSON(toJSONText(receipt));
  }
}

module.exports = {
  WorkGraph,
  RepositoryDiscoveryError,
  discoverRepositoryObservation,
};
