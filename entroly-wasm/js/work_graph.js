'use strict';

// Thin npm orchestration over the shared Rust/WASM AI Work Graph. No task-state
// inference, trust upgrades, coordination logic, or handoff verification lives
// here; those semantics are single-source-of-truth in entroly-engine.

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

class WorkGraph {
  constructor(repoId, inner = null) {
    const Native = wasmWorkGraph();
    this._inner = inner || new Native(String(repoId || ''));
  }

  static fromJSON(serialized) {
    const Native = wasmWorkGraph();
    return new WorkGraph('', Native.fromJSON(toJSONText(serialized)));
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
    return fromJSONText(this._inner.resumeJSON(id, Number(maxEvidence), Boolean(pretty)));
  }

  coordination(nowMs = Date.now(), pretty = false) {
    return fromJSONText(this._inner.coordinationJSON(Number(nowMs), Boolean(pretty)));
  }

  handoff(workstreamId, fromAgent, toAgent, generatedAtMs = Date.now(), pretty = false) {
    return fromJSONText(this._inner.handoffJSON(
      String(workstreamId),
      String(fromAgent),
      String(toAgent),
      Number(generatedAtMs),
      Boolean(pretty),
    ));
  }

  verifyHandoff(receipt) {
    return this._inner.verifyHandoffJSON(toJSONText(receipt));
  }
}

module.exports = { WorkGraph };
