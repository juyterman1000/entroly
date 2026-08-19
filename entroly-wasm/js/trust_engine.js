'use strict';

// Thin Node transport over the shared Rust/WASM Trust Engine. All profile,
// scoring, commitment, and guardrail semantics stay in entroly-engine.
function wasmTrustEngine() {
  const { WasmTrustEngine } = require('../pkg/entroly_wasm');
  if (!WasmTrustEngine) {
    throw new Error('Rust Trust Engine is unavailable; rebuild entroly-wasm');
  }
  return WasmTrustEngine;
}

class TrustEngine {
  constructor(profile = 'rag') {
    this._inner = new (wasmTrustEngine())(String(profile));
  }

  assessClaim(evidence, claim) {
    return JSON.parse(this._inner.assessClaimJSON(String(evidence), String(claim)));
  }

  fileCriticality(path) {
    return this._inner.fileCriticality(String(path));
  }

  hasSafetySignal(content) {
    return Boolean(this._inner.hasSafetySignal(String(content)));
  }
}

module.exports = { TrustEngine };
