'use strict';

// Thin Node transport over the shared Rust/WASM Governance Engine.
// All identity token signing, policy evaluation, risk scoring, audit chain
// hashing, provenance IDs, and supply-chain scanning stay in entroly-engine.
//
// This module provides JS-friendly wrappers that:
//   1. Accept/return plain JS objects (not JSON strings)
//   2. Provide sensible defaults for optional parameters
//   3. Mirror the Python governance/ API surface for cross-language parity

let wasmExports = null;

function ensureWasm() {
  if (wasmExports) return wasmExports;
  try {
    wasmExports = require('../pkg/entroly_wasm');
  } catch {
    throw new Error(
      'Governance WASM module unavailable. Run `wasm-pack build --target nodejs` in entroly-wasm/'
    );
  }
  return wasmExports;
}

// ── Identity ─────────────────────────────────────────────────────────────────

/**
 * Compute an HMAC-style identity token for the given identity payload.
 *
 * @param {object} payload - AgentIdentityPayload fields
 * @param {string} key     - Operator signing key
 * @returns {string} Token string (prefixed `egov1:...`), or empty if key is empty
 */
function computeIdentityToken(payload, key) {
  const wasm = ensureWasm();
  return wasm.governanceComputeIdentityToken(JSON.stringify(payload), key);
}

/**
 * Verify an identity token against a key and payload.
 *
 * @param {object} payload - AgentIdentityPayload fields
 * @param {string} token   - Token to verify
 * @param {string} key     - Operator signing key
 * @returns {boolean} True if the token is valid
 */
function verifyIdentityToken(payload, token, key) {
  const wasm = ensureWasm();
  return wasm.governanceVerifyIdentityToken(JSON.stringify(payload), token, key);
}

// ── Policy Evaluation ─────────────────────────────────────────────────────────

/**
 * Evaluate a policy against an identity for a given scope and resource.
 *
 * @param {object} identity  - AgentIdentity JSON
 * @param {string} scope     - e.g. "write", "write:src/", "deploy"
 * @param {object} policy    - GovernancePolicy JSON
 * @param {object} [options] - { resource, riskLevel }
 * @returns {object} PolicyEvaluation result
 */
function evaluatePolicy(identity, scope, policy, options = {}) {
  const wasm = ensureWasm();
  const { resource = '', riskLevel = 'low' } = options;
  const resultJson = wasm.governanceEvaluatePolicy(
    JSON.stringify(identity),
    scope,
    resource,
    riskLevel,
    JSON.stringify(policy)
  );
  return JSON.parse(resultJson);
}

// ── Risk Scoring ──────────────────────────────────────────────────────────────

/**
 * Compute a composite risk score from named signals.
 *
 * @param {Array<{name: string, raw_score: number, weight: number, note: string}>} signals
 * @returns {object} RiskAssessment
 */
function computeRisk(signals) {
  const wasm = ensureWasm();
  return JSON.parse(wasm.governanceComputeRisk(JSON.stringify(signals)));
}

/**
 * Compute standard risk signals from a change description.
 *
 * @param {object} changeInput - ChangeRiskInput fields
 * @returns {Array} Risk signals
 */
function standardRiskSignals(changeInput) {
  const wasm = ensureWasm();
  return JSON.parse(wasm.governanceStandardRiskSignals(JSON.stringify(changeInput)));
}

// ── Audit Chain ───────────────────────────────────────────────────────────────

/**
 * Compute the chain hash for a new audit entry.
 * Deterministic across Python, Node, and Rust.
 *
 * @param {string} prevChainHash - Previous chain hash (empty for first entry)
 * @param {string} eventId       - Event ID
 * @param {string} payloadJson   - Canonical JSON of the event payload
 * @returns {string} SHA-256 chain hash
 */
function computeAuditChainHash(prevChainHash, eventId, payloadJson) {
  const wasm = ensureWasm();
  return wasm.governanceComputeAuditChainHash(prevChainHash, eventId, payloadJson);
}

/**
 * Verify an audit chain.
 *
 * @param {Array<object>} entries - Audit entries
 * @returns {{ ok: boolean, verified?: number, error?: string }}
 */
function verifyAuditChain(entries) {
  const wasm = ensureWasm();
  return JSON.parse(wasm.governanceVerifyAuditChain(JSON.stringify(entries)));
}

// ── Provenance ────────────────────────────────────────────────────────────────

/**
 * Compute a deterministic provenance node ID.
 *
 * @param {string} entityType
 * @param {string} actorId
 * @param {string} subjectId
 * @param {number} createdAtMs - Unix timestamp in milliseconds
 * @param {string} correlationId
 * @returns {string} SHA-256 node ID
 */
function provenanceNodeId(entityType, actorId, subjectId, createdAtMs, correlationId) {
  const wasm = ensureWasm();
  return wasm.governanceProvenanceNodeId(entityType, actorId, subjectId, createdAtMs, correlationId);
}

// ── Supply-Chain Scanning ─────────────────────────────────────────────────────

/**
 * Scan an MCP tool schema for supply-chain threats.
 *
 * @param {string} toolName       - Name of the tool
 * @param {string} schemaJson     - Raw JSON of the tool schema
 * @param {string} [knownGoodHash] - Optional SHA-256 of last known-good schema
 * @returns {object} SupplyChainScanResult
 */
function scanToolSchema(toolName, schemaJson, knownGoodHash = null) {
  const wasm = ensureWasm();
  return JSON.parse(wasm.governanceScanToolSchema(toolName, schemaJson, knownGoodHash));
}

// ── Anonymous Identity ────────────────────────────────────────────────────────

/**
 * Create an anonymous read-only identity (fail-closed fallback).
 */
function anonymousIdentity() {
  return {
    payload: {
      agent_id: 'anonymous',
      agent_type: 'unknown',
      created_at_ms: 0,
      model: '',
      organization: '',
      scopes: ['read'],
      session_id: '',
      team: '',
      user: '',
    },
    identity_token: '',
    verified: false,
  };
}

module.exports = {
  computeIdentityToken,
  verifyIdentityToken,
  evaluatePolicy,
  computeRisk,
  standardRiskSignals,
  computeAuditChainHash,
  verifyAuditChain,
  provenanceNodeId,
  scanToolSchema,
  anonymousIdentity,
};
