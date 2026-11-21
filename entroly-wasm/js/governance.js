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
 * Compute the legacy chain hash, which commits to the payload alone.
 *
 * Retained so archives written before `computeAuditChainHashV2` keep
 * verifying. Do not use it for new entries: it leaves the acting agent, the
 * timestamp and `allowed` — the authorization decision itself — outside the
 * commitment, so those fields can be rewritten and the chain still verifies.
 *
 * @param {string} prevChainHash - Previous chain hash (empty for first entry)
 * @param {string} eventId       - Event ID
 * @param {string} payloadJson   - Canonical JSON of the event payload
 * @returns {string} SHA-256 chain hash
 * @deprecated Use {@link computeAuditChainHashV2} for new entries.
 */
function computeAuditChainHash(prevChainHash, eventId, payloadJson) {
  const wasm = ensureWasm();
  return wasm.governanceComputeAuditChainHash(prevChainHash, eventId, payloadJson);
}

/**
 * Compute the chain hash covering the whole entry.
 *
 * Commits to every identity and decision field, not just the payload.
 * Deterministic across Python, Node and Rust: all three call one engine
 * function, so a chain written from Node verifies in Python unchanged.
 *
 * @param {string} prevChainHash - Previous chain hash (empty for first entry)
 * @param {object} entry         - The audit entry about to be stored
 * @returns {string} SHA-256 chain hash
 */
function computeAuditChainHashV2(prevChainHash, entry) {
  const wasm = ensureWasm();
  return wasm.governanceComputeAuditChainHashV2(prevChainHash, JSON.stringify(entry));
}

/**
 * Verify an audit chain.
 *
 * `legacy_covered` counts entries sealed only by the older payload-only hash.
 * It is reported separately rather than folded into `verified`, because those
 * entries carry a weaker guarantee: check `fully_sealed` before treating a
 * chain as proof of what happened.
 *
 * @param {Array<object>} entries - Audit entries
 * @returns {{ ok: boolean, verified?: number, legacy_covered?: number,
 *             fully_sealed?: boolean, error?: string }}
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
  computeAuditChainHashV2,
  verifyAuditChain,
  provenanceNodeId,
  scanToolSchema,
  anonymousIdentity,
};
