//! Thin PyO3 boundary over `entroly_engine::governance`.
//!
//! All governance computation (identity token signing, policy evaluation,
//! risk scoring, audit chain hashing, provenance IDs, supply-chain scanning)
//! lives in `entroly-engine`. This file only:
//!   - Converts Python types to Rust types
//!   - Serializes Rust results to JSON strings (Python parses JSON)
//!   - Maps GovernanceError → PyValueError
//!
//! Python callers import these via `entroly_core` and deserialize the JSON
//! into the typed domain objects in `entroly/governance/`.

use entroly_engine::governance::{
    compute_audit_chain_hash, compute_risk, evaluate_policy, scan_tool_schema,
    standard_risk_signals, verify_audit_chain, AgentIdentity, AgentIdentityPayload,
    AuditEntry, ChangeRiskInput, GovernancePolicy, ProvenanceNode, RiskLevel, RiskSignal,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;


fn py_err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn to_json<T: serde::Serialize>(val: &T) -> PyResult<String> {
    serde_json::to_string(val).map_err(py_err)
}

// ── Identity ─────────────────────────────────────────────────────────────────

/// Compute an HMAC-style identity token for the given payload JSON and key.
///
/// Returns the token string (prefixed `egov1:...`), or empty string if key is empty.
/// Called by Python's `governance/identity.py` to produce signed identity tokens.
#[pyfunction]
pub(crate) fn governance_compute_identity_token(
    payload_json: &str,
    key: &str,
) -> PyResult<String> {
    let payload: AgentIdentityPayload =
        serde_json::from_str(payload_json).map_err(py_err)?;
    payload.compute_token(key).map_err(py_err)
}

/// Verify an identity token against a key and payload JSON.
///
/// Returns `true` if the token is valid (constant-time comparison), `false` otherwise.
#[pyfunction]
pub(crate) fn governance_verify_identity_token(
    payload_json: &str,
    token: &str,
    key: &str,
) -> PyResult<bool> {
    let payload: AgentIdentityPayload =
        serde_json::from_str(payload_json).map_err(py_err)?;
    payload.verify_token(token, key).map_err(py_err)
}

// ── Policy Evaluation ─────────────────────────────────────────────────────────

/// Evaluate a policy against an identity for a given scope and resource.
///
/// Args:
///   identity_json: JSON of AgentIdentity
///   scope: e.g. "write", "write:src/", "deploy"
///   resource: resource path being accessed (empty string if N/A)
///   risk_level: "low" | "medium" | "high" | "critical"
///   policy_json: JSON of GovernancePolicy
///
/// Returns: JSON of PolicyEvaluation
#[pyfunction]
#[pyo3(signature = (identity_json, scope, resource="", risk_level="low", policy_json="{}"))]
pub(crate) fn governance_evaluate_policy(
    identity_json: &str,
    scope: &str,
    resource: &str,
    risk_level: &str,
    policy_json: &str,
) -> PyResult<String> {
    let identity: AgentIdentity =
        serde_json::from_str(identity_json).map_err(py_err)?;
    let policy: GovernancePolicy =
        serde_json::from_str(policy_json).map_err(py_err)?;
    let rl = parse_risk_level(risk_level)?;
    let result = evaluate_policy(&identity, scope, resource, rl, &policy);
    to_json(&result)
}

// ── Risk Scoring ──────────────────────────────────────────────────────────────

/// Compute a composite risk score from a list of named signals.
///
/// Args:
///   signals_json: JSON array of `{name, raw_score, weight, note}` objects
///
/// Returns: JSON of RiskAssessment
#[pyfunction]
pub(crate) fn governance_compute_risk(signals_json: &str) -> PyResult<String> {
    let signals: Vec<RiskSignal> =
        serde_json::from_str(signals_json).map_err(py_err)?;
    let assessment = compute_risk(signals);
    to_json(&assessment)
}

/// Compute standard risk signals from a change description.
///
/// Args:
///   change_json: JSON of ChangeRiskInput fields
///
/// Returns: JSON array of RiskSignal
#[pyfunction]
pub(crate) fn governance_standard_risk_signals(change_json: &str) -> PyResult<String> {
    let input: ChangeRiskInput =
        serde_json::from_str(change_json).map_err(py_err)?;
    let signals = standard_risk_signals(&input);
    to_json(&signals)
}

// ── Audit Chain ───────────────────────────────────────────────────────────────

/// Compute the chain hash for a new audit entry.
///
/// This is the Rust implementation used by all three distributions (Python,
/// Node, native) to guarantee identical chain hashes.
#[pyfunction]
pub(crate) fn governance_compute_audit_chain_hash(
    prev_chain_hash: &str,
    event_id: &str,
    payload_json: &str,
) -> String {
    compute_audit_chain_hash(prev_chain_hash, event_id, payload_json)
}

/// Verify an audit chain from a JSON array of AuditEntry objects.
///
/// Returns: `{"ok": true, "verified": N}` or `{"ok": false, "error": "..."}`.
#[pyfunction]
pub(crate) fn governance_verify_audit_chain(entries_json: &str) -> PyResult<String> {
    let entries: Vec<AuditEntry> =
        serde_json::from_str(entries_json).map_err(py_err)?;
    match verify_audit_chain(&entries) {
        Ok(n) => Ok(format!("{{\"ok\":true,\"verified\":{n}}}")),
        Err(msg) => Ok(format!("{{\"ok\":false,\"error\":{}}}", serde_json::json!(msg))),
    }
}

// ── Provenance ────────────────────────────────────────────────────────────────

/// Compute a deterministic provenance node ID.
///
/// All three distributions call this to guarantee identical node IDs for
/// the same event.
#[pyfunction]
pub(crate) fn governance_provenance_node_id(
    entity_type: &str,
    actor_id: &str,
    subject_id: &str,
    created_at_ms: u64,
    correlation_id: &str,
) -> String {
    ProvenanceNode::compute_id(entity_type, actor_id, subject_id, created_at_ms, correlation_id)
}

// ── Supply-Chain Scanning ─────────────────────────────────────────────────────

/// Scan an MCP tool schema for supply-chain threats.
///
/// Args:
///   tool_name: name of the tool being registered
///   schema_json: raw JSON of the tool schema
///   known_good_hash: optional SHA-256 of the last known-good schema
///
/// Returns: JSON of SupplyChainScanResult
#[pyfunction]
#[pyo3(signature = (tool_name, schema_json, known_good_hash=None))]
pub(crate) fn governance_scan_tool_schema(
    tool_name: &str,
    schema_json: &str,
    known_good_hash: Option<&str>,
) -> PyResult<String> {
    let result = scan_tool_schema(tool_name, schema_json, known_good_hash);
    to_json(&result)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_risk_level(s: &str) -> PyResult<RiskLevel> {
    match s {
        "low" => Ok(RiskLevel::Low),
        "medium" => Ok(RiskLevel::Medium),
        "high" => Ok(RiskLevel::High),
        "critical" => Ok(RiskLevel::Critical),
        other => Err(PyValueError::new_err(format!(
            "invalid risk_level {other:?}; expected low|medium|high|critical"
        ))),
    }
}

// ── Module Registration ───────────────────────────────────────────────────────

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(governance_compute_identity_token, m)?)?;
    m.add_function(wrap_pyfunction!(governance_verify_identity_token, m)?)?;
    m.add_function(wrap_pyfunction!(governance_evaluate_policy, m)?)?;
    m.add_function(wrap_pyfunction!(governance_compute_risk, m)?)?;
    m.add_function(wrap_pyfunction!(governance_standard_risk_signals, m)?)?;
    m.add_function(wrap_pyfunction!(governance_compute_audit_chain_hash, m)?)?;
    m.add_function(wrap_pyfunction!(governance_verify_audit_chain, m)?)?;
    m.add_function(wrap_pyfunction!(governance_provenance_node_id, m)?)?;
    m.add_function(wrap_pyfunction!(governance_scan_tool_schema, m)?)?;
    Ok(())
}
