//! Thin wasm-bindgen boundary over `entroly_engine::governance`.
//!
//! All governance computation lives in `entroly-engine`. This file only
//! converts JS strings to Rust types and serializes results back to JSON.
//! The JS orchestration layer in `js/governance.js` parses the JSON.

use entroly_engine::governance::{
    compute_audit_chain_hash, compute_risk, evaluate_policy, scan_tool_schema,
    standard_risk_signals, verify_audit_chain, AgentIdentity, AgentIdentityPayload,
    AuditEntry, ChangeRiskInput, GovernancePolicy, ProvenanceNode, RiskLevel, RiskSignal,
};
use wasm_bindgen::prelude::*;

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn to_json<T: serde::Serialize>(val: &T) -> Result<String, JsValue> {
    serde_json::to_string(val).map_err(js_err)
}

// ── Identity ─────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = governanceComputeIdentityToken)]
pub fn governance_compute_identity_token(
    payload_json: &str,
    key: &str,
) -> Result<String, JsValue> {
    let payload: AgentIdentityPayload =
        serde_json::from_str(payload_json).map_err(js_err)?;
    payload.compute_token(key).map_err(js_err)
}

#[wasm_bindgen(js_name = governanceVerifyIdentityToken)]
pub fn governance_verify_identity_token(
    payload_json: &str,
    token: &str,
    key: &str,
) -> Result<bool, JsValue> {
    let payload: AgentIdentityPayload =
        serde_json::from_str(payload_json).map_err(js_err)?;
    payload.verify_token(token, key).map_err(js_err)
}

// ── Policy Evaluation ─────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = governanceEvaluatePolicy)]
pub fn governance_evaluate_policy(
    identity_json: &str,
    scope: &str,
    resource: &str,
    risk_level: &str,
    policy_json: &str,
) -> Result<String, JsValue> {
    let identity: AgentIdentity =
        serde_json::from_str(identity_json).map_err(js_err)?;
    let policy: GovernancePolicy =
        serde_json::from_str(policy_json).map_err(js_err)?;
    let rl = parse_risk_level(risk_level)?;
    let result = evaluate_policy(&identity, scope, resource, rl, &policy);
    to_json(&result)
}

// ── Risk Scoring ──────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = governanceComputeRisk)]
pub fn governance_compute_risk(signals_json: &str) -> Result<String, JsValue> {
    let signals: Vec<RiskSignal> =
        serde_json::from_str(signals_json).map_err(js_err)?;
    let assessment = compute_risk(signals);
    to_json(&assessment)
}

#[wasm_bindgen(js_name = governanceStandardRiskSignals)]
pub fn governance_standard_risk_signals(change_json: &str) -> Result<String, JsValue> {
    let input: ChangeRiskInput =
        serde_json::from_str(change_json).map_err(js_err)?;
    let signals = standard_risk_signals(&input);
    to_json(&signals)
}

// ── Audit Chain ───────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = governanceComputeAuditChainHash)]
pub fn governance_compute_audit_chain_hash_js(
    prev_chain_hash: &str,
    event_id: &str,
    payload_json: &str,
) -> String {
    compute_audit_chain_hash(prev_chain_hash, event_id, payload_json)
}

#[wasm_bindgen(js_name = governanceVerifyAuditChain)]
pub fn governance_verify_audit_chain(entries_json: &str) -> Result<String, JsValue> {
    let entries: Vec<AuditEntry> =
        serde_json::from_str(entries_json).map_err(js_err)?;
    match verify_audit_chain(&entries) {
        Ok(n) => Ok(format!("{{\"ok\":true,\"verified\":{n}}}")),
        Err(msg) => Ok(format!("{{\"ok\":false,\"error\":{}}}", serde_json::json!(msg))),
    }
}

// ── Provenance ────────────────────────────────────────────────────────────────

#[wasm_bindgen(js_name = governanceProvenanceNodeId)]
pub fn governance_provenance_node_id(
    entity_type: &str,
    actor_id: &str,
    subject_id: &str,
    created_at_ms: u64,
    correlation_id: &str,
) -> String {
    ProvenanceNode::compute_id(entity_type, actor_id, subject_id, created_at_ms, correlation_id)
}

// ── Supply-Chain Scanning ─────────────────────────────────────────────────────

#[wasm_bindgen(js_name = governanceScanToolSchema)]
pub fn governance_scan_tool_schema(
    tool_name: &str,
    schema_json: &str,
    known_good_hash: Option<String>,
) -> Result<String, JsValue> {
    let result = scan_tool_schema(
        tool_name,
        schema_json,
        known_good_hash.as_deref(),
    );
    to_json(&result)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn parse_risk_level(s: &str) -> Result<RiskLevel, JsValue> {
    match s {
        "low" => Ok(RiskLevel::Low),
        "medium" => Ok(RiskLevel::Medium),
        "high" => Ok(RiskLevel::High),
        "critical" => Ok(RiskLevel::Critical),
        other => Err(JsValue::from_str(&format!(
            "invalid risk_level {other:?}; expected low|medium|high|critical"
        ))),
    }
}
