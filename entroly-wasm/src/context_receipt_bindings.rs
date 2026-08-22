//! wasm-bindgen boundary for the canonical Context Receipt envelope.
//!
//! The mirror of `entroly-core/src/context_receipt_bindings.rs`. Both call the
//! same `entroly_engine::engine_contracts` functions and neither decides
//! identity, canonicalisation or commitment, so a Node caller and a Python
//! caller given equivalent input produce a byte-identical `receipt_commitment`.
//!
//! That is what turns "npm agents can participate in Work Graph continuity"
//! into "any supported runtime can prove what evidence it received".

use entroly_engine::engine_contracts::{ContextReceiptEnvelope, CONTEXT_RECEIPT_SCHEMA_VERSION};
use wasm_bindgen::prelude::*;

fn js_err(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}

/// Build a canonical receipt envelope and return it as canonical JSON.
///
/// Reference lists arrive as JSON arrays rather than `Box<[JsString]>` so there
/// is one decoding path shared with the other bindings here, and so an empty
/// list is expressed the same way in both runtimes.
#[wasm_bindgen(js_name = contextReceiptBuildJSON)]
#[allow(clippy::too_many_arguments)]
pub fn context_receipt_build_json(
    repository_id: &str,
    repository_commitment: &str,
    graph_commitment: &str,
    work_scope_id: &str,
    source_commitment: Option<String>,
    selected_refs_json: Option<String>,
    omitted_refs_json: Option<String>,
    pinned_refs_json: Option<String>,
    recoverable_refs_json: Option<String>,
    recovery_handles_json: Option<String>,
    evidence_ids_json: Option<String>,
    budget_tokens: Option<u32>,
    selection_policy: Option<String>,
    execution_id: Option<String>,
    created_at_ms: Option<f64>,
) -> Result<String, JsValue> {
    fn refs(value: Option<String>) -> Result<Vec<String>, JsValue> {
        match value {
            None => Ok(Vec::new()),
            Some(text) if text.trim().is_empty() => Ok(Vec::new()),
            Some(text) => serde_json::from_str(&text).map_err(js_err),
        }
    }

    // JavaScript numbers are f64. Reject anything that is not an exact
    // integer instead of silently truncating a timestamp into a different
    // receipt — a truncated millisecond is a different commitment.
    let created_at_ms = match created_at_ms {
        None => 0,
        Some(value) => {
            const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
            if !value.is_finite() || value.fract() != 0.0 || value.abs() > MAX_SAFE_INTEGER {
                return Err(JsValue::from_str(
                    "created_at_ms must be a finite JavaScript-safe integer",
                ));
            }
            value as i64
        }
    };

    let envelope = ContextReceiptEnvelope::new(
        repository_id.to_owned(),
        repository_commitment.to_owned(),
        graph_commitment.to_owned(),
        work_scope_id.to_owned(),
        source_commitment.unwrap_or_default(),
        refs(selected_refs_json)?,
        refs(omitted_refs_json)?,
        refs(pinned_refs_json)?,
        refs(recoverable_refs_json)?,
        refs(recovery_handles_json)?,
        refs(evidence_ids_json)?,
        budget_tokens.unwrap_or(0),
        selection_policy.unwrap_or_default(),
        execution_id.unwrap_or_default(),
        created_at_ms,
    )
    .map_err(js_err)?;
    envelope.to_json().map_err(js_err)
}

/// Parse and verify an envelope, returning its canonical JSON.
///
/// Fails closed on a commitment that does not recompute and on an unrecognised
/// schema version — a caller cannot obtain an unverified envelope here.
#[wasm_bindgen(js_name = contextReceiptVerifyJSON)]
pub fn context_receipt_verify_json(receipt_json: &str) -> Result<String, JsValue> {
    let envelope = ContextReceiptEnvelope::from_json_verified(receipt_json).map_err(js_err)?;
    envelope.to_json().map_err(js_err)
}

/// The commitment carried by a verified envelope.
#[wasm_bindgen(js_name = contextReceiptCommitment)]
pub fn context_receipt_commitment(receipt_json: &str) -> Result<String, JsValue> {
    let envelope = ContextReceiptEnvelope::from_json_verified(receipt_json).map_err(js_err)?;
    Ok(envelope.receipt_commitment)
}

/// Project a verified envelope into the Work Graph reference form, as JSON.
#[wasm_bindgen(js_name = contextReceiptGraphRefJSON)]
pub fn context_receipt_graph_ref_json(
    receipt_json: &str,
    workstream_id: &str,
    agent_id: &str,
    session_id: &str,
) -> Result<String, JsValue> {
    let envelope = ContextReceiptEnvelope::from_json_verified(receipt_json).map_err(js_err)?;
    let graph_ref = envelope
        .to_graph_ref(
            workstream_id.to_owned(),
            agent_id.to_owned(),
            session_id.to_owned(),
        )
        .map_err(js_err)?;
    serde_json::to_string(&graph_ref).map_err(js_err)
}

/// Schema version this build implements.
#[wasm_bindgen(js_name = contextReceiptSchemaVersion)]
pub fn context_receipt_schema_version() -> u32 {
    CONTEXT_RECEIPT_SCHEMA_VERSION
}
