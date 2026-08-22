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

// ── Recovery handles ──────────────────────────────────────────────────────
//
// The mirror of the PyO3 recovery surface. Same engine functions, so the two
// runtimes agree on which claims are refused as well as on which ids are
// produced — a contract that accepted different inputs in different runtimes
// would not be one contract.

use entroly_engine::engine_contracts::{
    RecoveryDisposition, RecoveryHandle, RecoveryIntegrityState, RECOVERY_HANDLE_SCHEMA_VERSION,
};

fn parse_disposition(token: &str) -> Result<RecoveryDisposition, JsValue> {
    match token {
        "included" => Ok(RecoveryDisposition::Included),
        "compressed" => Ok(RecoveryDisposition::Compressed),
        "omitted_but_recoverable" => Ok(RecoveryDisposition::OmittedButRecoverable),
        "omitted_and_unavailable" => Ok(RecoveryDisposition::OmittedAndUnavailable),
        other => Err(JsValue::from_str(&format!(
            "unknown recovery disposition {other:?}; expected one of included, \
             compressed, omitted_but_recoverable, omitted_and_unavailable"
        ))),
    }
}

fn integrity_token(state: RecoveryIntegrityState) -> &'static str {
    match state {
        RecoveryIntegrityState::Verified => "verified",
        RecoveryIntegrityState::CommitmentMismatch => "commitment_mismatch",
        RecoveryIntegrityState::NotRecoverable => "not_recoverable",
    }
}

/// Build a recovery handle and return it as canonical JSON.
///
/// Throws when a disposition promises recovery without the means to honour it.
#[wasm_bindgen(js_name = recoveryHandleBuildJSON)]
#[allow(clippy::too_many_arguments)]
pub fn recovery_handle_build_json(
    repository_id: &str,
    receipt_id: &str,
    disposition: &str,
    source_ref: Option<String>,
    source_commitment: Option<String>,
    fragment_commitment: Option<String>,
    byte_start: Option<f64>,
    byte_end: Option<f64>,
    version: Option<String>,
    storage_locator: Option<String>,
    observed_at_ms: Option<f64>,
) -> Result<String, JsValue> {
    // Byte offsets and timestamps arrive as f64. A truncated offset addresses
    // different bytes than the caller meant, which is a different handle.
    fn exact_u64(value: Option<f64>, name: &str) -> Result<u64, JsValue> {
        const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
        match value {
            None => Ok(0),
            Some(v)
                if v.is_finite() && v.fract() == 0.0 && (0.0..=MAX_SAFE_INTEGER).contains(&v) =>
            {
                Ok(v as u64)
            }
            Some(_) => Err(JsValue::from_str(&format!(
                "{name} must be a non-negative JavaScript-safe integer"
            ))),
        }
    }
    fn exact_i64(value: Option<f64>, name: &str) -> Result<i64, JsValue> {
        const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
        match value {
            None => Ok(0),
            Some(v) if v.is_finite() && v.fract() == 0.0 && v.abs() <= MAX_SAFE_INTEGER => {
                Ok(v as i64)
            }
            Some(_) => Err(JsValue::from_str(&format!(
                "{name} must be a finite JavaScript-safe integer"
            ))),
        }
    }

    let handle = RecoveryHandle::new(
        repository_id.to_owned(),
        receipt_id.to_owned(),
        parse_disposition(disposition)?,
        source_ref.unwrap_or_default(),
        source_commitment.unwrap_or_default(),
        fragment_commitment.unwrap_or_default(),
        exact_u64(byte_start, "byte_start")?,
        exact_u64(byte_end, "byte_end")?,
        version.unwrap_or_default(),
        storage_locator.unwrap_or_default(),
        exact_i64(observed_at_ms, "observed_at_ms")?,
    )
    .map_err(js_err)?;
    handle.to_json().map_err(js_err)
}

/// Parse and check a handle, returning its canonical JSON.
#[wasm_bindgen(js_name = recoveryHandleVerifyJSON)]
pub fn recovery_handle_verify_json(handle_json: &str) -> Result<String, JsValue> {
    let handle = RecoveryHandle::from_json_verified(handle_json).map_err(js_err)?;
    handle.to_json().map_err(js_err)
}

/// Check recovered bytes against the handle's commitment.
///
/// Returns `verified`, `commitment_mismatch` or `not_recoverable`.
#[wasm_bindgen(js_name = recoveryHandleVerifyBytes)]
pub fn recovery_handle_verify_bytes(handle_json: &str, payload: &[u8]) -> Result<String, JsValue> {
    let handle = RecoveryHandle::from_json_verified(handle_json).map_err(js_err)?;
    Ok(integrity_token(handle.verify_recovered(payload)).to_string())
}

/// Schema version this build implements for recovery handles.
#[wasm_bindgen(js_name = recoveryHandleSchemaVersion)]
pub fn recovery_handle_schema_version() -> u32 {
    RECOVERY_HANDLE_SCHEMA_VERSION
}
