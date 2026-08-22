//! PyO3 boundary for the canonical Context Receipt envelope.
//!
//! Transport only. The envelope's identity, canonicalisation, commitment and
//! tamper checks all live in `entroly_engine::engine_contracts`; nothing here
//! decides any of them. That is the point: the WASM binding calls the same
//! functions, so a Python caller and a Node caller given equivalent input
//! produce a byte-identical `receipt_commitment`.
//!
//! JSON is the interchange shape rather than a rich Python object, for the same
//! reason the Work Graph bindings use it — a structured object would need its
//! field order and number formatting reproduced identically in two runtimes,
//! which is a second contract to keep in step. Canonical JSON produced by the
//! engine has exactly one producer.

use entroly_engine::engine_contracts::{ContextReceiptEnvelope, CONTEXT_RECEIPT_SCHEMA_VERSION};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn contract_err(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// Build a canonical receipt envelope and return it as canonical JSON.
///
/// Returns JSON rather than a handle because callers store, transmit and hash
/// receipts far more often than they inspect them, and the JSON *is* the
/// committed form.
#[pyfunction]
#[pyo3(signature = (
    repository_id,
    repository_commitment,
    graph_commitment,
    work_scope_id,
    source_commitment = String::new(),
    selected_refs = Vec::new(),
    omitted_refs = Vec::new(),
    pinned_refs = Vec::new(),
    recoverable_refs = Vec::new(),
    recovery_handles = Vec::new(),
    evidence_ids = Vec::new(),
    budget_tokens = 0,
    selection_policy = String::new(),
    execution_id = String::new(),
    created_at_ms = 0,
))]
#[allow(clippy::too_many_arguments)]
pub fn context_receipt_build_json(
    repository_id: String,
    repository_commitment: String,
    graph_commitment: String,
    work_scope_id: String,
    source_commitment: String,
    selected_refs: Vec<String>,
    omitted_refs: Vec<String>,
    pinned_refs: Vec<String>,
    recoverable_refs: Vec<String>,
    recovery_handles: Vec<String>,
    evidence_ids: Vec<String>,
    budget_tokens: u32,
    selection_policy: String,
    execution_id: String,
    created_at_ms: i64,
) -> PyResult<String> {
    let envelope = ContextReceiptEnvelope::new(
        repository_id,
        repository_commitment,
        graph_commitment,
        work_scope_id,
        source_commitment,
        selected_refs,
        omitted_refs,
        pinned_refs,
        recoverable_refs,
        recovery_handles,
        evidence_ids,
        budget_tokens,
        selection_policy,
        execution_id,
        created_at_ms,
    )
    .map_err(contract_err)?;
    envelope.to_json().map_err(contract_err)
}

/// Parse and verify an envelope, returning its canonical JSON.
///
/// Fails closed on a commitment that does not recompute and on an unrecognised
/// schema version, matching `from_json_verified` exactly — a caller cannot
/// obtain an unverified envelope through this boundary.
#[pyfunction]
pub fn context_receipt_verify_json(receipt_json: &str) -> PyResult<String> {
    let envelope =
        ContextReceiptEnvelope::from_json_verified(receipt_json).map_err(contract_err)?;
    envelope.to_json().map_err(contract_err)
}

/// The commitment carried by a verified envelope.
#[pyfunction]
pub fn context_receipt_commitment(receipt_json: &str) -> PyResult<String> {
    let envelope =
        ContextReceiptEnvelope::from_json_verified(receipt_json).map_err(contract_err)?;
    Ok(envelope.receipt_commitment)
}

/// Project a verified envelope into the Work Graph reference form, as JSON.
///
/// Section 8's rule that the graph references a receipt rather than duplicating
/// it — enforced here rather than trusted to callers.
#[pyfunction]
pub fn context_receipt_graph_ref_json(
    receipt_json: &str,
    workstream_id: String,
    agent_id: String,
    session_id: String,
) -> PyResult<String> {
    let envelope =
        ContextReceiptEnvelope::from_json_verified(receipt_json).map_err(contract_err)?;
    let graph_ref = envelope
        .to_graph_ref(workstream_id, agent_id, session_id)
        .map_err(contract_err)?;
    serde_json::to_string(&graph_ref).map_err(contract_err)
}

/// Schema version this build implements, so a host can refuse a newer receipt
/// before attempting to interpret it.
#[pyfunction]
pub fn context_receipt_schema_version() -> u32 {
    CONTEXT_RECEIPT_SCHEMA_VERSION
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(context_receipt_build_json, m)?)?;
    m.add_function(wrap_pyfunction!(context_receipt_verify_json, m)?)?;
    m.add_function(wrap_pyfunction!(context_receipt_commitment, m)?)?;
    m.add_function(wrap_pyfunction!(context_receipt_graph_ref_json, m)?)?;
    m.add_function(wrap_pyfunction!(context_receipt_schema_version, m)?)?;
    Ok(())
}
