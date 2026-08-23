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
    m.add_function(wrap_pyfunction!(recovery_handle_build_json, m)?)?;
    m.add_function(wrap_pyfunction!(recovery_handle_verify_json, m)?)?;
    m.add_function(wrap_pyfunction!(recovery_handle_verify_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(recovery_handle_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(memory_record_build_json, m)?)?;
    m.add_function(wrap_pyfunction!(memory_record_admissibility, m)?)?;
    m.add_function(wrap_pyfunction!(memory_record_verify_json, m)?)?;
    m.add_function(wrap_pyfunction!(memory_record_schema_version, m)?)?;
    Ok(())
}

// ── Recovery handles ──────────────────────────────────────────────────────
//
// Kept in this module rather than a separate one: a recovery handle only exists
// because a receipt promised material it did not deliver, so the two contracts
// are one surface from a caller's point of view.

use entroly_engine::engine_contracts::{
    RecoveryDisposition, RecoveryHandle, RecoveryIntegrityState, RECOVERY_HANDLE_SCHEMA_VERSION,
};

fn parse_disposition(token: &str) -> PyResult<RecoveryDisposition> {
    match token {
        "included" => Ok(RecoveryDisposition::Included),
        "compressed" => Ok(RecoveryDisposition::Compressed),
        "omitted_but_recoverable" => Ok(RecoveryDisposition::OmittedButRecoverable),
        "omitted_and_unavailable" => Ok(RecoveryDisposition::OmittedAndUnavailable),
        other => Err(PyValueError::new_err(format!(
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
/// Raises when a disposition promises recovery without the means to honour it —
/// that refusal is the contract, not a validation nicety.
#[pyfunction]
#[pyo3(signature = (
    repository_id,
    receipt_id,
    disposition,
    source_ref = String::new(),
    source_commitment = String::new(),
    fragment_commitment = String::new(),
    byte_start = 0,
    byte_end = 0,
    version = String::new(),
    storage_locator = String::new(),
    observed_at_ms = 0,
))]
#[allow(clippy::too_many_arguments)]
pub fn recovery_handle_build_json(
    repository_id: String,
    receipt_id: String,
    disposition: &str,
    source_ref: String,
    source_commitment: String,
    fragment_commitment: String,
    byte_start: u64,
    byte_end: u64,
    version: String,
    storage_locator: String,
    observed_at_ms: i64,
) -> PyResult<String> {
    let handle = RecoveryHandle::new(
        repository_id,
        receipt_id,
        parse_disposition(disposition)?,
        source_ref,
        source_commitment,
        fragment_commitment,
        byte_start,
        byte_end,
        version,
        storage_locator,
        observed_at_ms,
    )
    .map_err(contract_err)?;
    handle.to_json().map_err(contract_err)
}

/// Parse and check a handle, returning its canonical JSON.
#[pyfunction]
pub fn recovery_handle_verify_json(handle_json: &str) -> PyResult<String> {
    let handle = RecoveryHandle::from_json_verified(handle_json).map_err(contract_err)?;
    handle.to_json().map_err(contract_err)
}

/// Check recovered bytes against the handle's commitment.
///
/// Returns `verified`, `commitment_mismatch` or `not_recoverable`. Recovered
/// material may only be used on `verified` — the caller cannot get that answer
/// without the bytes actually hashing to what was promised.
#[pyfunction]
pub fn recovery_handle_verify_bytes(handle_json: &str, payload: &[u8]) -> PyResult<String> {
    let handle = RecoveryHandle::from_json_verified(handle_json).map_err(contract_err)?;
    Ok(integrity_token(handle.verify_recovered(payload)).to_string())
}

/// Schema version this build implements for recovery handles.
#[pyfunction]
pub fn recovery_handle_schema_version() -> u32 {
    RECOVERY_HANDLE_SCHEMA_VERSION
}

// ── Provenance-bearing memory ─────────────────────────────────────────────

use entroly_engine::engine_contracts::{
    MemoryAdmissibility, MemoryRecord, MEMORY_RECORD_SCHEMA_VERSION,
};
use entroly_engine::work_graph::TrustLevel;
use std::collections::BTreeSet;

fn parse_trust(token: &str) -> PyResult<TrustLevel> {
    match token {
        "untrusted" => Ok(TrustLevel::Untrusted),
        "inferred" => Ok(TrustLevel::Inferred),
        "observed" => Ok(TrustLevel::Observed),
        "verified" => Ok(TrustLevel::Verified),
        other => Err(PyValueError::new_err(format!(
            "unknown trust level {other:?}; expected untrusted, inferred, observed or verified"
        ))),
    }
}

fn admissibility_token(verdict: MemoryAdmissibility) -> &'static str {
    match verdict {
        MemoryAdmissibility::Admissible => "admissible",
        MemoryAdmissibility::Contradicted => "contradicted",
        MemoryAdmissibility::Superseded => "superseded",
        MemoryAdmissibility::Expired => "expired",
        MemoryAdmissibility::Unsupported => "unsupported",
    }
}

/// Build a memory record and return it as canonical JSON.
#[pyfunction]
#[pyo3(signature = (
    repository_id,
    content_reference,
    trust_state,
    task_id = String::new(),
    workstream_id = String::new(),
    source_agent = String::new(),
    source_session = String::new(),
    source_execution = String::new(),
    content_commitment = String::new(),
    evidence_ids = Vec::new(),
    created_at_ms = 0,
    observed_at_ms = 0,
    valid_until_ms = 0,
    supersedes = Vec::new(),
    contradicted_by = Vec::new(),
    recovery_handle = String::new(),
))]
#[allow(clippy::too_many_arguments)]
pub fn memory_record_build_json(
    repository_id: String,
    content_reference: String,
    trust_state: &str,
    task_id: String,
    workstream_id: String,
    source_agent: String,
    source_session: String,
    source_execution: String,
    content_commitment: String,
    evidence_ids: Vec<String>,
    created_at_ms: i64,
    observed_at_ms: i64,
    valid_until_ms: i64,
    supersedes: Vec<String>,
    contradicted_by: Vec<String>,
    recovery_handle: String,
) -> PyResult<String> {
    let record = MemoryRecord::new(
        repository_id,
        task_id,
        workstream_id,
        source_agent,
        source_session,
        source_execution,
        content_reference,
        content_commitment,
        evidence_ids,
        parse_trust(trust_state)?,
        created_at_ms,
        observed_at_ms,
        valid_until_ms,
        supersedes,
        contradicted_by,
        recovery_handle,
    )
    .map_err(contract_err)?;
    record.to_json().map_err(contract_err)
}

/// May this memory be injected, and why.
///
/// Takes `now_ms` and an optional set of superseded ids rather than reading a
/// clock or a store: a verdict that depends on ambient state is not replayable.
/// Note there is no score parameter — section 10's "do not let similarity score
/// imply truth" is enforced by the signature, not by a warning.
#[pyfunction]
#[pyo3(signature = (record_json, now_ms, superseded_ids = Vec::new()))]
pub fn memory_record_admissibility(
    record_json: &str,
    now_ms: i64,
    superseded_ids: Vec<String>,
) -> PyResult<String> {
    let record = MemoryRecord::from_json_verified(record_json).map_err(contract_err)?;
    let superseded: BTreeSet<String> = superseded_ids.into_iter().collect();
    Ok(admissibility_token(record.admissibility_in_set(now_ms, &superseded)).to_string())
}

/// Parse and check a memory record, returning its canonical JSON.
#[pyfunction]
pub fn memory_record_verify_json(record_json: &str) -> PyResult<String> {
    let record = MemoryRecord::from_json_verified(record_json).map_err(contract_err)?;
    record.to_json().map_err(contract_err)
}

/// Schema version this build implements for memory records.
#[pyfunction]
pub fn memory_record_schema_version() -> u32 {
    MEMORY_RECORD_SCHEMA_VERSION
}
