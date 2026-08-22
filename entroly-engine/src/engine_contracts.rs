//! Narrow integration contracts between the AI Work Graph, Context Engine, and
//! Trust Engine.
//!
//! These types deliberately carry references, commitments, and bounded IDs —
//! never giant raw contexts or agent transcripts. The Work Graph remains the
//! source of work-state truth; existing context/recovery systems remain the
//! source of context bytes. This module only defines the stable seam between
//! those systems.

use crate::work_graph::{EvidenceRef, ResumeView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

const MAX_CONTRACT_ID_BYTES: usize = 512;
const MAX_SCOPE_PATH_BYTES: usize = 4_096;
const MAX_RECOVERY_HANDLE_BYTES: usize = 4_096;
const MAX_COMMITMENT_BYTES: usize = 256;
const MAX_SCOPE_ITEMS: usize = 512;
const MAX_RECOVERY_HANDLES: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineContractError {
    EmptyField(&'static str),
    FieldTooLong {
        field: &'static str,
        max_bytes: usize,
    },
    TooManyItems {
        field: &'static str,
        max_items: usize,
    },
    InvalidTimestamp(&'static str),
    /// Canonical encoding failed. Carries the underlying message rather than the
    /// source error so the enum stays `Clone + PartialEq` like the rest.
    Serialization(String),
    /// A persisted contract declares a schema this build does not implement.
    /// Fail closed: an unknown schema cannot be validated, so it is refused
    /// rather than interpreted under today's rules.
    UnsupportedSchema {
        field: &'static str,
        found: u32,
        expected: u32,
    },
    /// A carried commitment does not match a recomputation of its own payload.
    /// This is the tamper signal.
    CommitmentMismatch(&'static str),
}

impl fmt::Display for EngineContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(formatter, "{field} cannot be empty"),
            Self::FieldTooLong { field, max_bytes } => {
                write!(formatter, "{field} exceeds {max_bytes} bytes")
            }
            Self::TooManyItems { field, max_items } => {
                write!(formatter, "{field} exceeds {max_items} items")
            }
            Self::InvalidTimestamp(field) => write!(formatter, "{field} cannot be negative"),
            Self::Serialization(detail) => {
                write!(formatter, "canonical serialization failed: {detail}")
            }
            Self::UnsupportedSchema {
                field,
                found,
                expected,
            } => write!(
                formatter,
                "{field} {found} is not supported by this build (expected {expected})"
            ),
            Self::CommitmentMismatch(field) => write!(
                formatter,
                "{field} does not match a recomputation of its payload"
            ),
        }
    }
}

impl std::error::Error for EngineContractError {}

fn validate_text(
    field: &'static str,
    value: &str,
    max_bytes: usize,
    allow_empty: bool,
) -> Result<(), EngineContractError> {
    if !allow_empty && value.is_empty() {
        return Err(EngineContractError::EmptyField(field));
    }
    if value.len() > max_bytes {
        return Err(EngineContractError::FieldTooLong { field, max_bytes });
    }
    Ok(())
}

fn canonical_strings(
    field: &'static str,
    values: impl IntoIterator<Item = String>,
    max_items: usize,
    max_bytes: usize,
) -> Result<Vec<String>, EngineContractError> {
    let mut unique = BTreeSet::new();
    for value in values {
        validate_text(field, &value, max_bytes, false)?;
        unique.insert(value);
        if unique.len() > max_items {
            return Err(EngineContractError::TooManyItems { field, max_items });
        }
    }
    Ok(unique.into_iter().collect())
}

fn canonical_ids(
    field: &'static str,
    values: impl IntoIterator<Item = String>,
    max_items: usize,
) -> Result<Vec<String>, EngineContractError> {
    canonical_strings(field, values, max_items, MAX_CONTRACT_ID_BYTES)
}

fn evidence_ids(evidence: &[EvidenceRef]) -> Result<Vec<String>, EngineContractError> {
    canonical_ids(
        "evidence_ids",
        evidence.iter().map(|item| item.evidence_id.clone()),
        MAX_SCOPE_ITEMS,
    )
}

/// Bounded, text-light view of the current work that Context/Trust can consume.
///
/// It contains durable graph identifiers and paths but intentionally excludes
/// decision prose, agent messages, failure text, and raw evidence payloads.
/// Those remain recoverable through the referenced Work Graph/evidence store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkScope {
    pub repo_id: String,
    pub graph_revision: u64,
    pub graph_commitment: String,
    pub workstream_id: String,
    pub task_ids: Vec<String>,
    pub agent_ids: Vec<String>,
    pub changed_paths: Vec<String>,
    pub commit_ids: Vec<String>,
    pub evidence_ids: Vec<String>,
}

impl WorkScope {
    /// Derive a deterministic integration scope from a Rust-owned resume view.
    ///
    /// This never interprets labels or prose. It only canonicalizes IDs/paths
    /// already selected by the Work Graph and fails closed if the bounded
    /// integration envelope would be exceeded.
    pub fn from_resume(resume: &ResumeView) -> Result<Self, EngineContractError> {
        validate_text("repo_id", &resume.repo_id, MAX_CONTRACT_ID_BYTES, false)?;
        validate_text(
            "graph_commitment",
            &resume.graph_commitment,
            MAX_COMMITMENT_BYTES,
            false,
        )?;
        validate_text(
            "workstream_id",
            &resume.selected_workstream.node_id,
            MAX_CONTRACT_ID_BYTES,
            false,
        )?;

        let mut changed_paths = resume.selected_workstream.changed_paths.clone();
        changed_paths.extend(resume.changed_paths.iter().cloned());

        let mut commits = resume.selected_workstream.commit_ids.clone();
        commits.extend(resume.commits.iter().cloned());

        let mut evidence = resume.selected_workstream.evidence_ids.clone();
        evidence.extend(evidence_ids(&resume.evidence)?);

        Ok(Self {
            repo_id: resume.repo_id.clone(),
            graph_revision: resume.graph_revision,
            graph_commitment: resume.graph_commitment.clone(),
            workstream_id: resume.selected_workstream.node_id.clone(),
            task_ids: canonical_ids(
                "task_ids",
                resume.selected_workstream.task_ids.clone(),
                MAX_SCOPE_ITEMS,
            )?,
            agent_ids: canonical_ids(
                "agent_ids",
                resume.selected_workstream.agent_ids.clone(),
                MAX_SCOPE_ITEMS,
            )?,
            changed_paths: canonical_strings(
                "changed_paths",
                changed_paths,
                MAX_SCOPE_ITEMS,
                MAX_SCOPE_PATH_BYTES,
            )?,
            commit_ids: canonical_ids("commit_ids", commits, MAX_SCOPE_ITEMS)?,
            evidence_ids: canonical_ids("evidence_ids", evidence, MAX_SCOPE_ITEMS)?,
        })
    }
}

/// Reference-only linkage from a context delivery/receipt back into work state.
///
/// The full Context Receipt may contain selected/omitted text and ranking detail
/// in its owning context subsystem. The Work Graph gets only identifiers,
/// commitments, evidence links, and recovery handles so graph state stays
/// bounded and does not become a second context store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextReceiptRef {
    pub receipt_id: String,
    pub workstream_id: String,
    #[serde(default)]
    pub agent_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub model_execution_id: String,
    pub reproducibility_hash: String,
    pub selected_evidence_ids: Vec<String>,
    pub recovery_handles: Vec<String>,
    pub observed_at_ms: i64,
}

impl ContextReceiptRef {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        receipt_id: String,
        workstream_id: String,
        agent_id: String,
        session_id: String,
        model_execution_id: String,
        reproducibility_hash: String,
        selected_evidence_ids: Vec<String>,
        recovery_handles: Vec<String>,
        observed_at_ms: i64,
    ) -> Result<Self, EngineContractError> {
        validate_text("receipt_id", &receipt_id, MAX_CONTRACT_ID_BYTES, false)?;
        validate_text(
            "workstream_id",
            &workstream_id,
            MAX_CONTRACT_ID_BYTES,
            false,
        )?;
        validate_text("agent_id", &agent_id, MAX_CONTRACT_ID_BYTES, true)?;
        validate_text("session_id", &session_id, MAX_CONTRACT_ID_BYTES, true)?;
        validate_text(
            "model_execution_id",
            &model_execution_id,
            MAX_CONTRACT_ID_BYTES,
            true,
        )?;
        validate_text(
            "reproducibility_hash",
            &reproducibility_hash,
            MAX_COMMITMENT_BYTES,
            false,
        )?;
        if observed_at_ms < 0 {
            return Err(EngineContractError::InvalidTimestamp("observed_at_ms"));
        }

        Ok(Self {
            receipt_id,
            workstream_id,
            agent_id,
            session_id,
            model_execution_id,
            reproducibility_hash,
            selected_evidence_ids: canonical_ids(
                "selected_evidence_ids",
                selected_evidence_ids,
                MAX_SCOPE_ITEMS,
            )?,
            recovery_handles: canonical_strings(
                "recovery_handles",
                recovery_handles,
                MAX_RECOVERY_HANDLES,
                MAX_RECOVERY_HANDLE_BYTES,
            )?,
            observed_at_ms,
        })
    }
}

// ── Canonical Context Receipt envelope ────────────────────────────────────

/// Schema version for [`ContextReceiptEnvelope`].
///
/// Part of the committed payload, so a version change is a commitment change
/// and old receipts cannot silently validate against new rules.
pub const CONTEXT_RECEIPT_SCHEMA_VERSION: u32 = 1;

const MAX_RECEIPT_REFS: usize = 4_096;
const MAX_POLICY_BYTES: usize = 1_024;

/// The invariant core of a Context Receipt, shared by every runtime.
///
/// # Why this exists
///
/// A rich receipt already exists in the Python host layer: selected text,
/// ranking reasons, risk summary, warnings. That receipt is useful and belongs
/// where it is — but its `reproducibility_hash` is computed over the whole
/// enriched record *including selected content*, so no other runtime can
/// reproduce it. A Node caller could participate in Work Graph continuity and
/// still not prove what evidence it received, because the two runtimes had no
/// receipt contract in common.
///
/// This envelope is that contract, and only that: identity, commitments,
/// references, budget and policy. No selected text, no ranking prose, no host
/// metadata. Those stay in the host receipt, which may embed this envelope and
/// add whatever it likes around it.
///
/// # Determinism
///
/// Equivalent input must produce a byte-identical `receipt_commitment` in every
/// runtime, so:
///
/// * reference lists are canonicalised — sorted and deduplicated — via the same
///   [`canonical_strings`] the rest of this module uses. The commitment attests
///   to *which* evidence was involved, not the order a ranker happened to emit;
///   ordering is presentation and belongs in the host receipt.
/// * the committed payload has a fixed field order and no floating-point
///   fields, so serialisation cannot vary by map iteration or float formatting.
/// * `receipt_id` is *derived* from the commitment rather than supplied. A
///   caller-chosen id is one more thing two runtimes can disagree about for no
///   benefit, and the Python host receipt already derives its id the same way.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextReceiptEnvelope {
    pub schema_version: u32,
    /// Derived: `cr_` followed by the first 16 hex characters of the commitment.
    pub receipt_id: String,
    pub repository_id: String,
    /// Commitment to the repository state the context was compiled from.
    pub repository_commitment: String,
    /// Work Graph commitment current when the context was compiled.
    pub graph_commitment: String,
    pub work_scope_id: String,
    /// Commitment to the source corpus the selection ran over.
    pub source_commitment: String,
    pub selected_refs: Vec<String>,
    pub omitted_refs: Vec<String>,
    pub pinned_refs: Vec<String>,
    /// Omitted material that remains exactly recoverable. A ref appearing here
    /// is a promise; see `RecoveryHandle` work for how it is honoured.
    pub recoverable_refs: Vec<String>,
    pub recovery_handles: Vec<String>,
    pub evidence_ids: Vec<String>,
    pub budget_tokens: u32,
    pub selection_policy: String,
    pub execution_id: String,
    pub created_at_ms: i64,
    /// SHA-256 over the canonical payload. Excluded from its own input.
    pub receipt_commitment: String,
}

/// The committed payload. Field order here *is* the canonical encoding.
///
/// `receipt_id` and `receipt_commitment` are absent by construction: both are
/// derived from this payload, so including either would be circular.
#[derive(Serialize)]
struct ContextReceiptPayload<'a> {
    schema_version: u32,
    repository_id: &'a str,
    repository_commitment: &'a str,
    graph_commitment: &'a str,
    work_scope_id: &'a str,
    source_commitment: &'a str,
    selected_refs: &'a [String],
    omitted_refs: &'a [String],
    pinned_refs: &'a [String],
    recoverable_refs: &'a [String],
    recovery_handles: &'a [String],
    evidence_ids: &'a [String],
    budget_tokens: u32,
    selection_policy: &'a str,
    execution_id: &'a str,
    created_at_ms: i64,
}

impl ContextReceiptEnvelope {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
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
    ) -> Result<Self, EngineContractError> {
        validate_text(
            "repository_id",
            &repository_id,
            MAX_CONTRACT_ID_BYTES,
            false,
        )?;
        validate_text(
            "repository_commitment",
            &repository_commitment,
            MAX_COMMITMENT_BYTES,
            false,
        )?;
        validate_text(
            "graph_commitment",
            &graph_commitment,
            MAX_COMMITMENT_BYTES,
            false,
        )?;
        validate_text(
            "work_scope_id",
            &work_scope_id,
            MAX_CONTRACT_ID_BYTES,
            false,
        )?;
        validate_text(
            "source_commitment",
            &source_commitment,
            MAX_COMMITMENT_BYTES,
            true,
        )?;
        validate_text(
            "selection_policy",
            &selection_policy,
            MAX_POLICY_BYTES,
            true,
        )?;
        validate_text("execution_id", &execution_id, MAX_CONTRACT_ID_BYTES, true)?;
        if created_at_ms < 0 {
            return Err(EngineContractError::InvalidTimestamp("created_at_ms"));
        }

        let selected_refs = canonical_strings(
            "selected_refs",
            selected_refs,
            MAX_RECEIPT_REFS,
            MAX_SCOPE_PATH_BYTES,
        )?;
        let omitted_refs = canonical_strings(
            "omitted_refs",
            omitted_refs,
            MAX_RECEIPT_REFS,
            MAX_SCOPE_PATH_BYTES,
        )?;
        let pinned_refs = canonical_strings(
            "pinned_refs",
            pinned_refs,
            MAX_RECEIPT_REFS,
            MAX_SCOPE_PATH_BYTES,
        )?;
        let recoverable_refs = canonical_strings(
            "recoverable_refs",
            recoverable_refs,
            MAX_RECEIPT_REFS,
            MAX_SCOPE_PATH_BYTES,
        )?;
        let recovery_handles = canonical_strings(
            "recovery_handles",
            recovery_handles,
            MAX_RECOVERY_HANDLES,
            MAX_RECOVERY_HANDLE_BYTES,
        )?;
        let evidence_ids = canonical_strings(
            "evidence_ids",
            evidence_ids,
            MAX_RECEIPT_REFS,
            MAX_CONTRACT_ID_BYTES,
        )?;

        let mut envelope = ContextReceiptEnvelope {
            schema_version: CONTEXT_RECEIPT_SCHEMA_VERSION,
            receipt_id: String::new(),
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
            receipt_commitment: String::new(),
        };
        envelope.receipt_commitment = envelope.compute_commitment()?;
        envelope.receipt_id = derive_receipt_id(&envelope.receipt_commitment);
        Ok(envelope)
    }

    fn compute_commitment(&self) -> Result<String, EngineContractError> {
        contract_sha256_json(&ContextReceiptPayload {
            schema_version: self.schema_version,
            repository_id: &self.repository_id,
            repository_commitment: &self.repository_commitment,
            graph_commitment: &self.graph_commitment,
            work_scope_id: &self.work_scope_id,
            source_commitment: &self.source_commitment,
            selected_refs: &self.selected_refs,
            omitted_refs: &self.omitted_refs,
            pinned_refs: &self.pinned_refs,
            recoverable_refs: &self.recoverable_refs,
            recovery_handles: &self.recovery_handles,
            evidence_ids: &self.evidence_ids,
            budget_tokens: self.budget_tokens,
            selection_policy: &self.selection_policy,
            execution_id: &self.execution_id,
            created_at_ms: self.created_at_ms,
        })
    }

    /// Recompute the commitment and compare it to the carried one.
    ///
    /// This is the tamper check. An envelope that crossed a process, a file or a
    /// network is untrusted until this passes — mutate any committed field and
    /// the recomputation diverges. It also catches a receipt whose `receipt_id`
    /// no longer derives from its own commitment.
    pub fn verify_commitment(&self) -> Result<bool, EngineContractError> {
        let recomputed = self.compute_commitment()?;
        Ok(recomputed == self.receipt_commitment
            && self.receipt_id == derive_receipt_id(&recomputed))
    }

    /// Canonical JSON for transport. Field order is the struct's declaration
    /// order, which is what makes cross-runtime byte comparison meaningful.
    pub fn to_json(&self) -> Result<String, EngineContractError> {
        serde_json::to_string(self)
            .map_err(|error| EngineContractError::Serialization(error.to_string()))
    }

    /// Parse and verify in one step. Fails closed: an envelope whose commitment
    /// does not recompute is rejected rather than returned for the caller to
    /// forget to check.
    pub fn from_json_verified(json_text: &str) -> Result<Self, EngineContractError> {
        let envelope: ContextReceiptEnvelope = serde_json::from_str(json_text)
            .map_err(|error| EngineContractError::Serialization(error.to_string()))?;
        if envelope.schema_version != CONTEXT_RECEIPT_SCHEMA_VERSION {
            return Err(EngineContractError::UnsupportedSchema {
                field: "schema_version",
                found: envelope.schema_version,
                expected: CONTEXT_RECEIPT_SCHEMA_VERSION,
            });
        }
        if !envelope.verify_commitment()? {
            return Err(EngineContractError::CommitmentMismatch(
                "receipt_commitment",
            ));
        }
        Ok(envelope)
    }

    /// Reference-only projection for the Work Graph.
    ///
    /// Section 8's rule that the graph references a receipt rather than
    /// duplicating it, made mechanical: the graph gets ids, commitments and
    /// handles, and never the receipt body.
    pub fn to_graph_ref(
        &self,
        workstream_id: String,
        agent_id: String,
        session_id: String,
    ) -> Result<ContextReceiptRef, EngineContractError> {
        ContextReceiptRef::new(
            self.receipt_id.clone(),
            workstream_id,
            agent_id,
            session_id,
            self.execution_id.clone(),
            self.receipt_commitment.clone(),
            self.evidence_ids.clone(),
            self.recovery_handles.clone(),
            self.created_at_ms,
        )
    }
}

fn derive_receipt_id(commitment: &str) -> String {
    let take = commitment.len().min(16);
    format!("cr_{}", &commitment[..take])
}

fn contract_sha256_json<T: Serialize>(value: &T) -> Result<String, EngineContractError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| EngineContractError::Serialization(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The cross-runtime anchor.
    ///
    /// Every binding calls the same engine function, so this one value is what
    /// "byte-equal commitments across runtimes" means in practice: PyO3 and
    /// WASM each assert it independently, and drift in either breaks its own
    /// test rather than silently diverging from the other.
    ///
    /// Changing this value is a schema change. If a code change moves it, that
    /// is either a bug or a deliberate version bump -- never an incidental edit.
    pub(crate) const GOLDEN_RECEIPT_COMMITMENT: &str =
        "672457349ba403bc885ea2104162fe212fb8e9bddf51a884df27d33c37a77c84";

    fn golden_envelope() -> ContextReceiptEnvelope {
        ContextReceiptEnvelope::new(
            "repo:golden".to_string(),
            "sha256:repo-golden".to_string(),
            "sha256:graph-golden".to_string(),
            "workstream:golden".to_string(),
            "sha256:source-golden".to_string(),
            vec!["ref:alpha".to_string(), "ref:beta".to_string()],
            vec!["ref:omitted".to_string()],
            vec!["ref:pinned".to_string()],
            vec!["ref:recoverable".to_string()],
            vec!["handle:alpha".to_string()],
            vec!["evidence:alpha".to_string()],
            4096,
            "knapsack/v1".to_string(),
            "exec:golden".to_string(),
            1_700_000_000_000,
        )
        .expect("golden fixture must be valid")
    }

    #[test]
    fn golden_vector_pins_the_cross_runtime_commitment() {
        let envelope = golden_envelope();
        assert_eq!(envelope.receipt_commitment, GOLDEN_RECEIPT_COMMITMENT);
        assert_eq!(envelope.receipt_id, "cr_672457349ba403bc");
        assert!(envelope.verify_commitment().expect("verify"));
    }



    // ── ContextReceiptEnvelope ───────────────────────────────────────────

    fn sample_envelope() -> ContextReceiptEnvelope {
        ContextReceiptEnvelope::new(
            "repo:demo".to_string(),
            "sha256:repo".to_string(),
            "sha256:graph".to_string(),
            "workstream:1".to_string(),
            "sha256:source".to_string(),
            vec!["ref:b".to_string(), "ref:a".to_string()],
            vec!["ref:omitted".to_string()],
            vec!["ref:pinned".to_string()],
            vec!["ref:recoverable".to_string()],
            vec!["handle:1".to_string()],
            vec!["evidence:1".to_string()],
            4096,
            "knapsack/v1".to_string(),
            "exec:1".to_string(),
            1_700_000_000_000,
        )
        .expect("valid envelope")
    }

    #[test]
    fn envelope_commitment_is_deterministic() {
        // The property the whole contract rests on: equivalent input, identical
        // commitment. Without this there is no cross-runtime receipt.
        assert_eq!(
            sample_envelope().receipt_commitment,
            sample_envelope().receipt_commitment
        );
    }

    #[test]
    fn reference_order_and_duplicates_do_not_change_the_commitment() {
        // Refs are canonicalised, so a runtime that enumerates in a different
        // order still commits to the same set. The commitment attests to which
        // evidence was involved; ranking order is presentation and lives in the
        // host receipt.
        let ordered = ContextReceiptEnvelope::new(
            "repo:demo".to_string(),
            "sha256:repo".to_string(),
            "sha256:graph".to_string(),
            "workstream:1".to_string(),
            "sha256:source".to_string(),
            vec![
                "ref:a".to_string(),
                "ref:b".to_string(),
                "ref:a".to_string(),
            ],
            vec!["ref:omitted".to_string()],
            vec!["ref:pinned".to_string()],
            vec!["ref:recoverable".to_string()],
            vec!["handle:1".to_string()],
            vec!["evidence:1".to_string()],
            4096,
            "knapsack/v1".to_string(),
            "exec:1".to_string(),
            1_700_000_000_000,
        )
        .expect("valid envelope");

        assert_eq!(
            ordered.receipt_commitment,
            sample_envelope().receipt_commitment
        );
        assert_eq!(
            ordered.selected_refs,
            vec!["ref:a".to_string(), "ref:b".to_string()]
        );
    }

    #[test]
    fn a_changed_field_changes_the_commitment() {
        let mut other = sample_envelope();
        other.budget_tokens = 8192;
        let recomputed = other.compute_commitment().expect("recompute");
        assert_ne!(recomputed, sample_envelope().receipt_commitment);
    }

    #[test]
    fn receipt_id_is_derived_from_the_commitment() {
        let envelope = sample_envelope();
        assert!(envelope.receipt_id.starts_with("cr_"));
        assert_eq!(
            envelope.receipt_id,
            format!("cr_{}", &envelope.receipt_commitment[..16])
        );
    }

    #[test]
    fn tampering_with_a_committed_field_is_detected() {
        let mut envelope = sample_envelope();
        assert!(envelope.verify_commitment().expect("verify"));

        envelope.selected_refs.push("ref:smuggled".to_string());
        assert!(
            !envelope.verify_commitment().expect("verify"),
            "an added reference must invalidate the commitment"
        );
    }

    #[test]
    fn tampering_with_the_receipt_id_alone_is_detected() {
        // The id derives from the commitment, so desynchronising them is also
        // tampering even though every committed field is untouched.
        let mut envelope = sample_envelope();
        envelope.receipt_id = "cr_0000000000000000".to_string();
        assert!(!envelope.verify_commitment().expect("verify"));
    }

    #[test]
    fn json_round_trip_verifies() {
        let envelope = sample_envelope();
        let json = envelope.to_json().expect("serialize");
        let parsed = ContextReceiptEnvelope::from_json_verified(&json).expect("verified parse");
        assert_eq!(parsed, envelope);
    }

    #[test]
    fn a_tampered_payload_fails_closed_on_parse() {
        // Parsing must refuse rather than hand back an unverified envelope for
        // the caller to forget to check.
        let envelope = sample_envelope();
        let json = envelope
            .to_json()
            .expect("serialize")
            .replace("\"budget_tokens\":4096", "\"budget_tokens\":999999");
        match ContextReceiptEnvelope::from_json_verified(&json) {
            Err(EngineContractError::CommitmentMismatch(field)) => {
                assert_eq!(field, "receipt_commitment")
            }
            other => panic!("expected a commitment mismatch, got {other:?}"),
        }
    }

    #[test]
    fn an_unknown_schema_is_refused_rather_than_interpreted() {
        let envelope = sample_envelope();
        let json = envelope
            .to_json()
            .expect("serialize")
            .replace("\"schema_version\":1", "\"schema_version\":99");
        match ContextReceiptEnvelope::from_json_verified(&json) {
            Err(EngineContractError::UnsupportedSchema {
                found, expected, ..
            }) => {
                assert_eq!(found, 99);
                assert_eq!(expected, CONTEXT_RECEIPT_SCHEMA_VERSION);
            }
            other => panic!("expected an unsupported-schema refusal, got {other:?}"),
        }
    }

    #[test]
    fn required_identity_fields_are_enforced() {
        let missing_repo = ContextReceiptEnvelope::new(
            String::new(),
            "sha256:repo".to_string(),
            "sha256:graph".to_string(),
            "workstream:1".to_string(),
            String::new(),
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            String::new(),
            String::new(),
            0,
        );
        assert_eq!(
            missing_repo,
            Err(EngineContractError::EmptyField("repository_id"))
        );
    }

    #[test]
    fn a_negative_timestamp_is_rejected() {
        let bad = ContextReceiptEnvelope::new(
            "repo:demo".to_string(),
            "sha256:repo".to_string(),
            "sha256:graph".to_string(),
            "workstream:1".to_string(),
            String::new(),
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            String::new(),
            String::new(),
            -1,
        );
        assert_eq!(
            bad,
            Err(EngineContractError::InvalidTimestamp("created_at_ms"))
        );
    }

    #[test]
    fn the_graph_projection_carries_references_only() {
        // Section 8's rule made mechanical: the graph gets ids, commitments and
        // handles -- never the receipt body.
        let envelope = sample_envelope();
        let graph_ref = envelope
            .to_graph_ref(
                "workstream:1".to_string(),
                "agent:claude".to_string(),
                "session:1".to_string(),
            )
            .expect("projection");

        assert_eq!(graph_ref.receipt_id, envelope.receipt_id);
        assert_eq!(graph_ref.reproducibility_hash, envelope.receipt_commitment);
        assert_eq!(graph_ref.recovery_handles, envelope.recovery_handles);

        let json = serde_json::to_string(&graph_ref).expect("serialize ref");
        for body_field in [
            "selected_refs",
            "omitted_refs",
            "selection_policy",
            "budget_tokens",
        ] {
            assert!(
                !json.contains(body_field),
                "graph reference leaked receipt body field {body_field}"
            );
        }
    }
    use crate::work_graph::{EvidenceKind, NodeKind, TrustLevel, WorkItemView, WorkStatus};
    use std::collections::BTreeMap;

    fn resume_fixture() -> ResumeView {
        ResumeView {
            repo_id: "repo:test".into(),
            graph_revision: 7,
            graph_commitment: "sha256:graph".into(),
            selected_workstream: WorkItemView {
                node_id: "workstream:auth".into(),
                kind: NodeKind::Workstream,
                label: "auth work".into(),
                status: WorkStatus::InProgress,
                trust: TrustLevel::Observed,
                updated_at_ms: 10,
                task_ids: vec!["task:b".into(), "task:a".into(), "task:a".into()],
                agent_ids: vec!["agent:codex".into(), "agent:claude".into()],
                changed_paths: vec!["src/auth.rs".into()],
                commit_ids: vec!["commit:2".into()],
                decision_ids: vec![],
                failure_ids: vec![],
                verification_ids: vec![],
                evidence_ids: vec!["evidence:git".into()],
            },
            task_labels: vec!["prose must not leak into scope".into()],
            agents: vec!["display-name".into()],
            decisions: vec!["secret decision prose".into()],
            failures: vec!["failure prose".into()],
            verification: vec!["verification prose".into()],
            changed_paths: vec!["tests/auth.rs".into(), "src/auth.rs".into()],
            commits: vec!["commit:1".into(), "commit:2".into()],
            evidence: vec![EvidenceRef {
                evidence_id: "evidence:test".into(),
                kind: EvidenceKind::TestResult,
                source_ref: "test://auth".into(),
                digest: "sha256:test".into(),
                locator: String::new(),
                trust: TrustLevel::Verified,
                observed_at_ms: 9,
                attributes: BTreeMap::new(),
            }],
        }
    }

    #[test]
    fn resume_scope_is_deterministic_bounded_and_text_light() {
        let scope = WorkScope::from_resume(&resume_fixture()).unwrap();
        assert_eq!(scope.task_ids, vec!["task:a", "task:b"]);
        assert_eq!(scope.agent_ids, vec!["agent:claude", "agent:codex"]);
        assert_eq!(scope.changed_paths, vec!["src/auth.rs", "tests/auth.rs"]);
        assert_eq!(scope.commit_ids, vec!["commit:1", "commit:2"]);
        assert_eq!(scope.evidence_ids, vec!["evidence:git", "evidence:test"]);

        let json = serde_json::to_string(&scope).unwrap();
        assert!(!json.contains("secret decision prose"));
        assert!(!json.contains("failure prose"));
        assert!(!json.contains("display-name"));
    }

    #[test]
    fn path_and_locator_bounds_are_distinct_from_id_bounds() {
        let mut resume = resume_fixture();
        resume.changed_paths = vec![format!("src/{}", "a".repeat(1_000))];
        assert!(WorkScope::from_resume(&resume).is_ok());

        let receipt = ContextReceiptRef::new(
            "receipt:1".into(),
            "workstream:auth".into(),
            String::new(),
            String::new(),
            String::new(),
            "sha256:x".into(),
            vec![],
            vec![format!("recover://{}", "b".repeat(1_000))],
            1,
        );
        assert!(receipt.is_ok());
    }

    #[test]
    fn receipt_reference_never_needs_raw_context() {
        let receipt = ContextReceiptRef::new(
            "receipt:1".into(),
            "workstream:auth".into(),
            "agent:claude".into(),
            "session:1".into(),
            "execution:1".into(),
            "sha256:receipt".into(),
            vec!["evidence:test".into(), "evidence:git".into()],
            vec!["recover:chunk-2".into(), "recover:chunk-1".into()],
            42,
        )
        .unwrap();

        assert_eq!(
            receipt.selected_evidence_ids,
            vec!["evidence:git", "evidence:test"]
        );
        assert_eq!(
            receipt.recovery_handles,
            vec!["recover:chunk-1", "recover:chunk-2"]
        );
        let json = serde_json::to_string(&receipt).unwrap();
        assert!(!json.contains("selected_context"));
        assert!(!json.contains("omitted_context"));
        assert!(!json.contains("text_preview"));
    }

    #[test]
    fn receipt_reference_rejects_unbounded_or_invalid_identity() {
        let empty = ContextReceiptRef::new(
            String::new(),
            "workstream:auth".into(),
            String::new(),
            String::new(),
            String::new(),
            "sha256:x".into(),
            vec![],
            vec![],
            1,
        );
        assert!(matches!(
            empty,
            Err(EngineContractError::EmptyField("receipt_id"))
        ));

        let negative_time = ContextReceiptRef::new(
            "receipt:1".into(),
            "workstream:auth".into(),
            String::new(),
            String::new(),
            String::new(),
            "sha256:x".into(),
            vec![],
            vec![],
            -1,
        );
        assert!(matches!(
            negative_time,
            Err(EngineContractError::InvalidTimestamp("observed_at_ms"))
        ));

        let too_many = (0..=MAX_SCOPE_ITEMS)
            .map(|index| format!("evidence:{index}"))
            .collect();
        let result = ContextReceiptRef::new(
            "receipt:1".into(),
            "workstream:auth".into(),
            String::new(),
            String::new(),
            String::new(),
            "sha256:x".into(),
            too_many,
            vec![],
            1,
        );
        assert!(matches!(
            result,
            Err(EngineContractError::TooManyItems {
                field: "selected_evidence_ids",
                ..
            })
        ));
    }
}
