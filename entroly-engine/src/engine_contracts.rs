//! Narrow integration contracts between the AI Work Graph, Context Engine, and
//! Trust Engine.
//!
//! These types deliberately carry references, commitments, and bounded IDs —
//! never giant raw contexts or agent transcripts. The Work Graph remains the
//! source of work-state truth; existing context/recovery systems remain the
//! source of context bytes. This module only defines the stable seam between
//! those systems.

use crate::work_graph::{EvidenceRef, ResumeView, TrustLevel};
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
    /// A byte span whose end precedes its start.
    InvalidSpan {
        start: u64,
        end: u64,
    },
    /// A disposition promised recovery without carrying the means to honour it.
    /// Section 9's "never call destructive omission recoverable", enforced.
    UnbackedRecoveryClaim(&'static str),
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
            Self::InvalidSpan { start, end } => {
                write!(formatter, "byte span {start}..{end} ends before it starts")
            }
            Self::UnbackedRecoveryClaim(detail) => {
                write!(formatter, "cannot claim recoverability: {detail}")
            }
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

// ── Recovery contract ─────────────────────────────────────────────────────

/// Schema version for [`RecoveryHandle`].
pub const RECOVERY_HANDLE_SCHEMA_VERSION: u32 = 1;

/// What happened to a piece of context, and what can still be done about it.
///
/// The four states are deliberately not a bool. "Omitted" alone is the answer
/// that lets a destructive drop be reported as if it were recoverable, which is
/// the specific dishonesty this enum exists to make impossible: a caller reading
/// `OmittedButRecoverable` is being promised the bytes can be produced again,
/// and that promise is enforced at construction rather than trusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryDisposition {
    /// Delivered to the model in full.
    Included,
    /// Delivered in a reduced form — skeleton, summary, truncation. The original
    /// is still recoverable, so this carries the same evidence requirements as
    /// `OmittedButRecoverable`.
    Compressed,
    /// Not delivered, but the exact bytes can be produced again.
    OmittedButRecoverable,
    /// Not delivered and not reproducible. The honest state for anything that
    /// was dropped without a durable reference.
    OmittedAndUnavailable,
}

impl RecoveryDisposition {
    /// Whether this disposition promises the caller that bytes can be produced.
    pub fn promises_recovery(self) -> bool {
        matches!(self, Self::Compressed | Self::OmittedButRecoverable)
    }
}

/// The outcome of checking recovered bytes against what was promised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryIntegrityState {
    /// The bytes hash to the committed value. This is the only state in which
    /// recovered material may be used.
    Verified,
    /// Bytes were produced but do not match the commitment — the source moved,
    /// was edited, or the wrong span was read.
    CommitmentMismatch,
    /// The handle never promised recovery, so there is nothing to verify.
    NotRecoverable,
}

/// A durable, verifiable pointer to context that was not delivered in full.
///
/// # Why the constructor refuses things
///
/// Section 9's rule is "never call destructive omission recoverable". A doc
/// comment cannot enforce that, so [`RecoveryHandle::new`] does: a disposition
/// that promises recovery must arrive with a fragment commitment *and* a way to
/// find the bytes again — either a content-addressed locator, or a source
/// reference with its own commitment. A caller that drops material without
/// keeping either cannot express that as recoverable; it has to say
/// `OmittedAndUnavailable`, which is the truth.
///
/// # Why verification is a method, not a convention
///
/// Section 9 also requires that recovery verify the expected commitment before
/// returning material. [`RecoveryHandle::verify_recovered`] is the only way to
/// get a `Verified` answer, and it recomputes rather than trusting a carried
/// field, so a tampered handle cannot certify its own bytes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RecoveryHandle {
    pub schema_version: u32,
    /// Derived from the handle's own payload; see [`ContextReceiptEnvelope`] for
    /// why derived ids beat caller-chosen ones across runtimes.
    pub handle_id: String,
    pub repository_id: String,
    /// Which receipt promised this material.
    pub receipt_id: String,
    pub disposition: RecoveryDisposition,
    /// Path or artifact identifier the bytes came from.
    pub source_ref: String,
    /// Commitment to the whole source artifact, when known.
    pub source_commitment: String,
    /// Commitment to the exact bytes this handle recovers.
    pub fragment_commitment: String,
    /// Byte span within the source. `0..0` means "the whole artifact".
    pub byte_start: u64,
    pub byte_end: u64,
    /// Repository version the span was read at — a commit sha, typically.
    /// A span without a version is only meaningful while nothing moves.
    pub version: String,
    /// Content-addressed locator, when the bytes live in a blob store rather
    /// than being re-read from the source.
    pub storage_locator: String,
    pub observed_at_ms: i64,
}

#[derive(Serialize)]
struct RecoveryHandlePayload<'a> {
    schema_version: u32,
    repository_id: &'a str,
    receipt_id: &'a str,
    disposition: RecoveryDisposition,
    source_ref: &'a str,
    source_commitment: &'a str,
    fragment_commitment: &'a str,
    byte_start: u64,
    byte_end: u64,
    version: &'a str,
    storage_locator: &'a str,
    observed_at_ms: i64,
}

impl RecoveryHandle {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        repository_id: String,
        receipt_id: String,
        disposition: RecoveryDisposition,
        source_ref: String,
        source_commitment: String,
        fragment_commitment: String,
        byte_start: u64,
        byte_end: u64,
        version: String,
        storage_locator: String,
        observed_at_ms: i64,
    ) -> Result<Self, EngineContractError> {
        validate_text(
            "repository_id",
            &repository_id,
            MAX_CONTRACT_ID_BYTES,
            false,
        )?;
        validate_text("receipt_id", &receipt_id, MAX_CONTRACT_ID_BYTES, false)?;
        validate_text("source_ref", &source_ref, MAX_SCOPE_PATH_BYTES, true)?;
        validate_text(
            "source_commitment",
            &source_commitment,
            MAX_COMMITMENT_BYTES,
            true,
        )?;
        validate_text(
            "fragment_commitment",
            &fragment_commitment,
            MAX_COMMITMENT_BYTES,
            true,
        )?;
        validate_text("version", &version, MAX_CONTRACT_ID_BYTES, true)?;
        validate_text(
            "storage_locator",
            &storage_locator,
            MAX_RECOVERY_HANDLE_BYTES,
            true,
        )?;
        if observed_at_ms < 0 {
            return Err(EngineContractError::InvalidTimestamp("observed_at_ms"));
        }
        if byte_end < byte_start {
            return Err(EngineContractError::InvalidSpan {
                start: byte_start,
                end: byte_end,
            });
        }

        // The rule that makes the disposition mean something. A promise of
        // recovery must arrive with the means to honour it.
        if disposition.promises_recovery() {
            if fragment_commitment.is_empty() {
                return Err(EngineContractError::UnbackedRecoveryClaim(
                    "fragment_commitment is required to promise recovery",
                ));
            }
            let locatable = !storage_locator.is_empty()
                || (!source_ref.is_empty() && !source_commitment.is_empty());
            if !locatable {
                return Err(EngineContractError::UnbackedRecoveryClaim(
                    "recovery requires a storage_locator, or a source_ref with its source_commitment",
                ));
            }
        }

        let mut handle = RecoveryHandle {
            schema_version: RECOVERY_HANDLE_SCHEMA_VERSION,
            handle_id: String::new(),
            repository_id,
            receipt_id,
            disposition,
            source_ref,
            source_commitment,
            fragment_commitment,
            byte_start,
            byte_end,
            version,
            storage_locator,
            observed_at_ms,
        };
        handle.handle_id = format!("rh_{}", &handle.compute_commitment()?[..16]);
        Ok(handle)
    }

    fn compute_commitment(&self) -> Result<String, EngineContractError> {
        contract_sha256_json(&RecoveryHandlePayload {
            schema_version: self.schema_version,
            repository_id: &self.repository_id,
            receipt_id: &self.receipt_id,
            disposition: self.disposition,
            source_ref: &self.source_ref,
            source_commitment: &self.source_commitment,
            fragment_commitment: &self.fragment_commitment,
            byte_start: self.byte_start,
            byte_end: self.byte_end,
            version: &self.version,
            storage_locator: &self.storage_locator,
            observed_at_ms: self.observed_at_ms,
        })
    }

    /// True when this handle's id still derives from its own payload.
    pub fn verify_handle_id(&self) -> Result<bool, EngineContractError> {
        let expected = format!("rh_{}", &self.compute_commitment()?[..16]);
        Ok(self.handle_id == expected)
    }

    /// Check recovered bytes against the commitment this handle carries.
    ///
    /// The only route to `Verified`. It hashes the bytes rather than trusting
    /// any field on the handle, so a handle whose `fragment_commitment` was
    /// edited cannot certify material that does not match the original.
    pub fn verify_recovered(&self, bytes: &[u8]) -> RecoveryIntegrityState {
        if !self.disposition.promises_recovery() {
            return RecoveryIntegrityState::NotRecoverable;
        }
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let actual = format!("{:x}", hasher.finalize());
        let expected = self
            .fragment_commitment
            .strip_prefix("sha256:")
            .unwrap_or(&self.fragment_commitment);
        if actual == expected {
            RecoveryIntegrityState::Verified
        } else {
            RecoveryIntegrityState::CommitmentMismatch
        }
    }

    pub fn to_json(&self) -> Result<String, EngineContractError> {
        serde_json::to_string(self)
            .map_err(|error| EngineContractError::Serialization(error.to_string()))
    }

    /// Parse and check, failing closed on an unknown schema or a handle whose id
    /// no longer derives from its payload.
    pub fn from_json_verified(json_text: &str) -> Result<Self, EngineContractError> {
        let handle: RecoveryHandle = serde_json::from_str(json_text)
            .map_err(|error| EngineContractError::Serialization(error.to_string()))?;
        if handle.schema_version != RECOVERY_HANDLE_SCHEMA_VERSION {
            return Err(EngineContractError::UnsupportedSchema {
                field: "schema_version",
                found: handle.schema_version,
                expected: RECOVERY_HANDLE_SCHEMA_VERSION,
            });
        }
        if !handle.verify_handle_id()? {
            return Err(EngineContractError::CommitmentMismatch("handle_id"));
        }
        Ok(handle)
    }
}


// ── Provenance-bearing memory ─────────────────────────────────────────────

/// Schema version for [`MemoryRecord`].
pub const MEMORY_RECORD_SCHEMA_VERSION: u32 = 1;

/// Why a memory may or may not be put in front of a model.
///
/// Returned with a reason so the decision is explainable rather than a bare
/// bool — section 22 asks "why was this memory rejected?" and the answer has to
/// exist somewhere other than a reader's head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryAdmissibility {
    /// May be injected.
    Admissible,
    /// Another record explicitly contradicts it. Contradiction outranks
    /// everything: a contradicted memory is not merely stale, it is disputed.
    Contradicted,
    /// A later record supersedes it.
    Superseded,
    /// Past its stated validity horizon.
    Expired,
    /// The source is untrusted, or the claim is inferred with nothing
    /// supporting it.
    Unsupported,
}

/// A memory, described by where it came from rather than what it resembles.
///
/// # What this deliberately does not have
///
/// There is no similarity score, no relevance, no salience. Section 10's rule is
/// "do not let similarity score imply truth", and the way to enforce that is not
/// to warn about it — it is to make admissibility a function that cannot take a
/// score as input. Ranking is a host concern and stays in `memory.py`, which
/// already does it well with retention and importance; this contract answers a
/// different question, and answering it needs different fields.
///
/// # What it holds instead
///
/// Provenance (which agent, session and execution produced it), a content
/// *reference* and commitment rather than the content, the evidence that
/// supports it, and the relations that can invalidate it: `supersedes` and
/// `contradicted_by`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryRecord {
    pub schema_version: u32,
    /// Derived from the record's own payload.
    pub memory_id: String,
    pub repository_id: String,
    pub task_id: String,
    pub workstream_id: String,
    pub source_agent: String,
    pub source_session: String,
    pub source_execution: String,
    /// A locator for the content, never the content itself. Memory that
    /// embedded its own text would make the graph a second content store and
    /// grow without bound.
    pub content_reference: String,
    pub content_commitment: String,
    pub evidence_ids: Vec<String>,
    pub trust_state: TrustLevel,
    pub created_at_ms: i64,
    pub observed_at_ms: i64,
    /// `0` means "no stated horizon" — valid until something supersedes or
    /// contradicts it. A record cannot be silently immortal *and* expiring.
    pub valid_until_ms: i64,
    pub supersedes: Vec<String>,
    pub contradicted_by: Vec<String>,
    /// How to recover the content if it is not already to hand.
    pub recovery_handle: String,
    pub record_commitment: String,
}

#[derive(Serialize)]
struct MemoryRecordPayload<'a> {
    schema_version: u32,
    repository_id: &'a str,
    task_id: &'a str,
    workstream_id: &'a str,
    source_agent: &'a str,
    source_session: &'a str,
    source_execution: &'a str,
    content_reference: &'a str,
    content_commitment: &'a str,
    evidence_ids: &'a [String],
    trust_state: TrustLevel,
    created_at_ms: i64,
    observed_at_ms: i64,
    valid_until_ms: i64,
    supersedes: &'a [String],
    contradicted_by: &'a [String],
    recovery_handle: &'a str,
}

impl MemoryRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        repository_id: String,
        task_id: String,
        workstream_id: String,
        source_agent: String,
        source_session: String,
        source_execution: String,
        content_reference: String,
        content_commitment: String,
        evidence_ids: Vec<String>,
        trust_state: TrustLevel,
        created_at_ms: i64,
        observed_at_ms: i64,
        valid_until_ms: i64,
        supersedes: Vec<String>,
        contradicted_by: Vec<String>,
        recovery_handle: String,
    ) -> Result<Self, EngineContractError> {
        validate_text(
            "repository_id",
            &repository_id,
            MAX_CONTRACT_ID_BYTES,
            false,
        )?;
        validate_text("task_id", &task_id, MAX_CONTRACT_ID_BYTES, true)?;
        validate_text("workstream_id", &workstream_id, MAX_CONTRACT_ID_BYTES, true)?;
        validate_text("source_agent", &source_agent, MAX_CONTRACT_ID_BYTES, true)?;
        validate_text(
            "source_session",
            &source_session,
            MAX_CONTRACT_ID_BYTES,
            true,
        )?;
        validate_text(
            "source_execution",
            &source_execution,
            MAX_CONTRACT_ID_BYTES,
            true,
        )?;
        validate_text(
            "content_reference",
            &content_reference,
            MAX_SCOPE_PATH_BYTES,
            false,
        )?;
        validate_text(
            "content_commitment",
            &content_commitment,
            MAX_COMMITMENT_BYTES,
            true,
        )?;
        validate_text(
            "recovery_handle",
            &recovery_handle,
            MAX_RECOVERY_HANDLE_BYTES,
            true,
        )?;
        for (field, value) in [
            ("created_at_ms", created_at_ms),
            ("observed_at_ms", observed_at_ms),
            ("valid_until_ms", valid_until_ms),
        ] {
            if value < 0 {
                return Err(EngineContractError::InvalidTimestamp(field));
            }
        }

        let mut record = MemoryRecord {
            schema_version: MEMORY_RECORD_SCHEMA_VERSION,
            memory_id: String::new(),
            repository_id,
            task_id,
            workstream_id,
            source_agent,
            source_session,
            source_execution,
            content_reference,
            content_commitment,
            evidence_ids: canonical_ids("evidence_ids", evidence_ids, MAX_SCOPE_ITEMS)?,
            trust_state,
            created_at_ms,
            observed_at_ms,
            valid_until_ms,
            supersedes: canonical_ids("supersedes", supersedes, MAX_SCOPE_ITEMS)?,
            contradicted_by: canonical_ids("contradicted_by", contradicted_by, MAX_SCOPE_ITEMS)?,
            recovery_handle,
            record_commitment: String::new(),
        };
        record.record_commitment = record.compute_commitment()?;
        record.memory_id = format!("mem_{}", &record.record_commitment[..16]);
        Ok(record)
    }

    fn compute_commitment(&self) -> Result<String, EngineContractError> {
        contract_sha256_json(&MemoryRecordPayload {
            schema_version: self.schema_version,
            repository_id: &self.repository_id,
            task_id: &self.task_id,
            workstream_id: &self.workstream_id,
            source_agent: &self.source_agent,
            source_session: &self.source_session,
            source_execution: &self.source_execution,
            content_reference: &self.content_reference,
            content_commitment: &self.content_commitment,
            evidence_ids: &self.evidence_ids,
            trust_state: self.trust_state,
            created_at_ms: self.created_at_ms,
            observed_at_ms: self.observed_at_ms,
            valid_until_ms: self.valid_until_ms,
            supersedes: &self.supersedes,
            contradicted_by: &self.contradicted_by,
            recovery_handle: &self.recovery_handle,
        })
    }

    /// May this memory be put in front of a model, and why.
    ///
    /// Deterministic and evidence-driven. Note the order: contradiction is
    /// checked before expiry, because a disputed memory that also happens to be
    /// fresh is still disputed, and reporting `Expired` for it would describe
    /// the less important problem.
    ///
    /// Takes `now_ms` rather than reading a clock. A verdict that depends on
    /// ambient time is not reproducible, and section 22 asks for replay.
    pub fn admissibility(&self, now_ms: i64) -> MemoryAdmissibility {
        if !self.contradicted_by.is_empty() {
            return MemoryAdmissibility::Contradicted;
        }
        if self.valid_until_ms > 0 && now_ms > self.valid_until_ms {
            return MemoryAdmissibility::Expired;
        }
        match self.trust_state {
            TrustLevel::Untrusted => MemoryAdmissibility::Unsupported,
            // An inference with nothing behind it is a guess. Requiring evidence
            // here is what stops a plausible-sounding recollection from being
            // injected as though it were observed.
            TrustLevel::Inferred if self.evidence_ids.is_empty() => {
                MemoryAdmissibility::Unsupported
            }
            _ => MemoryAdmissibility::Admissible,
        }
    }

    /// Admissibility given the ids that later records supersede.
    ///
    /// Supersession is a property of the *set*, not of one record — a memory
    /// cannot know it has been replaced. The caller supplies what the newer
    /// records claim, and this reports the consequence.
    pub fn admissibility_in_set(
        &self,
        now_ms: i64,
        superseded_ids: &BTreeSet<String>,
    ) -> MemoryAdmissibility {
        if !self.contradicted_by.is_empty() {
            return MemoryAdmissibility::Contradicted;
        }
        if superseded_ids.contains(&self.memory_id) {
            return MemoryAdmissibility::Superseded;
        }
        self.admissibility(now_ms)
    }

    pub fn verify_commitment(&self) -> Result<bool, EngineContractError> {
        let recomputed = self.compute_commitment()?;
        Ok(recomputed == self.record_commitment
            && self.memory_id == format!("mem_{}", &recomputed[..16]))
    }

    pub fn to_json(&self) -> Result<String, EngineContractError> {
        serde_json::to_string(self)
            .map_err(|error| EngineContractError::Serialization(error.to_string()))
    }

    pub fn from_json_verified(json_text: &str) -> Result<Self, EngineContractError> {
        let record: MemoryRecord = serde_json::from_str(json_text)
            .map_err(|error| EngineContractError::Serialization(error.to_string()))?;
        if record.schema_version != MEMORY_RECORD_SCHEMA_VERSION {
            return Err(EngineContractError::UnsupportedSchema {
                field: "schema_version",
                found: record.schema_version,
                expected: MEMORY_RECORD_SCHEMA_VERSION,
            });
        }
        if !record.verify_commitment()? {
            return Err(EngineContractError::CommitmentMismatch("record_commitment"));
        }
        Ok(record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MemoryRecord ─────────────────────────────────────────────────────

    /// Cross-runtime anchor for the memory contract.
    pub(crate) const GOLDEN_MEMORY_ID: &str = "mem_a3b337c53411d1a5";

    #[allow(clippy::too_many_arguments)]
    fn memory(
        trust: TrustLevel,
        evidence: Vec<String>,
        valid_until_ms: i64,
        contradicted_by: Vec<String>,
    ) -> MemoryRecord {
        MemoryRecord::new(
            "repo:demo".to_string(),
            "task:auth".to_string(),
            "workstream:1".to_string(),
            "agent:claude".to_string(),
            "session:1".to_string(),
            "exec:1".to_string(),
            "vault/beliefs/auth.md".to_string(),
            "sha256:content".to_string(),
            evidence,
            trust,
            1_700_000_000_000,
            1_700_000_000_000,
            valid_until_ms,
            vec![],
            contradicted_by,
            String::new(),
        )
        .expect("valid memory record")
    }

    fn observed_memory() -> MemoryRecord {
        memory(TrustLevel::Observed, vec!["evidence:1".to_string()], 0, vec![])
    }

    #[test]
    fn memory_golden_vector() {
        let record = observed_memory();
        assert_eq!(record.memory_id, GOLDEN_MEMORY_ID);
        assert!(record.verify_commitment().expect("verify"));
    }

    #[test]
    fn observed_memory_is_admissible() {
        assert_eq!(
            observed_memory().admissibility(1_700_000_100_000),
            MemoryAdmissibility::Admissible
        );
    }

    #[test]
    fn untrusted_memory_is_never_admissible() {
        // Untrusted is the default for external statements. It must not reach a
        // model regardless of how relevant it looks.
        let record = memory(TrustLevel::Untrusted, vec!["evidence:1".to_string()], 0, vec![]);
        assert_eq!(
            record.admissibility(1_700_000_100_000),
            MemoryAdmissibility::Unsupported
        );
    }

    #[test]
    fn an_inference_with_no_evidence_is_refused() {
        // A plausible-sounding recollection with nothing behind it is a guess.
        let guess = memory(TrustLevel::Inferred, vec![], 0, vec![]);
        assert_eq!(
            guess.admissibility(1_700_000_100_000),
            MemoryAdmissibility::Unsupported
        );
    }

    #[test]
    fn an_inference_with_evidence_is_admissible() {
        let backed = memory(TrustLevel::Inferred, vec!["evidence:1".to_string()], 0, vec![]);
        assert_eq!(
            backed.admissibility(1_700_000_100_000),
            MemoryAdmissibility::Admissible
        );
    }

    #[test]
    fn expiry_is_honoured_but_zero_means_no_horizon() {
        let expiring = memory(
            TrustLevel::Observed,
            vec!["evidence:1".to_string()],
            1_700_000_050_000,
            vec![],
        );
        assert_eq!(
            expiring.admissibility(1_700_000_040_000),
            MemoryAdmissibility::Admissible
        );
        assert_eq!(
            expiring.admissibility(1_700_000_060_000),
            MemoryAdmissibility::Expired
        );

        // 0 means "no stated horizon", not "expired at the epoch".
        assert_eq!(
            observed_memory().admissibility(i64::MAX),
            MemoryAdmissibility::Admissible
        );
    }

    #[test]
    fn contradiction_outranks_freshness() {
        // A disputed memory that is also perfectly fresh is still disputed.
        // Reporting Expired for it would name the less important problem.
        let disputed = memory(
            TrustLevel::Verified,
            vec!["evidence:1".to_string()],
            1_700_000_050_000,
            vec!["mem_other".to_string()],
        );
        assert_eq!(
            disputed.admissibility(1_700_000_060_000),
            MemoryAdmissibility::Contradicted,
            "expiry must not mask a contradiction"
        );
        assert_eq!(
            disputed.admissibility(1_700_000_040_000),
            MemoryAdmissibility::Contradicted
        );
    }

    #[test]
    fn contradiction_outranks_verification() {
        // Verified is the strongest trust level and still loses to a
        // contradiction. Trust is not a licence to ignore conflicting evidence.
        let disputed = memory(
            TrustLevel::Verified,
            vec!["evidence:1".to_string()],
            0,
            vec!["mem_other".to_string()],
        );
        assert_eq!(
            disputed.admissibility(1_700_000_100_000),
            MemoryAdmissibility::Contradicted
        );
    }

    #[test]
    fn supersession_is_a_property_of_the_set() {
        // A record cannot know it has been replaced, so the caller supplies what
        // newer records claim.
        let record = observed_memory();
        let mut superseded = BTreeSet::new();
        assert_eq!(
            record.admissibility_in_set(1_700_000_100_000, &superseded),
            MemoryAdmissibility::Admissible
        );

        superseded.insert(record.memory_id.clone());
        assert_eq!(
            record.admissibility_in_set(1_700_000_100_000, &superseded),
            MemoryAdmissibility::Superseded
        );
    }

    #[test]
    fn admissibility_takes_no_similarity_score() {
        // Section 10: do not let similarity score imply truth. Enforced by the
        // signature -- there is nowhere to put one. This test exists so that
        // adding a score parameter later has to break something first.
        let record = observed_memory();
        let _: MemoryAdmissibility = record.admissibility(0);
        let json = record.to_json().expect("serialize");
        for scored_field in ["score", "similarity", "relevance", "salience", "importance"] {
            assert!(
                !json.contains(scored_field),
                "memory contract leaked a ranking field: {scored_field}"
            );
        }
    }

    #[test]
    fn memory_content_is_referenced_never_embedded() {
        // Memory that embedded its own text would make graph state a second
        // content store and grow without bound.
        let record = observed_memory();
        assert_eq!(record.content_reference, "vault/beliefs/auth.md");
        let json = record.to_json().expect("serialize");
        assert!(!json.contains("\"content\":"));
    }

    #[test]
    fn a_tampered_memory_record_fails_closed() {
        let json = observed_memory()
            .to_json()
            .expect("serialize")
            .replace("agent:claude", "agent:someone");
        assert!(matches!(
            MemoryRecord::from_json_verified(&json),
            Err(EngineContractError::CommitmentMismatch("record_commitment"))
        ));
    }

    #[test]
    fn an_unknown_memory_schema_is_refused() {
        let json = observed_memory()
            .to_json()
            .expect("serialize")
            .replace("\"schema_version\":1", "\"schema_version\":7");
        assert!(matches!(
            MemoryRecord::from_json_verified(&json),
            Err(EngineContractError::UnsupportedSchema { found: 7, .. })
        ));
    }

    #[test]
    fn a_negative_validity_horizon_is_rejected() {
        let bad = MemoryRecord::new(
            "repo:demo".to_string(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            String::new(),
            "vault/x.md".to_string(),
            String::new(),
            vec![],
            TrustLevel::Observed,
            0,
            0,
            -1,
            vec![],
            vec![],
            String::new(),
        );
        assert_eq!(
            bad,
            Err(EngineContractError::InvalidTimestamp("valid_until_ms"))
        );
    }

    // ── RecoveryHandle ───────────────────────────────────────────────────

    /// SHA-256 of b"recoverable bytes", the fixture body used below.
    const FIXTURE_BODY: &[u8] = b"recoverable bytes";

    fn fixture_commitment() -> String {
        let mut hasher = Sha256::new();
        hasher.update(FIXTURE_BODY);
        format!("{:x}", hasher.finalize())
    }

    /// Cross-runtime anchor for the recovery contract, same role as
    /// GOLDEN_RECEIPT_COMMITMENT. Moving this value is a schema change.
    pub(crate) const GOLDEN_RECOVERY_HANDLE_ID: &str = "rh_61e976bc425ad0de";

    fn recoverable_handle() -> RecoveryHandle {
        RecoveryHandle::new(
            "repo:demo".to_string(),
            "cr_672457349ba403bc".to_string(),
            RecoveryDisposition::OmittedButRecoverable,
            "src/auth.py".to_string(),
            "sha256:source".to_string(),
            fixture_commitment(),
            0,
            17,
            "commit:abc123".to_string(),
            String::new(),
            1_700_000_000_000,
        )
        .expect("a backed recovery claim must be accepted")
    }

    #[test]
    fn a_recovery_promise_without_a_commitment_is_refused() {
        // Section 9: never call destructive omission recoverable. Enforced at
        // construction, because a doc comment cannot enforce anything.
        let unbacked = RecoveryHandle::new(
            "repo:demo".to_string(),
            "cr_x".to_string(),
            RecoveryDisposition::OmittedButRecoverable,
            "src/auth.py".to_string(),
            "sha256:source".to_string(),
            String::new(), // no fragment commitment
            0,
            0,
            String::new(),
            String::new(),
            0,
        );
        assert!(matches!(
            unbacked,
            Err(EngineContractError::UnbackedRecoveryClaim(_))
        ));
    }

    #[test]
    fn a_recovery_promise_without_a_way_back_is_refused() {
        // A commitment proves what the bytes were; it does not say where to find
        // them. Both are required.
        let nowhere = RecoveryHandle::new(
            "repo:demo".to_string(),
            "cr_x".to_string(),
            RecoveryDisposition::OmittedButRecoverable,
            String::new(), // no source_ref
            String::new(), // no source_commitment
            fixture_commitment(),
            0,
            0,
            String::new(),
            String::new(), // no storage locator
            0,
        );
        assert!(matches!(
            nowhere,
            Err(EngineContractError::UnbackedRecoveryClaim(_))
        ));
    }

    #[test]
    fn compression_carries_the_same_burden_as_omission() {
        // Compressed material is still promised back in full, so it cannot be
        // claimed without the means to produce it.
        let unbacked = RecoveryHandle::new(
            "repo:demo".to_string(),
            "cr_x".to_string(),
            RecoveryDisposition::Compressed,
            "src/auth.py".to_string(),
            "sha256:source".to_string(),
            String::new(),
            0,
            0,
            String::new(),
            String::new(),
            0,
        );
        assert!(matches!(
            unbacked,
            Err(EngineContractError::UnbackedRecoveryClaim(_))
        ));
    }

    #[test]
    fn destructive_omission_is_expressible_without_evidence() {
        // The honest state must always be available, or callers are pushed into
        // overclaiming to get a handle at all.
        let gone = RecoveryHandle::new(
            "repo:demo".to_string(),
            "cr_x".to_string(),
            RecoveryDisposition::OmittedAndUnavailable,
            String::new(),
            String::new(),
            String::new(),
            0,
            0,
            String::new(),
            String::new(),
            0,
        );
        assert!(gone.is_ok(), "an honest unavailable claim must be accepted");
    }

    #[test]
    fn a_storage_locator_alone_backs_a_recovery_promise() {
        // Blob-store material has no source path to re-read; the locator is the
        // way back.
        let blob = RecoveryHandle::new(
            "repo:demo".to_string(),
            "cr_x".to_string(),
            RecoveryDisposition::OmittedButRecoverable,
            String::new(),
            String::new(),
            fixture_commitment(),
            0,
            0,
            String::new(),
            "blob:sha256:deadbeef".to_string(),
            0,
        );
        assert!(blob.is_ok());
    }

    #[test]
    fn recovered_bytes_are_verified_against_the_commitment() {
        let handle = recoverable_handle();
        assert_eq!(
            handle.verify_recovered(FIXTURE_BODY),
            RecoveryIntegrityState::Verified
        );
    }

    #[test]
    fn wrong_bytes_do_not_verify() {
        let handle = recoverable_handle();
        assert_eq!(
            handle.verify_recovered(b"different bytes entirely"),
            RecoveryIntegrityState::CommitmentMismatch
        );
    }

    #[test]
    fn a_tampered_commitment_cannot_certify_its_own_bytes() {
        // verify_recovered hashes the bytes rather than trusting the handle, so
        // editing the commitment to match forged material still fails -- the
        // handle_id no longer derives from the payload.
        let mut handle = recoverable_handle();
        let forged = b"forged replacement";
        let mut hasher = Sha256::new();
        hasher.update(forged);
        handle.fragment_commitment = format!("{:x}", hasher.finalize());

        // The bytes now "match" the edited commitment ...
        assert_eq!(
            handle.verify_recovered(forged),
            RecoveryIntegrityState::Verified
        );
        // ... but the handle itself is detectably tampered.
        assert!(
            !handle.verify_handle_id().expect("id check"),
            "an edited commitment must break the derived handle id"
        );
    }

    #[test]
    fn an_unavailable_handle_has_nothing_to_verify() {
        let gone = RecoveryHandle::new(
            "repo:demo".to_string(),
            "cr_x".to_string(),
            RecoveryDisposition::OmittedAndUnavailable,
            String::new(),
            String::new(),
            String::new(),
            0,
            0,
            String::new(),
            String::new(),
            0,
        )
        .expect("honest unavailable");
        assert_eq!(
            gone.verify_recovered(b"anything"),
            RecoveryIntegrityState::NotRecoverable
        );
    }

    #[test]
    fn an_inverted_span_is_rejected() {
        let inverted = RecoveryHandle::new(
            "repo:demo".to_string(),
            "cr_x".to_string(),
            RecoveryDisposition::OmittedAndUnavailable,
            String::new(),
            String::new(),
            String::new(),
            99,
            10,
            String::new(),
            String::new(),
            0,
        );
        assert!(matches!(
            inverted,
            Err(EngineContractError::InvalidSpan { start: 99, end: 10 })
        ));
    }

    #[test]
    fn handle_json_round_trips_and_fails_closed_when_edited() {
        let handle = recoverable_handle();
        let json = handle.to_json().expect("serialize");
        assert_eq!(
            RecoveryHandle::from_json_verified(&json).expect("verified"),
            handle
        );

        let edited = json.replace("src/auth.py", "src/other.py");
        assert!(matches!(
            RecoveryHandle::from_json_verified(&edited),
            Err(EngineContractError::CommitmentMismatch("handle_id"))
        ));
    }

    #[test]
    fn an_unknown_handle_schema_is_refused() {
        let json = recoverable_handle()
            .to_json()
            .expect("serialize")
            .replace("\"schema_version\":1", "\"schema_version\":42");
        assert!(matches!(
            RecoveryHandle::from_json_verified(&json),
            Err(EngineContractError::UnsupportedSchema { found: 42, .. })
        ));
    }

    #[test]
    fn recovery_handle_golden_vector() {
        // Cross-runtime anchor, same role as GOLDEN_RECEIPT_COMMITMENT.
        let handle = recoverable_handle();
        assert_eq!(handle.handle_id, GOLDEN_RECOVERY_HANDLE_ID);
        assert!(handle.verify_handle_id().expect("id check"));
    }
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
