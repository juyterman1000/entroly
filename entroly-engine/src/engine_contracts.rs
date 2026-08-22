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

#[cfg(test)]
mod tests {
    use super::*;
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
