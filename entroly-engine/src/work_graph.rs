//! Evidence-backed temporal AI Work Graph.
//!
//! The work graph is deliberately I/O-free. Python, Node, MCP, and native
//! hosts observe Git/filesystem/provider state and submit normalized
//! observations; this module owns the semantics, inference rules, event
//! ordering, trust labels, deduplication, coordination analysis, resume views,
//! and handoff commitments. Keeping those rules here prevents pip and npm from
//! developing different ideas of what "unfinished work" means.
//!
//! Design invariants:
//! - append-only events are the source of truth; materialized nodes/edges are
//!   always rebuildable from them;
//! - event IDs and graph commitments are SHA-256 over canonical serde payloads;
//! - a clean repository is a null control and never invents an unfinished task;
//! - an agent statement is an observation, not automatically a verified fact;
//! - `Completed` requires explicit completion *and* passing verification;
//! - contradictions/failures block work rather than being averaged away;
//! - coordination leases are advisory and never become filesystem locks;
//! - persisted graph documents are integrity-checked on import.

use crate::coordination_index::{candidate_pairs, CoordinationScope};
use crate::engine_contracts::{
    ContextReceiptEnvelope, ExecutionState, MemoryAdmissibility, MemoryRecord,
    ModelExecutionOutcome, RoutingDecision, VerificationFreshness, VerificationRecord,
    VerificationVerdict, WorkContinuationProof, WorkScope,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

pub const WORK_GRAPH_SCHEMA_VERSION: u32 = 1;
const MAX_EVENTS: usize = 50_000;
const MAX_OPERATIONS_PER_EVENT: usize = 4_096;
const MAX_CHANGES_PER_EVENT: usize = 512;
const MAX_CHANGES_PER_OBSERVATION: usize = 16_384;
const MAX_NODES: usize = 100_000;
const MAX_EDGES: usize = 250_000;
const MAX_EVIDENCE: usize = 250_000;
const MAX_ID_LEN: usize = 512;
const MAX_LABEL_LEN: usize = 8_192;
const MAX_SOURCE_REF_LEN: usize = 8_192;
const MAX_ATTRIBUTE_KEYS: usize = 128;
const MAX_ATTRIBUTE_BYTES: usize = 128 * 1024;
const MAX_SCOPE_ITEMS: usize = 2_048;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkGraphError {
    InvalidInput(String),
    LimitExceeded(String),
    RepoMismatch { expected: String, actual: String },
    IntegrityMismatch { expected: String, actual: String },
    Serialization(String),
    UnknownNode(String),
}

impl fmt::Display for WorkGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(msg) => write!(f, "invalid work-graph input: {msg}"),
            Self::LimitExceeded(msg) => write!(f, "work-graph limit exceeded: {msg}"),
            Self::RepoMismatch { expected, actual } => {
                write!(
                    f,
                    "work-graph repo mismatch: expected {expected:?}, got {actual:?}"
                )
            }
            Self::IntegrityMismatch { expected, actual } => write!(
                f,
                "work-graph integrity mismatch: expected {expected}, got {actual}"
            ),
            Self::Serialization(msg) => write!(f, "work-graph serialization error: {msg}"),
            Self::UnknownNode(id) => write!(f, "unknown work-graph node: {id}"),
        }
    }
}

impl Error for WorkGraphError {}

impl From<serde_json::Error> for WorkGraphError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    /// Untrusted external statement or transport metadata.
    #[default]
    Untrusted,
    /// Derived conservatively from other observations.
    Inferred,
    /// Directly observed durable state (for example Git status).
    Observed,
    /// Independently checked evidence (for example a passing test or receipt).
    Verified,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Repository,
    File,
    Symbol,
    Task,
    Workstream,
    Agent,
    Session,
    Model,
    ModelExecution,
    Change,
    Commit,
    PullRequest,
    Test,
    CiRun,
    Decision,
    Claim,
    Memory,
    Evidence,
    Receipt,
    Handoff,
    Failure,
    WorkLease,
}

impl NodeKind {
    fn token(self) -> &'static str {
        match self {
            Self::Repository => "repository",
            Self::File => "file",
            Self::Symbol => "symbol",
            Self::Task => "task",
            Self::Workstream => "workstream",
            Self::Agent => "agent",
            Self::Session => "session",
            Self::Model => "model",
            Self::ModelExecution => "model_execution",
            Self::Change => "change",
            Self::Commit => "commit",
            Self::PullRequest => "pull_request",
            Self::Test => "test",
            Self::CiRun => "ci_run",
            Self::Decision => "decision",
            Self::Claim => "claim",
            Self::Memory => "memory",
            Self::Evidence => "evidence",
            Self::Receipt => "receipt",
            Self::Handoff => "handoff",
            Self::Failure => "failure",
            Self::WorkLease => "work_lease",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    Contains,
    Defines,
    Calls,
    Imports,
    DependsOn,
    WorksOn,
    DelegatedTo,
    Changed,
    Touches,
    Affects,
    SupportedBy,
    ContradictedBy,
    VerifiedBy,
    ProducedBy,
    Continues,
    HandedOffTo,
    Blocks,
    ConflictsWith,
    Supersedes,
    RoutedTo,
    PartOf,
    RecoversTo,
    References,
}

impl EdgeKind {
    fn token(self) -> &'static str {
        match self {
            Self::Contains => "contains",
            Self::Defines => "defines",
            Self::Calls => "calls",
            Self::Imports => "imports",
            Self::DependsOn => "depends_on",
            Self::WorksOn => "works_on",
            Self::DelegatedTo => "delegated_to",
            Self::Changed => "changed",
            Self::Touches => "touches",
            Self::Affects => "affects",
            Self::SupportedBy => "supported_by",
            Self::ContradictedBy => "contradicted_by",
            Self::VerifiedBy => "verified_by",
            Self::ProducedBy => "produced_by",
            Self::Continues => "continues",
            Self::HandedOffTo => "handed_off_to",
            Self::Blocks => "blocks",
            Self::ConflictsWith => "conflicts_with",
            Self::Supersedes => "supersedes",
            Self::RoutedTo => "routed_to",
            Self::PartOf => "part_of",
            Self::RecoversTo => "recovers_to",
            Self::References => "references",
        }
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum WorkStatus {
    #[default]
    Unknown,
    Planned,
    InProgress,
    Blocked,
    NeedsVerification,
    Completed,
    Abandoned,
}

impl WorkStatus {
    fn is_unfinished(self) -> bool {
        matches!(
            self,
            Self::Planned | Self::InProgress | Self::Blocked | Self::NeedsVerification
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    GitStatus,
    GitCommit,
    Checkpoint,
    TestResult,
    CiResult,
    Receipt,
    RavsOutcome,
    Memory,
    AgentStatement,
    UserStatement,
    RepositoryFact,
    RuntimeObservation,
    Other,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum VerificationState {
    Passed,
    Failed,
    Skipped,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FileChangeKind {
    Added,
    Modified,
    Deleted,
    Renamed,
    Copied,
    Untracked,
    Unmerged,
    Unknown,
}

impl FileChangeKind {
    fn token(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Modified => "modified",
            Self::Deleted => "deleted",
            Self::Renamed => "renamed",
            Self::Copied => "copied",
            Self::Untracked => "untracked",
            Self::Unmerged => "unmerged",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClaimState {
    Grounded,
    Unsupported,
    Contradicted,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EvidenceRef {
    #[serde(default)]
    pub evidence_id: String,
    pub kind: EvidenceKind,
    #[serde(default)]
    pub source_ref: String,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub locator: String,
    #[serde(default)]
    pub trust: TrustLevel,
    #[serde(default)]
    pub observed_at_ms: i64,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkNode {
    pub node_id: String,
    pub kind: NodeKind,
    pub label: String,
    #[serde(default)]
    pub trust: TrustLevel,
    #[serde(default)]
    pub status: WorkStatus,
    /// Trust level of the status assertion. Kept separate from node trust so
    /// a later low-trust observation cannot downgrade verified work state.
    #[serde(default)]
    pub status_trust: TrustLevel,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
    #[serde(default)]
    pub evidence_ids: BTreeSet<String>,
    #[serde(default)]
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkEdge {
    pub edge_id: String,
    pub from_node: String,
    pub to_node: String,
    pub kind: EdgeKind,
    #[serde(default)]
    pub trust: TrustLevel,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
    #[serde(default)]
    pub evidence_ids: BTreeSet<String>,
    #[serde(default)]
    pub updated_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WorkOperation {
    AddEvidence {
        evidence: EvidenceRef,
    },
    UpsertNode {
        node: WorkNode,
    },
    UpsertEdge {
        edge: WorkEdge,
    },
    SetStatus {
        node_id: String,
        status: WorkStatus,
        #[serde(default)]
        trust: TrustLevel,
        #[serde(default)]
        reason: String,
        #[serde(default)]
        evidence_ids: BTreeSet<String>,
    },
    AttachEvidence {
        node_id: String,
        evidence_ids: BTreeSet<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkEvent {
    #[serde(default)]
    pub event_id: String,
    pub observed_at_ms: i64,
    pub source_kind: EvidenceKind,
    #[serde(default)]
    pub source_ref: String,
    #[serde(default)]
    pub actor_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub operations: Vec<WorkOperation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskHint {
    #[serde(default)]
    pub task_id: String,
    pub title: String,
    #[serde(default = "default_observed_trust")]
    pub trust: TrustLevel,
    #[serde(default)]
    pub explicit_status: WorkStatus,
    #[serde(default)]
    pub remaining_work: Vec<String>,
    #[serde(default = "default_task_evidence_kind")]
    pub source_kind: EvidenceKind,
    #[serde(default)]
    pub source_ref: String,
}

fn default_observed_trust() -> TrustLevel {
    TrustLevel::Observed
}

fn default_task_evidence_kind() -> EvidenceKind {
    EvidenceKind::UserStatement
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct BranchObservation {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub head_sha: String,
    #[serde(default)]
    pub base_ref: String,
    #[serde(default)]
    pub default_branch: String,
    #[serde(default)]
    pub ahead_by: u64,
    #[serde(default)]
    pub behind_by: u64,
    #[serde(default)]
    pub merge_in_progress: bool,
    #[serde(default)]
    pub rebase_in_progress: bool,
    #[serde(default)]
    pub detached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileChangeObservation {
    pub path: String,
    pub kind: FileChangeKind,
    #[serde(default)]
    pub staged: bool,
    #[serde(default)]
    pub conflicted: bool,
    #[serde(default)]
    pub old_path: String,
    #[serde(default)]
    pub content_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommitObservation {
    pub sha: String,
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub timestamp_ms: i64,
    #[serde(default)]
    pub parent_shas: Vec<String>,
    #[serde(default)]
    pub changed_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VerificationObservation {
    pub verification_id: String,
    pub name: String,
    pub state: VerificationState,
    pub evidence_kind: EvidenceKind,
    #[serde(default)]
    pub source_ref: String,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub observed_at_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecisionObservation {
    #[serde(default)]
    pub decision_id: String,
    pub text: String,
    #[serde(default)]
    pub source_ref: String,
    #[serde(default = "default_decision_evidence_kind")]
    pub source_kind: EvidenceKind,
    #[serde(default)]
    pub trust: TrustLevel,
}

fn default_decision_evidence_kind() -> EvidenceKind {
    EvidenceKind::Checkpoint
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClaimObservation {
    #[serde(default)]
    pub claim_id: String,
    pub text: String,
    pub state: ClaimState,
    /// Trust in the *claim assessment*, not in the claim text itself.
    /// Adapters should use `Verified` only for independently checked outcomes
    /// (for example WITNESS/RAVS evidence), never for a raw agent assertion.
    #[serde(default)]
    pub trust: TrustLevel,
    #[serde(default)]
    pub risk: f64,
    #[serde(default)]
    pub source_ref: String,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkLeaseObservation {
    #[serde(default)]
    pub lease_id: String,
    pub agent_id: String,
    #[serde(default)]
    pub task_id: String,
    #[serde(default)]
    pub scope_paths: Vec<String>,
    #[serde(default)]
    pub scope_symbols: Vec<String>,
    pub expires_at_ms: i64,
    #[serde(default)]
    pub source_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelExecutionObservation {
    #[serde(default)]
    pub execution_id: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub success: Option<bool>,
    #[serde(default)]
    pub latency_ms: u64,
    #[serde(default)]
    pub cost_micro_usd: u64,
    #[serde(default)]
    pub source_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepositoryObservation {
    pub repo_id: String,
    pub observed_at_ms: i64,
    #[serde(default)]
    pub repository_label: String,
    #[serde(default)]
    pub agent_id: String,
    #[serde(default)]
    pub session_id: String,
    #[serde(default)]
    pub task_hint: Option<TaskHint>,
    #[serde(default)]
    pub branch: BranchObservation,
    #[serde(default)]
    pub changes: Vec<FileChangeObservation>,
    #[serde(default)]
    pub commits: Vec<CommitObservation>,
    #[serde(default)]
    pub verifications: Vec<VerificationObservation>,
    #[serde(default)]
    pub decisions: Vec<DecisionObservation>,
    #[serde(default)]
    pub claims: Vec<ClaimObservation>,
    #[serde(default)]
    pub leases: Vec<WorkLeaseObservation>,
    #[serde(default)]
    pub model_executions: Vec<ModelExecutionObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkGraphSummary {
    pub schema_version: u32,
    pub repo_id: String,
    pub revision: u64,
    pub graph_commitment: String,
    pub event_count: usize,
    pub node_count: usize,
    pub edge_count: usize,
    pub evidence_count: usize,
    pub unfinished_count: usize,
    pub blocked_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkItemView {
    pub node_id: String,
    pub kind: NodeKind,
    pub label: String,
    pub status: WorkStatus,
    pub trust: TrustLevel,
    pub updated_at_ms: i64,
    #[serde(default)]
    pub task_ids: Vec<String>,
    #[serde(default)]
    pub agent_ids: Vec<String>,
    #[serde(default)]
    pub changed_paths: Vec<String>,
    #[serde(default)]
    pub symbol_ids: Vec<String>,
    #[serde(default)]
    pub commit_ids: Vec<String>,
    #[serde(default)]
    pub decision_ids: Vec<String>,
    #[serde(default)]
    pub failure_ids: Vec<String>,
    #[serde(default)]
    pub verification_ids: Vec<String>,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
}

/// A claim, carried to the successor with the trust that qualifies it.
///
/// Claims were the one evidence kind `resume` never returned. They are also the
/// kind most likely to say something a successor must not miss: a claim records
/// what a previous agent asserted about the work, together with whether that
/// assertion was grounded in evidence or merely inferred.
///
/// The trust level travels with the text on purpose. A bare string cannot
/// distinguish "the tests prove this" from "the previous agent believed this",
/// and presenting the second as the first is the fail-open direction for a
/// system whose whole claim is evidence-bounded reconstruction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClaimView {
    pub label: String,
    pub trust: TrustLevel,
    /// `grounded`, `contradicted`, `unverified`, ... as recorded on the node.
    pub claim_state: Option<String>,
    /// Caller-supplied risk weight, when one was recorded.
    pub risk: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResumeView {
    pub repo_id: String,
    pub graph_revision: u64,
    pub graph_commitment: String,
    pub selected_workstream: WorkItemView,
    pub task_labels: Vec<String>,
    pub agents: Vec<String>,
    pub decisions: Vec<String>,
    /// Claims scoped to the selected workstream, highest risk first.
    ///
    /// Added because they were being dropped entirely: a durable, engine-
    /// verified claim reading "this work is INCOMPLETE" with risk 0.9 was
    /// stored, correctly edged to its workstream, and then never shown to the
    /// agent picking the work up. Found by handing a repository to a second
    /// agent with no handoff and asking what it could reconstruct.
    #[serde(default)]
    pub claims: Vec<ClaimView>,
    pub failures: Vec<String>,
    pub verification: Vec<String>,
    pub changed_paths: Vec<String>,
    pub commits: Vec<String>,
    pub evidence: Vec<EvidenceRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoordinationConflict {
    pub lease_a: String,
    pub lease_b: String,
    pub agent_a: String,
    pub agent_b: String,
    pub task_a: String,
    pub task_b: String,
    pub overlapping_paths: Vec<String>,
    pub overlapping_symbols: Vec<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoordinationReport {
    pub as_of_ms: i64,
    pub active_leases: usize,
    pub conflicts: Vec<CoordinationConflict>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HandoffReceipt {
    pub schema_version: u32,
    pub repo_id: String,
    pub graph_revision: u64,
    pub graph_commitment: String,
    pub workstream_id: String,
    pub from_agent: String,
    pub to_agent: String,
    pub generated_at_ms: i64,
    pub node_ids: Vec<String>,
    pub edge_ids: Vec<String>,
    pub evidence_ids: Vec<String>,
    pub payload_commitment: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkGraphDocument {
    schema_version: u32,
    repo_id: String,
    events: Vec<WorkEvent>,
    graph_commitment: String,
}

#[derive(Debug, Clone)]
pub struct WorkGraph {
    repo_id: String,
    events: Vec<WorkEvent>,
    /// Derived membership index for the append-only event log. Never serialized.
    /// Keeping this separate removes an O(N) scan from every long-session append.
    event_ids: BTreeSet<String>,
    /// The most recent passive repository snapshot and its final event ID.
    /// Only consecutive equal snapshots collapse: A -> B -> A remains an
    /// auditable history, while UI polling stays O(1). Never serialized.
    last_passive_snapshot: Option<(String, String)>,
    /// SHA-256 state for the canonical commitment bytes through the open events array.
    /// Derived runtime state only; persisted WorkGraphDocument remains unchanged.
    commitment_hasher: Sha256,
    nodes: BTreeMap<String, WorkNode>,
    edges: BTreeMap<String, WorkEdge>,
    evidence: BTreeMap<String, EvidenceRef>,
    /// Materialized undirected adjacency for bounded work-context traversal.
    /// This is derived state and is never serialized.
    adjacency: BTreeMap<String, BTreeSet<String>>,
    graph_commitment: String,
}

impl WorkGraph {
    pub fn new(repo_id: impl Into<String>) -> Result<Self, WorkGraphError> {
        let repo_id = repo_id.into();
        validate_text("repo_id", &repo_id, MAX_ID_LEN, false)?;
        let mut graph = Self {
            repo_id,
            events: Vec::new(),
            event_ids: BTreeSet::new(),
            last_passive_snapshot: None,
            commitment_hasher: Sha256::new(),
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            evidence: BTreeMap::new(),
            adjacency: BTreeMap::new(),
            graph_commitment: String::new(),
        };
        graph.refresh_commitment()?;
        Ok(graph)
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn revision(&self) -> u64 {
        self.events.len() as u64
    }

    pub fn graph_commitment(&self) -> &str {
        &self.graph_commitment
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn apply_event(&mut self, mut event: WorkEvent) -> Result<String, WorkGraphError> {
        validate_event(&mut event)?;
        if event.event_id.is_empty() {
            event.event_id = event_commitment(&event)?;
        } else {
            let expected = event_commitment(&event)?;
            if event.event_id != expected {
                return Err(WorkGraphError::IntegrityMismatch {
                    expected,
                    actual: event.event_id.clone(),
                });
            }
        }
        let id = event.event_id.clone();
        if self.event_ids.contains(&id) {
            return Ok(id);
        }
        if self.events.len() >= MAX_EVENTS {
            return Err(WorkGraphError::LimitExceeded(format!(
                "event count exceeds {MAX_EVENTS}"
            )));
        }
        self.validate_event_references_and_capacity(&event)?;

        let append_in_order = self.events.last().is_none_or(|last| {
            (last.observed_at_ms, last.event_id.as_str())
                <= (event.observed_at_ms, event.event_id.as_str())
        });
        self.events.push(event.clone());
        self.event_ids.insert(id.clone());

        let result = if append_in_order {
            self.apply_materialized(&event)
                .and_then(|_| self.append_commitment_event(&event))
        } else {
            self.rebuild()
        };
        if let Err(error) = result {
            self.events.retain(|existing| existing.event_id != id);
            // Restore the last known-good materialization before returning.
            self.rebuild()?;
            return Err(error);
        }
        self.refresh_last_passive_snapshot();
        Ok(id)
    }

    pub fn apply_event_json(&mut self, json_text: &str) -> Result<String, WorkGraphError> {
        let event: WorkEvent = serde_json::from_str(json_text)?;
        self.apply_event(event)
    }

    pub fn observe_repository(
        &mut self,
        observation: RepositoryObservation,
    ) -> Result<String, WorkGraphError> {
        if observation.repo_id != self.repo_id {
            return Err(WorkGraphError::RepoMismatch {
                expected: self.repo_id.clone(),
                actual: observation.repo_id.clone(),
            });
        }
        if observation.changes.len() > MAX_CHANGES_PER_OBSERVATION {
            return Err(WorkGraphError::LimitExceeded(format!(
                "file changes per observation exceed {MAX_CHANGES_PER_OBSERVATION}"
            )));
        }
        let passive_source_ref = passive_repository_snapshot_fingerprint(&observation)?
            .map(|fingerprint| format!("repo-snapshot:{fingerprint}"));
        if let (Some(source_ref), Some((last_source_ref, last_event_id))) =
            (&passive_source_ref, &self.last_passive_snapshot)
        {
            if source_ref == last_source_ref {
                return Ok(last_event_id.clone());
            }
        }

        if observation.changes.len() > MAX_CHANGES_PER_EVENT {
            let previous = self.clone();
            let mut last_event_id = String::new();
            for chunk in observation.changes.chunks(MAX_CHANGES_PER_EVENT) {
                let mut bounded = observation.clone();
                bounded.changes = chunk.to_vec();
                match self.observe_repository_single(bounded, passive_source_ref.as_deref()) {
                    Ok(event_id) => last_event_id = event_id,
                    Err(error) => {
                        *self = previous;
                        return Err(error);
                    }
                }
            }
            return Ok(last_event_id);
        }
        self.observe_repository_single(observation, passive_source_ref.as_deref())
    }

    fn observe_repository_single(
        &mut self,
        observation: RepositoryObservation,
        passive_source_ref: Option<&str>,
    ) -> Result<String, WorkGraphError> {
        let mut event = observation_to_event(&self.repo_id, observation)?;
        if let Some(source_ref) = passive_source_ref {
            event.source_ref = source_ref.to_string();
        }
        self.apply_event(event)
    }

    pub fn observe_repository_json(&mut self, json_text: &str) -> Result<String, WorkGraphError> {
        let observation: RepositoryObservation = serde_json::from_str(json_text)?;
        self.observe_repository(observation)
    }

    pub fn merge(&mut self, other: &WorkGraph) -> Result<usize, WorkGraphError> {
        if self.repo_id != other.repo_id {
            return Err(WorkGraphError::RepoMismatch {
                expected: self.repo_id.clone(),
                actual: other.repo_id.clone(),
            });
        }
        let before = self.events.len();
        let mut candidate = self.events.clone();
        let mut existing = self.event_ids.clone();
        for event in &other.events {
            if existing.insert(event.event_id.clone()) {
                candidate.push(event.clone());
            }
        }
        if candidate.len() > MAX_EVENTS {
            return Err(WorkGraphError::LimitExceeded(format!(
                "merged event count exceeds {MAX_EVENTS}"
            )));
        }
        let previous = std::mem::replace(&mut self.events, candidate);
        if let Err(error) = self.rebuild() {
            self.events = previous;
            self.rebuild()?;
            return Err(error);
        }
        Ok(self.events.len() - before)
    }

    pub fn merge_json(&mut self, json_text: &str) -> Result<usize, WorkGraphError> {
        let other = Self::from_json(json_text)?;
        self.merge(&other)
    }

    pub fn export_json(&self, pretty: bool) -> Result<String, WorkGraphError> {
        let doc = WorkGraphDocument {
            schema_version: WORK_GRAPH_SCHEMA_VERSION,
            repo_id: self.repo_id.clone(),
            events: self.events.clone(),
            graph_commitment: self.graph_commitment.clone(),
        };
        if pretty {
            serde_json::to_string_pretty(&doc).map_err(Into::into)
        } else {
            serde_json::to_string(&doc).map_err(Into::into)
        }
    }

    pub fn from_json(json_text: &str) -> Result<Self, WorkGraphError> {
        let doc: WorkGraphDocument = serde_json::from_str(json_text)?;
        if doc.schema_version != WORK_GRAPH_SCHEMA_VERSION {
            return Err(WorkGraphError::InvalidInput(format!(
                "unsupported work graph schema {}",
                doc.schema_version
            )));
        }
        let mut graph = Self::new(doc.repo_id)?;
        if doc.events.len() > MAX_EVENTS {
            return Err(WorkGraphError::LimitExceeded(format!(
                "event count exceeds {MAX_EVENTS}"
            )));
        }
        graph.events = doc.events;
        let mut seen_event_ids = BTreeSet::new();
        for event in &mut graph.events {
            validate_event(event)?;
            let expected = event_commitment(event)?;
            if event.event_id != expected {
                return Err(WorkGraphError::IntegrityMismatch {
                    expected,
                    actual: event.event_id.clone(),
                });
            }
            if !seen_event_ids.insert(event.event_id.clone()) {
                return Err(WorkGraphError::InvalidInput(format!(
                    "duplicate event id in persisted graph: {}",
                    event.event_id
                )));
            }
        }
        graph.rebuild()?;
        if graph.graph_commitment != doc.graph_commitment {
            return Err(WorkGraphError::IntegrityMismatch {
                expected: doc.graph_commitment,
                actual: graph.graph_commitment,
            });
        }
        Ok(graph)
    }

    pub fn summary(&self) -> WorkGraphSummary {
        let unfinished = self.unfinished_work();
        WorkGraphSummary {
            schema_version: WORK_GRAPH_SCHEMA_VERSION,
            repo_id: self.repo_id.clone(),
            revision: self.revision(),
            graph_commitment: self.graph_commitment.clone(),
            event_count: self.events.len(),
            node_count: self.nodes.len(),
            edge_count: self.edges.len(),
            evidence_count: self.evidence.len(),
            unfinished_count: unfinished.len(),
            blocked_count: unfinished
                .iter()
                .filter(|item| item.status == WorkStatus::Blocked)
                .count(),
        }
    }

    pub fn summary_json(&self) -> Result<String, WorkGraphError> {
        serde_json::to_string(&self.summary()).map_err(Into::into)
    }

    pub fn snapshot_json(&self, pretty: bool) -> Result<String, WorkGraphError> {
        #[derive(Serialize)]
        struct Snapshot<'a> {
            schema_version: u32,
            repo_id: &'a str,
            revision: u64,
            graph_commitment: &'a str,
            nodes: Vec<&'a WorkNode>,
            edges: Vec<&'a WorkEdge>,
            evidence: Vec<&'a EvidenceRef>,
        }
        let snapshot = Snapshot {
            schema_version: WORK_GRAPH_SCHEMA_VERSION,
            repo_id: &self.repo_id,
            revision: self.revision(),
            graph_commitment: &self.graph_commitment,
            nodes: self.nodes.values().collect(),
            edges: self.edges.values().collect(),
            evidence: self.evidence.values().collect(),
        };
        if pretty {
            serde_json::to_string_pretty(&snapshot).map_err(Into::into)
        } else {
            serde_json::to_string(&snapshot).map_err(Into::into)
        }
    }

    pub fn unfinished_work(&self) -> Vec<WorkItemView> {
        let mut items: Vec<WorkItemView> = self
            .nodes
            .values()
            .filter(|node| node.kind == NodeKind::Workstream && node.status.is_unfinished())
            .map(|node| self.work_item_view(node))
            .collect();
        items.sort_by(|a, b| {
            b.updated_at_ms
                .cmp(&a.updated_at_ms)
                .then_with(|| a.node_id.cmp(&b.node_id))
        });
        items
    }

    pub fn unfinished_json(&self, pretty: bool) -> Result<String, WorkGraphError> {
        if pretty {
            serde_json::to_string_pretty(&self.unfinished_work()).map_err(Into::into)
        } else {
            serde_json::to_string(&self.unfinished_work()).map_err(Into::into)
        }
    }

    pub fn resume(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
    ) -> Result<ResumeView, WorkGraphError> {
        let selected = if let Some(id) = workstream_id.filter(|id| !id.is_empty()) {
            let node = self
                .nodes
                .get(id)
                .ok_or_else(|| WorkGraphError::UnknownNode(id.to_string()))?;
            if node.kind != NodeKind::Workstream {
                return Err(WorkGraphError::InvalidInput(format!(
                    "resume target {id:?} is not a workstream"
                )));
            }
            self.work_item_view(node)
        } else {
            self.unfinished_work().into_iter().next().ok_or_else(|| {
                WorkGraphError::UnknownNode("no unfinished workstream".to_string())
            })?
        };

        let related = self.connected_node_ids(&selected.node_id, 2, 10_000);
        let task_labels = related
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .filter(|n| n.kind == NodeKind::Task)
            .map(|n| n.label.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let agents = related
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .filter(|n| n.kind == NodeKind::Agent)
            .map(|n| n.label.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let decisions = related
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .filter(|n| n.kind == NodeKind::Decision)
            .map(|n| n.label.clone())
            .collect();
        // Highest risk first: a successor reading a truncated view must see the
        // most consequential assertion, not whichever traversal reached first.
        // Unranked claims sort last rather than being treated as risk 0.
        let mut claims: Vec<ClaimView> = related
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .filter(|n| n.kind == NodeKind::Claim)
            .map(|n| ClaimView {
                label: n.label.clone(),
                trust: n.trust,
                claim_state: n
                    .attributes
                    .get("claim_state")
                    .and_then(|v| v.as_str())
                    .map(str::to_string),
                risk: n.attributes.get("risk").and_then(|v| v.as_f64()),
            })
            .collect();
        claims.sort_by(|a, b| {
            b.risk
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&a.risk.unwrap_or(f64::NEG_INFINITY))
                .then_with(|| a.label.cmp(&b.label))
        });

        let failures = related
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .filter(|n| n.kind == NodeKind::Failure)
            .map(|n| n.label.clone())
            .collect();
        let verification = related
            .iter()
            .filter_map(|id| self.nodes.get(id))
            .filter(|n| matches!(n.kind, NodeKind::Test | NodeKind::CiRun))
            .map(|n| n.label.clone())
            .collect();
        let evidence = self.context_evidence(&selected.node_id, max_evidence);

        Ok(ResumeView {
            repo_id: self.repo_id.clone(),
            graph_revision: self.revision(),
            graph_commitment: self.graph_commitment.clone(),
            selected_workstream: selected.clone(),
            task_labels,
            agents,
            decisions,
            claims,
            failures,
            verification,
            changed_paths: selected.changed_paths,
            commits: selected.commit_ids,
            evidence,
        })
    }

    pub fn resume_json(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
        pretty: bool,
    ) -> Result<String, WorkGraphError> {
        let view = self.resume(workstream_id, max_evidence)?;
        if pretty {
            serde_json::to_string_pretty(&view).map_err(Into::into)
        } else {
            serde_json::to_string(&view).map_err(Into::into)
        }
    }

    /// Derive the bounded, text-light Context/Trust integration scope from the
    /// exact Rust-owned resume view. Raw decision/failure prose and context
    /// bytes remain in their owning stores and never become graph payload here.
    pub fn context_scope(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
    ) -> Result<WorkScope, WorkGraphError> {
        let view = self.resume(workstream_id, max_evidence)?;
        WorkScope::from_resume(&view)
            .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))
    }

    pub fn context_scope_json(
        &self,
        workstream_id: Option<&str>,
        max_evidence: usize,
        pretty: bool,
    ) -> Result<String, WorkGraphError> {
        let scope = self.context_scope(workstream_id, max_evidence)?;
        if pretty {
            serde_json::to_string_pretty(&scope).map_err(Into::into)
        } else {
            serde_json::to_string(&scope).map_err(Into::into)
        }
    }

    /// Record a verified canonical Context Receipt as bounded graph evidence.
    /// The receipt must have been produced against this exact graph revision;
    /// recording it advances the graph, so a replay of the old receipt fails
    /// closed instead of silently attaching stale context twice.
    pub fn record_context_receipt(
        &mut self,
        receipt: ContextReceiptEnvelope,
        agent_id: String,
        session_id: String,
    ) -> Result<String, WorkGraphError> {
        if receipt.repository_id != self.repo_id {
            return Err(WorkGraphError::RepoMismatch {
                expected: self.repo_id.clone(),
                actual: receipt.repository_id,
            });
        }
        if receipt.graph_commitment != self.graph_commitment {
            return Err(WorkGraphError::IntegrityMismatch {
                expected: self.graph_commitment.clone(),
                actual: receipt.graph_commitment,
            });
        }
        let repo_node_id = stable_node_id(NodeKind::Repository, &self.repo_id, &self.repo_id);
        let current_repository_commitment = self
            .nodes
            .get(&repo_node_id)
            .and_then(|node| attr_string(&node.attributes, "head_sha"))
            .ok_or_else(|| {
                WorkGraphError::InvalidInput(
                    "cannot record a context receipt without a repository head commitment"
                        .to_string(),
                )
            })?;
        if receipt.repository_commitment != current_repository_commitment {
            return Err(WorkGraphError::IntegrityMismatch {
                expected: current_repository_commitment,
                actual: receipt.repository_commitment,
            });
        }
        let workstream = self
            .nodes
            .get(&receipt.work_scope_id)
            .ok_or_else(|| WorkGraphError::UnknownNode(receipt.work_scope_id.clone()))?;
        if workstream.kind != NodeKind::Workstream {
            return Err(WorkGraphError::InvalidInput(
                "context receipt work scope is not a workstream".to_string(),
            ));
        }
        let event = context_receipt_event(&self.repo_id, receipt, agent_id, session_id)?;
        self.apply_event(event)
    }

    pub fn record_context_receipt_json(
        &mut self,
        receipt_json: &str,
        agent_id: &str,
        session_id: &str,
    ) -> Result<String, WorkGraphError> {
        let receipt = ContextReceiptEnvelope::from_json_verified(receipt_json)
            .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))?;
        self.record_context_receipt(receipt, agent_id.to_string(), session_id.to_string())
    }

    /// Record a provenance-bearing memory without treating its text or trust
    /// label as verified work truth. Admissibility is computed by the canonical
    /// memory contract and persisted as inspectable graph data.
    pub fn record_memory(
        &mut self,
        memory: MemoryRecord,
        now_ms: i64,
        superseded_ids: BTreeSet<String>,
    ) -> Result<String, WorkGraphError> {
        if memory.repository_id != self.repo_id {
            return Err(WorkGraphError::RepoMismatch {
                expected: self.repo_id.clone(),
                actual: memory.repository_id,
            });
        }
        if !memory.workstream_id.is_empty() {
            let node = self
                .nodes
                .get(&memory.workstream_id)
                .ok_or_else(|| WorkGraphError::UnknownNode(memory.workstream_id.clone()))?;
            if node.kind != NodeKind::Workstream {
                return Err(WorkGraphError::InvalidInput(
                    "memory scope is not a workstream".to_string(),
                ));
            }
        }
        if !memory.task_id.is_empty() {
            let node = self
                .nodes
                .get(&memory.task_id)
                .ok_or_else(|| WorkGraphError::UnknownNode(memory.task_id.clone()))?;
            if node.kind != NodeKind::Task {
                return Err(WorkGraphError::InvalidInput(
                    "memory task scope is not a task".to_string(),
                ));
            }
        }
        let admissibility = memory.admissibility_in_set(now_ms, &superseded_ids);
        let event = memory_record_event(&self.repo_id, memory, admissibility, now_ms)?;
        self.apply_event(event)
    }

    pub fn record_memory_json(
        &mut self,
        memory_json: &str,
        now_ms: i64,
        superseded_ids_json: &str,
    ) -> Result<String, WorkGraphError> {
        let memory = MemoryRecord::from_json_verified(memory_json)
            .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))?;
        let superseded: BTreeSet<String> = if superseded_ids_json.trim().is_empty() {
            BTreeSet::new()
        } else {
            serde_json::from_str(superseded_ids_json)?
        };
        self.record_memory(memory, now_ms, superseded)
    }

    /// Append one verified route → execution → verification chain as a single
    /// atomic Work Event. The contracts are parsed and commitment-checked by
    /// the caller-facing JSON method below; this method additionally checks
    /// their cross-contract identity and current repository version.
    pub fn record_execution_chain(
        &mut self,
        route: RoutingDecision,
        outcome: ModelExecutionOutcome,
        verification: VerificationRecord,
        invalidated_commitments: BTreeSet<String>,
    ) -> Result<String, WorkGraphError> {
        if route.repository_id != self.repo_id
            || outcome.repository_id != self.repo_id
            || verification.repository_id != self.repo_id
        {
            let actual = if route.repository_id != self.repo_id {
                route.repository_id.clone()
            } else if outcome.repository_id != self.repo_id {
                outcome.repository_id.clone()
            } else {
                verification.repository_id.clone()
            };
            return Err(WorkGraphError::RepoMismatch {
                expected: self.repo_id.clone(),
                actual,
            });
        }
        if !outcome.verify_route(&route) {
            return Err(WorkGraphError::InvalidInput(
                "model execution outcome does not match its routing decision".to_string(),
            ));
        }
        if verification.subject_id != outcome.outcome_id
            || verification.subject_commitment != outcome.outcome_commitment
            || verification.observed_at_ms < outcome.completed_at_ms
        {
            return Err(WorkGraphError::InvalidInput(
                "verification record does not bind the exact execution outcome".to_string(),
            ));
        }
        let workstream = self
            .nodes
            .get(&route.workstream_id)
            .ok_or_else(|| WorkGraphError::UnknownNode(route.workstream_id.clone()))?;
        if workstream.kind != NodeKind::Workstream {
            return Err(WorkGraphError::InvalidInput(
                "execution chain target is not a workstream".to_string(),
            ));
        }
        let task_matches = self
            .connected_node_ids(&route.workstream_id, 1, MAX_SCOPE_ITEMS)
            .contains(&route.task_id);
        if !task_matches {
            return Err(WorkGraphError::InvalidInput(
                "routing decision task is not linked to its workstream".to_string(),
            ));
        }
        let repo_node_id = stable_node_id(NodeKind::Repository, &self.repo_id, &self.repo_id);
        let current_repository_commitment = self
            .nodes
            .get(&repo_node_id)
            .and_then(|node| attr_string(&node.attributes, "head_sha"))
            .ok_or_else(|| {
                WorkGraphError::InvalidInput(
                    "cannot record exact-version verification without a repository head commitment"
                        .to_string(),
                )
            })?;
        let event = execution_chain_event(
            &self.repo_id,
            route,
            outcome,
            verification,
            &current_repository_commitment,
            &invalidated_commitments,
        )?;
        self.apply_event(event)
    }

    pub fn record_execution_chain_json(
        &mut self,
        route_json: &str,
        outcome_json: &str,
        verification_json: &str,
        invalidated_commitments_json: &str,
    ) -> Result<String, WorkGraphError> {
        let route = RoutingDecision::from_json_verified(route_json)
            .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))?;
        let outcome = ModelExecutionOutcome::from_json_verified(outcome_json)
            .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))?;
        let verification = VerificationRecord::from_json_verified(verification_json)
            .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))?;
        let invalidated: BTreeSet<String> = if invalidated_commitments_json.trim().is_empty() {
            BTreeSet::new()
        } else {
            serde_json::from_str(invalidated_commitments_json)?
        };
        self.record_execution_chain(route, outcome, verification, invalidated)
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn resolve_continuation_manifest(
        &self,
        node_ids: &BTreeSet<String>,
        evidence_ids: &BTreeSet<String>,
        context_receipt_commitments: Vec<String>,
        routing_commitments: Vec<String>,
        execution_outcome_commitments: Vec<String>,
        verification_commitments: Vec<String>,
        memory_commitments: Vec<String>,
        recovery_handle_ids: Vec<String>,
    ) -> Result<
        (
            Vec<String>,
            Vec<String>,
            Vec<String>,
            Vec<String>,
            Vec<String>,
            Vec<String>,
        ),
        WorkGraphError,
    > {
        let mut receipts = BTreeSet::new();
        let mut routes = BTreeSet::new();
        let mut outcomes = BTreeSet::new();
        let mut verifications = BTreeSet::new();
        let mut memories = BTreeSet::new();

        for evidence_id in evidence_ids {
            let evidence = self.evidence.get(evidence_id).ok_or_else(|| {
                WorkGraphError::InvalidInput(format!(
                    "continuation evidence {evidence_id:?} is absent from the graph"
                ))
            })?;
            if evidence.digest.is_empty() {
                continue;
            }
            if evidence.source_ref.starts_with("context-receipt:") {
                receipts.insert(evidence.digest.clone());
            } else if evidence.source_ref.starts_with("routing:") {
                routes.insert(evidence.digest.clone());
            } else if evidence.source_ref.starts_with("execution:") {
                outcomes.insert(evidence.digest.clone());
            } else if evidence.source_ref.starts_with("verification:") {
                verifications.insert(evidence.digest.clone());
            } else if evidence.source_ref.starts_with("memory:") {
                memories.insert(evidence.digest.clone());
            }
        }

        let mut recovery_handles = BTreeSet::new();
        for node_id in node_ids {
            let Some(node) = self.nodes.get(node_id) else {
                continue;
            };
            if let Some(Value::Array(values)) = node.attributes.get("recovery_handles") {
                recovery_handles.extend(
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .filter(|value| !value.is_empty())
                        .map(ToOwned::to_owned),
                );
            }
            if let Some(Value::String(value)) = node.attributes.get("recovery_handle") {
                if !value.is_empty() {
                    recovery_handles.insert(value.clone());
                }
            }
        }

        fn resolve(
            field: &'static str,
            supplied: Vec<String>,
            discovered: BTreeSet<String>,
        ) -> Result<Vec<String>, WorkGraphError> {
            if supplied.is_empty() {
                return Ok(discovered.into_iter().collect());
            }
            if let Some(unknown) = supplied
                .iter()
                .find(|value| !discovered.contains(value.as_str()))
            {
                return Err(WorkGraphError::InvalidInput(format!(
                    "{field} contains a value not evidenced by this workstream: {unknown:?}"
                )));
            }
            Ok(supplied)
        }

        Ok((
            resolve(
                "context_receipt_commitments",
                context_receipt_commitments,
                receipts,
            )?,
            resolve("routing_commitments", routing_commitments, routes)?,
            resolve(
                "execution_outcome_commitments",
                execution_outcome_commitments,
                outcomes,
            )?,
            resolve(
                "verification_commitments",
                verification_commitments,
                verifications,
            )?,
            resolve("memory_commitments", memory_commitments, memories)?,
            resolve("recovery_handle_ids", recovery_handle_ids, recovery_handles)?,
        ))
    }

    /// Build a graph-bound continuation proof only after the ordinary handoff
    /// receipt is verified against this exact materialization.
    #[allow(clippy::too_many_arguments)]
    pub fn continuation_proof(
        &self,
        handoff: &HandoffReceipt,
        context_receipt_commitments: Vec<String>,
        routing_commitments: Vec<String>,
        execution_outcome_commitments: Vec<String>,
        verification_commitments: Vec<String>,
        memory_commitments: Vec<String>,
        outstanding_work_refs: Vec<String>,
        recovery_handle_ids: Vec<String>,
        created_at_ms: i64,
    ) -> Result<WorkContinuationProof, WorkGraphError> {
        if !self.verify_handoff_receipt_against_graph(handoff)? {
            return Err(WorkGraphError::IntegrityMismatch {
                expected: self.graph_commitment.clone(),
                actual: handoff.graph_commitment.clone(),
            });
        }
        let node_ids = handoff.node_ids.iter().cloned().collect();
        let evidence_ids = handoff.evidence_ids.iter().cloned().collect();
        let (
            context_receipt_commitments,
            routing_commitments,
            execution_outcome_commitments,
            verification_commitments,
            memory_commitments,
            recovery_handle_ids,
        ) = self.resolve_continuation_manifest(
            &node_ids,
            &evidence_ids,
            context_receipt_commitments,
            routing_commitments,
            execution_outcome_commitments,
            verification_commitments,
            memory_commitments,
            recovery_handle_ids,
        )?;
        WorkContinuationProof::new(
            self.repo_id.clone(),
            self.revision(),
            self.graph_commitment.clone(),
            handoff.workstream_id.clone(),
            handoff.from_agent.clone(),
            handoff.to_agent.clone(),
            handoff.payload_commitment.clone(),
            context_receipt_commitments,
            routing_commitments,
            execution_outcome_commitments,
            verification_commitments,
            memory_commitments,
            outstanding_work_refs,
            recovery_handle_ids,
            created_at_ms,
        )
        .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))
    }

    pub fn continuation_proof_json(
        &self,
        handoff_json: &str,
        manifest_json: &str,
    ) -> Result<String, WorkGraphError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Manifest {
            #[serde(default)]
            context_receipt_commitments: Vec<String>,
            #[serde(default)]
            routing_commitments: Vec<String>,
            #[serde(default)]
            execution_outcome_commitments: Vec<String>,
            #[serde(default)]
            verification_commitments: Vec<String>,
            #[serde(default)]
            memory_commitments: Vec<String>,
            #[serde(default)]
            outstanding_work_refs: Vec<String>,
            #[serde(default)]
            recovery_handle_ids: Vec<String>,
            created_at_ms: i64,
        }
        let handoff: HandoffReceipt = serde_json::from_str(handoff_json)?;
        let manifest: Manifest = serde_json::from_str(manifest_json)?;
        self.continuation_proof(
            &handoff,
            manifest.context_receipt_commitments,
            manifest.routing_commitments,
            manifest.execution_outcome_commitments,
            manifest.verification_commitments,
            manifest.memory_commitments,
            manifest.outstanding_work_refs,
            manifest.recovery_handle_ids,
            manifest.created_at_ms,
        )?
        .to_json()
        .map_err(|error| WorkGraphError::Serialization(error.to_string()))
    }

    /// Reconstruct a bounded continuation proof when an agent was interrupted
    /// before it could seal a handoff.  The empty source/handoff fields are
    /// deliberate evidence: previous-agent identity and intent are not
    /// invented.  Resumable references still have to come from a caller-owned
    /// manifest and the proof remains bound to this exact graph materialization.
    #[allow(clippy::too_many_arguments)]
    pub fn reconstructed_continuation_proof(
        &self,
        workstream_id: &str,
        to_agent: &str,
        context_receipt_commitments: Vec<String>,
        routing_commitments: Vec<String>,
        execution_outcome_commitments: Vec<String>,
        verification_commitments: Vec<String>,
        memory_commitments: Vec<String>,
        mut outstanding_work_refs: Vec<String>,
        recovery_handle_ids: Vec<String>,
        created_at_ms: i64,
    ) -> Result<WorkContinuationProof, WorkGraphError> {
        validate_text("to_agent", to_agent, MAX_ID_LEN, false)?;
        let workstream = self
            .nodes
            .get(workstream_id)
            .ok_or_else(|| WorkGraphError::UnknownNode(workstream_id.to_string()))?;
        if workstream.kind != NodeKind::Workstream {
            return Err(WorkGraphError::InvalidInput(format!(
                "continuation target {workstream_id:?} is not a workstream"
            )));
        }
        if !workstream.status.is_unfinished() {
            return Err(WorkGraphError::InvalidInput(
                "cannot reconstruct continuation for finished work".to_string(),
            ));
        }
        let related = self.connected_node_ids(workstream_id, 2, 20_000);
        let mut related_evidence_ids = BTreeSet::new();
        for node_id in &related {
            if let Some(node) = self.nodes.get(node_id) {
                related_evidence_ids.extend(node.evidence_ids.iter().cloned());
            }
        }
        for edge in self.edges.values() {
            if related.contains(&edge.from_node) && related.contains(&edge.to_node) {
                related_evidence_ids.extend(edge.evidence_ids.iter().cloned());
            }
        }
        let (
            context_receipt_commitments,
            routing_commitments,
            execution_outcome_commitments,
            verification_commitments,
            memory_commitments,
            recovery_handle_ids,
        ) = self.resolve_continuation_manifest(
            &related,
            &related_evidence_ids,
            context_receipt_commitments,
            routing_commitments,
            execution_outcome_commitments,
            verification_commitments,
            memory_commitments,
            recovery_handle_ids,
        )?;
        // Preserve the uncertainty that an explicit handoff would otherwise
        // resolve.  This is a bounded machine-readable fact, not invented prose.
        outstanding_work_refs.push("unknown:previous-agent-intent".to_string());
        WorkContinuationProof::new(
            self.repo_id.clone(),
            self.revision(),
            self.graph_commitment.clone(),
            workstream_id.to_string(),
            String::new(),
            to_agent.to_string(),
            String::new(),
            context_receipt_commitments,
            routing_commitments,
            execution_outcome_commitments,
            verification_commitments,
            memory_commitments,
            outstanding_work_refs,
            recovery_handle_ids,
            created_at_ms,
        )
        .map_err(|error| WorkGraphError::InvalidInput(error.to_string()))
    }

    pub fn reconstructed_continuation_proof_json(
        &self,
        workstream_id: &str,
        to_agent: &str,
        manifest_json: &str,
    ) -> Result<String, WorkGraphError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Manifest {
            #[serde(default)]
            context_receipt_commitments: Vec<String>,
            #[serde(default)]
            routing_commitments: Vec<String>,
            #[serde(default)]
            execution_outcome_commitments: Vec<String>,
            #[serde(default)]
            verification_commitments: Vec<String>,
            #[serde(default)]
            memory_commitments: Vec<String>,
            #[serde(default)]
            outstanding_work_refs: Vec<String>,
            #[serde(default)]
            recovery_handle_ids: Vec<String>,
            created_at_ms: i64,
        }
        let manifest: Manifest = serde_json::from_str(manifest_json)?;
        self.reconstructed_continuation_proof(
            workstream_id,
            to_agent,
            manifest.context_receipt_commitments,
            manifest.routing_commitments,
            manifest.execution_outcome_commitments,
            manifest.verification_commitments,
            manifest.memory_commitments,
            manifest.outstanding_work_refs,
            manifest.recovery_handle_ids,
            manifest.created_at_ms,
        )?
        .to_json()
        .map_err(|error| WorkGraphError::Serialization(error.to_string()))
    }

    pub fn coordination_report(&self, now_ms: i64) -> CoordinationReport {
        #[derive(Clone)]
        struct Lease {
            id: String,
            agent: String,
            task: String,
            expires: i64,
            paths: Vec<String>,
            symbols: Vec<String>,
        }
        let mut leases = Vec::new();
        for node in self
            .nodes
            .values()
            .filter(|n| n.kind == NodeKind::WorkLease)
        {
            let expires = attr_i64(&node.attributes, "expires_at_ms").unwrap_or(0);
            if expires <= now_ms {
                continue;
            }
            let agent = attr_string(&node.attributes, "agent_id").unwrap_or_default();
            if agent.is_empty() {
                continue;
            }
            leases.push(Lease {
                id: node.node_id.clone(),
                agent,
                task: attr_string(&node.attributes, "task_id").unwrap_or_default(),
                expires,
                paths: attr_strings(&node.attributes, "scope_paths"),
                symbols: attr_strings(&node.attributes, "scope_symbols"),
            });
        }
        leases.sort_by(|a, b| a.id.cmp(&b.id));
        let scopes: Vec<CoordinationScope<'_>> = leases
            .iter()
            .map(|lease| CoordinationScope {
                agent: &lease.agent,
                paths: &lease.paths,
                symbols: &lease.symbols,
            })
            .collect();
        let mut conflicts = Vec::new();
        for (i, j) in candidate_pairs(&scopes) {
            let a = &leases[i];
            let b = &leases[j];
            // Candidate generation is only a performance filter. Keep the
            // pre-existing exact overlap functions authoritative so conflict
            // semantics cannot drift with the index.
            if a.agent == b.agent || a.expires <= now_ms || b.expires <= now_ms {
                continue;
            }
            let overlapping_paths = overlap_paths(&a.paths, &b.paths);
            let overlapping_symbols = overlap_exact(&a.symbols, &b.symbols);
            if overlapping_paths.is_empty() && overlapping_symbols.is_empty() {
                continue;
            }
            conflicts.push(CoordinationConflict {
                lease_a: a.id.clone(),
                lease_b: b.id.clone(),
                agent_a: a.agent.clone(),
                agent_b: b.agent.clone(),
                task_a: a.task.clone(),
                task_b: b.task.clone(),
                overlapping_paths,
                overlapping_symbols,
                reason: "active advisory work scopes overlap".to_string(),
            });
        }
        CoordinationReport {
            as_of_ms: now_ms,
            active_leases: leases.len(),
            conflicts,
        }
    }

    pub fn coordination_json(&self, now_ms: i64, pretty: bool) -> Result<String, WorkGraphError> {
        let report = self.coordination_report(now_ms);
        if pretty {
            serde_json::to_string_pretty(&report).map_err(Into::into)
        } else {
            serde_json::to_string(&report).map_err(Into::into)
        }
    }

    pub fn context_evidence(&self, workstream_id: &str, max_evidence: usize) -> Vec<EvidenceRef> {
        let related = self.connected_node_ids(workstream_id, 2, 10_000);
        let mut ids = BTreeSet::new();
        for id in &related {
            if let Some(node) = self.nodes.get(id) {
                ids.extend(node.evidence_ids.iter().cloned());
            }
        }
        for edge in self.edges.values() {
            // Cross-boundary edges may belong to another workstream through a
            // shared agent/model hub. Only evidence fully contained in this
            // task-local subgraph is eligible for context selection.
            if related.contains(&edge.from_node) && related.contains(&edge.to_node) {
                ids.extend(edge.evidence_ids.iter().cloned());
            }
        }
        let mut evidence: Vec<EvidenceRef> = ids
            .iter()
            .filter_map(|id| self.evidence.get(id).cloned())
            .collect();
        evidence.sort_by(|a, b| {
            b.trust
                .cmp(&a.trust)
                .then_with(|| b.observed_at_ms.cmp(&a.observed_at_ms))
                .then_with(|| a.evidence_id.cmp(&b.evidence_id))
        });
        evidence.truncate(max_evidence.min(10_000));
        evidence
    }

    pub fn handoff_receipt(
        &self,
        workstream_id: &str,
        from_agent: &str,
        to_agent: &str,
        generated_at_ms: i64,
    ) -> Result<HandoffReceipt, WorkGraphError> {
        validate_text("from_agent", from_agent, MAX_ID_LEN, false)?;
        validate_text("to_agent", to_agent, MAX_ID_LEN, false)?;
        let node = self
            .nodes
            .get(workstream_id)
            .ok_or_else(|| WorkGraphError::UnknownNode(workstream_id.to_string()))?;
        if node.kind != NodeKind::Workstream {
            return Err(WorkGraphError::InvalidInput(format!(
                "handoff target {workstream_id:?} is not a workstream"
            )));
        }
        let related = self.connected_node_ids(workstream_id, 2, 20_000);
        let node_ids: Vec<String> = related.iter().cloned().collect();
        let edge_ids: Vec<String> = self
            .edges
            .values()
            .filter(|edge| related.contains(&edge.from_node) && related.contains(&edge.to_node))
            .map(|edge| edge.edge_id.clone())
            .collect();
        let mut evidence_ids = BTreeSet::new();
        for node_id in &node_ids {
            if let Some(n) = self.nodes.get(node_id) {
                evidence_ids.extend(n.evidence_ids.iter().cloned());
            }
        }
        for edge_id in &edge_ids {
            if let Some(e) = self.edges.get(edge_id) {
                evidence_ids.extend(e.evidence_ids.iter().cloned());
            }
        }
        let mut receipt = HandoffReceipt {
            schema_version: WORK_GRAPH_SCHEMA_VERSION,
            repo_id: self.repo_id.clone(),
            graph_revision: self.revision(),
            graph_commitment: self.graph_commitment.clone(),
            workstream_id: workstream_id.to_string(),
            from_agent: from_agent.to_string(),
            to_agent: to_agent.to_string(),
            generated_at_ms,
            node_ids,
            edge_ids,
            evidence_ids: evidence_ids.into_iter().collect(),
            payload_commitment: String::new(),
        };
        receipt.payload_commitment = handoff_commitment(&receipt)?;
        Ok(receipt)
    }

    pub fn verify_handoff_receipt(receipt: &HandoffReceipt) -> Result<bool, WorkGraphError> {
        Ok(!receipt.payload_commitment.is_empty()
            && handoff_commitment(receipt)? == receipt.payload_commitment)
    }

    /// Verify a handoff against this exact materialized graph snapshot.
    ///
    /// The static verifier proves only that the receipt is internally
    /// tamper-evident. This method additionally proves that its repository,
    /// revision, graph commitment, workstream and referenced graph objects all
    /// exist in the graph being asked to accept the handoff.
    pub fn verify_handoff_receipt_against_graph(
        &self,
        receipt: &HandoffReceipt,
    ) -> Result<bool, WorkGraphError> {
        if !Self::verify_handoff_receipt(receipt)?
            || receipt.schema_version != WORK_GRAPH_SCHEMA_VERSION
            || receipt.repo_id != self.repo_id
            || receipt.graph_revision != self.revision()
            || receipt.graph_commitment != self.graph_commitment
        {
            return Ok(false);
        }
        let Some(workstream) = self.nodes.get(&receipt.workstream_id) else {
            return Ok(false);
        };
        if workstream.kind != NodeKind::Workstream
            || receipt
                .node_ids
                .binary_search(&receipt.workstream_id)
                .is_err()
        {
            return Ok(false);
        }
        if receipt.node_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || receipt.edge_ids.windows(2).any(|pair| pair[0] >= pair[1])
            || receipt
                .evidence_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Ok(false);
        }
        if receipt
            .node_ids
            .iter()
            .any(|id| !self.nodes.contains_key(id))
            || receipt
                .edge_ids
                .iter()
                .any(|id| !self.edges.contains_key(id))
            || receipt
                .evidence_ids
                .iter()
                .any(|id| !self.evidence.contains_key(id))
        {
            return Ok(false);
        }
        for edge_id in &receipt.edge_ids {
            let edge = &self.edges[edge_id];
            if receipt.node_ids.binary_search(&edge.from_node).is_err()
                || receipt.node_ids.binary_search(&edge.to_node).is_err()
            {
                return Ok(false);
            }
        }
        let mut required_evidence = BTreeSet::new();
        for node_id in &receipt.node_ids {
            required_evidence.extend(self.nodes[node_id].evidence_ids.iter().cloned());
        }
        for edge_id in &receipt.edge_ids {
            required_evidence.extend(self.edges[edge_id].evidence_ids.iter().cloned());
        }
        Ok(required_evidence
            .iter()
            .all(|id| receipt.evidence_ids.binary_search(id).is_ok()))
    }

    pub fn handoff_json(
        &self,
        workstream_id: &str,
        from_agent: &str,
        to_agent: &str,
        generated_at_ms: i64,
        pretty: bool,
    ) -> Result<String, WorkGraphError> {
        let receipt = self.handoff_receipt(workstream_id, from_agent, to_agent, generated_at_ms)?;
        if pretty {
            serde_json::to_string_pretty(&receipt).map_err(Into::into)
        } else {
            serde_json::to_string(&receipt).map_err(Into::into)
        }
    }

    fn rebuild(&mut self) -> Result<(), WorkGraphError> {
        self.events.sort_by(|a, b| {
            a.observed_at_ms
                .cmp(&b.observed_at_ms)
                .then_with(|| a.event_id.cmp(&b.event_id))
        });
        self.event_ids.clear();
        self.last_passive_snapshot = None;
        self.nodes.clear();
        self.edges.clear();
        self.evidence.clear();
        self.adjacency.clear();
        let events = self.events.clone();
        for event in &events {
            if !self.event_ids.insert(event.event_id.clone()) {
                return Err(WorkGraphError::InvalidInput(format!(
                    "duplicate event id in work graph: {}",
                    event.event_id
                )));
            }
            self.apply_materialized(event)?;
        }
        if self.nodes.len() > MAX_NODES {
            return Err(WorkGraphError::LimitExceeded(format!(
                "node count exceeds {MAX_NODES}"
            )));
        }
        if self.edges.len() > MAX_EDGES {
            return Err(WorkGraphError::LimitExceeded(format!(
                "edge count exceeds {MAX_EDGES}"
            )));
        }
        if self.evidence.len() > MAX_EVIDENCE {
            return Err(WorkGraphError::LimitExceeded(format!(
                "evidence count exceeds {MAX_EVIDENCE}"
            )));
        }
        self.refresh_last_passive_snapshot();
        self.refresh_commitment()
    }

    fn refresh_last_passive_snapshot(&mut self) {
        self.last_passive_snapshot = self.events.last().and_then(|event| {
            passive_snapshot_group(event).map(|source_ref| (source_ref, event.event_id.clone()))
        });
    }

    fn apply_materialized(&mut self, event: &WorkEvent) -> Result<(), WorkGraphError> {
        // Materialize in semantic phases rather than trusting operation order.
        // Raw event producers may serialize SetStatus before UpsertNode, but
        // references within one event must still resolve deterministically.
        for operation in &event.operations {
            if let WorkOperation::AddEvidence { evidence } = operation {
                let mut incoming = evidence.clone();
                if incoming.evidence_id.is_empty() {
                    incoming.evidence_id = stable_evidence_id(&incoming)?;
                }
                match self.evidence.get_mut(&incoming.evidence_id) {
                    Some(existing) => {
                        let may_overwrite = incoming.trust >= existing.trust;
                        existing.trust = existing.trust.max(incoming.trust);
                        existing.observed_at_ms =
                            existing.observed_at_ms.max(incoming.observed_at_ms);
                        if may_overwrite {
                            for (key, value) in incoming.attributes {
                                existing.attributes.insert(key, value);
                            }
                        }
                    }
                    None => {
                        self.evidence.insert(incoming.evidence_id.clone(), incoming);
                    }
                }
            }
        }

        for operation in &event.operations {
            if let WorkOperation::UpsertNode { node } = operation {
                let mut incoming = node.clone();
                incoming.updated_at_ms = incoming.updated_at_ms.max(event.observed_at_ms);
                match self.nodes.get_mut(&incoming.node_id) {
                    Some(existing) => {
                        let may_overwrite = incoming.trust >= existing.trust;
                        if may_overwrite && !incoming.label.is_empty() {
                            existing.label = incoming.label;
                        }
                        if incoming.status != WorkStatus::Unknown
                            && incoming.status_trust >= existing.status_trust
                        {
                            existing.status = incoming.status;
                            existing.status_trust = incoming.status_trust;
                        }
                        existing.trust = existing.trust.max(incoming.trust);
                        existing.updated_at_ms = existing.updated_at_ms.max(incoming.updated_at_ms);
                        existing.evidence_ids.extend(incoming.evidence_ids);
                        if may_overwrite {
                            for (key, value) in incoming.attributes {
                                existing.attributes.insert(key, value);
                            }
                        }
                    }
                    None => {
                        self.nodes.insert(incoming.node_id.clone(), incoming);
                    }
                }
            }
        }

        for operation in &event.operations {
            if let WorkOperation::UpsertEdge { edge } = operation {
                let mut incoming = edge.clone();
                incoming.updated_at_ms = incoming.updated_at_ms.max(event.observed_at_ms);
                let from = incoming.from_node.clone();
                let to = incoming.to_node.clone();
                match self.edges.get_mut(&incoming.edge_id) {
                    Some(existing) => {
                        let may_overwrite = incoming.trust >= existing.trust;
                        existing.trust = existing.trust.max(incoming.trust);
                        existing.updated_at_ms = existing.updated_at_ms.max(incoming.updated_at_ms);
                        existing.evidence_ids.extend(incoming.evidence_ids);
                        if may_overwrite {
                            for (key, value) in incoming.attributes {
                                existing.attributes.insert(key, value);
                            }
                        }
                    }
                    None => {
                        self.edges.insert(incoming.edge_id.clone(), incoming);
                    }
                }
                self.adjacency
                    .entry(from.clone())
                    .or_default()
                    .insert(to.clone());
                self.adjacency.entry(to).or_default().insert(from);
            }
        }

        for operation in &event.operations {
            match operation {
                WorkOperation::SetStatus {
                    node_id,
                    status,
                    trust,
                    reason,
                    evidence_ids,
                } => {
                    let node = self
                        .nodes
                        .get_mut(node_id)
                        .ok_or_else(|| WorkGraphError::UnknownNode(node_id.clone()))?;
                    node.trust = node.trust.max(*trust);
                    node.updated_at_ms = node.updated_at_ms.max(event.observed_at_ms);
                    node.evidence_ids.extend(evidence_ids.iter().cloned());
                    if *trust >= node.status_trust {
                        node.status = *status;
                        node.status_trust = *trust;
                        if !reason.is_empty() {
                            node.attributes
                                .insert("status_reason".to_string(), Value::String(reason.clone()));
                        }
                    }
                }
                WorkOperation::AttachEvidence {
                    node_id,
                    evidence_ids,
                } => {
                    let node = self
                        .nodes
                        .get_mut(node_id)
                        .ok_or_else(|| WorkGraphError::UnknownNode(node_id.clone()))?;
                    node.evidence_ids.extend(evidence_ids.iter().cloned());
                    node.updated_at_ms = node.updated_at_ms.max(event.observed_at_ms);
                }
                WorkOperation::AddEvidence { .. }
                | WorkOperation::UpsertNode { .. }
                | WorkOperation::UpsertEdge { .. } => {}
            }
        }
        Ok(())
    }

    fn validate_event_references_and_capacity(
        &self,
        event: &WorkEvent,
    ) -> Result<(), WorkGraphError> {
        let mut node_ids: BTreeSet<String> = self.nodes.keys().cloned().collect();
        let mut evidence_trust: BTreeMap<String, TrustLevel> = self
            .evidence
            .iter()
            .map(|(id, evidence)| (id.clone(), evidence.trust))
            .collect();
        let mut new_nodes = BTreeSet::new();
        let mut new_edges = BTreeSet::new();
        let mut new_evidence = BTreeSet::new();

        for operation in &event.operations {
            match operation {
                WorkOperation::AddEvidence { evidence } => {
                    let id = if evidence.evidence_id.is_empty() {
                        stable_evidence_id(evidence)?
                    } else {
                        evidence.evidence_id.clone()
                    };
                    evidence_trust
                        .entry(id.clone())
                        .and_modify(|trust| *trust = (*trust).max(evidence.trust))
                        .or_insert(evidence.trust);
                    if !self.evidence.contains_key(&id) {
                        new_evidence.insert(id);
                    }
                }
                WorkOperation::UpsertNode { node } => {
                    node_ids.insert(node.node_id.clone());
                    if !self.nodes.contains_key(&node.node_id) {
                        new_nodes.insert(node.node_id.clone());
                    }
                }
                WorkOperation::UpsertEdge { edge } => {
                    if !self.edges.contains_key(&edge.edge_id) {
                        new_edges.insert(edge.edge_id.clone());
                    }
                }
                WorkOperation::SetStatus { .. } | WorkOperation::AttachEvidence { .. } => {}
            }
        }

        let projected_nodes = self.nodes.len().saturating_add(new_nodes.len());
        let projected_edges = self.edges.len().saturating_add(new_edges.len());
        let projected_evidence = self.evidence.len().saturating_add(new_evidence.len());
        if projected_nodes > MAX_NODES {
            return Err(WorkGraphError::LimitExceeded(format!(
                "node count exceeds {MAX_NODES}"
            )));
        }
        if projected_edges > MAX_EDGES {
            return Err(WorkGraphError::LimitExceeded(format!(
                "edge count exceeds {MAX_EDGES}"
            )));
        }
        if projected_evidence > MAX_EVIDENCE {
            return Err(WorkGraphError::LimitExceeded(format!(
                "evidence count exceeds {MAX_EVIDENCE}"
            )));
        }

        let check_evidence = |ids: &BTreeSet<String>| -> Result<TrustLevel, WorkGraphError> {
            let mut strongest = TrustLevel::Untrusted;
            for id in ids {
                let Some(trust) = evidence_trust.get(id) else {
                    return Err(WorkGraphError::InvalidInput(format!(
                        "event references unknown evidence id {id:?}"
                    )));
                };
                strongest = strongest.max(*trust);
            }
            Ok(strongest)
        };

        for operation in &event.operations {
            match operation {
                WorkOperation::AddEvidence { .. } => {}
                WorkOperation::UpsertNode { node } => {
                    let strongest = check_evidence(&node.evidence_ids)?;
                    if node.status != WorkStatus::Unknown && node.status_trust > strongest {
                        return Err(WorkGraphError::InvalidInput(format!(
                            "node {} status trust {:?} exceeds supporting evidence trust {:?}",
                            node.node_id, node.status_trust, strongest
                        )));
                    }
                }
                WorkOperation::UpsertEdge { edge } => {
                    if !node_ids.contains(&edge.from_node) || !node_ids.contains(&edge.to_node) {
                        return Err(WorkGraphError::InvalidInput(format!(
                            "edge {} references an unknown node",
                            edge.edge_id
                        )));
                    }
                    check_evidence(&edge.evidence_ids)?;
                }
                WorkOperation::SetStatus {
                    node_id,
                    status,
                    trust,
                    evidence_ids: ids,
                    ..
                } => {
                    if !node_ids.contains(node_id) {
                        return Err(WorkGraphError::UnknownNode(node_id.clone()));
                    }
                    if *status != WorkStatus::Unknown && ids.is_empty() {
                        return Err(WorkGraphError::InvalidInput(format!(
                            "status update for {node_id} has no supporting evidence"
                        )));
                    }
                    let strongest = check_evidence(ids)?;
                    if *trust > strongest {
                        return Err(WorkGraphError::InvalidInput(format!(
                            "status trust {:?} for {node_id} exceeds supporting evidence trust {:?}",
                            trust, strongest
                        )));
                    }
                }
                WorkOperation::AttachEvidence {
                    node_id,
                    evidence_ids: ids,
                } => {
                    if !node_ids.contains(node_id) {
                        return Err(WorkGraphError::UnknownNode(node_id.clone()));
                    }
                    check_evidence(ids)?;
                }
            }
        }
        Ok(())
    }

    fn commitment_prefix_hasher(repo_id: &str) -> Result<Sha256, WorkGraphError> {
        let mut hasher = Sha256::new();
        hasher.update(b"{\"schema_version\":");
        hasher.update(WORK_GRAPH_SCHEMA_VERSION.to_string().as_bytes());
        hasher.update(b",\"repo_id\":");
        hasher.update(serde_json::to_vec(repo_id)?);
        hasher.update(b",\"events\":[");
        Ok(hasher)
    }

    fn finalize_commitment(hasher: &Sha256) -> String {
        let mut final_hasher = hasher.clone();
        final_hasher.update(b"]}");
        format!("{:x}", final_hasher.finalize())
    }

    fn refresh_commitment(&mut self) -> Result<(), WorkGraphError> {
        let mut hasher = Self::commitment_prefix_hasher(&self.repo_id)?;
        for (index, event) in self.events.iter().enumerate() {
            if index > 0 {
                hasher.update(b",");
            }
            hasher.update(serde_json::to_vec(event)?);
        }
        self.graph_commitment = Self::finalize_commitment(&hasher);
        self.commitment_hasher = hasher;
        Ok(())
    }

    fn append_commitment_event(&mut self, event: &WorkEvent) -> Result<(), WorkGraphError> {
        let event_bytes = serde_json::to_vec(event)?;
        if self.events.len() > 1 {
            self.commitment_hasher.update(b",");
        }
        self.commitment_hasher.update(event_bytes);
        self.graph_commitment = Self::finalize_commitment(&self.commitment_hasher);
        Ok(())
    }

    fn connected_node_ids(
        &self,
        start: &str,
        max_depth: usize,
        max_nodes: usize,
    ) -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        let Some(start_node) = self.nodes.get(start) else {
            return seen;
        };
        let start_is_repository = start_node.kind == NodeKind::Repository;
        seen.insert(start.to_string());
        let mut queue = VecDeque::from([(start.to_string(), 0usize)]);
        while let Some((node_id, depth)) = queue.pop_front() {
            if depth >= max_depth || seen.len() >= max_nodes {
                continue;
            }
            let Some(neighbors) = self.adjacency.get(&node_id) else {
                continue;
            };
            for next in neighbors {
                if seen.len() >= max_nodes {
                    break;
                }
                // Repository is a high-degree ownership hub. Traversing through
                // it from a workstream would leak sibling workstreams into
                // resume/handoff context. Repository queries may still traverse
                // it when the repository itself is the starting point.
                let next_kind = self.nodes.get(next).map(|node| node.kind);
                if !start_is_repository && next_kind == Some(NodeKind::Repository) {
                    continue;
                }
                if seen.insert(next.clone()) {
                    // Agent/session/model nodes are useful context, but they are
                    // identity/routing hubs shared by unrelated workstreams.
                    // Include the hub itself without expanding through it.
                    let is_shared_hub = matches!(
                        next_kind,
                        Some(NodeKind::Agent) | Some(NodeKind::Session) | Some(NodeKind::Model)
                    );
                    if !is_shared_hub {
                        queue.push_back((next.clone(), depth + 1));
                    }
                }
            }
        }
        seen
    }

    fn work_item_view(&self, node: &WorkNode) -> WorkItemView {
        let related = self.connected_node_ids(&node.node_id, 2, 10_000);
        // Symbols are one hop beyond the changed file. Keep all other work-item
        // fields on the existing two-hop scope so imported boundary files are
        // not mislabeled as changed paths.
        let symbol_related = self.connected_node_ids(&node.node_id, 3, 10_000);
        let mut task_ids = BTreeSet::new();
        let mut agent_ids = BTreeSet::new();
        let mut changed_paths = BTreeSet::new();
        let mut symbol_ids = BTreeSet::new();
        let mut commit_ids = BTreeSet::new();
        let mut decision_ids = BTreeSet::new();
        let mut failure_ids = BTreeSet::new();
        let mut verification_ids = BTreeSet::new();
        let mut evidence_ids = node.evidence_ids.clone();
        for id in &related {
            let Some(n) = self.nodes.get(id) else {
                continue;
            };
            evidence_ids.extend(n.evidence_ids.iter().cloned());
            match n.kind {
                NodeKind::Task => {
                    task_ids.insert(n.node_id.clone());
                }
                NodeKind::Agent => {
                    agent_ids.insert(n.node_id.clone());
                }
                NodeKind::File => {
                    if let Some(path) = attr_string(&n.attributes, "path") {
                        changed_paths.insert(path);
                    }
                }
                NodeKind::Commit => {
                    commit_ids.insert(n.node_id.clone());
                }
                NodeKind::Decision => {
                    decision_ids.insert(n.node_id.clone());
                }
                NodeKind::Failure => {
                    failure_ids.insert(n.node_id.clone());
                }
                NodeKind::Test | NodeKind::CiRun => {
                    verification_ids.insert(n.node_id.clone());
                }
                _ => {}
            }
        }
        for id in symbol_related {
            if self.nodes.get(&id).map(|item| item.kind) == Some(NodeKind::Symbol) {
                symbol_ids.insert(id);
            }
        }
        WorkItemView {
            node_id: node.node_id.clone(),
            kind: node.kind,
            label: node.label.clone(),
            status: node.status,
            trust: node.trust,
            updated_at_ms: node.updated_at_ms,
            task_ids: task_ids.into_iter().collect(),
            agent_ids: agent_ids.into_iter().collect(),
            changed_paths: changed_paths.into_iter().collect(),
            symbol_ids: symbol_ids.into_iter().collect(),
            commit_ids: commit_ids.into_iter().collect(),
            decision_ids: decision_ids.into_iter().collect(),
            failure_ids: failure_ids.into_iter().collect(),
            verification_ids: verification_ids.into_iter().collect(),
            evidence_ids: evidence_ids.into_iter().collect(),
        }
    }
}

fn canonicalize_repository_observation(obs: &mut RepositoryObservation) {
    obs.changes.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| a.kind.token().cmp(b.kind.token()))
            .then_with(|| a.old_path.cmp(&b.old_path))
            .then_with(|| a.staged.cmp(&b.staged))
            .then_with(|| a.conflicted.cmp(&b.conflicted))
            .then_with(|| a.content_digest.cmp(&b.content_digest))
    });
    obs.changes.dedup();
    for commit in &mut obs.commits {
        commit.parent_shas.sort();
        commit.parent_shas.dedup();
        commit.changed_paths.sort();
        commit.changed_paths.dedup();
    }
    obs.commits.sort_by(|a, b| {
        a.timestamp_ms
            .cmp(&b.timestamp_ms)
            .then_with(|| a.sha.cmp(&b.sha))
    });
    obs.commits.dedup();
    obs.verifications.sort_by(|a, b| {
        a.observed_at_ms
            .cmp(&b.observed_at_ms)
            .then_with(|| a.verification_id.cmp(&b.verification_id))
            .then_with(|| a.name.cmp(&b.name))
            .then_with(|| a.source_ref.cmp(&b.source_ref))
    });
    obs.verifications.dedup();
    obs.decisions.sort_by(|a, b| {
        a.decision_id
            .cmp(&b.decision_id)
            .then_with(|| a.text.cmp(&b.text))
            .then_with(|| a.source_ref.cmp(&b.source_ref))
    });
    obs.decisions.dedup();
    for claim in &mut obs.claims {
        claim.evidence_ids.sort();
        claim.evidence_ids.dedup();
    }
    obs.claims.sort_by(|a, b| {
        a.claim_id
            .cmp(&b.claim_id)
            .then_with(|| a.text.cmp(&b.text))
            .then_with(|| a.source_ref.cmp(&b.source_ref))
    });
    obs.claims.dedup();
    for lease in &mut obs.leases {
        lease.scope_paths.sort();
        lease.scope_paths.dedup();
        lease.scope_symbols.sort();
        lease.scope_symbols.dedup();
    }
    obs.leases.sort_by(|a, b| {
        a.lease_id
            .cmp(&b.lease_id)
            .then_with(|| a.agent_id.cmp(&b.agent_id))
            .then_with(|| a.task_id.cmp(&b.task_id))
            .then_with(|| a.expires_at_ms.cmp(&b.expires_at_ms))
    });
    obs.leases.dedup();
    obs.model_executions.sort_by(|a, b| {
        a.execution_id
            .cmp(&b.execution_id)
            .then_with(|| a.provider.cmp(&b.provider))
            .then_with(|| a.model.cmp(&b.model))
            .then_with(|| a.source_ref.cmp(&b.source_ref))
    });
    obs.model_executions.dedup();
    if let Some(task) = obs.task_hint.as_mut() {
        task.remaining_work.sort();
        task.remaining_work.dedup();
    }
}

fn valid_passive_content_digest(change: &FileChangeObservation) -> bool {
    if change.staged || change.conflicted {
        return false;
    }
    if change.kind == FileChangeKind::Deleted {
        return change.content_digest == "worktree:deleted";
    }
    let Some(hex) = change.content_digest.strip_prefix("git-blob:") else {
        return false;
    };
    matches!(hex.len(), 40 | 64) && hex.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn passive_snapshot_group(event: &WorkEvent) -> Option<String> {
    if event.source_kind != EvidenceKind::RepositoryFact
        || !event.source_ref.starts_with("repo-snapshot:")
    {
        return None;
    }
    Some(
        event
            .source_ref
            .strip_suffix(":scope")
            .unwrap_or(&event.source_ref)
            .to_string(),
    )
}

fn passive_repository_snapshot_fingerprint(
    observation: &RepositoryObservation,
) -> Result<Option<String>, WorkGraphError> {
    // Only passive repository/checkpoint observations may collapse. Active
    // operations are audit history and must remain distinct even when their
    // payload happens to repeat.
    if !observation.agent_id.is_empty()
        || !observation.session_id.is_empty()
        || !observation.verifications.is_empty()
        || !observation.claims.is_empty()
        || !observation.leases.is_empty()
        || !observation.model_executions.is_empty()
    {
        return Ok(None);
    }
    if observation
        .task_hint
        .as_ref()
        .is_some_and(|task| task.source_kind != EvidenceKind::Checkpoint)
    {
        return Ok(None);
    }
    if observation
        .decisions
        .iter()
        .any(|decision| decision.source_kind != EvidenceKind::Checkpoint)
    {
        return Ok(None);
    }
    // A timestamp-only equality decision is safe only when every worktree
    // change has exact content identity. Staged/conflicted/special/oversized
    // paths deliberately fail closed in the adapters and therefore remain
    // separate audit events.
    if observation
        .changes
        .iter()
        .any(|change| !valid_passive_content_digest(change))
    {
        return Ok(None);
    }

    let mut semantic = observation.clone();
    semantic.observed_at_ms = 0;
    canonicalize_repository_observation(&mut semantic);
    Ok(Some(sha256_json(&semantic)?))
}

fn context_receipt_event(
    repo_id: &str,
    receipt: ContextReceiptEnvelope,
    agent_id: String,
    session_id: String,
) -> Result<WorkEvent, WorkGraphError> {
    validate_text("receipt agent_id", &agent_id, MAX_ID_LEN, true)?;
    validate_text("receipt session_id", &session_id, MAX_ID_LEN, true)?;
    let evidence = with_evidence_id(EvidenceRef {
        evidence_id: String::new(),
        kind: EvidenceKind::Receipt,
        source_ref: format!("context-receipt:{}", receipt.receipt_id),
        digest: receipt.receipt_commitment.clone(),
        locator: receipt.source_commitment.clone(),
        trust: TrustLevel::Verified,
        observed_at_ms: receipt.created_at_ms,
        attributes: BTreeMap::new(),
    })?;
    let receipt_node_id = stable_node_id(NodeKind::Receipt, repo_id, &receipt.receipt_id);
    let mut attributes = BTreeMap::new();
    attributes.insert(
        "receipt_id".to_string(),
        Value::String(receipt.receipt_id.clone()),
    );
    attributes.insert(
        "receipt_commitment".to_string(),
        Value::String(receipt.receipt_commitment.clone()),
    );
    attributes.insert(
        "repository_commitment".to_string(),
        Value::String(receipt.repository_commitment.clone()),
    );
    attributes.insert(
        "source_commitment".to_string(),
        Value::String(receipt.source_commitment.clone()),
    );
    attributes.insert(
        "selected_refs".to_string(),
        serde_json::to_value(&receipt.selected_refs)?,
    );
    attributes.insert(
        "omitted_refs".to_string(),
        serde_json::to_value(&receipt.omitted_refs)?,
    );
    attributes.insert(
        "pinned_refs".to_string(),
        serde_json::to_value(&receipt.pinned_refs)?,
    );
    attributes.insert(
        "recoverable_refs".to_string(),
        serde_json::to_value(&receipt.recoverable_refs)?,
    );
    attributes.insert(
        "recovery_handles".to_string(),
        serde_json::to_value(&receipt.recovery_handles)?,
    );
    attributes.insert(
        "evidence_ids".to_string(),
        serde_json::to_value(&receipt.evidence_ids)?,
    );
    attributes.insert(
        "budget_tokens".to_string(),
        Value::from(receipt.budget_tokens),
    );
    attributes.insert(
        "selection_policy".to_string(),
        Value::String(receipt.selection_policy.clone()),
    );
    attributes.insert(
        "execution_id".to_string(),
        Value::String(receipt.execution_id.clone()),
    );
    let operations = vec![
        WorkOperation::AddEvidence {
            evidence: evidence.clone(),
        },
        WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: receipt_node_id.clone(),
                kind: NodeKind::Receipt,
                label: receipt.receipt_id.clone(),
                trust: TrustLevel::Verified,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes,
                evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                updated_at_ms: receipt.created_at_ms,
            },
        },
        edge_op(
            &receipt_node_id,
            &receipt.work_scope_id,
            EdgeKind::PartOf,
            TrustLevel::Verified,
            receipt.created_at_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ),
        WorkOperation::AttachEvidence {
            node_id: receipt.work_scope_id,
            evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
        },
    ];
    Ok(WorkEvent {
        event_id: String::new(),
        observed_at_ms: receipt.created_at_ms,
        source_kind: EvidenceKind::Receipt,
        source_ref: format!("context-receipt:{}", receipt.receipt_id),
        actor_id: agent_id,
        session_id,
        operations,
    })
}

fn memory_record_event(
    repo_id: &str,
    memory: MemoryRecord,
    admissibility: MemoryAdmissibility,
    now_ms: i64,
) -> Result<WorkEvent, WorkGraphError> {
    let evidence = with_evidence_id(EvidenceRef {
        evidence_id: String::new(),
        kind: EvidenceKind::Memory,
        source_ref: format!("memory:{}", memory.memory_id),
        digest: memory.record_commitment.clone(),
        locator: memory.content_reference.clone(),
        trust: TrustLevel::Observed,
        observed_at_ms: now_ms,
        attributes: BTreeMap::new(),
    })?;
    let memory_node_id = stable_node_id(NodeKind::Memory, repo_id, &memory.memory_id);
    let (status, status_trust) = match admissibility {
        MemoryAdmissibility::Admissible => (WorkStatus::Unknown, TrustLevel::Untrusted),
        MemoryAdmissibility::Contradicted => (WorkStatus::Blocked, TrustLevel::Observed),
        MemoryAdmissibility::Superseded | MemoryAdmissibility::Expired => {
            (WorkStatus::Abandoned, TrustLevel::Observed)
        }
        MemoryAdmissibility::Unsupported => (WorkStatus::NeedsVerification, TrustLevel::Observed),
    };
    let mut attributes = BTreeMap::new();
    attributes.insert(
        "memory_id".to_string(),
        Value::String(memory.memory_id.clone()),
    );
    attributes.insert(
        "record_commitment".to_string(),
        Value::String(memory.record_commitment.clone()),
    );
    attributes.insert(
        "content_reference".to_string(),
        Value::String(memory.content_reference.clone()),
    );
    attributes.insert(
        "content_commitment".to_string(),
        Value::String(memory.content_commitment.clone()),
    );
    attributes.insert(
        "admissibility".to_string(),
        serde_json::to_value(admissibility)?,
    );
    attributes.insert(
        "trust_state".to_string(),
        serde_json::to_value(memory.trust_state)?,
    );
    attributes.insert(
        "source_agent".to_string(),
        Value::String(memory.source_agent.clone()),
    );
    attributes.insert(
        "source_session".to_string(),
        Value::String(memory.source_session.clone()),
    );
    attributes.insert(
        "source_execution".to_string(),
        Value::String(memory.source_execution.clone()),
    );
    attributes.insert(
        "evidence_ids".to_string(),
        serde_json::to_value(&memory.evidence_ids)?,
    );
    attributes.insert(
        "supersedes".to_string(),
        serde_json::to_value(&memory.supersedes)?,
    );
    attributes.insert(
        "contradicted_by".to_string(),
        serde_json::to_value(&memory.contradicted_by)?,
    );
    attributes.insert(
        "valid_until_ms".to_string(),
        Value::from(memory.valid_until_ms),
    );
    attributes.insert(
        "recovery_handle".to_string(),
        Value::String(memory.recovery_handle.clone()),
    );

    let mut operations = vec![
        WorkOperation::AddEvidence {
            evidence: evidence.clone(),
        },
        WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: memory_node_id.clone(),
                kind: NodeKind::Memory,
                label: memory.content_reference.clone(),
                trust: TrustLevel::Observed,
                status,
                status_trust,
                attributes,
                evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                updated_at_ms: now_ms,
            },
        },
    ];
    if !memory.workstream_id.is_empty() {
        operations.push(edge_op(
            &memory_node_id,
            &memory.workstream_id,
            EdgeKind::PartOf,
            TrustLevel::Observed,
            now_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ));
    }
    if !memory.task_id.is_empty() {
        operations.push(edge_op(
            &memory_node_id,
            &memory.task_id,
            EdgeKind::References,
            TrustLevel::Observed,
            now_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ));
    }
    for superseded_id in &memory.supersedes {
        let old_node_id = stable_node_id(NodeKind::Memory, repo_id, superseded_id);
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: old_node_id.clone(),
                kind: NodeKind::Memory,
                label: superseded_id.clone(),
                trust: TrustLevel::Observed,
                status: WorkStatus::Abandoned,
                status_trust: TrustLevel::Observed,
                attributes: BTreeMap::new(),
                evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                updated_at_ms: now_ms,
            },
        });
        operations.push(edge_op(
            &memory_node_id,
            &old_node_id,
            EdgeKind::Supersedes,
            TrustLevel::Observed,
            now_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ));
    }
    Ok(WorkEvent {
        event_id: String::new(),
        observed_at_ms: now_ms,
        source_kind: EvidenceKind::Memory,
        source_ref: format!("memory:{}", memory.memory_id),
        actor_id: memory.source_agent,
        session_id: memory.source_session,
        operations,
    })
}

fn execution_chain_event(
    repo_id: &str,
    route: RoutingDecision,
    outcome: ModelExecutionOutcome,
    verification: VerificationRecord,
    current_repository_commitment: &str,
    invalidated_commitments: &BTreeSet<String>,
) -> Result<WorkEvent, WorkGraphError> {
    let freshness = verification.freshness(
        current_repository_commitment,
        verification.observed_at_ms,
        invalidated_commitments,
    );
    let verification_is_current = freshness == VerificationFreshness::Current;
    let verification_is_decisive = verification_is_current
        && matches!(
            verification.verdict,
            VerificationVerdict::Passed | VerificationVerdict::Failed
        );
    let verification_trust = if verification_is_decisive {
        TrustLevel::Verified
    } else {
        TrustLevel::Observed
    };

    let route_evidence = with_evidence_id(EvidenceRef {
        evidence_id: String::new(),
        kind: EvidenceKind::RuntimeObservation,
        source_ref: format!("routing:{}", route.routing_id),
        digest: route.decision_commitment.clone(),
        locator: route.policy_version.clone(),
        trust: TrustLevel::Observed,
        observed_at_ms: route.decided_at_ms,
        attributes: BTreeMap::new(),
    })?;
    let outcome_evidence = with_evidence_id(EvidenceRef {
        evidence_id: String::new(),
        kind: EvidenceKind::RuntimeObservation,
        source_ref: format!("execution:{}", outcome.outcome_id),
        digest: outcome.outcome_commitment.clone(),
        locator: outcome.response_commitment.clone(),
        trust: TrustLevel::Observed,
        observed_at_ms: outcome.completed_at_ms,
        attributes: BTreeMap::new(),
    })?;
    let mut verification_evidence_attributes = BTreeMap::new();
    verification_evidence_attributes
        .insert("freshness".to_string(), serde_json::to_value(freshness)?);
    verification_evidence_attributes.insert(
        "verified_repository_commitment".to_string(),
        Value::String(verification.verified_repository_commitment.clone()),
    );
    verification_evidence_attributes.insert(
        "dependency_commitments".to_string(),
        serde_json::to_value(&verification.dependency_commitments)?,
    );
    let verification_evidence = with_evidence_id(EvidenceRef {
        evidence_id: String::new(),
        kind: EvidenceKind::TestResult,
        source_ref: format!("verification:{}", verification.verification_id),
        digest: verification.record_commitment.clone(),
        locator: verification.subject_id.clone(),
        trust: verification_trust,
        observed_at_ms: verification.observed_at_ms,
        attributes: verification_evidence_attributes,
    })?;

    let route_node_id = stable_node_id(NodeKind::Decision, repo_id, &route.routing_id);
    let execution_node_id = stable_node_id(NodeKind::ModelExecution, repo_id, &outcome.outcome_id);
    let verification_node_id =
        stable_node_id(NodeKind::Test, repo_id, &verification.verification_id);
    let model_node_id = stable_node_id(
        NodeKind::Model,
        repo_id,
        &format!("{}:{}:{}", route.provider, route.model, route.runtime),
    );

    let mut operations = vec![
        WorkOperation::AddEvidence {
            evidence: route_evidence.clone(),
        },
        WorkOperation::AddEvidence {
            evidence: outcome_evidence.clone(),
        },
        WorkOperation::AddEvidence {
            evidence: verification_evidence.clone(),
        },
    ];

    let mut model_attributes = BTreeMap::new();
    model_attributes.insert(
        "provider".to_string(),
        Value::String(route.provider.clone()),
    );
    model_attributes.insert("model".to_string(), Value::String(route.model.clone()));
    model_attributes.insert("runtime".to_string(), Value::String(route.runtime.clone()));
    operations.push(WorkOperation::UpsertNode {
        node: WorkNode {
            node_id: model_node_id.clone(),
            kind: NodeKind::Model,
            label: format!("{}:{}", route.provider, route.model),
            trust: TrustLevel::Observed,
            status: WorkStatus::Unknown,
            status_trust: TrustLevel::Untrusted,
            attributes: model_attributes,
            evidence_ids: BTreeSet::from([route_evidence.evidence_id.clone()]),
            updated_at_ms: route.decided_at_ms,
        },
    });

    let mut route_attributes = BTreeMap::new();
    route_attributes.insert(
        "routing_id".to_string(),
        Value::String(route.routing_id.clone()),
    );
    route_attributes.insert(
        "decision_commitment".to_string(),
        Value::String(route.decision_commitment.clone()),
    );
    route_attributes.insert(
        "context_budget_tokens".to_string(),
        Value::from(route.context_budget_tokens),
    );
    route_attributes.insert(
        "policy_version".to_string(),
        Value::String(route.policy_version.clone()),
    );
    route_attributes.insert(
        "reason_codes".to_string(),
        serde_json::to_value(&route.reason_codes)?,
    );
    route_attributes.insert(
        "feature_commitments".to_string(),
        serde_json::to_value(&route.feature_commitments)?,
    );
    route_attributes.insert(
        "fallback_route_ids".to_string(),
        serde_json::to_value(&route.fallback_route_ids)?,
    );
    route_attributes.insert(
        "receipt_id".to_string(),
        Value::String(route.receipt_id.clone()),
    );
    route_attributes.insert(
        "evidence_ids".to_string(),
        serde_json::to_value(&route.evidence_ids)?,
    );
    operations.push(WorkOperation::UpsertNode {
        node: WorkNode {
            node_id: route_node_id.clone(),
            kind: NodeKind::Decision,
            label: format!(
                "route {} to {}:{}",
                route.routing_id, route.provider, route.model
            ),
            trust: TrustLevel::Observed,
            status: WorkStatus::Unknown,
            status_trust: TrustLevel::Untrusted,
            attributes: route_attributes,
            evidence_ids: BTreeSet::from([route_evidence.evidence_id.clone()]),
            updated_at_ms: route.decided_at_ms,
        },
    });

    let (execution_status, execution_status_trust) = match outcome.state {
        ExecutionState::Succeeded => (WorkStatus::Completed, TrustLevel::Observed),
        ExecutionState::Failed => (WorkStatus::Blocked, TrustLevel::Observed),
        ExecutionState::Cancelled => (WorkStatus::Abandoned, TrustLevel::Observed),
        ExecutionState::Unknown => (WorkStatus::Unknown, TrustLevel::Untrusted),
    };
    let mut outcome_attributes = BTreeMap::new();
    outcome_attributes.insert(
        "outcome_id".to_string(),
        Value::String(outcome.outcome_id.clone()),
    );
    outcome_attributes.insert(
        "routing_id".to_string(),
        Value::String(outcome.routing_id.clone()),
    );
    outcome_attributes.insert(
        "outcome_commitment".to_string(),
        Value::String(outcome.outcome_commitment.clone()),
    );
    outcome_attributes.insert("latency_ms".to_string(), Value::from(outcome.latency_ms));
    outcome_attributes.insert(
        "input_tokens".to_string(),
        Value::from(outcome.input_tokens),
    );
    outcome_attributes.insert(
        "output_tokens".to_string(),
        Value::from(outcome.output_tokens),
    );
    outcome_attributes.insert(
        "cost_micro_usd".to_string(),
        Value::from(outcome.cost_micro_usd),
    );
    outcome_attributes.insert(
        "verification_state".to_string(),
        serde_json::to_value(outcome.verification_state)?,
    );
    outcome_attributes.insert(
        "error_code".to_string(),
        Value::String(outcome.error_code.clone()),
    );
    outcome_attributes.insert(
        "evidence_ids".to_string(),
        serde_json::to_value(&outcome.evidence_ids)?,
    );
    operations.push(WorkOperation::UpsertNode {
        node: WorkNode {
            node_id: execution_node_id.clone(),
            kind: NodeKind::ModelExecution,
            label: format!("{}:{} execution", outcome.provider, outcome.model),
            trust: TrustLevel::Observed,
            status: execution_status,
            status_trust: execution_status_trust,
            attributes: outcome_attributes,
            evidence_ids: BTreeSet::from([outcome_evidence.evidence_id.clone()]),
            updated_at_ms: outcome.completed_at_ms,
        },
    });

    let verification_status = match (verification.verdict, freshness) {
        (VerificationVerdict::Passed, VerificationFreshness::Current) => WorkStatus::Completed,
        (VerificationVerdict::Failed, VerificationFreshness::Current) => WorkStatus::Blocked,
        (VerificationVerdict::Passed | VerificationVerdict::Failed, _) => {
            WorkStatus::NeedsVerification
        }
        _ => WorkStatus::Unknown,
    };
    let verification_status_trust = if verification_status == WorkStatus::Unknown {
        TrustLevel::Untrusted
    } else {
        verification_trust
    };
    let mut verification_attributes = BTreeMap::new();
    verification_attributes.insert(
        "verification_id".to_string(),
        Value::String(verification.verification_id.clone()),
    );
    verification_attributes.insert(
        "record_commitment".to_string(),
        Value::String(verification.record_commitment.clone()),
    );
    verification_attributes.insert(
        "verified_repository_commitment".to_string(),
        Value::String(verification.verified_repository_commitment.clone()),
    );
    verification_attributes.insert(
        "verdict".to_string(),
        serde_json::to_value(verification.verdict)?,
    );
    verification_attributes.insert("freshness".to_string(), serde_json::to_value(freshness)?);
    verification_attributes.insert(
        "dependency_commitments".to_string(),
        serde_json::to_value(&verification.dependency_commitments)?,
    );
    verification_attributes.insert(
        "evidence_ids".to_string(),
        serde_json::to_value(&verification.evidence_ids)?,
    );
    operations.push(WorkOperation::UpsertNode {
        node: WorkNode {
            node_id: verification_node_id.clone(),
            kind: NodeKind::Test,
            label: format!("execution verification {}", verification.verification_id),
            trust: verification_trust,
            status: verification_status,
            status_trust: verification_status_trust,
            attributes: verification_attributes,
            evidence_ids: BTreeSet::from([verification_evidence.evidence_id.clone()]),
            updated_at_ms: verification.observed_at_ms,
        },
    });

    for (from, to, kind, evidence_id, trust) in [
        (
            route.task_id.as_str(),
            route_node_id.as_str(),
            EdgeKind::RoutedTo,
            route_evidence.evidence_id.as_str(),
            TrustLevel::Observed,
        ),
        (
            route_node_id.as_str(),
            model_node_id.as_str(),
            EdgeKind::RoutedTo,
            route_evidence.evidence_id.as_str(),
            TrustLevel::Observed,
        ),
        (
            route_node_id.as_str(),
            route.workstream_id.as_str(),
            EdgeKind::PartOf,
            route_evidence.evidence_id.as_str(),
            TrustLevel::Observed,
        ),
        (
            execution_node_id.as_str(),
            route_node_id.as_str(),
            EdgeKind::ProducedBy,
            outcome_evidence.evidence_id.as_str(),
            TrustLevel::Observed,
        ),
        (
            execution_node_id.as_str(),
            route.workstream_id.as_str(),
            EdgeKind::PartOf,
            outcome_evidence.evidence_id.as_str(),
            TrustLevel::Observed,
        ),
        (
            execution_node_id.as_str(),
            verification_node_id.as_str(),
            EdgeKind::VerifiedBy,
            verification_evidence.evidence_id.as_str(),
            verification_trust,
        ),
    ] {
        operations.push(edge_op(
            from,
            to,
            kind,
            trust,
            verification.observed_at_ms,
            &[evidence_id.to_string()],
        ));
    }

    if !route.receipt_id.is_empty() {
        let receipt_node_id = stable_node_id(NodeKind::Receipt, repo_id, &route.receipt_id);
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: receipt_node_id.clone(),
                kind: NodeKind::Receipt,
                label: route.receipt_id.clone(),
                trust: TrustLevel::Observed,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes: BTreeMap::new(),
                evidence_ids: BTreeSet::from([route_evidence.evidence_id.clone()]),
                updated_at_ms: route.decided_at_ms,
            },
        });
        operations.push(edge_op(
            &route_node_id,
            &receipt_node_id,
            EdgeKind::References,
            TrustLevel::Observed,
            route.decided_at_ms,
            std::slice::from_ref(&route_evidence.evidence_id),
        ));
    }

    match (verification.verdict, freshness) {
        (VerificationVerdict::Failed, VerificationFreshness::Current) => {
            operations.push(WorkOperation::SetStatus {
                node_id: route.workstream_id.clone(),
                status: WorkStatus::Blocked,
                trust: TrustLevel::Verified,
                reason: "current exact-version execution verification failed".to_string(),
                evidence_ids: BTreeSet::from([verification_evidence.evidence_id.clone()]),
            });
        }
        (VerificationVerdict::Passed, VerificationFreshness::Current) => {
            operations.push(WorkOperation::AttachEvidence {
                node_id: route.workstream_id.clone(),
                evidence_ids: BTreeSet::from([verification_evidence.evidence_id.clone()]),
            });
        }
        (VerificationVerdict::Passed | VerificationVerdict::Failed, _) => {
            operations.push(WorkOperation::SetStatus {
                node_id: route.workstream_id.clone(),
                status: WorkStatus::NeedsVerification,
                trust: TrustLevel::Observed,
                reason: "execution verification is stale or transitively invalidated".to_string(),
                evidence_ids: BTreeSet::from([verification_evidence.evidence_id.clone()]),
            });
        }
        _ => {
            operations.push(WorkOperation::AttachEvidence {
                node_id: route.workstream_id.clone(),
                evidence_ids: BTreeSet::from([verification_evidence.evidence_id.clone()]),
            });
        }
    }

    Ok(WorkEvent {
        event_id: String::new(),
        observed_at_ms: verification.observed_at_ms,
        source_kind: EvidenceKind::RuntimeObservation,
        source_ref: format!("execution-chain:{}", outcome.outcome_id),
        actor_id: String::new(),
        session_id: String::new(),
        operations,
    })
}

fn observation_to_event(
    repo_id: &str,
    mut obs: RepositoryObservation,
) -> Result<WorkEvent, WorkGraphError> {
    validate_text("repo_id", &obs.repo_id, MAX_ID_LEN, false)?;
    validate_text(
        "repository_label",
        &obs.repository_label,
        MAX_LABEL_LEN,
        true,
    )?;
    validate_text("agent_id", &obs.agent_id, MAX_ID_LEN, true)?;
    validate_text("session_id", &obs.session_id, MAX_ID_LEN, true)?;
    if obs.changes.len() > MAX_CHANGES_PER_EVENT {
        return Err(WorkGraphError::LimitExceeded(
            "too many file changes in one observation".to_string(),
        ));
    }
    if obs.leases.len() > MAX_SCOPE_ITEMS {
        return Err(WorkGraphError::LimitExceeded(
            "too many work leases".to_string(),
        ));
    }

    // Keep one canonicalization rule for event construction and passive
    // semantic fingerprints. This is the parity boundary shared by every
    // adapter that submits RepositoryObservation.
    canonicalize_repository_observation(&mut obs);

    let mut operations = Vec::new();
    let repo_node_id = stable_node_id(NodeKind::Repository, repo_id, repo_id);
    let repo_evidence = EvidenceRef {
        evidence_id: String::new(),
        kind: EvidenceKind::RepositoryFact,
        source_ref: format!("repo:{repo_id}"),
        digest: obs.branch.head_sha.clone(),
        locator: String::new(),
        trust: TrustLevel::Observed,
        observed_at_ms: obs.observed_at_ms,
        attributes: BTreeMap::new(),
    };
    let repo_evidence = with_evidence_id(repo_evidence)?;
    operations.push(WorkOperation::AddEvidence {
        evidence: repo_evidence.clone(),
    });
    let mut repo_attrs = BTreeMap::new();
    repo_attrs.insert("repo_id".to_string(), Value::String(repo_id.to_string()));
    if !obs.branch.name.is_empty() {
        repo_attrs.insert("branch".to_string(), Value::String(obs.branch.name.clone()));
    }
    if !obs.branch.head_sha.is_empty() {
        repo_attrs.insert(
            "head_sha".to_string(),
            Value::String(obs.branch.head_sha.clone()),
        );
    }
    operations.push(WorkOperation::UpsertNode {
        node: WorkNode {
            node_id: repo_node_id.clone(),
            kind: NodeKind::Repository,
            label: if obs.repository_label.is_empty() {
                repo_id.to_string()
            } else {
                obs.repository_label.clone()
            },
            trust: TrustLevel::Observed,
            status: WorkStatus::Unknown,
            status_trust: TrustLevel::Untrusted,
            attributes: repo_attrs,
            evidence_ids: BTreeSet::from([repo_evidence.evidence_id.clone()]),
            updated_at_ms: obs.observed_at_ms,
        },
    });

    let had_explicit_task = obs.task_hint.is_some();
    let meaningful_git = !obs.changes.is_empty()
        || !obs.commits.is_empty()
        || obs.branch.ahead_by > 0
        || obs.branch.merge_in_progress
        || obs.branch.rebase_in_progress;
    let meaningful_non_git = obs.task_hint.is_some()
        || !obs.decisions.is_empty()
        || !obs.claims.is_empty()
        || !obs.leases.is_empty()
        || !obs.model_executions.is_empty()
        || !obs.verifications.is_empty();
    let meaningful = meaningful_git || meaningful_non_git;

    if !meaningful {
        return Ok(WorkEvent {
            event_id: String::new(),
            observed_at_ms: obs.observed_at_ms,
            source_kind: EvidenceKind::RepositoryFact,
            source_ref: format!("repo:{repo_id}"),
            actor_id: obs.agent_id,
            session_id: obs.session_id,
            operations,
        });
    }

    let (
        task_id,
        task_label,
        task_trust,
        task_explicit_status,
        remaining_work,
        task_source_kind,
        task_source,
    ) = if let Some(task) = obs.task_hint.take() {
        validate_text("task title", &task.title, MAX_LABEL_LEN, false)?;
        let task_id = if task.task_id.is_empty() {
            stable_node_id(NodeKind::Task, repo_id, &task.title)
        } else {
            validate_text("task_id", &task.task_id, MAX_ID_LEN, false)?;
            stable_node_id(NodeKind::Task, repo_id, &task.task_id)
        };
        (
            task_id,
            task.title,
            task.trust,
            task.explicit_status,
            task.remaining_work,
            task.source_kind,
            task.source_ref,
        )
    } else {
        let branch_label = if !obs.branch.name.is_empty() {
            obs.branch.name.clone()
        } else {
            "detached-head".to_string()
        };
        let label = if meaningful_git {
            format!("Repository work on {branch_label}")
        } else {
            "Repository work".to_string()
        };
        (
            stable_node_id(NodeKind::Task, repo_id, &format!("inferred:{branch_label}")),
            label,
            TrustLevel::Inferred,
            WorkStatus::Unknown,
            Vec::new(),
            EvidenceKind::RepositoryFact,
            "inferred:repository-state".to_string(),
        )
    };

    let workstream_key = format!("{}:{}", obs.branch.name, task_id);
    let workstream_id = stable_node_id(NodeKind::Workstream, repo_id, &workstream_key);
    let task_evidence = with_evidence_id(EvidenceRef {
        evidence_id: String::new(),
        kind: task_source_kind,
        source_ref: if task_source.is_empty() {
            "work-graph:task-inference".to_string()
        } else {
            task_source
        },
        digest: sha256_text(&task_label),
        locator: String::new(),
        trust: task_trust,
        observed_at_ms: obs.observed_at_ms,
        attributes: BTreeMap::new(),
    })?;
    operations.push(WorkOperation::AddEvidence {
        evidence: task_evidence.clone(),
    });
    let mut task_attrs = BTreeMap::new();
    if !remaining_work.is_empty() {
        task_attrs.insert(
            "remaining_work".to_string(),
            serde_json::to_value(&remaining_work)?,
        );
    }
    operations.push(WorkOperation::UpsertNode {
        node: WorkNode {
            node_id: task_id.clone(),
            kind: NodeKind::Task,
            label: task_label.clone(),
            trust: task_trust,
            status: WorkStatus::Unknown,
            status_trust: TrustLevel::Untrusted,
            attributes: task_attrs,
            evidence_ids: BTreeSet::from([task_evidence.evidence_id.clone()]),
            updated_at_ms: obs.observed_at_ms,
        },
    });

    let mut workstream_attrs = BTreeMap::new();
    workstream_attrs.insert("branch".to_string(), Value::String(obs.branch.name.clone()));
    workstream_attrs.insert(
        "head_sha".to_string(),
        Value::String(obs.branch.head_sha.clone()),
    );
    workstream_attrs.insert(
        "base_ref".to_string(),
        Value::String(obs.branch.base_ref.clone()),
    );
    workstream_attrs.insert("ahead_by".to_string(), Value::from(obs.branch.ahead_by));
    workstream_attrs.insert("behind_by".to_string(), Value::from(obs.branch.behind_by));
    workstream_attrs.insert(
        "remaining_work".to_string(),
        serde_json::to_value(&remaining_work)?,
    );
    operations.push(WorkOperation::UpsertNode {
        node: WorkNode {
            node_id: workstream_id.clone(),
            kind: NodeKind::Workstream,
            label: task_label.clone(),
            trust: task_trust,
            status: WorkStatus::Unknown,
            status_trust: TrustLevel::Untrusted,
            attributes: workstream_attrs,
            evidence_ids: BTreeSet::from([
                repo_evidence.evidence_id.clone(),
                task_evidence.evidence_id.clone(),
            ]),
            updated_at_ms: obs.observed_at_ms,
        },
    });
    operations.push(edge_op(
        &repo_node_id,
        &workstream_id,
        EdgeKind::Contains,
        TrustLevel::Observed,
        obs.observed_at_ms,
        std::slice::from_ref(&repo_evidence.evidence_id),
    ));
    operations.push(edge_op(
        &workstream_id,
        &task_id,
        EdgeKind::PartOf,
        task_trust,
        obs.observed_at_ms,
        std::slice::from_ref(&task_evidence.evidence_id),
    ));

    let mut agent_node_id = String::new();
    if !obs.agent_id.is_empty() {
        agent_node_id = stable_node_id(NodeKind::Agent, repo_id, &obs.agent_id);
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: agent_node_id.clone(),
                kind: NodeKind::Agent,
                label: obs.agent_id.clone(),
                trust: TrustLevel::Observed,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes: BTreeMap::new(),
                evidence_ids: BTreeSet::new(),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &agent_node_id,
            &workstream_id,
            EdgeKind::WorksOn,
            TrustLevel::Observed,
            obs.observed_at_ms,
            &[],
        ));
    }

    if !obs.session_id.is_empty() {
        let session_id = stable_node_id(NodeKind::Session, repo_id, &obs.session_id);
        let mut attrs = BTreeMap::new();
        attrs.insert(
            "session_id".to_string(),
            Value::String(obs.session_id.clone()),
        );
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: session_id.clone(),
                kind: NodeKind::Session,
                label: obs.session_id.clone(),
                trust: TrustLevel::Observed,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes: attrs,
                evidence_ids: BTreeSet::new(),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &session_id,
            &workstream_id,
            EdgeKind::PartOf,
            TrustLevel::Observed,
            obs.observed_at_ms,
            &[],
        ));
        if !agent_node_id.is_empty() {
            operations.push(edge_op(
                &session_id,
                &agent_node_id,
                EdgeKind::ProducedBy,
                TrustLevel::Observed,
                obs.observed_at_ms,
                &[],
            ));
        }
    }

    let mut blocked = obs.branch.merge_in_progress || obs.branch.rebase_in_progress;
    let mut blocked_trust = if blocked {
        TrustLevel::Observed
    } else {
        TrustLevel::Untrusted
    };
    let mut has_passing_verification = false;
    let mut status_evidence_ids = BTreeSet::from([
        repo_evidence.evidence_id.clone(),
        task_evidence.evidence_id.clone(),
    ]);

    for change in &obs.changes {
        validate_path(&change.path)?;
        if !change.old_path.is_empty() {
            validate_path(&change.old_path)?;
        }
        let source_ref = format!("git-status:{}", obs.branch.head_sha);
        let mut ev_attrs = BTreeMap::new();
        ev_attrs.insert("path".to_string(), Value::String(change.path.clone()));
        ev_attrs.insert(
            "change_kind".to_string(),
            serde_json::to_value(change.kind)?,
        );
        if !change.old_path.is_empty() {
            ev_attrs.insert(
                "old_path".to_string(),
                Value::String(change.old_path.clone()),
            );
        }
        let evidence = with_evidence_id(EvidenceRef {
            evidence_id: String::new(),
            kind: EvidenceKind::GitStatus,
            source_ref,
            digest: change.content_digest.clone(),
            locator: change.path.clone(),
            trust: TrustLevel::Observed,
            observed_at_ms: obs.observed_at_ms,
            attributes: ev_attrs,
        })?;
        operations.push(WorkOperation::AddEvidence {
            evidence: evidence.clone(),
        });
        let file_id = stable_node_id(NodeKind::File, repo_id, &change.path);
        let mut file_attrs = BTreeMap::new();
        file_attrs.insert("path".to_string(), Value::String(change.path.clone()));
        file_attrs.insert(
            "change_kind".to_string(),
            serde_json::to_value(change.kind)?,
        );
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: file_id.clone(),
                kind: NodeKind::File,
                label: change.path.clone(),
                trust: TrustLevel::Observed,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes: file_attrs,
                evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        if !change.old_path.is_empty() {
            let old_file_id = stable_node_id(NodeKind::File, repo_id, &change.old_path);
            let mut old_file_attrs = BTreeMap::new();
            old_file_attrs.insert("path".to_string(), Value::String(change.old_path.clone()));
            old_file_attrs.insert("renamed_to".to_string(), Value::String(change.path.clone()));
            operations.push(WorkOperation::UpsertNode {
                node: WorkNode {
                    node_id: old_file_id.clone(),
                    kind: NodeKind::File,
                    label: change.old_path.clone(),
                    trust: TrustLevel::Observed,
                    status: WorkStatus::Unknown,
                    status_trust: TrustLevel::Untrusted,
                    attributes: old_file_attrs,
                    evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                    updated_at_ms: obs.observed_at_ms,
                },
            });
            operations.push(edge_op(
                &file_id,
                &old_file_id,
                EdgeKind::Supersedes,
                TrustLevel::Observed,
                obs.observed_at_ms,
                std::slice::from_ref(&evidence.evidence_id),
            ));
        }
        let change_key = format!(
            "{}:{:?}:{}:{}",
            obs.branch.head_sha, change.kind, change.old_path, change.path
        );
        let change_id = stable_node_id(NodeKind::Change, repo_id, &change_key);
        let mut change_attrs = BTreeMap::new();
        change_attrs.insert("path".to_string(), Value::String(change.path.clone()));
        change_attrs.insert("staged".to_string(), Value::Bool(change.staged));
        change_attrs.insert("conflicted".to_string(), Value::Bool(change.conflicted));
        if !change.old_path.is_empty() {
            change_attrs.insert(
                "old_path".to_string(),
                Value::String(change.old_path.clone()),
            );
        }
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: change_id.clone(),
                kind: NodeKind::Change,
                label: format!("{:?} {}", change.kind, change.path),
                trust: TrustLevel::Observed,
                status: WorkStatus::InProgress,
                status_trust: TrustLevel::Observed,
                attributes: change_attrs,
                evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &workstream_id,
            &change_id,
            EdgeKind::Changed,
            TrustLevel::Observed,
            obs.observed_at_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ));
        operations.push(edge_op(
            &change_id,
            &file_id,
            EdgeKind::Touches,
            TrustLevel::Observed,
            obs.observed_at_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ));
        if change.conflicted || change.kind == FileChangeKind::Unmerged {
            blocked = true;
            blocked_trust = blocked_trust.max(TrustLevel::Observed);
            status_evidence_ids.insert(evidence.evidence_id.clone());
            let failure_id = stable_node_id(
                NodeKind::Failure,
                repo_id,
                &format!("conflict:{}", change.path),
            );
            operations.push(WorkOperation::UpsertNode {
                node: WorkNode {
                    node_id: failure_id.clone(),
                    kind: NodeKind::Failure,
                    label: format!("merge conflict: {}", change.path),
                    trust: TrustLevel::Observed,
                    status: WorkStatus::Blocked,
                    status_trust: TrustLevel::Observed,
                    attributes: BTreeMap::new(),
                    evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                    updated_at_ms: obs.observed_at_ms,
                },
            });
            operations.push(edge_op(
                &failure_id,
                &workstream_id,
                EdgeKind::Blocks,
                TrustLevel::Observed,
                obs.observed_at_ms,
                std::slice::from_ref(&evidence.evidence_id),
            ));
        }
    }

    for commit in &obs.commits {
        validate_text("commit sha", &commit.sha, MAX_ID_LEN, false)?;
        let evidence = with_evidence_id(EvidenceRef {
            evidence_id: String::new(),
            kind: EvidenceKind::GitCommit,
            source_ref: format!("git-commit:{}", commit.sha),
            digest: commit.sha.clone(),
            locator: String::new(),
            trust: TrustLevel::Observed,
            observed_at_ms: if commit.timestamp_ms == 0 {
                obs.observed_at_ms
            } else {
                commit.timestamp_ms
            },
            attributes: BTreeMap::new(),
        })?;
        operations.push(WorkOperation::AddEvidence {
            evidence: evidence.clone(),
        });
        let commit_id = stable_node_id(NodeKind::Commit, repo_id, &commit.sha);
        let mut attrs = BTreeMap::new();
        attrs.insert("sha".to_string(), Value::String(commit.sha.clone()));
        attrs.insert(
            "subject".to_string(),
            Value::String(bounded(&commit.subject, MAX_LABEL_LEN)),
        );
        attrs.insert(
            "changed_paths".to_string(),
            serde_json::to_value(&commit.changed_paths)?,
        );
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: commit_id.clone(),
                kind: NodeKind::Commit,
                label: if commit.subject.is_empty() {
                    commit.sha.clone()
                } else {
                    bounded(&commit.subject, MAX_LABEL_LEN)
                },
                trust: TrustLevel::Observed,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes: attrs,
                evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &commit_id,
            &workstream_id,
            EdgeKind::PartOf,
            TrustLevel::Observed,
            obs.observed_at_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ));
        if !agent_node_id.is_empty() {
            operations.push(edge_op(
                &commit_id,
                &agent_node_id,
                EdgeKind::ProducedBy,
                TrustLevel::Observed,
                obs.observed_at_ms,
                std::slice::from_ref(&evidence.evidence_id),
            ));
        }
        for path in &commit.changed_paths {
            if validate_path(path).is_err() {
                continue;
            }
            let file_id = stable_node_id(NodeKind::File, repo_id, path);
            let mut attrs = BTreeMap::new();
            attrs.insert("path".to_string(), Value::String(path.clone()));
            operations.push(WorkOperation::UpsertNode {
                node: WorkNode {
                    node_id: file_id.clone(),
                    kind: NodeKind::File,
                    label: path.clone(),
                    trust: TrustLevel::Observed,
                    status: WorkStatus::Unknown,
                    status_trust: TrustLevel::Untrusted,
                    attributes: attrs,
                    evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                    updated_at_ms: obs.observed_at_ms,
                },
            });
            operations.push(edge_op(
                &commit_id,
                &file_id,
                EdgeKind::Changed,
                TrustLevel::Observed,
                obs.observed_at_ms,
                std::slice::from_ref(&evidence.evidence_id),
            ));
        }
    }

    for verification in &obs.verifications {
        let ev_kind = match verification.evidence_kind {
            EvidenceKind::TestResult | EvidenceKind::CiResult | EvidenceKind::RavsOutcome => {
                verification.evidence_kind
            }
            _ => EvidenceKind::TestResult,
        };
        let evidence = with_evidence_id(EvidenceRef {
            evidence_id: String::new(),
            kind: ev_kind,
            source_ref: verification.source_ref.clone(),
            digest: verification.digest.clone(),
            locator: verification.name.clone(),
            trust: TrustLevel::Verified,
            observed_at_ms: if verification.observed_at_ms == 0 {
                obs.observed_at_ms
            } else {
                verification.observed_at_ms
            },
            attributes: BTreeMap::new(),
        })?;
        operations.push(WorkOperation::AddEvidence {
            evidence: evidence.clone(),
        });
        let kind = if ev_kind == EvidenceKind::CiResult {
            NodeKind::CiRun
        } else {
            NodeKind::Test
        };
        let verification_key = if verification.verification_id.is_empty() {
            format!("{}:{}", verification.name, verification.source_ref)
        } else {
            verification.verification_id.clone()
        };
        let verification_id = stable_node_id(kind, repo_id, &verification_key);
        let mut attrs = BTreeMap::new();
        attrs.insert(
            "verification_state".to_string(),
            serde_json::to_value(verification.state)?,
        );
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: verification_id.clone(),
                kind,
                label: verification.name.clone(),
                trust: TrustLevel::Verified,
                status: match verification.state {
                    VerificationState::Passed => WorkStatus::Completed,
                    VerificationState::Failed => WorkStatus::Blocked,
                    _ => WorkStatus::Unknown,
                },
                status_trust: if matches!(
                    verification.state,
                    VerificationState::Passed | VerificationState::Failed
                ) {
                    TrustLevel::Verified
                } else {
                    TrustLevel::Untrusted
                },
                attributes: attrs,
                evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &workstream_id,
            &verification_id,
            EdgeKind::VerifiedBy,
            TrustLevel::Verified,
            obs.observed_at_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ));
        match verification.state {
            VerificationState::Passed => {
                has_passing_verification = true;
                status_evidence_ids.insert(evidence.evidence_id.clone());
            }
            VerificationState::Failed => {
                status_evidence_ids.insert(evidence.evidence_id.clone());
                blocked = true;
                blocked_trust = TrustLevel::Verified;
                let failure_id = stable_node_id(
                    NodeKind::Failure,
                    repo_id,
                    &format!("verification:{verification_id}"),
                );
                operations.push(WorkOperation::UpsertNode {
                    node: WorkNode {
                        node_id: failure_id.clone(),
                        kind: NodeKind::Failure,
                        label: format!("verification failed: {}", verification.name),
                        trust: TrustLevel::Verified,
                        status: WorkStatus::Blocked,
                        status_trust: TrustLevel::Verified,
                        attributes: BTreeMap::new(),
                        evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                        updated_at_ms: obs.observed_at_ms,
                    },
                });
                operations.push(edge_op(
                    &failure_id,
                    &workstream_id,
                    EdgeKind::Blocks,
                    TrustLevel::Verified,
                    obs.observed_at_ms,
                    std::slice::from_ref(&evidence.evidence_id),
                ));
            }
            _ => {}
        }
    }

    for decision in &obs.decisions {
        validate_text("decision", &decision.text, MAX_LABEL_LEN, false)?;
        let evidence = with_evidence_id(EvidenceRef {
            evidence_id: String::new(),
            kind: decision.source_kind,
            source_ref: decision.source_ref.clone(),
            digest: sha256_text(&decision.text),
            locator: String::new(),
            trust: decision.trust,
            observed_at_ms: obs.observed_at_ms,
            attributes: BTreeMap::new(),
        })?;
        operations.push(WorkOperation::AddEvidence {
            evidence: evidence.clone(),
        });
        let decision_key = if decision.decision_id.is_empty() {
            format!("{task_id}:{}", decision.text)
        } else {
            decision.decision_id.clone()
        };
        let decision_id = stable_node_id(NodeKind::Decision, repo_id, &decision_key);
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: decision_id.clone(),
                kind: NodeKind::Decision,
                label: decision.text.clone(),
                trust: decision.trust,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes: BTreeMap::new(),
                evidence_ids: BTreeSet::from([evidence.evidence_id.clone()]),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &decision_id,
            &workstream_id,
            EdgeKind::PartOf,
            decision.trust,
            obs.observed_at_ms,
            std::slice::from_ref(&evidence.evidence_id),
        ));
    }

    for claim in &obs.claims {
        validate_text("claim", &claim.text, MAX_LABEL_LEN, false)?;
        let risk = if claim.risk.is_finite() {
            claim.risk.clamp(0.0, 1.0)
        } else {
            1.0
        };
        // Claim state never manufactures trust. The adapter must explicitly
        // state how the assessment was established (for example WITNESS/RAVS).
        // Raw agent/user statements therefore remain untrusted by default.
        let trust = claim.trust;
        let claim_key = if claim.claim_id.is_empty() {
            format!("{task_id}:{}", claim.text)
        } else {
            claim.claim_id.clone()
        };
        let claim_id = stable_node_id(NodeKind::Claim, repo_id, &claim_key);
        let mut attrs = BTreeMap::new();
        attrs.insert(
            "claim_state".to_string(),
            serde_json::to_value(claim.state)?,
        );
        attrs.insert("risk".to_string(), Value::from(risk));
        attrs.insert(
            "source_ref".to_string(),
            Value::String(claim.source_ref.clone()),
        );
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: claim_id.clone(),
                kind: NodeKind::Claim,
                label: claim.text.clone(),
                trust,
                status: if claim.state == ClaimState::Contradicted {
                    WorkStatus::Blocked
                } else {
                    WorkStatus::Unknown
                },
                status_trust: if claim.state == ClaimState::Contradicted {
                    trust
                } else {
                    TrustLevel::Untrusted
                },
                attributes: attrs,
                evidence_ids: claim.evidence_ids.iter().cloned().collect(),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &claim_id,
            &workstream_id,
            EdgeKind::PartOf,
            trust,
            obs.observed_at_ms,
            &claim.evidence_ids,
        ));
        if claim.state == ClaimState::Contradicted {
            let failure_id =
                stable_node_id(NodeKind::Failure, repo_id, &format!("claim:{claim_id}"));
            operations.push(WorkOperation::UpsertNode {
                node: WorkNode {
                    node_id: failure_id.clone(),
                    kind: NodeKind::Failure,
                    label: format!("contradicted prior claim: {}", bounded(&claim.text, 512)),
                    trust,
                    status: WorkStatus::Blocked,
                    status_trust: trust,
                    attributes: BTreeMap::new(),
                    evidence_ids: claim.evidence_ids.iter().cloned().collect(),
                    updated_at_ms: obs.observed_at_ms,
                },
            });
            operations.push(edge_op(
                &claim_id,
                &failure_id,
                EdgeKind::ContradictedBy,
                trust,
                obs.observed_at_ms,
                &claim.evidence_ids,
            ));
        }
    }

    for lease in &obs.leases {
        validate_text("lease agent", &lease.agent_id, MAX_ID_LEN, false)?;
        if lease.scope_paths.len() + lease.scope_symbols.len() > MAX_SCOPE_ITEMS {
            return Err(WorkGraphError::LimitExceeded(
                "lease scope too large".to_string(),
            ));
        }
        for path in &lease.scope_paths {
            validate_path(path)?;
        }
        let lease_key = if lease.lease_id.is_empty() {
            format!(
                "{}:{}:{:?}:{:?}",
                lease.agent_id, lease.task_id, lease.scope_paths, lease.scope_symbols
            )
        } else {
            lease.lease_id.clone()
        };
        let lease_id = stable_node_id(NodeKind::WorkLease, repo_id, &lease_key);
        let lease_source_ref = if lease.source_ref.is_empty() {
            format!("work-lease:{lease_key}")
        } else {
            lease.source_ref.clone()
        };
        let lease_digest = sha256_json(&(
            lease.agent_id.as_str(),
            lease.task_id.as_str(),
            &lease.scope_paths,
            &lease.scope_symbols,
            lease.expires_at_ms,
        ))?;
        let lease_evidence = with_evidence_id(EvidenceRef {
            evidence_id: String::new(),
            kind: EvidenceKind::RuntimeObservation,
            source_ref: lease_source_ref.clone(),
            digest: lease_digest,
            locator: String::new(),
            trust: TrustLevel::Observed,
            observed_at_ms: obs.observed_at_ms,
            attributes: BTreeMap::new(),
        })?;
        let lease_evidence_id = lease_evidence.evidence_id.clone();
        operations.push(WorkOperation::AddEvidence {
            evidence: lease_evidence,
        });
        let mut attrs = BTreeMap::new();
        attrs.insert(
            "agent_id".to_string(),
            Value::String(lease.agent_id.clone()),
        );
        attrs.insert(
            "task_id".to_string(),
            Value::String(if lease.task_id.is_empty() {
                task_id.clone()
            } else {
                lease.task_id.clone()
            }),
        );
        attrs.insert(
            "scope_paths".to_string(),
            serde_json::to_value(&lease.scope_paths)?,
        );
        attrs.insert(
            "scope_symbols".to_string(),
            serde_json::to_value(&lease.scope_symbols)?,
        );
        attrs.insert(
            "expires_at_ms".to_string(),
            Value::from(lease.expires_at_ms),
        );
        attrs.insert("source_ref".to_string(), Value::String(lease_source_ref));
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: lease_id.clone(),
                kind: NodeKind::WorkLease,
                label: format!("{} work lease", lease.agent_id),
                trust: TrustLevel::Observed,
                status: WorkStatus::InProgress,
                status_trust: TrustLevel::Observed,
                attributes: attrs,
                evidence_ids: BTreeSet::from([lease_evidence_id.clone()]),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &lease_id,
            &workstream_id,
            EdgeKind::PartOf,
            TrustLevel::Observed,
            obs.observed_at_ms,
            std::slice::from_ref(&lease_evidence_id),
        ));
        let lease_agent_id = stable_node_id(NodeKind::Agent, repo_id, &lease.agent_id);
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: lease_agent_id.clone(),
                kind: NodeKind::Agent,
                label: lease.agent_id.clone(),
                trust: TrustLevel::Observed,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes: BTreeMap::new(),
                evidence_ids: BTreeSet::new(),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &lease_agent_id,
            &lease_id,
            EdgeKind::WorksOn,
            TrustLevel::Observed,
            obs.observed_at_ms,
            std::slice::from_ref(&lease_evidence_id),
        ));
    }

    for execution in &obs.model_executions {
        validate_text("provider", &execution.provider, MAX_ID_LEN, false)?;
        validate_text("model", &execution.model, MAX_ID_LEN, false)?;
        let model_key = format!("{}:{}", execution.provider, execution.model);
        let model_id = stable_node_id(NodeKind::Model, repo_id, &model_key);
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: model_id.clone(),
                kind: NodeKind::Model,
                label: model_key,
                trust: TrustLevel::Observed,
                status: WorkStatus::Unknown,
                status_trust: TrustLevel::Untrusted,
                attributes: BTreeMap::new(),
                evidence_ids: BTreeSet::new(),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        let execution_key = if execution.execution_id.is_empty() {
            format!(
                "{}:{}:{}:{}",
                task_id, execution.provider, execution.model, execution.source_ref
            )
        } else {
            execution.execution_id.clone()
        };
        let execution_id = stable_node_id(NodeKind::ModelExecution, repo_id, &execution_key);
        let execution_source_ref = if execution.source_ref.is_empty() {
            format!("model-execution:{execution_key}")
        } else {
            execution.source_ref.clone()
        };
        let execution_digest = sha256_json(&(
            execution.provider.as_str(),
            execution.model.as_str(),
            execution.success,
            execution.latency_ms,
            execution.cost_micro_usd,
        ))?;
        let execution_evidence = with_evidence_id(EvidenceRef {
            evidence_id: String::new(),
            kind: EvidenceKind::RuntimeObservation,
            source_ref: execution_source_ref.clone(),
            digest: execution_digest,
            locator: String::new(),
            trust: TrustLevel::Observed,
            observed_at_ms: obs.observed_at_ms,
            attributes: BTreeMap::new(),
        })?;
        let execution_evidence_id = execution_evidence.evidence_id.clone();
        operations.push(WorkOperation::AddEvidence {
            evidence: execution_evidence,
        });
        let mut attrs = BTreeMap::new();
        attrs.insert(
            "provider".to_string(),
            Value::String(execution.provider.clone()),
        );
        attrs.insert("model".to_string(), Value::String(execution.model.clone()));
        attrs.insert("latency_ms".to_string(), Value::from(execution.latency_ms));
        attrs.insert(
            "cost_micro_usd".to_string(),
            Value::from(execution.cost_micro_usd),
        );
        if let Some(success) = execution.success {
            attrs.insert("success".to_string(), Value::Bool(success));
        }
        attrs.insert(
            "source_ref".to_string(),
            Value::String(execution_source_ref),
        );
        operations.push(WorkOperation::UpsertNode {
            node: WorkNode {
                node_id: execution_id.clone(),
                kind: NodeKind::ModelExecution,
                label: format!("{}:{} execution", execution.provider, execution.model),
                trust: TrustLevel::Observed,
                status: match execution.success {
                    Some(true) => WorkStatus::Completed,
                    Some(false) => WorkStatus::Blocked,
                    None => WorkStatus::Unknown,
                },
                status_trust: if execution.success.is_some() {
                    TrustLevel::Observed
                } else {
                    TrustLevel::Untrusted
                },
                attributes: attrs,
                evidence_ids: BTreeSet::from([execution_evidence_id.clone()]),
                updated_at_ms: obs.observed_at_ms,
            },
        });
        operations.push(edge_op(
            &task_id,
            &model_id,
            EdgeKind::RoutedTo,
            TrustLevel::Observed,
            obs.observed_at_ms,
            std::slice::from_ref(&execution_evidence_id),
        ));
        operations.push(edge_op(
            &execution_id,
            &workstream_id,
            EdgeKind::PartOf,
            TrustLevel::Observed,
            obs.observed_at_ms,
            std::slice::from_ref(&execution_evidence_id),
        ));
        operations.push(edge_op(
            &execution_id,
            &model_id,
            EdgeKind::ProducedBy,
            TrustLevel::Observed,
            obs.observed_at_ms,
            std::slice::from_ref(&execution_evidence_id),
        ));
    }

    let status = if task_explicit_status == WorkStatus::Abandoned {
        WorkStatus::Abandoned
    } else if blocked {
        WorkStatus::Blocked
    } else if task_explicit_status == WorkStatus::Completed {
        if remaining_work.is_empty() && has_passing_verification {
            WorkStatus::Completed
        } else {
            WorkStatus::NeedsVerification
        }
    } else if task_explicit_status != WorkStatus::Unknown {
        task_explicit_status
    } else if meaningful_git
        || !obs.decisions.is_empty()
        || !obs.leases.is_empty()
        || !obs.model_executions.is_empty()
    {
        WorkStatus::InProgress
    } else if had_explicit_task {
        WorkStatus::Planned
    } else {
        WorkStatus::InProgress
    };
    let status_reason = match status {
        WorkStatus::Blocked => "durable conflict or failing verification observed",
        WorkStatus::Completed => "explicit completion with passing verification",
        WorkStatus::NeedsVerification => {
            "completion was stated but independent verification is missing"
        }
        WorkStatus::Planned => "explicit task observed before durable work changes",
        WorkStatus::InProgress => "durable work evidence observed",
        WorkStatus::Abandoned => "task explicitly marked abandoned",
        WorkStatus::Unknown => "insufficient evidence",
    };
    let status_trust = match status {
        WorkStatus::Blocked => blocked_trust,
        WorkStatus::Completed => TrustLevel::Verified,
        WorkStatus::NeedsVerification | WorkStatus::Planned | WorkStatus::Abandoned => task_trust,
        WorkStatus::InProgress if meaningful_git => TrustLevel::Observed,
        WorkStatus::InProgress => task_trust.max(TrustLevel::Inferred),
        WorkStatus::Unknown => TrustLevel::Untrusted,
    };
    operations.push(WorkOperation::SetStatus {
        node_id: workstream_id.clone(),
        status,
        trust: status_trust,
        reason: status_reason.to_string(),
        evidence_ids: status_evidence_ids.clone(),
    });
    operations.push(WorkOperation::SetStatus {
        node_id: task_id,
        status,
        trust: status_trust,
        reason: status_reason.to_string(),
        evidence_ids: status_evidence_ids,
    });

    Ok(WorkEvent {
        event_id: String::new(),
        observed_at_ms: obs.observed_at_ms,
        source_kind: EvidenceKind::RepositoryFact,
        source_ref: format!("repo-observation:{}", obs.branch.head_sha),
        actor_id: obs.agent_id,
        session_id: obs.session_id,
        operations,
    })
}

fn edge_op(
    from: &str,
    to: &str,
    kind: EdgeKind,
    trust: TrustLevel,
    observed_at_ms: i64,
    evidence_ids: &[String],
) -> WorkOperation {
    WorkOperation::UpsertEdge {
        edge: WorkEdge {
            edge_id: stable_edge_id(from, kind, to),
            from_node: from.to_string(),
            to_node: to.to_string(),
            kind,
            trust,
            attributes: BTreeMap::new(),
            evidence_ids: evidence_ids.iter().cloned().collect(),
            updated_at_ms: observed_at_ms,
        },
    }
}

pub fn stable_node_id(kind: NodeKind, repo_id: &str, key: &str) -> String {
    let digest = sha256_text(&format!("node|{}|{repo_id}|{key}", kind.token()));
    format!("{}:{}", kind.token(), &digest[..24])
}

/// Canonical node identity from a kind *token*, for callers outside Rust.
///
/// `stable_node_id` takes a `NodeKind`, which a binding cannot construct from a
/// string without duplicating the token table. Parsing goes through the same
/// serde mapping the persisted format uses, so there is exactly one definition
/// of "what `file` means" in the codebase.
///
/// This exists because the identity function was unreachable from Python and
/// Node: `entroly/repository_intelligence` grew its own free-form `symbol_id`
/// scheme, so a `File` in the work graph and a `FileRecord` in repository
/// intelligence described the same artifact and could not be addressed as the
/// same node. Shared identity is a shared semantic; it belongs here.
pub fn stable_node_id_for_token(
    kind_token: &str,
    repo_id: &str,
    key: &str,
) -> Result<String, WorkGraphError> {
    let kind: NodeKind = serde_json::from_value(serde_json::Value::String(kind_token.to_string()))
        .map_err(|_| WorkGraphError::InvalidInput(format!("unknown node kind: {kind_token}")))?;
    Ok(stable_node_id(kind, repo_id, key))
}

/// Canonical edge identity from a kind token. See `stable_node_id_for_token`.
pub fn stable_edge_id_for_token(
    from: &str,
    kind_token: &str,
    to: &str,
) -> Result<String, WorkGraphError> {
    let kind: EdgeKind = serde_json::from_value(serde_json::Value::String(kind_token.to_string()))
        .map_err(|_| WorkGraphError::InvalidInput(format!("unknown edge kind: {kind_token}")))?;
    Ok(stable_edge_id(from, kind, to))
}

pub fn stable_edge_id(from: &str, kind: EdgeKind, to: &str) -> String {
    let digest = sha256_text(&format!("edge|{from}|{}|{to}", kind.token()));
    format!("edge:{}", &digest[..24])
}

fn stable_evidence_id(evidence: &EvidenceRef) -> Result<String, WorkGraphError> {
    #[derive(Serialize)]
    struct Identity<'a> {
        kind: EvidenceKind,
        source_ref: &'a str,
        digest: &'a str,
        locator: &'a str,
        observed_at_ms: i64,
    }
    let digest = sha256_json(&Identity {
        kind: evidence.kind,
        source_ref: &evidence.source_ref,
        digest: &evidence.digest,
        locator: &evidence.locator,
        observed_at_ms: evidence.observed_at_ms,
    })?;
    Ok(format!("evidence:{}", &digest[..24]))
}

fn with_evidence_id(mut evidence: EvidenceRef) -> Result<EvidenceRef, WorkGraphError> {
    validate_evidence(&mut evidence)?;
    if evidence.evidence_id.is_empty() {
        evidence.evidence_id = stable_evidence_id(&evidence)?;
    }
    Ok(evidence)
}

fn event_commitment(event: &WorkEvent) -> Result<String, WorkGraphError> {
    #[derive(Serialize)]
    struct EventIdentity<'a> {
        observed_at_ms: i64,
        source_kind: EvidenceKind,
        source_ref: &'a str,
        actor_id: &'a str,
        session_id: &'a str,
        operations: &'a [WorkOperation],
    }
    sha256_json(&EventIdentity {
        observed_at_ms: event.observed_at_ms,
        source_kind: event.source_kind,
        source_ref: &event.source_ref,
        actor_id: &event.actor_id,
        session_id: &event.session_id,
        operations: &event.operations,
    })
}

fn handoff_commitment(receipt: &HandoffReceipt) -> Result<String, WorkGraphError> {
    #[derive(Serialize)]
    struct Payload<'a> {
        schema_version: u32,
        repo_id: &'a str,
        graph_revision: u64,
        graph_commitment: &'a str,
        workstream_id: &'a str,
        from_agent: &'a str,
        to_agent: &'a str,
        generated_at_ms: i64,
        node_ids: &'a [String],
        edge_ids: &'a [String],
        evidence_ids: &'a [String],
    }
    sha256_json(&Payload {
        schema_version: receipt.schema_version,
        repo_id: &receipt.repo_id,
        graph_revision: receipt.graph_revision,
        graph_commitment: &receipt.graph_commitment,
        workstream_id: &receipt.workstream_id,
        from_agent: &receipt.from_agent,
        to_agent: &receipt.to_agent,
        generated_at_ms: receipt.generated_at_ms,
        node_ids: &receipt.node_ids,
        edge_ids: &receipt.edge_ids,
        evidence_ids: &receipt.evidence_ids,
    })
}

fn validate_event(event: &mut WorkEvent) -> Result<(), WorkGraphError> {
    validate_text("event_id", &event.event_id, 128, true)?;
    validate_text(
        "event source_ref",
        &event.source_ref,
        MAX_SOURCE_REF_LEN,
        true,
    )?;
    validate_text("event actor_id", &event.actor_id, MAX_ID_LEN, true)?;
    validate_text("event session_id", &event.session_id, MAX_ID_LEN, true)?;
    if event.operations.len() > MAX_OPERATIONS_PER_EVENT {
        return Err(WorkGraphError::LimitExceeded(format!(
            "operations per event exceed {MAX_OPERATIONS_PER_EVENT}"
        )));
    }
    for operation in &mut event.operations {
        match operation {
            WorkOperation::AddEvidence { evidence } => validate_evidence(evidence)?,
            WorkOperation::UpsertNode { node } => validate_node(node)?,
            WorkOperation::UpsertEdge { edge } => validate_edge(edge)?,
            WorkOperation::SetStatus {
                node_id,
                reason,
                evidence_ids,
                ..
            } => {
                validate_text("status node_id", node_id, MAX_ID_LEN, false)?;
                validate_text("status reason", reason, MAX_LABEL_LEN, true)?;
                validate_id_set("status evidence", evidence_ids)?;
            }
            WorkOperation::AttachEvidence {
                node_id,
                evidence_ids,
            } => {
                validate_text("attachment node_id", node_id, MAX_ID_LEN, false)?;
                validate_id_set("attachment evidence", evidence_ids)?;
            }
        }
    }
    Ok(())
}

fn max_trust_for_evidence_kind(kind: EvidenceKind) -> TrustLevel {
    match kind {
        EvidenceKind::TestResult
        | EvidenceKind::CiResult
        | EvidenceKind::Receipt
        | EvidenceKind::RavsOutcome => TrustLevel::Verified,
        EvidenceKind::GitStatus
        | EvidenceKind::GitCommit
        | EvidenceKind::Checkpoint
        | EvidenceKind::Memory
        | EvidenceKind::AgentStatement
        | EvidenceKind::UserStatement
        | EvidenceKind::RepositoryFact
        | EvidenceKind::RuntimeObservation
        | EvidenceKind::Other => TrustLevel::Observed,
    }
}

fn validate_evidence(evidence: &mut EvidenceRef) -> Result<(), WorkGraphError> {
    validate_text("evidence_id", &evidence.evidence_id, MAX_ID_LEN, true)?;
    validate_text(
        "evidence source_ref",
        &evidence.source_ref,
        MAX_SOURCE_REF_LEN,
        true,
    )?;
    validate_text("evidence digest", &evidence.digest, MAX_ID_LEN, true)?;
    validate_text(
        "evidence locator",
        &evidence.locator,
        MAX_SOURCE_REF_LEN,
        true,
    )?;
    if evidence.trust > max_trust_for_evidence_kind(evidence.kind) {
        return Err(WorkGraphError::InvalidInput(format!(
            "evidence kind {:?} cannot directly assert trust {:?}",
            evidence.kind, evidence.trust
        )));
    }
    validate_attributes(&evidence.attributes)?;
    if !evidence.evidence_id.is_empty() {
        let expected = stable_evidence_id(evidence)?;
        if evidence.evidence_id != expected {
            return Err(WorkGraphError::IntegrityMismatch {
                expected,
                actual: evidence.evidence_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_node(node: &WorkNode) -> Result<(), WorkGraphError> {
    validate_text("node_id", &node.node_id, MAX_ID_LEN, false)?;
    validate_text("node label", &node.label, MAX_LABEL_LEN, true)?;
    let expected_prefix = format!("{}:", node.kind.token());
    if !node.node_id.starts_with(&expected_prefix) {
        return Err(WorkGraphError::InvalidInput(format!(
            "node id {:?} does not match node kind {:?}",
            node.node_id, node.kind
        )));
    }
    if node.status != WorkStatus::Unknown && node.status_trust > node.trust {
        return Err(WorkGraphError::InvalidInput(format!(
            "node {} status trust {:?} exceeds node trust {:?}",
            node.node_id, node.status_trust, node.trust
        )));
    }
    validate_attributes(&node.attributes)?;
    validate_id_set("node evidence", &node.evidence_ids)
}

fn validate_edge(edge: &WorkEdge) -> Result<(), WorkGraphError> {
    validate_text("edge_id", &edge.edge_id, MAX_ID_LEN, false)?;
    validate_text("edge from_node", &edge.from_node, MAX_ID_LEN, false)?;
    validate_text("edge to_node", &edge.to_node, MAX_ID_LEN, false)?;
    let expected = stable_edge_id(&edge.from_node, edge.kind, &edge.to_node);
    if edge.edge_id != expected {
        return Err(WorkGraphError::IntegrityMismatch {
            expected,
            actual: edge.edge_id.clone(),
        });
    }
    validate_attributes(&edge.attributes)?;
    validate_id_set("edge evidence", &edge.evidence_ids)
}

fn validate_id_set(name: &str, values: &BTreeSet<String>) -> Result<(), WorkGraphError> {
    if values.len() > MAX_SCOPE_ITEMS {
        return Err(WorkGraphError::LimitExceeded(format!(
            "{name} has too many IDs"
        )));
    }
    for value in values {
        validate_text(name, value, MAX_ID_LEN, false)?;
    }
    Ok(())
}

fn validate_attributes(attributes: &BTreeMap<String, Value>) -> Result<(), WorkGraphError> {
    if attributes.len() > MAX_ATTRIBUTE_KEYS {
        return Err(WorkGraphError::LimitExceeded(format!(
            "attribute key count exceeds {MAX_ATTRIBUTE_KEYS}"
        )));
    }
    for key in attributes.keys() {
        validate_text("attribute key", key, 256, false)?;
    }
    let bytes = serde_json::to_vec(attributes)?;
    if bytes.len() > MAX_ATTRIBUTE_BYTES {
        return Err(WorkGraphError::LimitExceeded(format!(
            "attribute payload exceeds {MAX_ATTRIBUTE_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_path(path: &str) -> Result<(), WorkGraphError> {
    validate_text("path", path, 8_192, false)?;
    if path.starts_with('/') || path.starts_with('\\') || path.contains('\0') {
        return Err(WorkGraphError::InvalidInput(format!(
            "work-graph paths must be repository-relative: {path:?}"
        )));
    }
    let normalized = path.replace('\\', "/");
    if normalized.split('/').any(|part| part == "..") {
        return Err(WorkGraphError::InvalidInput(format!(
            "work-graph path escapes repository: {path:?}"
        )));
    }
    Ok(())
}

fn validate_text(
    name: &str,
    text: &str,
    max_len: usize,
    allow_empty: bool,
) -> Result<(), WorkGraphError> {
    if !allow_empty && text.is_empty() {
        return Err(WorkGraphError::InvalidInput(format!(
            "{name} must not be empty"
        )));
    }
    if text.len() > max_len {
        return Err(WorkGraphError::LimitExceeded(format!(
            "{name} exceeds {max_len} bytes"
        )));
    }
    if text.contains('\0') {
        return Err(WorkGraphError::InvalidInput(format!("{name} contains NUL")));
    }
    Ok(())
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, WorkGraphError> {
    let bytes = serde_json::to_vec(value)?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_text(text: &str) -> String {
    sha256_bytes(text.as_bytes())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn bounded(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

fn attr_string(attrs: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    attrs.get(key)?.as_str().map(ToString::to_string)
}

fn attr_i64(attrs: &BTreeMap<String, Value>, key: &str) -> Option<i64> {
    attrs.get(key)?.as_i64()
}

fn attr_strings(attrs: &BTreeMap<String, Value>, key: &str) -> Vec<String> {
    attrs
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn normalize_scope_path(path: &str) -> String {
    path.replace('\\', "/").trim_matches('/').to_string()
}

fn paths_overlap(a: &str, b: &str) -> bool {
    let a = normalize_scope_path(a);
    let b = normalize_scope_path(b);
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a == b || a.starts_with(&(b.clone() + "/")) || b.starts_with(&(a.clone() + "/"))
}

fn overlap_paths(a: &[String], b: &[String]) -> Vec<String> {
    let mut result = BTreeSet::new();
    for left in a {
        for right in b {
            if paths_overlap(left, right) {
                let left_n = normalize_scope_path(left);
                let right_n = normalize_scope_path(right);
                result.insert(if left_n.len() >= right_n.len() {
                    left_n
                } else {
                    right_n
                });
            }
        }
    }
    result.into_iter().collect()
}

fn overlap_exact(a: &[String], b: &[String]) -> Vec<String> {
    let left: BTreeSet<&str> = a.iter().map(String::as_str).collect();
    let right: BTreeSet<&str> = b.iter().map(String::as_str).collect();
    left.intersection(&right)
        .map(|s| (*s).to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_observation() -> RepositoryObservation {
        RepositoryObservation {
            repo_id: "repo-1".to_string(),
            observed_at_ms: 1_000,
            repository_label: "repo".to_string(),
            agent_id: String::new(),
            session_id: String::new(),
            task_hint: None,
            branch: BranchObservation {
                name: "main".to_string(),
                head_sha: "abc".to_string(),
                default_branch: "main".to_string(),
                ..Default::default()
            },
            changes: vec![],
            commits: vec![],
            verifications: vec![],
            decisions: vec![],
            claims: vec![],
            leases: vec![],
            model_executions: vec![],
        }
    }

    fn passive_dirty_observation(digest: &str, observed_at_ms: i64) -> RepositoryObservation {
        let mut obs = clean_observation();
        obs.observed_at_ms = observed_at_ms;
        obs.branch.name = "feature/passive".to_string();
        obs.changes.push(FileChangeObservation {
            path: "src/auth.rs".to_string(),
            kind: FileChangeKind::Modified,
            staged: false,
            conflicted: false,
            old_path: String::new(),
            content_digest: digest.to_string(),
        });
        obs
    }

    #[test]
    fn identical_content_complete_passive_snapshots_do_not_grow_history() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let first = graph
            .observe_repository(passive_dirty_observation(
                "git-blob:1111111111111111111111111111111111111111",
                1_000,
            ))
            .unwrap();
        let commitment = graph.graph_commitment().to_string();
        let second = graph
            .observe_repository(passive_dirty_observation(
                "git-blob:1111111111111111111111111111111111111111",
                9_000,
            ))
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(graph.event_count(), 1);
        assert_eq!(graph.graph_commitment(), commitment);
    }

    fn canonical_full_graph_commitment(graph: &WorkGraph) -> String {
        #[derive(Serialize)]
        struct Commitment<'a> {
            schema_version: u32,
            repo_id: &'a str,
            events: &'a [WorkEvent],
        }
        sha256_json(&Commitment {
            schema_version: WORK_GRAPH_SCHEMA_VERSION,
            repo_id: &graph.repo_id,
            events: &graph.events,
        })
        .unwrap()
    }

    #[test]
    fn incremental_commitment_matches_canonical_serde_bytes() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        assert_eq!(
            graph.graph_commitment(),
            canonical_full_graph_commitment(&graph)
        );
        for i in 0..512u64 {
            let digest = format!("git-blob:{:040x}", i + 1);
            graph
                .observe_repository(passive_dirty_observation(&digest, 10_000 + i as i64))
                .unwrap();
            assert_eq!(
                graph.graph_commitment(),
                canonical_full_graph_commitment(&graph),
                "incremental commitment diverged after append {i}"
            );
        }

        let compact = graph.export_json(false).unwrap();
        let restored = WorkGraph::from_json(&compact).unwrap();
        assert_eq!(restored.graph_commitment(), graph.graph_commitment());
        assert_eq!(
            restored.graph_commitment(),
            canonical_full_graph_commitment(&restored)
        );
    }

    #[test]
    fn derived_event_id_index_tracks_append_dedupe_and_import() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        for i in 0..2_048u64 {
            let digest = format!("git-blob:{:040x}", i + 1);
            graph
                .observe_repository(passive_dirty_observation(&digest, 1_000 + i as i64))
                .unwrap();
        }
        assert_eq!(graph.event_ids.len(), graph.events.len());
        assert_eq!(graph.event_ids.len(), 2_048);

        let duplicate = graph.events[1_024].clone();
        let duplicate_id = duplicate.event_id.clone();
        let before = graph.graph_commitment().to_string();
        assert_eq!(graph.apply_event(duplicate).unwrap(), duplicate_id);
        assert_eq!(graph.event_count(), 2_048);
        assert_eq!(graph.event_ids.len(), 2_048);
        assert_eq!(graph.graph_commitment(), before);

        let restored = WorkGraph::from_json(&graph.export_json(false).unwrap()).unwrap();
        assert_eq!(restored.event_ids.len(), restored.events.len());
        assert_eq!(restored.event_ids, graph.event_ids);
        assert_eq!(restored.graph_commitment(), graph.graph_commitment());
    }

    #[test]
    fn passive_snapshot_byte_change_appends_new_event() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        graph
            .observe_repository(passive_dirty_observation(
                "git-blob:1111111111111111111111111111111111111111",
                1_000,
            ))
            .unwrap();
        graph
            .observe_repository(passive_dirty_observation(
                "git-blob:2222222222222222222222222222222222222222",
                2_000,
            ))
            .unwrap();
        assert_eq!(graph.event_count(), 2);
    }

    #[test]
    fn passive_change_away_and_back_remains_auditable() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        for (time, digest) in [
            (1_000, "git-blob:1111111111111111111111111111111111111111"),
            (2_000, "git-blob:2222222222222222222222222222222222222222"),
            (3_000, "git-blob:1111111111111111111111111111111111111111"),
        ] {
            graph
                .observe_repository(passive_dirty_observation(digest, time))
                .unwrap();
        }
        assert_eq!(graph.event_count(), 3);
    }

    #[test]
    fn passive_snapshot_scope_event_is_one_idempotent_poll_group() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let first =
            passive_dirty_observation("git-blob:1111111111111111111111111111111111111111", 1_000);
        graph.observe_repository(first.clone()).unwrap();
        let snapshot_source = graph.events.last().unwrap().source_ref.clone();
        graph
            .apply_event(WorkEvent {
                event_id: String::new(),
                observed_at_ms: 1_000,
                source_kind: EvidenceKind::RepositoryFact,
                source_ref: format!("{snapshot_source}:scope"),
                actor_id: String::new(),
                session_id: String::new(),
                operations: Vec::new(),
            })
            .unwrap();
        assert_eq!(graph.event_count(), 2);

        let mut repeated = first;
        repeated.observed_at_ms = 2_000;
        graph.observe_repository(repeated.clone()).unwrap();
        assert_eq!(graph.event_count(), 2);

        let mut restored = WorkGraph::from_json(&graph.export_json(false).unwrap()).unwrap();
        repeated.observed_at_ms = 3_000;
        restored.observe_repository(repeated).unwrap();
        assert_eq!(restored.event_count(), 2);

        restored
            .observe_repository(passive_dirty_observation(
                "git-blob:2222222222222222222222222222222222222222",
                4_000,
            ))
            .unwrap();
        assert_eq!(restored.event_count(), 3);
    }

    #[test]
    fn passive_snapshot_without_complete_digest_never_dedupes() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        graph
            .observe_repository(passive_dirty_observation("", 1_000))
            .unwrap();
        graph
            .observe_repository(passive_dirty_observation("", 2_000))
            .unwrap();
        assert_eq!(graph.event_count(), 2);
    }

    #[test]
    fn rename_lineage_keeps_new_symbol_scope_without_stale_changed_path() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut observation = clean_observation();
        observation.branch.name = "feature/rename".into();
        observation.branch.ahead_by = 1;
        observation.changes = vec![FileChangeObservation {
            path: "src/new_name.rs".into(),
            old_path: "src/old_name.rs".into(),
            kind: FileChangeKind::Renamed,
            staged: true,
            conflicted: false,
            content_digest: "git-blob:1111111111111111111111111111111111111111".into(),
        }];
        graph.observe_repository(observation).unwrap();

        let new_file_id = stable_node_id(NodeKind::File, "repo-1", "src/new_name.rs");
        let old_file_id = stable_node_id(NodeKind::File, "repo-1", "src/old_name.rs");
        let symbol_id = stable_node_id(
            NodeKind::Symbol,
            "repo-1",
            "src/new_name.rs::new_symbol::function",
        );
        graph
            .apply_event(WorkEvent {
                event_id: String::new(),
                observed_at_ms: 1_100,
                source_kind: EvidenceKind::RepositoryFact,
                source_ref: "repository-intelligence:repo-1".into(),
                actor_id: String::new(),
                session_id: String::new(),
                operations: vec![
                    WorkOperation::UpsertNode {
                        node: WorkNode {
                            node_id: symbol_id.clone(),
                            kind: NodeKind::Symbol,
                            label: "new_symbol".into(),
                            trust: TrustLevel::Observed,
                            status: WorkStatus::Unknown,
                            status_trust: TrustLevel::Untrusted,
                            attributes: BTreeMap::from([
                                ("path".into(), Value::String("src/new_name.rs".into())),
                                (
                                    "symbol_id".into(),
                                    Value::String("src/new_name.rs::new_symbol::function".into()),
                                ),
                            ]),
                            evidence_ids: BTreeSet::new(),
                            updated_at_ms: 1_100,
                        },
                    },
                    edge_op(
                        &new_file_id,
                        &symbol_id,
                        EdgeKind::Defines,
                        TrustLevel::Observed,
                        1_100,
                        &[],
                    ),
                ],
            })
            .unwrap();

        let workstream_id = graph.unfinished_work()[0].node_id.clone();
        let resume = graph.resume(Some(&workstream_id), 128).unwrap();
        assert_eq!(resume.changed_paths, vec!["src/new_name.rs"]);
        assert_eq!(
            resume.selected_workstream.symbol_ids,
            vec![symbol_id.clone()]
        );
        let scope = graph.context_scope(Some(&workstream_id), 128).unwrap();
        assert_eq!(scope.changed_paths, vec!["src/new_name.rs"]);
        assert_eq!(scope.symbol_ids, vec![symbol_id]);

        assert_eq!(
            attr_string(&graph.nodes[&old_file_id].attributes, "renamed_to"),
            Some("src/new_name.rs".into())
        );
        assert!(graph.edges.values().any(|edge| {
            edge.from_node == new_file_id
                && edge.to_node == old_file_id
                && edge.kind == EdgeKind::Supersedes
        }));
    }

    #[test]
    fn large_dirty_observation_is_bounded_atomic_and_poll_idempotent() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut observation = clean_observation();
        observation.branch.name = "feature/large-dirty".into();
        observation.branch.ahead_by = 1;
        observation.changes = (0..2_000)
            .map(|index| FileChangeObservation {
                path: format!("src/module_{index:05}.rs"),
                kind: FileChangeKind::Modified,
                staged: false,
                conflicted: false,
                old_path: String::new(),
                content_digest: format!("git-blob:{index:064x}"),
            })
            .collect();

        graph.observe_repository(observation.clone()).unwrap();
        assert_eq!(graph.event_count(), 4);
        assert_eq!(graph.unfinished_work().len(), 1);
        let commitment = graph.graph_commitment().to_string();

        graph.observe_repository(observation.clone()).unwrap();
        assert_eq!(graph.event_count(), 4);
        assert_eq!(graph.graph_commitment(), commitment);

        let mut restored = WorkGraph::from_json(&graph.export_json(false).unwrap()).unwrap();
        restored.observe_repository(observation).unwrap();
        assert_eq!(restored.event_count(), 4);
        assert_eq!(restored.graph_commitment(), commitment);

        let before = restored.export_json(false).unwrap();
        let mut oversized = clean_observation();
        oversized.changes = (0..=MAX_CHANGES_PER_OBSERVATION)
            .map(|index| FileChangeObservation {
                path: format!("oversized/{index:05}.rs"),
                kind: FileChangeKind::Modified,
                staged: false,
                conflicted: false,
                old_path: String::new(),
                content_digest: format!("git-blob:{index:064x}"),
            })
            .collect();
        assert!(matches!(
            restored.observe_repository(oversized),
            Err(WorkGraphError::LimitExceeded(message))
                if message.contains("file changes per observation")
        ));
        assert_eq!(restored.export_json(false).unwrap(), before);
    }

    #[test]
    fn repeated_active_verification_is_never_collapsed() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut first =
            passive_dirty_observation("git-blob:1111111111111111111111111111111111111111", 1_000);
        first.verifications.push(VerificationObservation {
            verification_id: "test:repeat".to_string(),
            name: "focused test".to_string(),
            state: VerificationState::Passed,
            evidence_kind: EvidenceKind::TestResult,
            source_ref: "pytest:test_repeat".to_string(),
            digest: "pass".to_string(),
            observed_at_ms: 1_000,
        });
        let mut second = first.clone();
        second.observed_at_ms = 2_000;
        second.verifications[0].observed_at_ms = 2_000;
        graph.observe_repository(first).unwrap();
        graph.observe_repository(second).unwrap();
        assert_eq!(graph.event_count(), 2);
    }

    #[test]
    fn repeated_model_execution_is_never_collapsed() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut first =
            passive_dirty_observation("git-blob:1111111111111111111111111111111111111111", 1_000);
        first.model_executions.push(ModelExecutionObservation {
            execution_id: "exec:repeat".to_string(),
            provider: "provider".to_string(),
            model: "model".to_string(),
            success: Some(true),
            latency_ms: 10,
            cost_micro_usd: 1,
            source_ref: "runtime:repeat".to_string(),
        });
        let mut second = first.clone();
        second.observed_at_ms = 2_000;
        graph.observe_repository(first).unwrap();
        graph.observe_repository(second).unwrap();
        assert_eq!(graph.event_count(), 2);
    }

    #[test]
    fn clean_repo_is_null_control() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        graph.observe_repository(clean_observation()).unwrap();
        assert!(graph.unfinished_work().is_empty());
        assert_eq!(
            graph.nodes.len(),
            1,
            "clean observation should create only repository node"
        );
    }

    #[test]
    fn explicit_task_without_durable_changes_is_planned() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Plan auth work".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::Checkpoint,
            explicit_status: WorkStatus::Unknown,
            remaining_work: vec!["implement refresh".to_string()],
            source_ref: "checkpoint:latest".to_string(),
        });
        graph.observe_repository(obs).unwrap();
        let unfinished = graph.unfinished_work();
        assert_eq!(unfinished.len(), 1);
        assert_eq!(unfinished[0].status, WorkStatus::Planned);
        assert!(graph
            .evidence
            .values()
            .any(|e| e.kind == EvidenceKind::Checkpoint));
    }

    #[test]
    fn dirty_repo_creates_in_progress_workstream() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.branch.name = "feature/auth".to_string();
        obs.agent_id = "claude".to_string();
        obs.changes.push(FileChangeObservation {
            path: "src/auth.rs".to_string(),
            kind: FileChangeKind::Modified,
            staged: false,
            conflicted: false,
            old_path: String::new(),
            content_digest: String::new(),
        });
        graph.observe_repository(obs).unwrap();
        let unfinished = graph.unfinished_work();
        assert_eq!(unfinished.len(), 1);
        assert_eq!(unfinished[0].status, WorkStatus::InProgress);
        assert!(unfinished[0]
            .changed_paths
            .contains(&"src/auth.rs".to_string()));
    }

    #[test]
    fn completion_requires_verification() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Finish auth".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::Checkpoint,
            explicit_status: WorkStatus::Completed,
            remaining_work: vec![],
            source_ref: "checkpoint:1".to_string(),
        });
        graph.observe_repository(obs.clone()).unwrap();
        assert_eq!(
            graph.unfinished_work()[0].status,
            WorkStatus::NeedsVerification
        );

        obs.observed_at_ms += 1;
        obs.verifications.push(VerificationObservation {
            verification_id: String::new(),
            name: "auth tests".to_string(),
            state: VerificationState::Passed,
            evidence_kind: EvidenceKind::TestResult,
            source_ref: "pytest:test_auth".to_string(),
            digest: "pass".to_string(),
            observed_at_ms: obs.observed_at_ms,
        });
        graph.observe_repository(obs).unwrap();
        assert!(graph.unfinished_work().is_empty());
        assert!(graph
            .nodes
            .values()
            .any(|n| n.kind == NodeKind::Task && n.status == WorkStatus::Completed));
    }

    #[test]
    fn failing_verification_blocks_work() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Fix stream".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::UserStatement,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec![],
            source_ref: "user".to_string(),
        });
        obs.verifications.push(VerificationObservation {
            verification_id: String::new(),
            name: "stream tests".to_string(),
            state: VerificationState::Failed,
            evidence_kind: EvidenceKind::TestResult,
            source_ref: "pytest:test_stream".to_string(),
            digest: "failure".to_string(),
            observed_at_ms: 1_000,
        });
        graph.observe_repository(obs).unwrap();
        let item = &graph.unfinished_work()[0];
        assert_eq!(item.status, WorkStatus::Blocked);
        assert!(!item.failure_ids.is_empty());
    }

    #[test]
    fn contradicted_claim_never_becomes_trusted_fact() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Inspect auth".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::UserStatement,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec![],
            source_ref: "user".to_string(),
        });
        obs.claims.push(ClaimObservation {
            claim_id: String::new(),
            text: "passwords are plaintext".to_string(),
            state: ClaimState::Contradicted,
            trust: TrustLevel::Untrusted,
            risk: 0.98,
            source_ref: "witness:cert".to_string(),
            evidence_ids: vec![],
        });
        graph.observe_repository(obs).unwrap();
        let claim = graph
            .nodes
            .values()
            .find(|n| n.kind == NodeKind::Claim)
            .unwrap();
        assert_eq!(claim.trust, TrustLevel::Untrusted);
        assert_eq!(claim.status, WorkStatus::Blocked);
    }

    #[test]
    fn overlapping_parallel_leases_are_reported_but_not_locked() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: "task:shared".to_string(),
            title: "Parallel work".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::UserStatement,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec![],
            source_ref: "user".to_string(),
        });
        obs.leases = vec![
            WorkLeaseObservation {
                lease_id: "lease:a".to_string(),
                agent_id: "claude".to_string(),
                task_id: "task:shared".to_string(),
                scope_paths: vec!["src/auth".to_string()],
                scope_symbols: vec!["Auth.refresh".to_string()],
                expires_at_ms: 5_000,
                source_ref: "agent".to_string(),
            },
            WorkLeaseObservation {
                lease_id: "lease:b".to_string(),
                agent_id: "codex".to_string(),
                task_id: "task:shared".to_string(),
                scope_paths: vec!["src/auth/token.rs".to_string()],
                scope_symbols: vec!["Auth.refresh".to_string()],
                expires_at_ms: 5_000,
                source_ref: "agent".to_string(),
            },
        ];
        graph.observe_repository(obs).unwrap();
        let report = graph.coordination_report(2_000);
        assert_eq!(report.active_leases, 2);
        assert_eq!(report.conflicts.len(), 1);
        assert_eq!(
            report.conflicts[0].overlapping_symbols,
            vec!["Auth.refresh"]
        );
    }

    /// Section 19 scenario E: two agents on disjoint paths and symbols must
    /// produce no conflict at all.
    ///
    /// This asserts it on the **shipping** coordination path. The only prior
    /// evidence for scenario E lived in `coordination_index`, which `lib.rs`
    /// declares `#[cfg(test)] mod` -- 308 lines that never reach a binding, so
    /// they cannot speak for what users actually run.
    ///
    /// A false conflict is not cosmetic: it tells two agents who could safely
    /// work in parallel that they cannot, which is the failure that makes
    /// coordination worth less than no coordination at all.
    #[test]
    fn disjoint_parallel_leases_produce_no_conflict() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: "task:shared".to_string(),
            title: "Parallel work".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::UserStatement,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec![],
            source_ref: "user".to_string(),
        });
        obs.leases = vec![
            WorkLeaseObservation {
                lease_id: "lease:a".to_string(),
                agent_id: "claude".to_string(),
                task_id: "task:shared".to_string(),
                scope_paths: vec!["src/auth".to_string()],
                scope_symbols: vec!["Auth.refresh".to_string()],
                expires_at_ms: 5_000,
                source_ref: "agent".to_string(),
            },
            WorkLeaseObservation {
                lease_id: "lease:b".to_string(),
                agent_id: "codex".to_string(),
                task_id: "task:shared".to_string(),
                scope_paths: vec!["src/billing".to_string()],
                scope_symbols: vec!["Invoice.total".to_string()],
                expires_at_ms: 5_000,
                source_ref: "agent".to_string(),
            },
        ];
        graph.observe_repository(obs).unwrap();

        let report = graph.coordination_report(2_000);
        assert_eq!(report.active_leases, 2, "both leases must still be active");
        assert!(
            report.conflicts.is_empty(),
            "disjoint scopes reported a conflict: {:?}",
            report.conflicts
        );
    }

    /// A sibling path that merely shares a textual prefix is not an overlap.
    ///
    /// `src/auth` and `src/auth.py` are different artifacts, and a bare
    /// `starts_with` would also collide `src/auth` with `src/authorization.rs`.
    /// `paths_overlap` requires the `/` boundary, which is correct -- and which
    /// a test in `tests/test_work_graph_multiprocess.py` contradicted until this
    /// session by asserting a conflict between exactly this pair (finding G15).
    #[test]
    fn prefix_sibling_paths_are_not_treated_as_overlapping() {
        assert!(!paths_overlap("src/auth", "src/auth.py"));
        assert!(!paths_overlap("src/auth", "src/authorization.rs"));
        assert!(!paths_overlap("src/auth", "src/authz/token.rs"));

        // Genuine containment must still hold.
        assert!(paths_overlap("src/auth", "src/auth/token.rs"));
        assert!(paths_overlap("src/auth/token.rs", "src/auth"));
        assert!(paths_overlap("src/auth", "src/auth"));

        // Separator style and stray slashes must not change the answer.
        assert!(paths_overlap(r"src\auth", "src/auth/token.rs"));
        assert!(paths_overlap("/src/auth/", "src/auth"));
        assert!(!paths_overlap("", "src/auth"));
    }

    #[test]
    fn event_merge_is_commutative_and_deduplicated() {
        let mut a = WorkGraph::new("repo-1").unwrap();
        let mut b = WorkGraph::new("repo-1").unwrap();
        let mut one = clean_observation();
        one.observed_at_ms = 2_000;
        one.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Task one".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::UserStatement,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec![],
            source_ref: "user".to_string(),
        });
        let mut two = one.clone();
        two.observed_at_ms = 1_000;
        two.task_hint.as_mut().unwrap().title = "Task two".to_string();
        a.observe_repository(one.clone()).unwrap();
        b.observe_repository(two.clone()).unwrap();
        a.merge(&b).unwrap();
        b.merge(&WorkGraph::from_json(&a.export_json(false).unwrap()).unwrap())
            .unwrap();
        assert_eq!(a.graph_commitment(), b.graph_commitment());
        assert_eq!(a.event_count(), 2);
    }

    #[test]
    fn persisted_document_detects_tampering() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Trust me".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::UserStatement,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec![],
            source_ref: "user".to_string(),
        });
        graph.observe_repository(obs).unwrap();
        let json = graph.export_json(false).unwrap();
        let restored = WorkGraph::from_json(&json).unwrap();
        assert_eq!(restored.graph_commitment(), graph.graph_commitment());
        let tampered = json.replace("Trust me", "Trust nobody");
        assert!(matches!(
            WorkGraph::from_json(&tampered),
            Err(WorkGraphError::IntegrityMismatch { .. })
        ));
    }

    #[test]
    fn handoff_commitment_is_stable_and_detects_mutation() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Handoff task".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::Checkpoint,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec!["finish tests".to_string()],
            source_ref: "checkpoint".to_string(),
        });
        graph.observe_repository(obs).unwrap();
        let id = graph.unfinished_work()[0].node_id.clone();
        let mut receipt = graph
            .handoff_receipt(&id, "claude", "codex", 2_000)
            .unwrap();
        assert!(WorkGraph::verify_handoff_receipt(&receipt).unwrap());
        receipt.to_agent = "other".to_string();
        assert!(!WorkGraph::verify_handoff_receipt(&receipt).unwrap());
    }

    /// A successor must be shown the claims, ranked, with their trust.
    ///
    /// `resume` returned decisions, failures and verification and silently
    /// omitted claims entirely. Handing a repository to a second agent with no
    /// handoff surfaced the cost: a durable, engine-verified claim reading
    /// "this work is INCOMPLETE" with risk 0.9 was stored, correctly edged to
    /// its workstream, and never shown. The agent picking the work up had to
    /// reconstruct the raw event log by hand to find it.
    ///
    /// Trust travels with the text because a bare string cannot separate "the
    /// tests prove this" from "the previous agent believed this", and showing
    /// the second as the first is the fail-open direction here.
    #[test]
    fn resume_surfaces_claims_ranked_by_risk_with_trust() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Fix expired refresh".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::Checkpoint,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec![],
            source_ref: "checkpoint".to_string(),
        });
        obs.claims.push(ClaimObservation {
            claim_id: String::new(),
            text: "low risk aside".to_string(),
            state: ClaimState::Unsupported,
            trust: TrustLevel::Inferred,
            risk: 0.10,
            source_ref: "agent".to_string(),
            evidence_ids: vec![],
        });
        obs.claims.push(ClaimObservation {
            claim_id: String::new(),
            text: "this work is INCOMPLETE".to_string(),
            state: ClaimState::Grounded,
            trust: TrustLevel::Verified,
            risk: 0.90,
            source_ref: "pytest".to_string(),
            evidence_ids: vec![],
        });
        graph.observe_repository(obs).unwrap();

        let view = graph.resume(None, 128).unwrap();

        assert_eq!(view.claims.len(), 2, "claims must reach the successor");
        // Highest risk first: a truncated read must not lose the one that matters.
        assert_eq!(view.claims[0].label, "this work is INCOMPLETE");
        assert_eq!(view.claims[0].risk, Some(0.90));
        assert_eq!(view.claims[0].trust, TrustLevel::Verified);
        assert_eq!(view.claims[0].claim_state.as_deref(), Some("grounded"));
        // The weaker claim keeps its weaker trust rather than being levelled up.
        assert_eq!(view.claims[1].trust, TrustLevel::Inferred);
    }

    #[test]
    fn resume_prioritizes_verified_evidence() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: String::new(),
            title: "Resume me".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::Checkpoint,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec!["one".to_string()],
            source_ref: "checkpoint".to_string(),
        });
        obs.verifications.push(VerificationObservation {
            verification_id: String::new(),
            name: "focused test".to_string(),
            state: VerificationState::Passed,
            evidence_kind: EvidenceKind::TestResult,
            source_ref: "pytest".to_string(),
            digest: "abc".to_string(),
            observed_at_ms: 1_000,
        });
        graph.observe_repository(obs).unwrap();
        let resume = graph.resume(None, 20).unwrap();
        assert!(!resume.evidence.is_empty());
        assert_eq!(resume.evidence[0].trust, TrustLevel::Verified);
    }

    #[test]
    fn lower_trust_observation_cannot_downgrade_verified_completion() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut completed = clean_observation();
        completed.task_hint = Some(TaskHint {
            task_id: "stable-task".to_string(),
            title: "Finish stable task".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::Checkpoint,
            explicit_status: WorkStatus::Completed,
            remaining_work: vec![],
            source_ref: "checkpoint:complete".to_string(),
        });
        completed.verifications.push(VerificationObservation {
            verification_id: "verification:stable".to_string(),
            name: "stable tests".to_string(),
            state: VerificationState::Passed,
            evidence_kind: EvidenceKind::TestResult,
            source_ref: "pytest:stable".to_string(),
            digest: "pass".to_string(),
            observed_at_ms: completed.observed_at_ms,
        });
        graph.observe_repository(completed).unwrap();
        let workstream_id = graph
            .nodes
            .values()
            .find(|node| node.kind == NodeKind::Workstream)
            .unwrap()
            .node_id
            .clone();
        assert_eq!(graph.nodes[&workstream_id].status, WorkStatus::Completed);
        assert_eq!(
            graph.nodes[&workstream_id].status_trust,
            TrustLevel::Verified
        );

        let mut stale = clean_observation();
        stale.observed_at_ms += 1;
        stale.task_hint = Some(TaskHint {
            task_id: "stable-task".to_string(),
            title: "Finish stable task".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::Checkpoint,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec!["stale note".to_string()],
            source_ref: "checkpoint:stale".to_string(),
        });
        graph.observe_repository(stale).unwrap();
        assert_eq!(graph.nodes[&workstream_id].status, WorkStatus::Completed);
        assert_eq!(
            graph.nodes[&workstream_id].status_trust,
            TrustLevel::Verified
        );
    }

    #[test]
    fn sibling_workstreams_do_not_leak_through_shared_agent_hub() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        for (offset, task_id, title, path) in [
            (0, "task-a", "Task A", "src/a.rs"),
            (1, "task-b", "Task B", "src/b.rs"),
        ] {
            let mut obs = clean_observation();
            obs.observed_at_ms += offset;
            obs.agent_id = "shared-agent".to_string();
            obs.task_hint = Some(TaskHint {
                task_id: task_id.to_string(),
                title: title.to_string(),
                trust: TrustLevel::Observed,
                source_kind: EvidenceKind::UserStatement,
                explicit_status: WorkStatus::InProgress,
                remaining_work: vec![],
                source_ref: format!("user:{task_id}"),
            });
            obs.changes.push(FileChangeObservation {
                path: path.to_string(),
                kind: FileChangeKind::Modified,
                staged: false,
                conflicted: false,
                old_path: String::new(),
                content_digest: format!("digest-{task_id}"),
            });
            graph.observe_repository(obs).unwrap();
        }
        let workstream_a = graph
            .nodes
            .values()
            .find(|node| node.kind == NodeKind::Workstream && node.label == "Task A")
            .unwrap();
        let view = graph.work_item_view(workstream_a);
        assert!(view.changed_paths.contains(&"src/a.rs".to_string()));
        assert!(!view.changed_paths.contains(&"src/b.rs".to_string()));
    }

    #[test]
    fn semantically_unordered_observations_have_identical_commitments() {
        let mut left = clean_observation();
        left.task_hint = Some(TaskHint {
            task_id: "ordered-task".to_string(),
            title: "Order-independent task".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::UserStatement,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec!["b".to_string(), "a".to_string()],
            source_ref: "user".to_string(),
        });
        left.changes = vec![
            FileChangeObservation {
                path: "src/z.rs".to_string(),
                kind: FileChangeKind::Modified,
                staged: false,
                conflicted: false,
                old_path: String::new(),
                content_digest: "z".to_string(),
            },
            FileChangeObservation {
                path: "src/a.rs".to_string(),
                kind: FileChangeKind::Added,
                staged: true,
                conflicted: false,
                old_path: String::new(),
                content_digest: "a".to_string(),
            },
        ];
        let mut right = left.clone();
        right.changes.reverse();
        right.task_hint.as_mut().unwrap().remaining_work.reverse();

        let mut graph_left = WorkGraph::new("repo-1").unwrap();
        let mut graph_right = WorkGraph::new("repo-1").unwrap();
        let event_left = graph_left.observe_repository(left).unwrap();
        let event_right = graph_right.observe_repository(right).unwrap();
        assert_eq!(event_left, event_right);
        assert_eq!(
            graph_left.graph_commitment(),
            graph_right.graph_commitment()
        );
    }

    #[test]
    fn persisted_graph_rejects_duplicate_events_even_if_payload_is_otherwise_valid() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: "duplicate-task".to_string(),
            title: "Duplicate task".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::UserStatement,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec![],
            source_ref: "user".to_string(),
        });
        graph.observe_repository(obs).unwrap();
        let mut value: Value = serde_json::from_str(&graph.export_json(false).unwrap()).unwrap();
        let events = value["events"].as_array_mut().unwrap();
        events.push(events[0].clone());
        let duplicate_json = serde_json::to_string(&value).unwrap();
        assert!(matches!(
            WorkGraph::from_json(&duplicate_json),
            Err(WorkGraphError::InvalidInput(message)) if message.contains("duplicate event id")
        ));
    }

    #[test]
    fn status_without_sufficient_supporting_evidence_is_rejected() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let node_id = stable_node_id(NodeKind::Task, "repo-1", "unsupported");
        let event = WorkEvent {
            event_id: String::new(),
            observed_at_ms: 1,
            source_kind: EvidenceKind::Other,
            source_ref: "test".to_string(),
            actor_id: String::new(),
            session_id: String::new(),
            operations: vec![WorkOperation::UpsertNode {
                node: WorkNode {
                    node_id,
                    kind: NodeKind::Task,
                    label: "unsupported".to_string(),
                    trust: TrustLevel::Verified,
                    status: WorkStatus::Completed,
                    status_trust: TrustLevel::Verified,
                    attributes: BTreeMap::new(),
                    evidence_ids: BTreeSet::new(),
                    updated_at_ms: 1,
                },
            }],
        };
        assert!(matches!(
            graph.apply_event(event),
            Err(WorkGraphError::InvalidInput(_))
        ));
    }

    #[test]
    fn graph_bound_handoff_verification_rejects_stale_or_foreign_snapshots() {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: "handoff-bound".to_string(),
            title: "Bound handoff".to_string(),
            trust: TrustLevel::Observed,
            source_kind: EvidenceKind::Checkpoint,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec!["finish".to_string()],
            source_ref: "checkpoint".to_string(),
        });
        graph.observe_repository(obs).unwrap();
        let workstream_id = graph.unfinished_work()[0].node_id.clone();
        let receipt = graph
            .handoff_receipt(&workstream_id, "claude", "codex", 2_000)
            .unwrap();
        assert!(graph
            .verify_handoff_receipt_against_graph(&receipt)
            .unwrap());

        let mut newer = clean_observation();
        newer.observed_at_ms = 3_000;
        graph.observe_repository(newer).unwrap();
        assert!(WorkGraph::verify_handoff_receipt(&receipt).unwrap());
        assert!(!graph
            .verify_handoff_receipt_against_graph(&receipt)
            .unwrap());

        let foreign = WorkGraph::new("repo-2").unwrap();
        assert!(!foreign
            .verify_handoff_receipt_against_graph(&receipt)
            .unwrap());
    }

    fn graph_with_execution_scope() -> (WorkGraph, String, String) {
        let mut graph = WorkGraph::new("repo-1").unwrap();
        let mut obs = clean_observation();
        obs.task_hint = Some(TaskHint {
            task_id: "execution-task".to_string(),
            title: "Execute verified route".to_string(),
            trust: TrustLevel::Observed,
            explicit_status: WorkStatus::InProgress,
            remaining_work: vec!["run model and verify output".to_string()],
            source_kind: EvidenceKind::UserStatement,
            source_ref: "user:task".to_string(),
        });
        obs.changes.push(FileChangeObservation {
            path: "src/auth.rs".to_string(),
            kind: FileChangeKind::Modified,
            staged: false,
            conflicted: false,
            old_path: String::new(),
            content_digest: "git-blob:1111111111111111111111111111111111111111".to_string(),
        });
        graph.observe_repository(obs).unwrap();
        let item = graph.unfinished_work()[0].clone();
        (graph, item.node_id, item.task_ids[0].clone())
    }

    fn execution_contracts(
        workstream_id: &str,
        task_id: &str,
        verified_head: &str,
    ) -> (RoutingDecision, ModelExecutionOutcome, VerificationRecord) {
        let route = RoutingDecision::new(
            "repo-1".into(),
            task_id.into(),
            workstream_id.into(),
            "openai".into(),
            "gpt-5".into(),
            "responses-api".into(),
            8_192,
            "policy:v1".into(),
            vec!["capability_match".into()],
            vec!["sha256:features".into()],
            vec![],
            "cr_example".into(),
            vec!["evidence:route".into()],
            1_100,
        )
        .unwrap();
        let outcome = ModelExecutionOutcome::new(
            route.routing_id.clone(),
            "repo-1".into(),
            task_id.into(),
            workstream_id.into(),
            "openai".into(),
            "gpt-5".into(),
            "responses-api".into(),
            "cr_example".into(),
            "sha256:request".into(),
            "sha256:response".into(),
            ExecutionState::Succeeded,
            crate::engine_contracts::OutcomeVerificationState::Passed,
            100,
            1_000,
            100,
            500,
            String::new(),
            vec!["evidence:outcome".into()],
            1_200,
        )
        .unwrap();
        let verification = VerificationRecord::new(
            "repo-1".into(),
            outcome.outcome_id.clone(),
            outcome.outcome_commitment.clone(),
            verified_head.into(),
            VerificationVerdict::Passed,
            vec!["evidence:test".into()],
            vec!["sha256:source".into()],
            1_300,
            0,
        )
        .unwrap();
        (route, outcome, verification)
    }

    #[test]
    fn execution_chain_closes_work_context_execution_and_trust_atomically() {
        let (mut graph, workstream_id, task_id) = graph_with_execution_scope();
        let (route, outcome, verification) = execution_contracts(&workstream_id, &task_id, "abc");
        graph
            .record_execution_chain(
                route.clone(),
                outcome.clone(),
                verification.clone(),
                BTreeSet::new(),
            )
            .unwrap();

        let snapshot: serde_json::Value =
            serde_json::from_str(&graph.snapshot_json(false).unwrap()).unwrap();
        let nodes = snapshot["nodes"].as_array().unwrap();
        assert!(nodes.iter().any(|node| {
            node["attributes"]["routing_id"] == serde_json::json!(route.routing_id)
        }));
        assert!(nodes.iter().any(|node| {
            node["attributes"]["outcome_id"] == serde_json::json!(outcome.outcome_id)
        }));
        assert!(nodes.iter().any(|node| {
            node["attributes"]["verification_id"] == serde_json::json!(verification.verification_id)
                && node["attributes"]["freshness"] == "current"
        }));
        assert_ne!(
            graph.unfinished_work()[0].status,
            WorkStatus::NeedsVerification
        );

        let handoff = graph
            .handoff_receipt(&workstream_id, "claude", "codex", 1_400)
            .unwrap();
        let proof = graph
            .continuation_proof(
                &handoff,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec!["run Linux CI".into()],
                vec![],
                1_500,
            )
            .unwrap();
        assert_eq!(proof.routing_commitments, vec![route.decision_commitment]);
        assert_eq!(
            proof.execution_outcome_commitments,
            vec![outcome.outcome_commitment]
        );
        assert_eq!(
            proof.verification_commitments,
            vec![verification.record_commitment]
        );
        assert_eq!(
            proof.state_for_graph("repo-1", graph.revision(), graph.graph_commitment()),
            crate::engine_contracts::ContinuationProofState::Valid
        );
    }

    #[test]
    fn interrupted_agent_gets_evidence_bounded_continuation_without_a_handoff() {
        let (graph, workstream_id, _) = graph_with_execution_scope();
        let proof = graph
            .reconstructed_continuation_proof(
                &workstream_id,
                "agent:codex",
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec!["run targeted tests".into()],
                vec![],
                1_500,
            )
            .unwrap();
        assert!(proof.from_agent.is_empty());
        assert!(proof.handoff_commitment.is_empty());
        assert!(proof
            .outstanding_work_refs
            .contains(&"unknown:previous-agent-intent".to_string()));
        assert_eq!(
            proof.state_for_graph("repo-1", graph.revision(), graph.graph_commitment()),
            crate::engine_contracts::ContinuationProofState::Valid
        );

        let completed = {
            let mut graph = graph;
            let evidence = with_evidence_id(EvidenceRef {
                evidence_id: String::new(),
                kind: EvidenceKind::TestResult,
                source_ref: "test:complete".into(),
                digest: "sha256:passing".into(),
                locator: String::new(),
                trust: TrustLevel::Verified,
                observed_at_ms: 1_600,
                attributes: BTreeMap::new(),
            })
            .unwrap();
            graph
                .apply_event(WorkEvent {
                    event_id: String::new(),
                    observed_at_ms: 1_600,
                    source_kind: EvidenceKind::TestResult,
                    source_ref: "test:complete".into(),
                    actor_id: String::new(),
                    session_id: String::new(),
                    operations: vec![
                        WorkOperation::AddEvidence {
                            evidence: evidence.clone(),
                        },
                        WorkOperation::SetStatus {
                            node_id: workstream_id.clone(),
                            status: WorkStatus::Completed,
                            trust: TrustLevel::Verified,
                            reason: "verified complete".into(),
                            evidence_ids: BTreeSet::from([evidence.evidence_id]),
                        },
                    ],
                })
                .unwrap();
            graph
        };
        assert!(matches!(
            completed.reconstructed_continuation_proof(
                &workstream_id,
                "agent:codex",
                vec![], vec![], vec![], vec![], vec![], vec!["x".into()], vec![], 1_700,
            ),
            Err(WorkGraphError::InvalidInput(message)) if message.contains("finished work")
        ));
    }

    #[test]
    fn stale_or_transitively_invalidated_verification_cannot_upgrade_work() {
        for (verified_head, invalidated) in [
            ("older-head", BTreeSet::new()),
            ("abc", BTreeSet::from(["sha256:source".to_string()])),
        ] {
            let (mut graph, workstream_id, task_id) = graph_with_execution_scope();
            let (route, outcome, verification) =
                execution_contracts(&workstream_id, &task_id, verified_head);
            graph
                .record_execution_chain(route, outcome, verification, invalidated)
                .unwrap();
            assert_eq!(
                graph.unfinished_work()[0].status,
                WorkStatus::NeedsVerification
            );
        }
    }

    #[test]
    fn context_receipt_and_memory_become_bounded_graph_evidence() {
        let (mut graph, workstream_id, task_id) = graph_with_execution_scope();
        let receipt = ContextReceiptEnvelope::new(
            "repo-1".into(),
            "abc".into(),
            graph.graph_commitment().into(),
            workstream_id.clone(),
            "sha256:sources".into(),
            vec!["src/auth.rs#0:20".into()],
            vec!["src/auth.rs#20:40".into()],
            vec!["evidence:test".into()],
            vec!["src/auth.rs#20:40".into()],
            vec!["rh_example".into()],
            vec!["evidence:test".into()],
            512,
            "work-scope/v1".into(),
            "execution:pending".into(),
            1_100,
        )
        .unwrap();
        graph
            .record_context_receipt(receipt.clone(), "agent:claude".into(), "session:1".into())
            .unwrap();
        assert!(graph
            .snapshot_json(false)
            .unwrap()
            .contains(&receipt.receipt_id));
        assert!(matches!(
            graph.record_context_receipt(
                receipt.clone(),
                "agent:claude".into(),
                "session:1".into()
            ),
            Err(WorkGraphError::IntegrityMismatch { .. })
        ));

        let stale_receipt = ContextReceiptEnvelope::new(
            "repo-1".into(),
            "older-head".into(),
            graph.graph_commitment().into(),
            workstream_id.clone(),
            "sha256:sources".into(),
            vec!["src/auth.rs#0:20".into()],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            512,
            "work-scope/v1".into(),
            "execution:pending".into(),
            1_150,
        )
        .unwrap();
        assert!(matches!(
            graph.record_context_receipt(
                stale_receipt,
                "agent:claude".into(),
                "session:1".into()
            ),
            Err(WorkGraphError::IntegrityMismatch { expected, actual })
                if expected == "abc" && actual == "older-head"
        ));

        let memory = MemoryRecord::new(
            "repo-1".into(),
            task_id,
            workstream_id.clone(),
            "agent:claude".into(),
            "session:1".into(),
            "execution:1".into(),
            "vault/auth-decision".into(),
            "sha256:memory-content".into(),
            vec!["evidence:test".into()],
            TrustLevel::Observed,
            1_200,
            1_200,
            0,
            vec![],
            vec![],
            "rh_memory".into(),
        )
        .unwrap();
        graph
            .record_memory(memory.clone(), 1_300, BTreeSet::new())
            .unwrap();
        let snapshot = graph.snapshot_json(false).unwrap();
        assert!(snapshot.contains(&memory.memory_id));
        assert!(snapshot.contains("\"admissibility\":\"admissible\""));
        assert!(!snapshot.contains("memory-content-bytes"));

        let handoff = graph
            .handoff_receipt(&workstream_id, "agent:claude", "agent:codex", 1_400)
            .unwrap();
        let proof = graph
            .continuation_proof(
                &handoff,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec!["run package tests".into()],
                vec![],
                1_500,
            )
            .unwrap();
        assert_eq!(
            proof.context_receipt_commitments,
            vec![receipt.receipt_commitment]
        );
        assert_eq!(proof.memory_commitments, vec![memory.record_commitment]);
        assert_eq!(
            proof.recovery_handle_ids,
            vec!["rh_example".to_string(), "rh_memory".to_string()]
        );
        assert!(matches!(
            graph.continuation_proof(
                &handoff,
                vec!["sha256:invented".into()],
                vec![],
                vec![],
                vec![],
                vec![],
                vec!["run package tests".into()],
                vec![],
                1_500,
            ),
            Err(WorkGraphError::InvalidInput(message))
                if message.contains("not evidenced by this workstream")
        ));
    }
}
