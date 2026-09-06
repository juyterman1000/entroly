"""
Governance Domain Model — Canonical entities for the enterprise control plane.
================================================================================

25 strongly-typed entities with stable IDs, timestamps, and correlation IDs.
Every entity supports serialization to dict/JSON for audit logging, provenance
recording, and cross-system integration.

Design principles:
  - Frozen dataclasses for immutability (events are facts, not mutable state)
  - String enums for wire-safe discriminators
  - UUID-based stable IDs (generated at creation, not assigned externally)
  - Every entity carries ``created_at`` and ``correlation_id``
  - ``to_dict()`` on every entity for audit/provenance serialization
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


# ── Helpers ──────────────────────────────────────────────────────────

def _new_id() -> str:
    """Generate a stable, sortable unique ID."""
    return str(uuid.uuid4())


def _now() -> float:
    return time.time()


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ── Enumerations ─────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionVerdict(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"
    REQUIRE_APPROVAL = "require_approval"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OutcomeStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    REVERTED = "reverted"
    UNKNOWN = "unknown"


class EvidenceKind(str, Enum):
    TEST_RUN = "test_run"
    SAST_SCAN = "sast_scan"
    DEPENDENCY_SCAN = "dependency_scan"
    BUILD = "build"
    TYPE_CHECK = "type_check"
    POLICY_CHECK = "policy_check"
    WITNESS_CHECK = "witness_check"
    MANUAL_REVIEW = "manual_review"
    DIFF_ANALYSIS = "diff_analysis"
    RUNTIME_CHECK = "runtime_check"


class SecuritySeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ProvenanceRelation(str, Enum):
    INITIATED = "initiated"
    USED = "used"
    READ = "read"
    WROTE = "wrote"
    INVOKED = "invoked"
    DERIVED_FROM = "derived_from"
    VERIFIED_BY = "verified_by"
    APPROVED_BY = "approved_by"
    BLOCKED_BY = "blocked_by"
    DEPLOYED_AS = "deployed_as"
    RESULTED_IN = "resulted_in"
    COSTED = "costed"
    AFFECTED = "affected"


# ── Core Identity Entities ───────────────────────────────────────────

@dataclass(frozen=True)
class AgentIdentity:
    """Cryptographically verifiable agent identity.

    Represents the full identity chain: organization → user → agent → session.
    HMAC-signed locally with the operator's key (no network calls).
    """
    agent_id: str
    agent_type: str                     # "claude-code" | "codex" | "cursor" | "human" | etc.
    organization: str = ""              # tenant/org boundary
    user: str = ""                      # human delegator
    team: str = ""                      # organizational unit
    session_id: str = ""
    model: str = ""                     # underlying LLM model
    scopes: frozenset[str] = frozenset()
    identity_token: str = ""            # HMAC-SHA256 signature
    created_at: float = field(default_factory=_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        return bool(self.identity_token)

    @property
    def is_human(self) -> bool:
        return self.agent_type == "human"

    def has_scope(self, scope: str) -> bool:
        if "admin" in self.scopes:
            return True
        if scope in self.scopes:
            return True
        action, _, path = scope.partition(":")
        if not path:
            return action in self.scopes
        for s in self.scopes:
            s_action, _, s_path = s.partition(":")
            if s_action == action and s_path and path.startswith(s_path):
                return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "organization": self.organization,
            "user": self.user,
            "team": self.team,
            "session_id": self.session_id,
            "model": self.model,
            "scopes": sorted(self.scopes),
            "identity_token": self.identity_token,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AgentIdentity":
        return cls(
            agent_id=str(data.get("agent_id", "")),
            agent_type=str(data.get("agent_type", "unknown")),
            organization=str(data.get("organization", "")),
            user=str(data.get("user", "")),
            team=str(data.get("team", "")),
            session_id=str(data.get("session_id", "")),
            model=str(data.get("model", "")),
            scopes=frozenset(data.get("scopes", [])),
            identity_token=str(data.get("identity_token", "")),
            created_at=float(data.get("created_at", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def anonymous(cls, *, agent_type: str = "unknown") -> "AgentIdentity":
        return cls(
            agent_id="anonymous",
            agent_type=agent_type,
            scopes=frozenset({"read"}),
        )


@dataclass(frozen=True)
class Agent:
    """Registered agent definition (independent of any session)."""
    id: str = field(default_factory=_new_id)
    name: str = ""
    agent_type: str = "unknown"
    organization: str = ""
    version: str = ""
    capabilities: frozenset[str] = frozenset()
    default_model: str = ""
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "agent_type": self.agent_type,
            "organization": self.organization, "version": self.version,
            "capabilities": sorted(self.capabilities),
            "default_model": self.default_model, "created_at": self.created_at,
        }


@dataclass(frozen=True)
class Session:
    """Agent working session."""
    id: str = field(default_factory=_new_id)
    agent_identity: AgentIdentity = field(default_factory=AgentIdentity.anonymous)
    task_id: str = ""
    started_at: float = field(default_factory=_now)
    correlation_id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "agent_identity": self.agent_identity.to_dict(),
            "task_id": self.task_id, "started_at": self.started_at,
            "correlation_id": self.correlation_id,
        }


@dataclass(frozen=True)
class Task:
    """Unit of work (feature, bugfix, review, deployment)."""
    id: str = field(default_factory=_new_id)
    title: str = ""
    intent: str = ""                    # "feature" | "bugfix" | "refactor" | "review" | etc.
    initiated_by: str = ""              # user ID
    organization: str = ""
    repository: str = ""
    created_at: float = field(default_factory=_now)
    correlation_id: str = field(default_factory=_new_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "intent": self.intent,
            "initiated_by": self.initiated_by, "organization": self.organization,
            "repository": self.repository, "created_at": self.created_at,
            "correlation_id": self.correlation_id,
        }


# ── Authorization Entities ───────────────────────────────────────────

@dataclass(frozen=True)
class Policy:
    """Versioned policy-as-code for authorization decisions.

    Evaluates: subject × action × resource × context × risk × environment.
    """
    id: str = field(default_factory=_new_id)
    name: str = ""
    version: str = "1"
    agent_type_pattern: str = "*"       # glob pattern
    allowed_scopes: frozenset[str] = frozenset({"read"})
    denied_paths: tuple[str, ...] = ()
    requires_approval_for: frozenset[str] = frozenset()
    max_risk_level: RiskLevel = RiskLevel.HIGH
    max_files_per_change: int = 50
    max_lines_per_change: int = 5000
    budget_limit_usd: float = 100.0
    conditions: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "version": self.version,
            "agent_type_pattern": self.agent_type_pattern,
            "allowed_scopes": sorted(self.allowed_scopes),
            "denied_paths": list(self.denied_paths),
            "requires_approval_for": sorted(self.requires_approval_for),
            "max_risk_level": self.max_risk_level.value,
            "max_files_per_change": self.max_files_per_change,
            "max_lines_per_change": self.max_lines_per_change,
            "budget_limit_usd": self.budget_limit_usd,
            "conditions": dict(self.conditions),
            "created_at": self.created_at,
        }


GovernancePolicy = Policy



@dataclass(frozen=True)
class Permission:
    """Evaluated permission result."""
    id: str = field(default_factory=_new_id)
    identity: AgentIdentity = field(default_factory=AgentIdentity.anonymous)
    scope: str = ""
    resource: str = ""
    allowed: bool = False
    policy_id: str = ""
    policy_version: str = ""
    reason: str = ""
    requires_approval: bool = False
    evaluated_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "agent_id": self.identity.agent_id,
            "agent_type": self.identity.agent_type,
            "scope": self.scope, "resource": self.resource,
            "allowed": self.allowed, "policy_id": self.policy_id,
            "policy_version": self.policy_version, "reason": self.reason,
            "requires_approval": self.requires_approval,
            "evaluated_at": self.evaluated_at,
        }


@dataclass(frozen=True)
class Approval:
    """Human approval record."""
    id: str = field(default_factory=_new_id)
    decision_id: str = ""
    approver: str = ""                  # user ID of approver
    status: ApprovalStatus = ApprovalStatus.PENDING
    reason: str = ""
    created_at: float = field(default_factory=_now)
    expires_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "decision_id": self.decision_id,
            "approver": self.approver, "status": self.status.value,
            "reason": self.reason, "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


# ── Execution Entities ───────────────────────────────────────────────

@dataclass(frozen=True)
class Resource:
    """File, API, database, or service being accessed."""
    id: str = field(default_factory=_new_id)
    resource_type: str = "file"         # "file" | "api" | "database" | "service"
    path: str = ""
    repository: str = ""
    sensitivity: str = "normal"         # "normal" | "sensitive" | "critical"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "resource_type": self.resource_type,
            "path": self.path, "repository": self.repository,
            "sensitivity": self.sensitivity,
        }


@dataclass(frozen=True)
class Tool:
    """MCP tool, skill, or capability."""
    id: str = field(default_factory=_new_id)
    name: str = ""
    tool_type: str = "mcp_tool"         # "mcp_tool" | "skill" | "plugin" | "builtin"
    server_id: str = ""                 # MCPServer ID if hosted externally
    schema_hash: str = ""               # SHA-256 of tool schema
    permissions_required: frozenset[str] = frozenset()
    trust_level: str = "untrusted"      # "trusted" | "verified" | "untrusted" | "revoked"
    first_seen_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "tool_type": self.tool_type,
            "server_id": self.server_id, "schema_hash": self.schema_hash,
            "permissions_required": sorted(self.permissions_required),
            "trust_level": self.trust_level,
            "first_seen_at": self.first_seen_at,
        }


@dataclass(frozen=True)
class MCPServer:
    """External MCP server identity and trust level."""
    id: str = field(default_factory=_new_id)
    name: str = ""
    uri: str = ""
    manifest_hash: str = ""
    tool_count: int = 0
    trust_level: str = "untrusted"
    last_scanned_at: float = 0.0
    vulnerabilities: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "uri": self.uri,
            "manifest_hash": self.manifest_hash, "tool_count": self.tool_count,
            "trust_level": self.trust_level,
            "last_scanned_at": self.last_scanned_at,
            "vulnerabilities": self.vulnerabilities,
        }


@dataclass(frozen=True)
class ToolInvocation:
    """Logged tool call with inputs, outputs, and governance metadata."""
    id: str = field(default_factory=_new_id)
    tool_id: str = ""
    tool_name: str = ""
    session_id: str = ""
    agent_identity: AgentIdentity = field(default_factory=AgentIdentity.anonymous)
    permission_id: str = ""             # Permission that authorized this
    inputs_hash: str = ""               # SHA-256 of inputs (not raw inputs)
    outputs_hash: str = ""              # SHA-256 of outputs
    duration_ms: float = 0.0
    success: bool = True
    error: str = ""
    created_at: float = field(default_factory=_now)
    correlation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "tool_id": self.tool_id, "tool_name": self.tool_name,
            "session_id": self.session_id,
            "agent_id": self.agent_identity.agent_id,
            "permission_id": self.permission_id,
            "inputs_hash": self.inputs_hash, "outputs_hash": self.outputs_hash,
            "duration_ms": self.duration_ms, "success": self.success,
            "error": self.error, "created_at": self.created_at,
            "correlation_id": self.correlation_id,
        }


@dataclass(frozen=True)
class Intent:
    """Classified purpose of an agent action."""
    id: str = field(default_factory=_new_id)
    intent_type: str = "unknown"        # "feature" | "bugfix" | "refactor" | "security" | etc.
    description: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "intent_type": self.intent_type,
            "description": self.description, "confidence": self.confidence,
        }


# ── Verification Entities ────────────────────────────────────────────

@dataclass(frozen=True)
class Change:
    """Code diff or artifact change produced by an agent."""
    id: str = field(default_factory=_new_id)
    session_id: str = ""
    agent_identity: AgentIdentity = field(default_factory=AgentIdentity.anonymous)
    files_added: tuple[str, ...] = ()
    files_modified: tuple[str, ...] = ()
    files_deleted: tuple[str, ...] = ()
    lines_added: int = 0
    lines_removed: int = 0
    diff_hash: str = ""
    intent: str = ""
    created_at: float = field(default_factory=_now)
    correlation_id: str = ""

    @property
    def file_count(self) -> int:
        return len(self.files_added) + len(self.files_modified) + len(self.files_deleted)

    @property
    def line_count(self) -> int:
        return self.lines_added + self.lines_removed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "session_id": self.session_id,
            "agent_id": self.agent_identity.agent_id,
            "files_added": list(self.files_added),
            "files_modified": list(self.files_modified),
            "files_deleted": list(self.files_deleted),
            "lines_added": self.lines_added, "lines_removed": self.lines_removed,
            "diff_hash": self.diff_hash, "intent": self.intent,
            "created_at": self.created_at, "correlation_id": self.correlation_id,
        }


@dataclass(frozen=True)
class Evidence:
    """Concrete verification artifact (test run, scan, build result)."""
    id: str = field(default_factory=_new_id)
    kind: EvidenceKind = EvidenceKind.TEST_RUN
    change_id: str = ""
    status: VerificationStatus = VerificationStatus.PENDING
    summary: str = ""                   # e.g. "4281 tests passed, 0 failed"
    details: Mapping[str, Any] = field(default_factory=dict)
    content_hash: str = ""              # SHA-256 of evidence content
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind.value,
            "change_id": self.change_id, "status": self.status.value,
            "summary": self.summary, "details": dict(self.details),
            "content_hash": self.content_hash, "created_at": self.created_at,
        }


@dataclass(frozen=True)
class Verification:
    """Composite verification result aggregating multiple Evidence items."""
    id: str = field(default_factory=_new_id)
    change_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    status: VerificationStatus = VerificationStatus.PENDING
    confidence: float = 0.0             # 0.0–1.0 composite
    summary: str = ""
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "change_id": self.change_id,
            "evidence_ids": list(self.evidence_ids),
            "status": self.status.value, "confidence": self.confidence,
            "summary": self.summary, "created_at": self.created_at,
        }


@dataclass(frozen=True)
class RiskAssessment:
    """Multi-signal risk score with human-readable explanation."""
    id: str = field(default_factory=_new_id)
    change_id: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0             # 0–100 normalized
    signals: Mapping[str, float] = field(default_factory=dict)
    explanation: str = ""
    recommended_action: str = ""        # "auto_merge" | "lightweight_review" | etc.
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "change_id": self.change_id,
            "risk_level": self.risk_level.value, "risk_score": self.risk_score,
            "signals": dict(self.signals), "explanation": self.explanation,
            "recommended_action": self.recommended_action,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class SecurityFinding:
    """Vulnerability or threat detection."""
    id: str = field(default_factory=_new_id)
    severity: SecuritySeverity = SecuritySeverity.INFO
    category: str = ""                  # "prompt_injection" | "cwe-79" | "dependency" | etc.
    source: str = ""                    # "acf" | "sast" | "supply_chain" | etc.
    file: str = ""
    line: int = 0
    description: str = ""
    remediation: str = ""
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "severity": self.severity.value,
            "category": self.category, "source": self.source,
            "file": self.file, "line": self.line,
            "description": self.description, "remediation": self.remediation,
            "created_at": self.created_at,
        }


# ── Decision Entities ────────────────────────────────────────────────

@dataclass(frozen=True)
class Decision:
    """Gate verdict: allow, deny, or escalate — with full reasoning."""
    id: str = field(default_factory=_new_id)
    change_id: str = ""
    verification_id: str = ""
    risk_assessment_id: str = ""
    policy_id: str = ""
    policy_version: str = ""
    verdict: DecisionVerdict = DecisionVerdict.DENY
    reason: str = ""
    evidence_summary: str = ""
    identity: AgentIdentity = field(default_factory=AgentIdentity.anonymous)
    approval_id: str = ""               # set if human approval was obtained
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "change_id": self.change_id,
            "verification_id": self.verification_id,
            "risk_assessment_id": self.risk_assessment_id,
            "policy_id": self.policy_id, "policy_version": self.policy_version,
            "verdict": self.verdict.value, "reason": self.reason,
            "evidence_summary": self.evidence_summary,
            "agent_id": self.identity.agent_id,
            "approval_id": self.approval_id, "created_at": self.created_at,
        }


# ── Traceability Entities ───────────────────────────────────────────

@dataclass(frozen=True)
class ProvenanceEvent:
    """Immutable node in the provenance DAG.

    Supports typed relationships: INITIATED, USED, READ, WROTE, INVOKED,
    DERIVED_FROM, VERIFIED_BY, APPROVED_BY, BLOCKED_BY, DEPLOYED_AS,
    RESULTED_IN, COSTED, AFFECTED.
    """
    id: str = field(default_factory=_new_id)
    event_type: str = ""                # matches entity type name
    relation: ProvenanceRelation = ProvenanceRelation.USED
    actor_id: str = ""                  # agent or user who caused this
    subject_id: str = ""                # entity being acted upon
    parent_ids: tuple[str, ...] = ()    # DAG edges to parent events
    content_hash: str = ""              # hash of the event content
    details: Mapping[str, Any] = field(default_factory=dict)
    session_id: str = ""
    correlation_id: str = ""
    trace_id: str = ""
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "event_type": self.event_type,
            "relation": self.relation.value,
            "actor_id": self.actor_id, "subject_id": self.subject_id,
            "parent_ids": list(self.parent_ids),
            "content_hash": self.content_hash, "details": dict(self.details),
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "trace_id": self.trace_id, "created_at": self.created_at,
        }


@dataclass(frozen=True)
class Deployment:
    """Promotion / deployment event."""
    id: str = field(default_factory=_new_id)
    change_id: str = ""
    decision_id: str = ""
    environment: str = ""               # "staging" | "production"
    deployed_by: str = ""               # agent or user ID
    deployed_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "change_id": self.change_id,
            "decision_id": self.decision_id, "environment": self.environment,
            "deployed_by": self.deployed_by, "deployed_at": self.deployed_at,
        }


@dataclass(frozen=True)
class Outcome:
    """Post-deployment or post-action result."""
    id: str = field(default_factory=_new_id)
    change_id: str = ""
    deployment_id: str = ""
    status: OutcomeStatus = OutcomeStatus.UNKNOWN
    defect_count: int = 0
    rollback: bool = False
    value_usd: float = 0.0             # estimated business value
    observed_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "change_id": self.change_id,
            "deployment_id": self.deployment_id, "status": self.status.value,
            "defect_count": self.defect_count, "rollback": self.rollback,
            "value_usd": self.value_usd, "observed_at": self.observed_at,
        }


# ── Economics Entities ───────────────────────────────────────────────

@dataclass(frozen=True)
class CostEvent:
    """Token/compute/time cost attribution."""
    id: str = field(default_factory=_new_id)
    session_id: str = ""
    agent_identity: AgentIdentity = field(default_factory=AgentIdentity.anonymous)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    compute_ms: float = 0.0
    cost_usd: float = 0.0
    task_id: str = ""
    change_id: str = ""
    outcome_id: str = ""                # links cost to verified outcome
    created_at: float = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "session_id": self.session_id,
            "agent_id": self.agent_identity.agent_id,
            "model": self.model, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "tool_calls": self.tool_calls,
            "compute_ms": self.compute_ms, "cost_usd": self.cost_usd,
            "task_id": self.task_id, "change_id": self.change_id,
            "outcome_id": self.outcome_id, "created_at": self.created_at,
        }


__all__ = [
    # Enums
    "RiskLevel", "DecisionVerdict", "VerificationStatus", "ApprovalStatus",
    "OutcomeStatus", "EvidenceKind", "SecuritySeverity", "ProvenanceRelation",
    # Core entities
    "AgentIdentity", "Agent", "Session", "Task",
    # Authorization
    "Policy", "Permission", "Approval",
    # Execution
    "Resource", "Tool", "MCPServer", "ToolInvocation", "Intent",
    # Verification
    "Change", "Evidence", "Verification", "RiskAssessment", "SecurityFinding",
    # Decision
    "Decision",
    # Traceability
    "ProvenanceEvent", "Deployment", "Outcome",
    # Economics
    "CostEvent",
]
