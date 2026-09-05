"""
Governance Policy Engine — ABAC policy evaluation for the control plane.
=========================================================================

Evaluates authorization decisions using Attribute-Based Access Control
(ABAC), considering:
  - subject (agent identity: type, organization, team, scopes)
  - action (requested scope)
  - resource (path, type, sensitivity)
  - context (risk level, environment, time, approval state)

Policies are versioned YAML-as-code loaded from:
  1. ENTROLY_POLICY_FILE environment variable
  2. ~/.entroly/governance/policies.yaml
  3. Built-in default deny-by-default policy

Every policy decision records: policy_id, version, inputs, decision,
reason, and timestamp — for the audit trail.

Design:
  - Deny by default
  - Explicit allow required
  - Human approval required for critical scopes
  - Policy decisions are explainable (human-readable reason)
  - Policies are versioned (v1, v2, ...)
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml  # type: ignore[import-untyped]
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from .domain import (
    AgentIdentity,
    Permission,
    Policy,
    RiskLevel,
    _new_id,
    _now,
)
from .events import (
    GovernanceEventBus,
    PermissionDeniedEvent,
    PermissionEvaluatedEvent,
    TracingContext,
    get_event_bus,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_POLICY_FILE_ENV = "ENTROLY_POLICY_FILE"
_DEFAULT_POLICY_PATH = Path("~/.entroly/governance/policies.yaml").expanduser()
POLICY_SCHEMA_VERSION = "entroly.governance.policy.v1"


# ── Exceptions ───────────────────────────────────────────────────────

class PolicyError(ValueError):
    """Raised when a policy is invalid or cannot be loaded."""


class PolicyDeniedError(PermissionError):
    """Raised when a policy denies an action and the caller requires allow."""


# ── Built-in Policies ────────────────────────────────────────────────

_READ_ONLY_POLICY = Policy(
    id="builtin-read-only",
    name="built-in: deny-by-default (read only)",
    version="1",
    agent_type_pattern="*",
    allowed_scopes=frozenset({"read"}),
    denied_paths=(),
    requires_approval_for=frozenset({"deploy", "admin", "execute", "write"}),
    max_risk_level=RiskLevel.LOW,
    max_files_per_change=0,
    max_lines_per_change=0,
    budget_limit_usd=0.0,
)

_DEVELOPER_POLICY = Policy(
    id="builtin-developer",
    name="built-in: developer (read + write, no deploy)",
    version="1",
    agent_type_pattern="*",
    allowed_scopes=frozenset({"read", "write", "execute", "review"}),
    denied_paths=(".env", "*.pem", "*.key", "secrets/**"),
    requires_approval_for=frozenset({"deploy", "admin"}),
    max_risk_level=RiskLevel.HIGH,
    max_files_per_change=50,
    max_lines_per_change=5000,
    budget_limit_usd=100.0,
)


# ── Policy Loading ───────────────────────────────────────────────────

def _policy_from_dict(data: Mapping[str, Any]) -> Policy:
    return Policy(
        id=str(data.get("id", _new_id())),
        name=str(data.get("name", "unnamed")),
        version=str(data.get("version", "1")),
        agent_type_pattern=str(data.get("agent_type_pattern", "*")),
        allowed_scopes=frozenset(data.get("allowed_scopes", ["read"])),
        denied_paths=tuple(data.get("denied_paths", [])),
        requires_approval_for=frozenset(data.get("requires_approval_for", [])),
        max_risk_level=RiskLevel(data.get("max_risk_level", "high")),
        max_files_per_change=int(data.get("max_files_per_change", 50)),
        max_lines_per_change=int(data.get("max_lines_per_change", 5000)),
        budget_limit_usd=float(data.get("budget_limit_usd", 100.0)),
        conditions=dict(data.get("conditions", {})),
    )


def load_policies(path: str | Path | None = None) -> list[Policy]:
    """Load versioned policies from YAML config. Fails safe to read-only."""
    if path is None:
        env_path = os.environ.get(_POLICY_FILE_ENV, "")
        path = Path(env_path) if env_path else _DEFAULT_POLICY_PATH

    path = Path(path)
    if not path.exists():
        logger.info(
            "No policy file at %s; using built-in deny-by-default policy. "
            "Create a policies.yaml to define agent permissions.",
            path,
        )
        return [_READ_ONLY_POLICY]

    if not _YAML_AVAILABLE:
        logger.warning("PyYAML not available; using built-in policies.")
        return [_READ_ONLY_POLICY]

    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except Exception as exc:
        raise PolicyError(f"Failed to load policy file {path}: {exc}") from exc

    if not isinstance(data, dict) or "policies" not in data:
        raise PolicyError(
            f"Policy file {path} must contain a 'policies' key with a list"
        )

    policies = []
    for entry in data.get("policies", []):
        try:
            policies.append(_policy_from_dict(entry))
        except (KeyError, TypeError, ValueError) as exc:
            raise PolicyError(f"Invalid policy entry: {exc}") from exc

    if not policies:
        raise PolicyError(f"Policy file {path} contains no policies")

    return policies


# ── Policy Matching ──────────────────────────────────────────────────

def find_matching_policy(
    identity: AgentIdentity,
    policies: list[Policy],
) -> Policy:
    """Find the first policy matching the agent type. Falls back to read-only."""
    for policy in policies:
        if fnmatch.fnmatch(identity.agent_type, policy.agent_type_pattern):
            return policy
    return _READ_ONLY_POLICY


def _path_denied(path: str, policy: Policy) -> bool:
    """Check if a path matches any deny pattern in the policy."""
    import os as _os
    basename = _os.path.basename(path)
    for pattern in policy.denied_paths:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(basename, pattern):
            return True
    return False


# ── Policy Evaluation ────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyDecision:
    """Result of a policy evaluation — every field is explainable."""
    allowed: bool
    policy: Policy
    scope_requested: str
    identity: AgentIdentity
    reason: str
    requires_approval: bool = False
    risk_level: RiskLevel = RiskLevel.LOW
    evaluated_at: float = field(default_factory=_now)

    def to_permission(self, resource: str = "") -> Permission:
        return Permission(
            identity=self.identity,
            scope=self.scope_requested,
            resource=resource,
            allowed=self.allowed,
            policy_id=self.policy.id,
            policy_version=self.policy.version,
            reason=self.reason,
            requires_approval=self.requires_approval,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "policy_id": self.policy.id,
            "policy_name": self.policy.name,
            "policy_version": self.policy.version,
            "scope_requested": self.scope_requested,
            "agent_id": self.identity.agent_id,
            "agent_type": self.identity.agent_type,
            "reason": self.reason,
            "requires_approval": self.requires_approval,
            "risk_level": self.risk_level.value,
            "evaluated_at": self.evaluated_at,
        }


def evaluate(
    identity: AgentIdentity,
    scope: str,
    *,
    resource: str = "",
    risk_level: RiskLevel = RiskLevel.LOW,
    policies: list[Policy] | None = None,
    bus: GovernanceEventBus | None = None,
) -> PolicyDecision:
    """Evaluate whether an identity has authorization for a scope.

    Checks (in order):
      1. Path denial patterns
      2. Risk level cap
      3. Scope allowlist
      4. Human approval requirement

    All decisions are emitted on the governance event bus.
    """
    if policies is None:
        policies = load_policies()
    if bus is None:
        bus = get_event_bus()

    policy = find_matching_policy(identity, policies)
    action = scope.partition(":")[0]

    tracing = TracingContext(
        agent_id=identity.agent_id,
        session_id=identity.session_id,
        organization_id=identity.organization,
        user_id=identity.user,
    )

    # 1. Path denial
    if resource and _path_denied(resource, policy):
        decision = PolicyDecision(
            allowed=False,
            policy=policy,
            scope_requested=scope,
            identity=identity,
            reason=f"Resource path {resource!r} is denied by policy {policy.name!r}",
            risk_level=risk_level,
        )
        _emit(bus, decision, tracing)
        return decision

    # 2. Risk level cap
    _RISK_ORDER = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    if _RISK_ORDER.index(risk_level) > _RISK_ORDER.index(policy.max_risk_level):
        decision = PolicyDecision(
            allowed=False,
            policy=policy,
            scope_requested=scope,
            identity=identity,
            reason=(
                f"Risk level {risk_level.value!r} exceeds policy maximum "
                f"{policy.max_risk_level.value!r} in {policy.name!r}"
            ),
            risk_level=risk_level,
        )
        _emit(bus, decision, tracing)
        return decision

    # 3. Scope check (policy + identity)
    scope_in_policy = _scope_allowed(scope, action, policy.allowed_scopes)
    scope_on_identity = identity.has_scope(scope)
    if not (scope_in_policy and scope_on_identity):
        reason = (
            f"Scope {scope!r} not in policy allowed_scopes for {policy.name!r}"
            if not scope_in_policy
            else f"Agent identity lacks scope {scope!r}"
        )
        decision = PolicyDecision(
            allowed=False,
            policy=policy,
            scope_requested=scope,
            identity=identity,
            reason=reason,
            risk_level=risk_level,
        )
        _emit(bus, decision, tracing)
        return decision

    # 4. Human approval required?
    requires_approval = action in policy.requires_approval_for
    if requires_approval and not identity.is_human:
        decision = PolicyDecision(
            allowed=False,
            policy=policy,
            scope_requested=scope,
            identity=identity,
            reason=f"Scope {scope!r} requires human approval per policy {policy.name!r}",
            requires_approval=True,
            risk_level=risk_level,
        )
        _emit(bus, decision, tracing)
        return decision

    decision = PolicyDecision(
        allowed=True,
        policy=policy,
        scope_requested=scope,
        identity=identity,
        reason=f"Allowed by policy {policy.name!r} v{policy.version}",
        requires_approval=False,
        risk_level=risk_level,
    )
    _emit(bus, decision, tracing)
    return decision


def _scope_allowed(scope: str, action: str, allowed: frozenset[str]) -> bool:
    if "admin" in allowed:
        return True
    if scope in allowed or action in allowed:
        return True
    # Path-prefix match: "write:src/" covers "write:src/foo.py"
    for s in allowed:
        a, _, p = s.partition(":")
        if a == action and p:
            req_path = scope.partition(":")[2]
            if req_path and req_path.startswith(p):
                return True
    return False


def _emit(
    bus: GovernanceEventBus,
    decision: PolicyDecision,
    tracing: TracingContext,
) -> None:
    payload = decision.to_dict()
    if decision.allowed:
        bus.emit(PermissionEvaluatedEvent(tracing=tracing, payload=payload))
    else:
        bus.emit(PermissionDeniedEvent(tracing=tracing, payload=payload))
        bus.emit(PermissionEvaluatedEvent(tracing=tracing, payload=payload))


def require(
    identity: AgentIdentity,
    scope: str,
    *,
    resource: str = "",
    risk_level: RiskLevel = RiskLevel.LOW,
    policies: list[Policy] | None = None,
    bus: GovernanceEventBus | None = None,
) -> PolicyDecision:
    """Like evaluate() but raises PolicyDeniedError if not allowed."""
    decision = evaluate(
        identity, scope, resource=resource,
        risk_level=risk_level, policies=policies, bus=bus,
    )
    if not decision.allowed:
        raise PolicyDeniedError(
            f"Agent {identity.agent_id!r} ({identity.agent_type}) "
            f"denied scope {scope!r}: {decision.reason}"
        )
    return decision


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "PolicyError",
    "PolicyDeniedError",
    "PolicyDecision",
    "load_policies",
    "find_matching_policy",
    "evaluate",
    "require",
]
