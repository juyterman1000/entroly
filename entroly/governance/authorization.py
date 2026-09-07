"""
Governance Authorization Service — Runtime authorization enforcement.
=====================================================================

Central authorization enforcement point that combines:
  1. Identity resolution
  2. Policy evaluation (ABAC)
  3. Budget enforcement
  4. Blast-radius checks
  5. Audit logging
  6. Event emission

Every agent action that needs authorization flows through this service.
The service is stateless and safe to instantiate per-request or as a
singleton.  The global singleton is backed by process-level policy cache.

SDK primitives::

    auth = AuthorizationService.from_environment()
    auth.request_permission("write:src/", resource="src/engine.py")
    auth.invoke_tool("entroly_optimize", identity=my_identity)
    auth.submit_change(change, identity=my_identity)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .domain import (
    AgentIdentity,
    Change,
    Permission,
    Policy,
    RiskLevel,
    ToolInvocation,
    _new_id,
    _now,
)
from .events import (
    GovernanceEventBus,
    ToolBlockedEvent,
    ToolInvokedEvent,
    TracingContext,
    get_event_bus,
)
from .identity import resolve_identity
from .policy import (
    PolicyDecision,
    PolicyDeniedError,
    evaluate,
    load_policies,
    require,
)

logger = logging.getLogger(__name__)


# ── Authorization Service ────────────────────────────────────────────

@dataclass
class AuthorizationStats:
    """Counters for authorization operations."""
    checks: int = 0
    allowed: int = 0
    denied: int = 0
    tool_invocations: int = 0
    tool_blocks: int = 0
    budget_enforcements: int = 0


class AuthorizationService:
    """Runtime authorization enforcement for the governance control plane.

    Wraps policy evaluation, budget checks, blast-radius enforcement,
    and audit logging into a single service callable from:
      - MCP server tool handlers (per-tool authorization)
      - Proxy middleware (per-request budget checks)
      - CLI change submission (blast-radius + risk gating)
    """

    def __init__(
        self,
        identity: AgentIdentity,
        policies: list[Policy],
        bus: GovernanceEventBus | None = None,
        *,
        spent_usd: float = 0.0,
    ) -> None:
        self._identity = identity
        self._policies = policies
        self._bus = bus or get_event_bus()
        self._stats = AuthorizationStats()
        self._spent_usd = spent_usd
        self._lock = threading.Lock()

    @classmethod
    def from_environment(
        cls,
        *,
        header_value: str | None = None,
        policy_path: str | None = None,
        bus: GovernanceEventBus | None = None,
    ) -> "AuthorizationService":
        """Create a service from environment variables and config files."""
        ibus = bus or get_event_bus()
        identity = resolve_identity(header_value=header_value, bus=ibus)
        policies = load_policies(policy_path)
        return cls(identity=identity, policies=policies, bus=ibus)

    # ── Core Authorization ───────────────────────────────────────────

    def check(
        self,
        scope: str,
        *,
        resource: str = "",
        risk_level: RiskLevel = RiskLevel.LOW,
    ) -> PolicyDecision:
        """Evaluate permission. Does not raise."""
        decision = evaluate(
            self._identity, scope,
            resource=resource,
            risk_level=risk_level,
            policies=self._policies,
            bus=self._bus,
        )
        with self._lock:
            self._stats.checks += 1
            if decision.allowed:
                self._stats.allowed += 1
            else:
                self._stats.denied += 1
        return decision

    def require_permission(
        self,
        scope: str,
        *,
        resource: str = "",
        risk_level: RiskLevel = RiskLevel.LOW,
    ) -> PolicyDecision:
        """Require permission; raise PolicyDeniedError if denied."""
        return require(
            self._identity, scope,
            resource=resource,
            risk_level=risk_level,
            policies=self._policies,
            bus=self._bus,
        )

    # ── Tool Authorization ───────────────────────────────────────────

    def authorize_tool(
        self,
        tool_name: str,
        *,
        tool_id: str = "",
        required_scope: str = "execute",
        correlation_id: str = "",
    ) -> ToolInvocation:
        """Authorize a tool invocation. Returns a ToolInvocation record.

        If the tool is not authorized, emits ToolBlockedEvent and raises
        PolicyDeniedError.
        """
        tracing = TracingContext(
            agent_id=self._identity.agent_id,
            session_id=self._identity.session_id,
            organization_id=self._identity.organization,
            tool_invocation_id=_new_id(),
        )

        decision = self.check(
            f"{required_scope}:{tool_name}",
            resource=tool_name,
        )

        invocation = ToolInvocation(
            tool_id=tool_id or tool_name,
            tool_name=tool_name,
            session_id=self._identity.session_id,
            agent_identity=self._identity,
            permission_id=decision.to_permission(tool_name).id,
            correlation_id=correlation_id,
        )

        if not decision.allowed:
            with self._lock:
                self._stats.tool_blocks += 1
            self._bus.emit(ToolBlockedEvent(
                tracing=tracing,
                payload={
                    "tool_name": tool_name,
                    "reason": decision.reason,
                    "agent_id": self._identity.agent_id,
                },
            ))
            raise PolicyDeniedError(
                f"Tool {tool_name!r} blocked: {decision.reason}"
            )

        with self._lock:
            self._stats.tool_invocations += 1
        self._bus.emit(ToolInvokedEvent(
            tracing=tracing,
            payload={
                "tool_name": tool_name,
                "tool_id": tool_id,
                "agent_id": self._identity.agent_id,
                "session_id": self._identity.session_id,
            },
        ))
        return invocation

    # ── Blast-Radius Enforcement ─────────────────────────────────────

    def check_blast_radius(
        self,
        change: Change,
    ) -> PolicyDecision:
        """Enforce blast-radius limits from the matching policy."""
        from .policy import find_matching_policy
        policy = find_matching_policy(self._identity, self._policies)

        if change.file_count > policy.max_files_per_change:
            return PolicyDecision(
                allowed=False,
                policy=policy,
                scope_requested="write",
                identity=self._identity,
                reason=(
                    f"Change touches {change.file_count} files, exceeding "
                    f"policy limit of {policy.max_files_per_change}"
                ),
                risk_level=RiskLevel.HIGH,
            )

        if change.line_count > policy.max_lines_per_change:
            return PolicyDecision(
                allowed=False,
                policy=policy,
                scope_requested="write",
                identity=self._identity,
                reason=(
                    f"Change modifies {change.line_count} lines, exceeding "
                    f"policy limit of {policy.max_lines_per_change}"
                ),
                risk_level=RiskLevel.HIGH,
            )

        from .policy import find_matching_policy as _fmp
        return PolicyDecision(
            allowed=True,
            policy=policy,
            scope_requested="write",
            identity=self._identity,
            reason="Blast radius within policy limits",
            risk_level=RiskLevel.LOW,
        )

    # ── Budget Enforcement ───────────────────────────────────────────

    def check_budget(self, cost_usd: float) -> PolicyDecision:
        """Check if a cost would exceed the budget cap in the policy."""
        from .policy import find_matching_policy
        from .events import BudgetExceededEvent

        policy = find_matching_policy(self._identity, self._policies)

        with self._lock:
            projected = self._spent_usd + cost_usd
            over_budget = projected > policy.budget_limit_usd

        if over_budget:
            with self._lock:
                self._stats.budget_enforcements += 1
            self._bus.emit(BudgetExceededEvent(
                tracing=TracingContext(
                    agent_id=self._identity.agent_id,
                    session_id=self._identity.session_id,
                    organization_id=self._identity.organization,
                ),
                payload={
                    "spent_usd": self._spent_usd,
                    "cost_usd": cost_usd,
                    "limit_usd": policy.budget_limit_usd,
                    "agent_id": self._identity.agent_id,
                },
            ))
            return PolicyDecision(
                allowed=False,
                policy=policy,
                scope_requested="execute",
                identity=self._identity,
                reason=(
                    f"Budget cap exceeded: ${self._spent_usd:.4f} spent + "
                    f"${cost_usd:.4f} projected > ${policy.budget_limit_usd:.2f} limit"
                ),
                risk_level=RiskLevel.MEDIUM,
            )

        return PolicyDecision(
            allowed=True,
            policy=policy,
            scope_requested="execute",
            identity=self._identity,
            reason=f"Within budget: ${projected:.4f} / ${policy.budget_limit_usd:.2f}",
            risk_level=RiskLevel.LOW,
        )

    def record_spend(self, cost_usd: float) -> None:
        """Record actual spend against the session budget."""
        with self._lock:
            self._spent_usd += cost_usd

    # ── Properties ──────────────────────────────────────────────────

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    def stats(self) -> AuthorizationStats:
        return self._stats

    @property
    def policies(self) -> list[Policy]:
        """The policies this service evaluates against.

        Exposed so callers can inspect the applicable policy without calling
        `load_policies` again. That function is uncached -- it stats the path
        and re-parses the YAML on every call -- so a per-request caller would
        add disk I/O to a hot path. Worse, it would read a *different* snapshot
        than the one `check` and `check_budget` evaluate against, letting a
        caller's pre-check and the service's decision disagree after an edit.
        """
        return self._policies

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self._identity.to_dict(),
            "stats": {
                "checks": self._stats.checks,
                "allowed": self._stats.allowed,
                "denied": self._stats.denied,
                "tool_invocations": self._stats.tool_invocations,
                "tool_blocks": self._stats.tool_blocks,
                "budget_enforcements": self._stats.budget_enforcements,
            },
            "spent_usd": self._spent_usd,
        }


# ── Global Singleton ─────────────────────────────────────────────────

_global_service: AuthorizationService | None = None
_service_lock = threading.Lock()


def get_authorization_service(
    *,
    header_value: str | None = None,
    reset: bool = False,
) -> AuthorizationService:
    """Get or create the process-global authorization service."""
    global _global_service
    if reset or _global_service is None:
        with _service_lock:
            if reset or _global_service is None:
                # Start recording as soon as governance is used.
                #
                # `install_audit_subscriber` existed and nothing called it, so
                # the audit log stayed empty unless an operator wired the bus
                # themselves. An audit trail that is empty by default is worse
                # than none: `govern audit verify` reports an intact chain over
                # a log nothing ever wrote to.
                #
                # Attached here rather than at import so it stays side-effect
                # free until governance is actually exercised -- the log
                # resolves its directory lazily and creates nothing until an
                # event arrives. Failure to attach must not stop authorization
                # working, so it degrades to a warning rather than raising.
                #
                # Ordered before construction, not after: `from_environment`
                # resolves the identity, which emits `identity.resolved`. A
                # subscriber attached afterwards misses it, and in a process
                # that builds the service once -- the normal case -- that is
                # the whole record of which agent this is.
                try:
                    from .audit import install_audit_subscriber

                    install_audit_subscriber()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "Governance audit subscriber not installed (%s); "
                        "decisions will not be recorded.",
                        exc,
                    )
                _global_service = AuthorizationService.from_environment(
                    header_value=header_value
                )
    return _global_service


def reset_authorization_service() -> None:
    """Reset the global service (for testing)."""
    global _global_service
    with _service_lock:
        _global_service = None


__all__ = [
    "AuthorizationService",
    "AuthorizationStats",
    "PolicyDeniedError",
    "get_authorization_service",
    "reset_authorization_service",
]
