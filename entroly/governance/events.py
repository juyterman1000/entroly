"""
Governance Event Bus — Event-driven backbone for the control plane.
====================================================================

Every governance action emits a typed event through a process-local bus.
Events are immutable records with correlation and trace IDs for distributed
tracing.  Subscribers receive events synchronously within the same process;
durable persistence is handled by the audit module.

Design:
  - Typed event classes (not generic dicts) for compile-time safety
  - Process-local pub/sub (no network, no external dependencies)
  - Thread-safe subscriber registration and dispatch
  - Every event carries request_id, trace_id, organization_id, user_id,
    agent_id, session_id, task_id for observability
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .domain import _new_id, _now

logger = logging.getLogger(__name__)


# ── Tracing Context ──────────────────────────────────────────────────

@dataclass(frozen=True)
class TracingContext:
    """Correlation context carried by every governance event.

    Provides the dimensions needed for distributed tracing and
    metric aggregation.
    """
    request_id: str = field(default_factory=_new_id)
    trace_id: str = field(default_factory=_new_id)
    organization_id: str = ""
    user_id: str = ""
    agent_id: str = ""
    session_id: str = ""
    task_id: str = ""
    tool_invocation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "organization_id": self.organization_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "tool_invocation_id": self.tool_invocation_id,
        }


# ── Event Types ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class GovernanceEvent:
    """Base event type for all governance events."""
    event_id: str = field(default_factory=_new_id)
    event_type: str = ""
    timestamp: float = field(default_factory=_now)
    tracing: TracingContext = field(default_factory=TracingContext)
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "tracing": self.tracing.to_dict(),
            "payload": dict(self.payload),
        }


# Specific event types — each one maps to a governance action.

@dataclass(frozen=True)
class IdentityResolvedEvent(GovernanceEvent):
    """Emitted when an agent identity is resolved (or falls back to anonymous)."""
    event_type: str = field(default="identity.resolved", init=False)


@dataclass(frozen=True)
class PermissionEvaluatedEvent(GovernanceEvent):
    """Emitted for every permission check (allowed or denied)."""
    event_type: str = field(default="permission.evaluated", init=False)


@dataclass(frozen=True)
class PermissionDeniedEvent(GovernanceEvent):
    """Emitted specifically when a permission is denied."""
    event_type: str = field(default="permission.denied", init=False)


@dataclass(frozen=True)
class ToolInvokedEvent(GovernanceEvent):
    """Emitted when a tool is invoked through the governance gateway."""
    event_type: str = field(default="tool.invoked", init=False)


@dataclass(frozen=True)
class ToolBlockedEvent(GovernanceEvent):
    """Emitted when a tool invocation is blocked by policy."""
    event_type: str = field(default="tool.blocked", init=False)


@dataclass(frozen=True)
class ChangeSubmittedEvent(GovernanceEvent):
    """Emitted when an agent submits a code change."""
    event_type: str = field(default="change.submitted", init=False)


@dataclass(frozen=True)
class VerificationCompletedEvent(GovernanceEvent):
    """Emitted when verification of a change completes."""
    event_type: str = field(default="verification.completed", init=False)


@dataclass(frozen=True)
class RiskAssessedEvent(GovernanceEvent):
    """Emitted when a risk assessment is computed."""
    event_type: str = field(default="risk.assessed", init=False)


@dataclass(frozen=True)
class GateDecisionEvent(GovernanceEvent):
    """Emitted when the evidence gate makes a decision."""
    event_type: str = field(default="gate.decision", init=False)


@dataclass(frozen=True)
class ApprovalRequestedEvent(GovernanceEvent):
    """Emitted when human approval is required."""
    event_type: str = field(default="approval.requested", init=False)


@dataclass(frozen=True)
class ApprovalReceivedEvent(GovernanceEvent):
    """Emitted when human approval is received."""
    event_type: str = field(default="approval.received", init=False)


@dataclass(frozen=True)
class SupplyChainScanEvent(GovernanceEvent):
    """Emitted when a supply-chain scan completes."""
    event_type: str = field(default="supply_chain.scanned", init=False)


@dataclass(frozen=True)
class SecurityFindingEvent(GovernanceEvent):
    """Emitted when a security finding is detected."""
    event_type: str = field(default="security.finding", init=False)


@dataclass(frozen=True)
class CostRecordedEvent(GovernanceEvent):
    """Emitted when a cost event is recorded."""
    event_type: str = field(default="cost.recorded", init=False)


@dataclass(frozen=True)
class ProvenanceRecordedEvent(GovernanceEvent):
    """Emitted when a provenance node is added to the DAG."""
    event_type: str = field(default="provenance.recorded", init=False)


@dataclass(frozen=True)
class BudgetExceededEvent(GovernanceEvent):
    """Emitted when an agent/team exceeds their budget."""
    event_type: str = field(default="budget.exceeded", init=False)


# ── Event Bus ────────────────────────────────────────────────────────

EventHandler = Callable[[GovernanceEvent], None]


class GovernanceEventBus:
    """Process-local, thread-safe event bus for governance events.

    Usage::

        bus = GovernanceEventBus()

        # Subscribe to specific event types
        bus.subscribe("permission.denied", my_alert_handler)
        bus.subscribe("*", my_audit_logger)   # wildcard = all events

        # Emit events
        bus.emit(PermissionDeniedEvent(
            tracing=TracingContext(agent_id="agent-1"),
            payload={"scope": "deploy:production", "reason": "policy denied"},
        ))
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._lock = threading.Lock()
        self._event_count = 0

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe a handler to an event type.  Use '*' for all events."""
        with self._lock:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler from an event type."""
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def emit(self, event: GovernanceEvent) -> None:
        """Emit an event to all matching subscribers.

        Handlers are called synchronously.  Exceptions in handlers are
        logged but never propagate to the emitter.
        """
        with self._lock:
            self._event_count += 1
            # Type-specific handlers
            handlers = list(self._handlers.get(event.event_type, []))
            # Wildcard handlers
            handlers.extend(self._handlers.get("*", []))

        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Error in governance event handler for %s",
                    event.event_type,
                )

    @property
    def event_count(self) -> int:
        return self._event_count

    def reset(self) -> None:
        """Clear all handlers (for testing)."""
        with self._lock:
            self._handlers.clear()
            self._event_count = 0


# ── Global Bus ───────────────────────────────────────────────────────

_global_bus: GovernanceEventBus | None = None
_bus_lock = threading.Lock()


def get_event_bus() -> GovernanceEventBus:
    """Get or create the process-global governance event bus."""
    global _global_bus
    if _global_bus is None:
        with _bus_lock:
            if _global_bus is None:
                _global_bus = GovernanceEventBus()
    return _global_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _global_bus
    with _bus_lock:
        if _global_bus is not None:
            _global_bus.reset()
        _global_bus = None


__all__ = [
    "TracingContext",
    "GovernanceEvent",
    "IdentityResolvedEvent",
    "PermissionEvaluatedEvent",
    "PermissionDeniedEvent",
    "ToolInvokedEvent",
    "ToolBlockedEvent",
    "ChangeSubmittedEvent",
    "VerificationCompletedEvent",
    "RiskAssessedEvent",
    "GateDecisionEvent",
    "ApprovalRequestedEvent",
    "ApprovalReceivedEvent",
    "SupplyChainScanEvent",
    "SecurityFindingEvent",
    "CostRecordedEvent",
    "ProvenanceRecordedEvent",
    "BudgetExceededEvent",
    "GovernanceEventBus",
    "EventHandler",
    "get_event_bus",
    "reset_event_bus",
]
