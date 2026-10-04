"""Explicit bounded-delegation policy for communication actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .models import (
    ActionType,
    AssuranceDecision,
    CommunicationActionProposal,
    CommunicationEvent,
)

PolicyMode = Literal["observe", "suggest", "approve", "bounded"]


@dataclass(frozen=True, slots=True)
class CommunicationPolicy:
    """Policy defaults can never authorize an external communication action."""

    mode: PolicyMode = "observe"
    auto_categories: tuple[str, ...] = ()
    auto_actions: tuple[ActionType, ...] = ()
    approval_categories: tuple[str, ...] = (
        "financial",
        "legal",
        "medical",
        "security",
        "employment",
        "sensitive",
        "emergency",
    )

    def evaluate(
        self,
        proposal: CommunicationActionProposal,
        *,
        source_events: Sequence[CommunicationEvent],
        already_handled: bool = False,
    ) -> tuple[AssuranceDecision, tuple[str, ...]]:
        if already_handled:
            return "already_handled", ("action:already_handled",)
        if proposal.action_type == "no_action":
            return "allow", ("action:no_external_effect",)
        if not proposal.source_event_ids:
            return "insufficient_context", ("evidence:missing",)

        found = {event.event_id for event in source_events}
        if found != set(proposal.source_event_ids):
            return "insufficient_context", ("evidence:incomplete",)

        for event in source_events:
            if (
                event.channel != proposal.channel
                or event.account_id != proposal.account_id
                or event.conversation_id != proposal.conversation_id
            ):
                return "deny", ("scope:cross_conversation",)

        if (
            proposal.action_type == "group_reply"
            and proposal.conversation_kind != "group"
        ):
            return "ambiguous", ("scope:group_not_verified",)
        if proposal.risk_class in {"high", "unknown"}:
            return "approval_required", (f"risk:{proposal.risk_class}",)
        if proposal.category in set(self.approval_categories):
            return "approval_required", (f"category:{proposal.category}",)
        if proposal.creates_commitment:
            return "approval_required", ("action:creates_commitment",)
        if self.mode != "bounded":
            return "approval_required", (f"policy:{self.mode}",)
        if proposal.action_type not in set(self.auto_actions):
            return "approval_required", ("policy:action_not_delegated",)
        if proposal.category not in set(self.auto_categories):
            return "approval_required", ("policy:category_not_delegated",)
        return "allow", ("policy:bounded_allow",)
