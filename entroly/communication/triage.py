"""Conservative deterministic triage for communication assurance.

This is intentionally a high-precision baseline, not a general semantic model.
Anything mixed, sensitive, action-bearing, long, or unknown stays human-visible.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence

from .models import CommunicationEvent

AttentionClass = Literal[
    "routine_candidate",
    "review",
    "urgent",
    "no_action",
]
RiskClass = Literal["low", "normal", "high", "unknown"]

_ROUTINE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "birthday_wish",
        (
            "happy birthday",
            "many happy returns",
            "birthday wishes",
            "wish you a very happy birthday",
        ),
    ),
    (
        "congratulations",
        (
            "congratulations",
            "congrats",
            "well done",
        ),
    ),
    (
        "holiday_wish",
        (
            "happy new year",
            "merry christmas",
            "happy diwali",
            "diwali wishes",
            "happy holi",
            "eid mubarak",
            "happy thanksgiving",
            "happy anniversary",
        ),
    ),
    (
        "thanks",
        (
            "thank you",
            "thanks",
            "thx",
        ),
    ),
)

_HIGH_RISK_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("financial", ("bank", "payment", "pay ", "paid ", "money", "transfer", "invoice", "refund", "$")),
    ("legal", ("contract", "legal", "lawyer", "attorney", "lawsuit", "sign this")),
    ("medical", ("doctor", "hospital", "medical", "medicine", "surgery", "emergency room")),
    ("security", ("password", "passcode", "otp", "verification code", "login", "security code", "2fa")),
    ("emergency", ("urgent", "emergency", "immediately", "asap", "right now")),
)

_REQUEST_PATTERNS = re.compile(
    r"\b("
    r"can you|could you|would you|will you|please|"
    r"send me|call me|let me know|confirm|book |reserve |"
    r"share |forward |reply |respond |check "
    r")\b",
    re.IGNORECASE,
)

_INBOUND_COMMITMENT_PATTERNS = re.compile(
    r"\b(i(?:'|’)ll|i will|we(?:'|’)ll|we will|i promise|we promise)\b",
    re.IGNORECASE,
)

_OUTBOUND_COMMITMENT_PATTERNS = re.compile(
    r"\b(i(?:'|’)ll|i will|we(?:'|’)ll|we will|i promise|we promise)\b",
    re.IGNORECASE,
)

_QUESTION_WORDS = re.compile(
    r"^\s*(who|what|when|where|why|how|which|can|could|would|will|are|is|do|did|should)\b",
    re.IGNORECASE,
)


def _normalized(text: str) -> str:
    lowered = " ".join(text.casefold().split())
    return lowered.strip(" \t\r\n.!,")


def _matches_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


@dataclass(frozen=True, slots=True)
class CommunicationAssessment:
    event_id: str
    category: str
    attention: AttentionClass
    risk_class: RiskClass
    reasons: tuple[str, ...]
    question: bool
    request: bool
    inbound_commitment: bool
    routine_social: bool
    reaction_candidate: bool
    direct_reply_candidate: bool
    group_ack_candidate: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_event(event: CommunicationEvent) -> CommunicationAssessment:
    """Return a conservative assessment linked to one exact evidence event."""

    text = _normalized(event.content)
    if event.direction != "inbound":
        return CommunicationAssessment(
            event_id=event.event_id,
            category="outbound",
            attention="no_action",
            risk_class="low",
            reasons=("direction:outbound",),
            question=False,
            request=False,
            inbound_commitment=False,
            routine_social=False,
            reaction_candidate=False,
            direct_reply_candidate=False,
            group_ack_candidate=False,
        )

    risk_categories = [
        category
        for category, phrases in _HIGH_RISK_PATTERNS
        if _matches_any(text, phrases)
    ]
    question = "?" in event.content or bool(_QUESTION_WORDS.search(text))
    request = bool(_REQUEST_PATTERNS.search(text))
    inbound_commitment = bool(_INBOUND_COMMITMENT_PATTERNS.search(text))

    routine_category = ""
    for category, phrases in _ROUTINE_PATTERNS:
        if _matches_any(text, phrases):
            routine_category = category
            break

    # A low-risk social acknowledgement must be compact and single-purpose.
    compact = len(event.content) <= 280 and event.content.count("\n") <= 2
    mixed = question or request or inbound_commitment or bool(risk_categories)
    routine_social = bool(routine_category and compact and not mixed)

    reasons: list[str] = []
    if risk_categories:
        reasons.extend(f"risk:{item}" for item in sorted(set(risk_categories)))
    if question:
        reasons.append("attention:question")
    if request:
        reasons.append("attention:request")
    if inbound_commitment:
        reasons.append("attention:commitment")
    if routine_social:
        reasons.append(f"routine:{routine_category}")
    if routine_category and not routine_social:
        reasons.append("routine:mixed_or_long")

    if risk_categories:
        attention: AttentionClass = "urgent" if "emergency" in risk_categories else "review"
        risk_class: RiskClass = "high"
        category = risk_categories[0]
    elif question or request or inbound_commitment:
        attention = "review"
        risk_class = "normal"
        category = "actionable"
    elif routine_social:
        attention = "routine_candidate"
        risk_class = "low"
        category = routine_category
    elif not text:
        attention = "no_action"
        risk_class = "low"
        category = "empty"
        reasons.append("content:empty")
    else:
        attention = "review"
        risk_class = "unknown"
        category = "other"
        reasons.append("attention:unclassified")

    return CommunicationAssessment(
        event_id=event.event_id,
        category=category,
        attention=attention,
        risk_class=risk_class,
        reasons=tuple(sorted(set(reasons))),
        question=question,
        request=request,
        inbound_commitment=inbound_commitment,
        routine_social=routine_social,
        reaction_candidate=routine_social,
        direct_reply_candidate=routine_social and event.conversation_kind == "direct",
        group_ack_candidate=routine_social and event.conversation_kind == "group",
    )


def outgoing_creates_commitment(payload: str) -> bool:
    return bool(_OUTBOUND_COMMITMENT_PATTERNS.search(payload or ""))


def combine_risk(assessments: Sequence[CommunicationAssessment]) -> RiskClass:
    if not assessments:
        return "unknown"
    values = {item.risk_class for item in assessments}
    if "high" in values:
        return "high"
    if "unknown" in values:
        return "unknown"
    if "normal" in values:
        return "normal"
    return "low"


def combine_category(assessments: Sequence[CommunicationAssessment]) -> str:
    if not assessments:
        return "other"
    categories = {item.category for item in assessments}
    if len(categories) == 1:
        return next(iter(categories))
    return "mixed"


@dataclass(frozen=True, slots=True)
class CommunicationEpisode:
    """Conservative same-scope aggregation of routine group communication."""

    category: str
    channel: str
    account_id: str
    conversation_id: str
    source_event_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]
    start_timestamp: float
    end_timestamp: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_group_episodes(
    events: Sequence[CommunicationEvent],
    *,
    window_seconds: float = 2 * 60 * 60,
    minimum_size: int = 2,
) -> list[CommunicationEpisode]:
    """Group only verified group-chat routine events within a bounded time window."""

    candidates: list[tuple[CommunicationEvent, CommunicationAssessment]] = []
    for event in events:
        assessment = assess_event(event)
        if (
            event.direction == "inbound"
            and event.conversation_kind == "group"
            and event.timestamp is not None
            and assessment.group_ack_candidate
        ):
            candidates.append((event, assessment))
    candidates.sort(key=lambda pair: (float(pair[0].timestamp or 0), pair[0].event_id))

    episodes: list[CommunicationEpisode] = []
    current: list[tuple[CommunicationEvent, CommunicationAssessment]] = []

    def flush() -> None:
        if len(current) < max(2, int(minimum_size)):
            current.clear()
            return
        first_event, first_assessment = current[0]
        episodes.append(
            CommunicationEpisode(
                category=first_assessment.category,
                channel=first_event.channel,
                account_id=first_event.account_id,
                conversation_id=first_event.conversation_id,
                source_event_ids=tuple(item[0].event_id for item in current),
                participant_ids=tuple(
                    sorted({item[0].sender_id for item in current if item[0].sender_id})
                ),
                start_timestamp=float(first_event.timestamp or 0),
                end_timestamp=float(current[-1][0].timestamp or 0),
            )
        )
        current.clear()

    for pair in candidates:
        event, assessment = pair
        if not current:
            current.append(pair)
            continue
        previous_event, previous_assessment = current[-1]
        same_scope = (
            event.channel == previous_event.channel
            and event.account_id == previous_event.account_id
            and event.conversation_id == previous_event.conversation_id
        )
        same_category = assessment.category == previous_assessment.category
        within_window = (
            float(event.timestamp or 0) - float(previous_event.timestamp or 0)
            <= window_seconds
        )
        if same_scope and same_category and within_window:
            current.append(pair)
        else:
            flush()
            current.append(pair)
    flush()
    return episodes


def build_digest(
    events: Sequence[CommunicationEvent],
    *,
    max_attention_items: int = 20,
) -> dict[str, object]:
    """Produce a bounded evidence-referenced secretary digest."""

    assessments = [assess_event(event) for event in events]
    by_id = {event.event_id: event for event in events}
    attention = [
        item
        for item in assessments
        if item.attention in {"review", "urgent"}
    ]
    routine = [item for item in assessments if item.attention == "routine_candidate"]
    episodes = build_group_episodes(events)

    attention_items: list[dict[str, object]] = []
    for item in sorted(
        attention,
        key=lambda value: (
            0 if value.attention == "urgent" else 1,
            -(by_id[value.event_id].timestamp or 0),
            value.event_id,
        ),
    )[: max(1, min(int(max_attention_items), 100))]:
        event = by_id[item.event_id]
        attention_items.append(
            {
                "event_id": event.event_id,
                "channel": event.channel,
                "account_id": event.account_id,
                "conversation_id": event.conversation_id,
                "conversation_kind": event.conversation_kind,
                "sender_id": event.sender_id,
                "message_id": event.message_id,
                "timestamp": event.timestamp,
                "category": item.category,
                "attention": item.attention,
                "risk_class": item.risk_class,
                "reasons": list(item.reasons),
                # Bounded excerpt for explicit owner review only.
                "excerpt": event.content[:280],
            }
        )

    routine_by_category: dict[str, int] = {}
    for item in routine:
        routine_by_category[item.category] = routine_by_category.get(item.category, 0) + 1

    return {
        "total_events": len(events),
        "attention_count": len(attention),
        "urgent_count": sum(item.attention == "urgent" for item in attention),
        "routine_candidate_count": len(routine),
        "routine_by_category": dict(sorted(routine_by_category.items())),
        "group_episodes": [episode.to_dict() for episode in episodes],
        "attention_items": attention_items,
        "truncated_attention_items": max(0, len(attention) - len(attention_items)),
    }
