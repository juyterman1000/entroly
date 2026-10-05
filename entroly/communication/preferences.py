"""Per-user/contact/group communication preferences.

Preferences shape drafts and low-risk recommendations. They never grant
communication authority by themselves. Explicit settings outrank inferred
settings; inferred settings must carry evidence and confidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from .models import (
    CommunicationEvent,
    CommunicationStateError,
    bounded_string,
    canonical_json,
    sha256_text,
)

PreferenceSource = Literal["default", "explicit", "inferred"]
PreferenceScope = Literal["owner", "contact", "group", "conversation"]

_ALLOWED_ROUTINE_ACTIONS = {"none", "react", "reply", "react_and_reply", "group_ack"}
_ALLOWED_FORMALITY = {"adaptive", "casual", "neutral", "formal"}
_ALLOWED_LENGTH = {"adaptive", "very_short", "short", "medium"}
_ALLOWED_EMOJI = {"adaptive", "none", "light", "expressive"}


@dataclass(frozen=True, slots=True)
class CommunicationTaste:
    """Communication style/presentation preferences, never execution authority."""

    profile_id: str
    scope_type: PreferenceScope
    scope_id: str
    source: PreferenceSource = "default"
    confidence: float = 0.0
    evidence_event_ids: tuple[str, ...] = ()
    preferred_language: str = "adaptive"
    formality: str = "adaptive"
    response_length: str = "adaptive"
    emoji_level: str = "adaptive"
    routine_action: str = "none"
    preferred_reaction: str = ""
    greeting_style: str = ""
    signoff_style: str = ""
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope_type not in {"owner", "contact", "group", "conversation"}:
            raise CommunicationStateError("invalid communication taste scope")
        if not self.scope_id:
            raise CommunicationStateError("communication taste scope_id is required")
        if self.source not in {"default", "explicit", "inferred"}:
            raise CommunicationStateError("invalid communication taste source")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise CommunicationStateError("communication taste confidence must be 0..1")
        if self.formality not in _ALLOWED_FORMALITY:
            raise CommunicationStateError("invalid communication taste formality")
        if self.response_length not in _ALLOWED_LENGTH:
            raise CommunicationStateError("invalid communication taste response_length")
        if self.emoji_level not in _ALLOWED_EMOJI:
            raise CommunicationStateError("invalid communication taste emoji_level")
        if self.routine_action not in _ALLOWED_ROUTINE_ACTIONS:
            raise CommunicationStateError("invalid communication taste routine_action")
        if self.source == "inferred":
            if not self.evidence_event_ids:
                raise CommunicationStateError(
                    "inferred communication taste requires evidence_event_ids"
                )
            if self.confidence <= 0:
                raise CommunicationStateError(
                    "inferred communication taste requires positive confidence"
                )
        if len(canonical_json(dict(self.notes)).encode("utf-8")) > 16 * 1024:
            raise CommunicationStateError("communication taste notes are too large")

    @classmethod
    def build(
        cls,
        *,
        scope_type: PreferenceScope,
        scope_id: str,
        source: PreferenceSource = "default",
        confidence: float = 0.0,
        evidence_event_ids: tuple[str, ...] = (),
        preferred_language: str = "adaptive",
        formality: str = "adaptive",
        response_length: str = "adaptive",
        emoji_level: str = "adaptive",
        routine_action: str = "none",
        preferred_reaction: str = "",
        greeting_style: str = "",
        signoff_style: str = "",
        notes: Mapping[str, Any] | None = None,
    ) -> "CommunicationTaste":
        scope_id = bounded_string(scope_id, 1024)
        evidence = tuple(sorted({bounded_string(item, 128) for item in evidence_event_ids if item}))
        material = {
            "scope_type": scope_type,
            "scope_id": scope_id,
            "source": source,
            "evidence_event_ids": evidence,
            "preferred_language": bounded_string(preferred_language, 32) or "adaptive",
            "formality": formality,
            "response_length": response_length,
            "emoji_level": emoji_level,
            "routine_action": routine_action,
            "preferred_reaction": bounded_string(preferred_reaction, 32),
            "greeting_style": bounded_string(greeting_style, 256),
            "signoff_style": bounded_string(signoff_style, 256),
            "notes": dict(notes or {}),
        }
        return cls(
            profile_id="taste_" + sha256_text(canonical_json(material))[:40],
            scope_type=scope_type,
            scope_id=scope_id,
            source=source,
            confidence=float(confidence),
            evidence_event_ids=evidence,
            preferred_language=material["preferred_language"],
            formality=formality,
            response_length=response_length,
            emoji_level=emoji_level,
            routine_action=routine_action,
            preferred_reaction=material["preferred_reaction"],
            greeting_style=material["greeting_style"],
            signoff_style=material["signoff_style"],
            notes=material["notes"],
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["notes"] = dict(self.notes)
        result["evidence_event_ids"] = list(self.evidence_event_ids)
        return result


def resolve_taste(*profiles: CommunicationTaste) -> dict[str, Any]:
    """Resolve presentation preferences without conferring action permission.

    Precedence:
      explicit narrow scope > explicit broad scope > inferred narrow scope >
      inferred broad scope > defaults.

    Callers should pass profiles from broad to narrow scope.
    """

    rank = {"default": 0, "inferred": 1, "explicit": 2}
    chosen: dict[str, tuple[tuple[int, int, float], Any]] = {}
    fields = (
        "preferred_language",
        "formality",
        "response_length",
        "emoji_level",
        "routine_action",
        "preferred_reaction",
        "greeting_style",
        "signoff_style",
    )
    for scope_index, profile in enumerate(profiles):
        for field_name in fields:
            value = getattr(profile, field_name)
            if value in {"", "adaptive"}:
                continue
            key = (rank[profile.source], scope_index, float(profile.confidence))
            current = chosen.get(field_name)
            if current is None or key > current[0]:
                chosen[field_name] = (key, value)

    return {
        field_name: chosen.get(field_name, ((0, 0, 0.0), "adaptive"))[1]
        for field_name in fields
    }


def inferred_taste_may_authorize_action(_: CommunicationTaste) -> bool:
    """Hard invariant: style inference cannot expand communication authority."""
    return False


_EMOJI_PATTERN = __import__("re").compile(
    "[\U0001F300-\U0001FAFF\u2600-\u27BF]"
)


def infer_taste_from_outbound(
    events: Sequence[CommunicationEvent],
    *,
    scope_type: PreferenceScope,
    scope_id: str,
    minimum_samples: int = 3,
) -> CommunicationTaste | None:
    """Infer narrow presentation taste from observed owner-authored messages.

    Only low-risk surface traits are inferred. The result never grants action
    authority and always carries exact evidence IDs.
    """
    usable = [
        event
        for event in events
        if event.direction == "outbound"
        and event.content.strip()
        and event.event_type == "message"
    ]
    # One physical message can be replayed; evidence identity, not row count,
    # determines sample support.
    by_id = {event.event_id: event for event in usable}
    usable = list(by_id.values())
    if len(usable) < max(3, int(minimum_samples)):
        return None

    lengths = [len(event.content.strip()) for event in usable]
    average_length = sum(lengths) / len(lengths)
    if average_length <= 24:
        response_length = "very_short"
    elif average_length <= 96:
        response_length = "short"
    else:
        response_length = "medium"

    emoji_messages = sum(
        bool(_EMOJI_PATTERN.search(event.content)) for event in usable
    )
    emoji_ratio = emoji_messages / len(usable)
    if emoji_ratio == 0:
        emoji_level = "none"
    elif emoji_ratio <= 0.35:
        emoji_level = "light"
    else:
        emoji_level = "expressive"

    confidence = min(0.95, 0.50 + 0.05 * len(usable))
    evidence_ids = tuple(sorted(by_id))
    return CommunicationTaste.build(
        scope_type=scope_type,
        scope_id=scope_id,
        source="inferred",
        confidence=confidence,
        evidence_event_ids=evidence_ids,
        response_length=response_length,
        emoji_level=emoji_level,
        # Permission-bearing fields stay neutral when inferred.
        routine_action="none",
        notes={
            "sample_count": len(usable),
            "average_response_chars": round(average_length, 2),
            "emoji_message_ratio": round(emoji_ratio, 4),
            "inference_version": 1,
        },
    )
