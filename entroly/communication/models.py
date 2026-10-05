"""Canonical communication-assurance records.

Raw channel events are observed evidence, not semantic memory.  This module is
channel-neutral: OpenClaw/WhatsApp adapters normalize into these records but no
provider-native identifier is parsed to infer hidden facts such as group kind.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

SCHEMA = "entroly.communication.v1"
MAX_CONTENT_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024

Direction = Literal["inbound", "outbound"]
ConversationKind = Literal["direct", "group", "unknown"]
EventType = Literal["message", "edit", "delete", "reaction", "system"]
DeliveryState = Literal["received", "sent", "failed", "unknown"]
IdentityStrength = Literal["provider_update", "message_id", "run_id", "derived"]
ActionType = Literal["send_message", "reply", "react", "group_reply", "no_action"]
RiskClass = Literal["low", "normal", "high", "unknown"]
AssuranceDecision = Literal[
    "allow",
    "approval_required",
    "deny",
    "already_handled",
    "ambiguous",
    "insufficient_context",
]


class CommunicationStateError(RuntimeError):
    """Invalid or unsafe communication state."""


class CommunicationStateConflict(CommunicationStateError):
    """A stable event/action identity was reused with different evidence."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bounded_string(value: Any, limit: int = 4096) -> str:
    if value is None:
        return ""
    return str(value).strip()[:limit]


def bounded_timestamp(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(timestamp) or timestamp < 0:
        return None
    # Hooks vary between seconds and milliseconds.  Normalize without guessing
    # a calendar date or altering provider identity.
    if timestamp > 100_000_000_000:
        timestamp /= 1000.0
    return timestamp


def bounded_metadata(value: Any) -> dict[str, Any]:
    metadata = dict(value) if isinstance(value, Mapping) else {}
    encoded = canonical_json(metadata).encode("utf-8")
    if len(encoded) > MAX_METADATA_BYTES:
        raise CommunicationStateError(
            f"communication metadata exceeds {MAX_METADATA_BYTES} bytes"
        )
    return metadata


@dataclass(frozen=True, slots=True)
class CommunicationEvent:
    """Immutable observed communication evidence."""

    event_id: str
    direction: Direction
    channel: str
    conversation_id: str
    conversation_kind: ConversationKind = "unknown"
    account_id: str = ""
    sender_id: str = ""
    recipient_id: str = ""
    message_id: str = ""
    reply_to_id: str = ""
    session_key: str = ""
    run_id: str = ""
    event_type: EventType = "message"
    timestamp: float | None = None
    content: str = ""
    delivery_state: DeliveryState = "unknown"
    identity_strength: IdentityStrength = "derived"
    provider_update_id: str = ""
    provider_update_kind: str = ""
    source: str = "adapter"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise CommunicationStateError(f"unsupported communication schema: {self.schema!r}")
        if not self.event_id:
            raise CommunicationStateError("event_id is required")
        if self.direction not in {"inbound", "outbound"}:
            raise CommunicationStateError("invalid direction")
        if not self.channel:
            raise CommunicationStateError("channel is required")
        if not self.conversation_id:
            raise CommunicationStateError(
                "conversation_id is required; refusing unscoped communication evidence"
            )
        if self.conversation_kind not in {"direct", "group", "unknown"}:
            raise CommunicationStateError("invalid conversation_kind")
        if self.event_type not in {"message", "edit", "delete", "reaction", "system"}:
            raise CommunicationStateError("invalid event_type")
        if self.delivery_state not in {"received", "sent", "failed", "unknown"}:
            raise CommunicationStateError("invalid delivery_state")
        if self.identity_strength not in {
            "provider_update",
            "message_id",
            "run_id",
            "derived",
        }:
            raise CommunicationStateError("invalid identity_strength")
        if len(self.content.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise CommunicationStateError(
                f"communication content exceeds {MAX_CONTENT_BYTES} bytes"
            )
        bounded_metadata(self.metadata)

    @property
    def content_sha256(self) -> str:
        return sha256_text(self.content)

    @property
    def commitment_sha256(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "schema": self.schema,
                    "event_id": self.event_id,
                    "direction": self.direction,
                    "channel": self.channel,
                    "account_id": self.account_id,
                    "conversation_id": self.conversation_id,
                    "conversation_kind": self.conversation_kind,
                    "sender_id": self.sender_id,
                    "recipient_id": self.recipient_id,
                    "message_id": self.message_id,
                    "reply_to_id": self.reply_to_id,
                    "session_key": self.session_key,
                    "run_id": self.run_id,
                    "event_type": self.event_type,
                    "timestamp": self.timestamp,
                    "content_sha256": self.content_sha256,
                    "delivery_state": self.delivery_state,
                    "identity_strength": self.identity_strength,
                    "provider_update_id": self.provider_update_id,
                    "provider_update_kind": self.provider_update_kind,
                    "source": self.source,
                    "metadata": dict(self.metadata),
                }
            )
        )

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        result = asdict(self)
        result["metadata"] = dict(self.metadata)
        result["content_sha256"] = self.content_sha256
        result["commitment_sha256"] = self.commitment_sha256
        if not include_content:
            result.pop("content", None)
        return result


def event_from_adapter(payload: Mapping[str, Any]) -> CommunicationEvent:
    """Normalize one adapter observation without parsing opaque provider IDs."""

    direction = bounded_string(payload.get("direction"), 16).lower()
    if direction not in {"inbound", "outbound"}:
        raise CommunicationStateError("adapter direction must be inbound or outbound")

    channel = bounded_string(payload.get("channel"), 128).lower()
    account_id = bounded_string(payload.get("account_id"), 256)
    conversation_id = bounded_string(payload.get("conversation_id"), 1024)
    if not channel:
        raise CommunicationStateError("adapter channel is required")
    if not conversation_id:
        raise CommunicationStateError(
            "adapter conversation_id is required; refusing to infer a privacy scope"
        )

    message_id = bounded_string(payload.get("message_id"), 1024)
    run_id = bounded_string(payload.get("run_id"), 256)
    provider_update = (
        dict(payload.get("provider_update"))
        if isinstance(payload.get("provider_update"), Mapping)
        else {}
    )
    provider_update_id = bounded_string(provider_update.get("id"), 1024)
    provider_update_kind = bounded_string(provider_update.get("kind"), 128)
    content = "" if payload.get("content") is None else str(payload.get("content"))
    timestamp = bounded_timestamp(payload.get("timestamp"))

    identity_strength: IdentityStrength
    if provider_update_id and provider_update_kind:
        identity_strength = "provider_update"
        identity = {
            "provider_update_kind": provider_update_kind,
            "provider_update_id": provider_update_id,
        }
    elif message_id:
        identity_strength = "message_id"
        identity = {"message_id": message_id}
    elif run_id:
        identity_strength = "run_id"
        identity = {"run_id": run_id, "content_sha256": sha256_text(content)}
    else:
        identity_strength = "derived"
        identity = {
            "sender_id": bounded_string(payload.get("sender_id"), 1024),
            "recipient_id": bounded_string(payload.get("recipient_id"), 1024),
            "timestamp": timestamp,
            "content_sha256": sha256_text(content),
        }

    event_id = "comm_" + sha256_text(
        canonical_json(
            {
                "schema": SCHEMA,
                "direction": direction,
                "channel": channel,
                "account_id": account_id,
                "conversation_id": conversation_id,
                **identity,
            }
        )
    )[:40]

    kind = bounded_string(payload.get("conversation_kind"), 16).lower() or "unknown"
    if kind not in {"direct", "group", "unknown"}:
        kind = "unknown"
    delivery = bounded_string(payload.get("delivery_state"), 16).lower()
    if not delivery:
        delivery = "received" if direction == "inbound" else "unknown"
    if delivery not in {"received", "sent", "failed", "unknown"}:
        delivery = "unknown"
    event_type = bounded_string(payload.get("event_type"), 16).lower() or "message"
    if event_type not in {"message", "edit", "delete", "reaction", "system"}:
        event_type = "message"

    return CommunicationEvent(
        event_id=event_id,
        direction=direction,  # type: ignore[arg-type]
        channel=channel,
        account_id=account_id,
        conversation_id=conversation_id,
        conversation_kind=kind,  # type: ignore[arg-type]
        sender_id=bounded_string(payload.get("sender_id"), 1024),
        recipient_id=bounded_string(payload.get("recipient_id"), 1024),
        message_id=message_id,
        reply_to_id=bounded_string(payload.get("reply_to_id"), 1024),
        session_key=bounded_string(payload.get("session_key"), 2048),
        run_id=run_id,
        event_type=event_type,  # type: ignore[arg-type]
        timestamp=timestamp,
        content=content,
        delivery_state=delivery,  # type: ignore[arg-type]
        identity_strength=identity_strength,
        provider_update_id=provider_update_id,
        provider_update_kind=provider_update_kind,
        source=bounded_string(payload.get("source"), 128) or "adapter",
        metadata=bounded_metadata(payload.get("metadata")),
    )


@dataclass(frozen=True, slots=True)
class CommunicationActionProposal:
    """Proposed external communication. It carries no execution authority."""

    action_id: str
    action_type: ActionType
    channel: str
    conversation_id: str
    source_event_ids: tuple[str, ...]
    payload_sha256: str = ""
    category: str = "other"
    risk_class: RiskClass = "unknown"
    creates_commitment: bool = False
    conversation_kind: ConversationKind = "unknown"
    account_id: str = ""

    @classmethod
    def build(
        cls,
        *,
        action_type: ActionType,
        channel: str,
        conversation_id: str,
        source_event_ids: Sequence[str],
        payload: str = "",
        category: str = "other",
        risk_class: RiskClass = "unknown",
        creates_commitment: bool = False,
        conversation_kind: ConversationKind = "unknown",
        account_id: str = "",
    ) -> "CommunicationActionProposal":
        if not channel or not conversation_id:
            raise CommunicationStateError("action scope is required")
        sources = tuple(sorted({str(item) for item in source_event_ids if str(item)}))
        material = {
            "action_type": action_type,
            "channel": channel,
            "account_id": account_id,
            "conversation_id": conversation_id,
            "source_event_ids": sources,
            "payload_sha256": sha256_text(payload) if payload else "",
            "category": category,
            "risk_class": risk_class,
            "creates_commitment": bool(creates_commitment),
            "conversation_kind": conversation_kind,
        }
        return cls(
            action_id="ca_" + sha256_text(canonical_json(material))[:40],
            action_type=action_type,
            channel=channel,
            account_id=account_id,
            conversation_id=conversation_id,
            source_event_ids=sources,
            payload_sha256=material["payload_sha256"],
            category=category,
            risk_class=risk_class,
            creates_commitment=bool(creates_commitment),
            conversation_kind=conversation_kind,
        )
