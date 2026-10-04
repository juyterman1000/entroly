"""Communication Assurance: durable evidence and bounded action policy."""

from .models import (
    SCHEMA,
    CommunicationActionProposal,
    CommunicationEvent,
    CommunicationStateConflict,
    CommunicationStateError,
    event_from_adapter,
)
from .policy import CommunicationPolicy
from .store import (
    DEFAULT_RETENTION_DAYS,
    CommunicationStore,
    default_store_path,
    normalize_retention_days,
    resolve_store_path,
)

__all__ = [
    "SCHEMA",
    "DEFAULT_RETENTION_DAYS",
    "CommunicationActionProposal",
    "CommunicationEvent",
    "CommunicationPolicy",
    "CommunicationStateConflict",
    "CommunicationStateError",
    "CommunicationStore",
    "default_store_path",
    "event_from_adapter",
    "normalize_retention_days",
    "resolve_store_path",
]
