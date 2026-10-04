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
from .preferences import (
    CommunicationTaste,
    inferred_taste_may_authorize_action,
    infer_taste_from_outbound,
    resolve_taste,
)
from .memory import (
    CommunicationMemory,
    default_communication_memory_path,
    resolve_communication_memory_path,
)
from .triage import (
    CommunicationAssessment,
    CommunicationEpisode,
    assess_event,
    build_digest,
    build_group_episodes,
    combine_category,
    combine_risk,
    outgoing_creates_commitment,
)
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
    "CommunicationTaste",
    "CommunicationMemory",
    "resolve_taste",
    "inferred_taste_may_authorize_action",
    "infer_taste_from_outbound",
    "default_communication_memory_path",
    "resolve_communication_memory_path",
    "CommunicationAssessment",
    "CommunicationEpisode",
    "assess_event",
    "build_digest",
    "build_group_episodes",
    "combine_category",
    "combine_risk",
    "outgoing_creates_commitment",
    "CommunicationStateConflict",
    "CommunicationStateError",
    "CommunicationStore",
    "default_store_path",
    "event_from_adapter",
    "normalize_retention_days",
    "resolve_store_path",
]
