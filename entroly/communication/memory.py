"""MemoryOS/Hippocampus adapter for learned communication taste.

Raw message evidence stays in CommunicationStore.  This adapter remembers only
derived preference observations.  Contact/group memories use MemoryOS
agent-partitions; the current global Hippocampus adapter is used only for
owner-global preference patterns to avoid cross-contact recall leakage.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Sequence

from ..memory_fabric import MemoryFabric
from .models import CommunicationStateError, canonical_json, sha256_text
from .preferences import CommunicationTaste, resolve_taste
from .store import default_store_path

_TASTE_SCHEMA = "entroly.communication.taste-memory.v1"


def default_communication_memory_path() -> Path:
    return default_store_path().parent / "taste-memory.json"


def resolve_communication_memory_path(
    path: str | os.PathLike[str] | None,
) -> Path:
    if path is None or not str(path).strip():
        return default_communication_memory_path()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise CommunicationStateError(
            "communication memory override must be an absolute path"
        )
    return candidate.absolute()


def _scope_agent_id(scope_type: str, scope_id: str) -> str:
    if not scope_type or not scope_id:
        raise CommunicationStateError("communication memory scope is required")
    digest = sha256_text(f"{scope_type}:{scope_id}")[:24]
    return f"communication:{scope_type}:{digest}"


def _taste_payload(taste: CommunicationTaste) -> str:
    return canonical_json(
        {
            "schema": _TASTE_SCHEMA,
            "source": taste.source,
            "confidence": float(taste.confidence),
            "evidence_event_ids": list(taste.evidence_event_ids),
            "preferred_language": taste.preferred_language,
            "formality": taste.formality,
            "response_length": taste.response_length,
            "emoji_level": taste.emoji_level,
            "routine_action": taste.routine_action,
            "preferred_reaction": taste.preferred_reaction,
            "greeting_style": taste.greeting_style,
            "signoff_style": taste.signoff_style,
            "notes": dict(taste.notes),
        }
    )


def _taste_from_payload(
    *,
    scope_type: str,
    scope_id: str,
    content: str,
) -> CommunicationTaste | None:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _TASTE_SCHEMA:
        return None
    source = str(payload.get("source") or "")
    if source not in {"default", "explicit", "inferred"}:
        return None
    evidence = payload.get("evidence_event_ids")
    evidence_ids = (
        tuple(str(item) for item in evidence if str(item))
        if isinstance(evidence, list)
        else ()
    )
    try:
        return CommunicationTaste.build(
            scope_type=scope_type,  # type: ignore[arg-type]
            scope_id=scope_id,
            source=source,  # type: ignore[arg-type]
            confidence=float(payload.get("confidence", 0.0)),
            evidence_event_ids=evidence_ids,
            preferred_language=str(payload.get("preferred_language") or "adaptive"),
            formality=str(payload.get("formality") or "adaptive"),
            response_length=str(payload.get("response_length") or "adaptive"),
            emoji_level=str(payload.get("emoji_level") or "adaptive"),
            routine_action=str(payload.get("routine_action") or "none"),
            preferred_reaction=str(payload.get("preferred_reaction") or ""),
            greeting_style=str(payload.get("greeting_style") or ""),
            signoff_style=str(payload.get("signoff_style") or ""),
            notes=payload.get("notes") if isinstance(payload.get("notes"), dict) else {},
        )
    except (CommunicationStateError, TypeError, ValueError):
        return None


class CommunicationMemory:
    """Durable learned communication preferences on Entroly MemoryFabric."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        enable_long_term: bool = True,
        enable_native: bool = True,
    ) -> None:
        self.path = resolve_communication_memory_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            self.path.parent.chmod(0o700)
        self._lock = threading.RLock()
        if self.path.exists():
            self.fabric = MemoryFabric.load(
                self.path,
                enable_long_term=enable_long_term,
                enable_native=enable_native,
            )
        else:
            self.fabric = MemoryFabric(
                enable_long_term=enable_long_term,
                enable_native=enable_native,
            )

    def remember_taste(
        self,
        taste: CommunicationTaste,
        *,
        mirror_owner_long_term: bool = True,
    ) -> dict[str, Any]:
        """Remember a preference observation without granting action authority."""
        if taste.source == "default":
            raise CommunicationStateError(
                "default taste does not need durable memory"
            )
        payload = _taste_payload(taste)
        agent_id = _scope_agent_id(taste.scope_type, taste.scope_id)
        tier = "semantic" if taste.source == "explicit" else "episodic"
        importance = (
            1.0
            if taste.source == "explicit"
            else max(0.5, min(0.95, float(taste.confidence)))
        )
        tags = [
            "communication",
            "taste",
            taste.source,
            f"scope:{taste.scope_type}",
        ]
        with self._lock:
            memory_id = self.fabric.remember(
                payload,
                agent_id=agent_id,
                importance=importance,
                tier=tier,  # type: ignore[arg-type]
                source=f"communication_taste:{taste.scope_type}",
                tags=tags,
            )
            # MemoryOS is the durable correctness layer.  Hippocampus is an
            # optional global recall accelerator, so only owner-global tastes
            # are mirrored there.
            long_term = {
                "remembered": False,
                "reason": "scope_partition_required",
                "count": 0,
            }
            evidence_is_strong = (
                taste.source == "explicit"
                or (
                    taste.source == "inferred"
                    and taste.confidence >= 0.85
                    and len(set(taste.evidence_event_ids)) >= 3
                )
            )
            if (
                mirror_owner_long_term
                and taste.scope_type == "owner"
                and evidence_is_strong
            ):
                long_term = self.fabric.remember_long_term(
                    payload,
                    source="communication_taste:owner",
                    importance=importance,
                )
            self.fabric.save(self.path)
        return {
            "memory_id": memory_id,
            "tier": tier,
            "agent_partition": agent_id,
            "long_term": long_term,
            "authority_expanded": False,
        }

    def recall_tastes(
        self,
        *,
        scope_type: str,
        scope_id: str,
        budget: int = 2048,
    ) -> list[CommunicationTaste]:
        """Recall only one exact MemoryOS partition; never global Hippocampus."""
        agent_id = _scope_agent_id(scope_type, scope_id)
        with self._lock:
            context = self.fabric.memory_os.recall(
                "communication taste preference style",
                agent_id=agent_id,
                budget=max(128, min(int(budget), 8192)),
                include_shared=False,
            )
        profiles: list[CommunicationTaste] = []
        for selected in context.selected:
            taste = _taste_from_payload(
                scope_type=scope_type,
                scope_id=scope_id,
                content=selected.content,
            )
            if taste is not None:
                profiles.append(taste)
        return profiles

    def resolve(
        self,
        scopes: Sequence[tuple[str, str]],
        *,
        budget_per_scope: int = 2048,
    ) -> dict[str, Any]:
        """Resolve broad-to-narrow taste without crossing memory partitions."""
        profiles: list[CommunicationTaste] = []
        for scope_type, scope_id in scopes:
            profiles.extend(
                self.recall_tastes(
                    scope_type=scope_type,
                    scope_id=scope_id,
                    budget=budget_per_scope,
                )
            )
        return {
            "resolved": resolve_taste(*profiles),
            "profiles": [profile.to_dict() for profile in profiles],
            "authority_expanded": False,
            "memory_layers": [
                layer.as_dict() for layer in self.fabric.capabilities()
            ],
        }

    def consolidate(self) -> dict[str, object]:
        """Run Entroly's existing MemoryOS + optional Hippocampus consolidation."""
        with self._lock:
            result = self.fabric.consolidate()
            self.fabric.save(self.path)
            return result

    def stats(self) -> dict[str, object]:
        with self._lock:
            return self.fabric.stats()
