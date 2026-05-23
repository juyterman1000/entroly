"""
Evolution Logger
================

Tracks repeated misses and system failures to feed the Self-Improvement
Loop (Canonical Flow ⑤).

When the epistemic router detects that the system repeatedly fails on
the same entity or query pattern, this logger:
  1. Records the miss with full context
  2. Clusters failures by entity/pattern
  3. Writes miss records to vault/evolution/
  4. Identifies skill gap candidates for dynamic skill creation
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MissRecord:
    """A single recorded miss / failure."""
    query: str
    entity_key: str
    intent: str
    timestamp: float = field(default_factory=time.time)
    flow_attempted: str = ""
    reason: str = ""
    # Source files relevant to this miss (for structural synthesis)
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "entity_key": self.entity_key,
            "intent": self.intent,
            "timestamp": self.timestamp,
            "flow_attempted": self.flow_attempted,
            "reason": self.reason,
            "source_files": self.source_files,
            "iso_time": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
        }


class EvolutionLogger:
    """
    Tracks failures and identifies skill gap candidates.

    Lives in the Evolution layer. Writes to vault/evolution/ when
    the system needs a new skill.
    """

    def __init__(
        self,
        vault_path: str | None = None,
        gap_threshold: int = 3,
    ):
        """
        Args:
            vault_path: Root of the Obsidian vault.
            gap_threshold: How many misses on the same entity before
                           declaring a skill gap.
        """
        self._vault_path = Path(vault_path) if vault_path else None
        self._gap_threshold = gap_threshold

        # In-memory miss log
        self._misses: list[MissRecord] = []

        # Entity -> miss count for clustering
        self._entity_misses: dict[str, int] = defaultdict(int)

        # Already-reported gaps (avoid spamming vault)
        self._reported_gaps: set[str] = set()

    def record_miss(
        self,
        query: str,
        entity_key: str,
        intent: str = "",
        flow_attempted: str = "",
        reason: str = "",
        source_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record a miss and check if it triggers a skill gap."""
        record = MissRecord(
            query=query,
            entity_key=entity_key,
            intent=intent,
            flow_attempted=flow_attempted,
            reason=reason,
            source_files=source_files or [],
        )
        self._misses.append(record)
        self._entity_misses[entity_key] += 1

        count = self._entity_misses[entity_key]
        is_gap = count >= self._gap_threshold and entity_key not in self._reported_gaps

        result: dict[str, Any] = {
            "status": "recorded",
            "entity_key": entity_key,
            "miss_count": count,
            "is_skill_gap": is_gap,
        }

        if is_gap:
            self._reported_gaps.add(entity_key)
            gap_report = self._write_skill_gap(entity_key, count)
            result["skill_gap_report"] = gap_report
            logger.info(
                f"EvolutionLogger: skill gap detected for '{entity_key}' "
                f"(miss_count={count})"
            )

        return result

    def stats(self) -> dict[str, Any]:
        """Return evolution statistics."""
        return {
            "total_misses": len(self._misses),
            "unique_entities": len(self._entity_misses),
            "skill_gaps_detected": len(self._reported_gaps),
            "top_miss_entities": dict(
                sorted(
                    self._entity_misses.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        }

    def get_pending_gaps(self) -> list[dict[str, Any]]:
        """Return all reported skill gaps with their context.

        Used by the EvolutionDaemon to iterate over gaps that need
        structural synthesis or LLM-based skill creation.
        """
        results = []
        for entity_key in self._reported_gaps:
            related = [m for m in self._misses if m.entity_key == entity_key]
            source_files: list[str] = []
            for m in related:
                source_files.extend(m.source_files)
            # Deduplicate preserving order
            seen: set[str] = set()
            unique_sources = []
            for s in source_files:
                if s not in seen:
                    seen.add(s)
                    unique_sources.append(s)

            results.append({
                "entity_key": entity_key,
                "miss_count": self._entity_misses[entity_key],
                "queries": [m.query for m in related[-10:]],
                "intents": list(set(m.intent for m in related if m.intent)),
                "source_files": unique_sources,
            })
        return results

    def _write_skill_gap(self, entity_key: str, miss_count: int) -> dict[str, Any]:
        """Write a skill gap report to vault/evolution/."""
        if not self._vault_path:
            return {"status": "no_vault", "entity_key": entity_key}

        evolution_dir = self._vault_path / "evolution"
        evolution_dir.mkdir(parents=True, exist_ok=True)

        # Gather all misses for this entity
        related = [m for m in self._misses if m.entity_key == entity_key]

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_key = entity_key.replace(":", "_").replace("/", "_")[:60]
        file_path = evolution_dir / f"gap_{timestamp}_{safe_key}.md"

        queries = "\n".join(f"- {m.query}" for m in related[-10:])
        intents = ", ".join(set(m.intent for m in related if m.intent))

        content = (
            f"---\n"
            f"type: skill_gap\n"
            f"entity: {entity_key}\n"
            f"miss_count: {miss_count}\n"
            f"detected_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"intents: {intents}\n"
            f"status: open\n"
            f"---\n\n"
            f"# Skill Gap: {entity_key}\n\n"
            f"The system has failed {miss_count} times on queries related "
            f"to `{entity_key}`.\n\n"
            f"## Recent Failing Queries\n\n{queries}\n\n"
            f"## Recommended Action\n\n"
            f"Create a new skill in `evolution/skills/{safe_key}/` with:\n"
            f"- `SKILL.md` — procedure for handling this entity\n"
            f"- `tool.py` — any custom logic needed\n"
            f"- `tests/` — validation that the skill works\n"
            f"- `metrics.json` — tracking reward/penalty\n"
        )

        file_path.write_text(content, encoding="utf-8")
        logger.info(f"EvolutionLogger: wrote skill gap report -> {file_path}")

        return {
            "status": "written",
            "path": str(file_path),
            "entity_key": entity_key,
            "miss_count": miss_count,
        }
