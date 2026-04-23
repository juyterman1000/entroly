---
claim_id: c4fd0f57-7047-4860-b43c-5891a886be47
entity: evolution_logger
status: inferred
confidence: 0.75
sources:
  - entroly\evolution_logger.py:30
  - entroly\evolution_logger.py:56
  - entroly\evolution_logger.py:41
  - entroly\evolution_logger.py:64
  - entroly\evolution_logger.py:87
  - entroly\evolution_logger.py:129
  - entroly\evolution_logger.py:144
last_checked: 2026-04-23T03:07:07.808125+00:00
derived_from:
  - belief_compiler
  - sast
---

# Module: evolution_logger

**Language:** python
**Lines of code:** 221

## Types
- `class MissRecord()` — A single recorded miss / failure.
- `class EvolutionLogger()` — Tracks failures and identifies skill gap candidates. Lives in the Evolution layer. Writes to vault/evolution/ when the system needs a new skill.

## Functions
- `def to_dict(self) -> dict[str, Any]`
- `def __init__(
        self,
        vault_path: str | None = None,
        gap_threshold: int = 3,
    )`
- `def record_miss(
        self,
        query: str,
        entity_key: str,
        intent: str = "",
        flow_attempted: str = "",
        reason: str = "",
        source_files: list[str] | None = None,
    ) -> dict[str, Any]`
- `def stats(self) -> dict[str, Any]` — Return evolution statistics.
- `def get_pending_gaps(self) -> list[dict[str, Any]]` — Return all reported skill gaps with their context. Used by the EvolutionDaemon to iterate over gaps that need structural synthesis or LLM-based skill creation.

## Dependencies
- `__future__`
- `collections`
- `dataclasses`
- `datetime`
- `logging`
- `pathlib`
- `time`
- `typing`
