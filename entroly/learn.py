"""
Failure mining — `entroly learn` command.

Mines session logs, vault data, and PRISM feedback for failure patterns,
clusters them by entity/file, and writes corrections to agent config files
(CLAUDE.md, .cursorrules, .github/copilot-instructions.md, etc.).

This is the user-facing "self-improvement" surface. The evolution daemon does
the same thing automatically; `entroly learn` gives the user control over what
gets learned and where corrections are written.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """A recurring failure pattern discovered from session data."""
    pattern_id: str
    entity: str           # file, function, or concept the failure relates to
    category: str         # error_type, wrong_assumption, missed_context, etc.
    description: str
    occurrences: int
    examples: list[str] = field(default_factory=list)
    suggested_fix: str = ""
    confidence: float = 0.0


@dataclass
class Correction:
    """A correction to be written to an agent config file."""
    target_file: str      # e.g., CLAUDE.md, .cursorrules
    section: str          # where in the file
    content: str          # the correction text
    pattern: FailurePattern
    applied: bool = False


# ---------------------------------------------------------------------------
# Failure mining
# ---------------------------------------------------------------------------

class FailureMiner:
    """Mines session data for recurring failure patterns."""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path(os.environ.get("ENTROLY_DIR", ".entroly"))
        self._root = Path(root)

    def mine(self, min_occurrences: int = 2) -> list[FailurePattern]:
        """
        Mine all available data sources for failure patterns.

        Sources:
          1. PRISM feedback (outcome=negative)
          2. Vault beliefs (status=invalidated or confidence < 0.3)
          3. Evolution daemon failures
          4. Session checkpoint errors
        """
        patterns: list[FailurePattern] = []
        patterns.extend(self._mine_prism())
        patterns.extend(self._mine_vault())
        patterns.extend(self._mine_evolution())
        patterns.extend(self._mine_checkpoints())

        # Cluster by entity and deduplicate
        clustered = self._cluster(patterns)

        # Filter by minimum occurrences
        return [p for p in clustered if p.occurrences >= min_occurrences]

    def _mine_prism(self) -> list[FailurePattern]:
        """Mine PRISM feedback for negative outcomes."""
        prism_dir = self._root / "prism"
        if not prism_dir.exists():
            return []

        patterns = []
        error_counts: dict[str, list[dict]] = collections.defaultdict(list)

        for path in prism_dir.glob("*.jsonl"):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("outcome") in ("negative", "failure", "error"):
                        entity = record.get("entity", record.get("file", "unknown"))
                        error_counts[entity].append(record)
            except (json.JSONDecodeError, OSError):
                continue

        for entity, records in error_counts.items():
            if len(records) >= 1:
                examples = [
                    r.get("error", r.get("description", ""))[:200]
                    for r in records[:5]
                ]
                patterns.append(FailurePattern(
                    pattern_id=f"prism_{_slugify(entity)}",
                    entity=entity,
                    category="prism_negative",
                    description=f"PRISM recorded {len(records)} negative outcomes for {entity}",
                    occurrences=len(records),
                    examples=[e for e in examples if e],
                    confidence=min(0.5 + len(records) * 0.1, 0.95),
                ))

        return patterns

    def _mine_vault(self) -> list[FailurePattern]:
        """Mine vault for invalidated or low-confidence beliefs."""
        vault_dir = self._root / "vault" / "beliefs"
        if not vault_dir.exists():
            return []

        patterns = []
        for path in vault_dir.glob("*.md"):
            try:
                content = path.read_text(encoding="utf-8")
                # Parse YAML frontmatter
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end > 0:
                        frontmatter = content[3:end]
                        if "status: invalidated" in frontmatter or "status: contradicted" in frontmatter:
                            entity = _extract_yaml_field(frontmatter, "entity")
                            confidence = float(_extract_yaml_field(frontmatter, "confidence") or "0")
                            patterns.append(FailurePattern(
                                pattern_id=f"vault_{path.stem}",
                                entity=entity or path.stem,
                                category="belief_invalidated",
                                description=f"Belief about {entity or path.stem} was invalidated",
                                occurrences=1,
                                confidence=1.0 - confidence,
                            ))
            except (OSError, ValueError):
                continue

        return patterns

    def _mine_evolution(self) -> list[FailurePattern]:
        """Mine evolution daemon for failed skill synthesis."""
        evo_dir = self._root / "vault" / "evolution" / "skills"
        if not evo_dir.exists():
            return []

        patterns = []
        for path in evo_dir.glob("*.json"):
            try:
                skill = json.loads(path.read_text(encoding="utf-8"))
                fitness = skill.get("fitness", 0.5)
                if fitness <= 0.3:
                    patterns.append(FailurePattern(
                        pattern_id=f"evo_{path.stem}",
                        entity=skill.get("entity", path.stem),
                        category="skill_failure",
                        description=f"Skill {path.stem} has fitness {fitness:.2f} (pruning threshold)",
                        occurrences=skill.get("test_count", 1),
                        confidence=1.0 - fitness,
                    ))
            except (json.JSONDecodeError, OSError):
                continue

        return patterns

    def _mine_checkpoints(self) -> list[FailurePattern]:
        """Mine session checkpoints for error patterns."""
        checkpoint_dir = self._root / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        error_entities: dict[str, int] = collections.defaultdict(int)
        for path in checkpoint_dir.glob("*.json"):
            try:
                cp = json.loads(path.read_text(encoding="utf-8"))
                errors = cp.get("errors", [])
                for err in errors:
                    entity = err.get("file", err.get("entity", "unknown"))
                    error_entities[entity] += 1
            except (json.JSONDecodeError, OSError):
                continue

        return [
            FailurePattern(
                pattern_id=f"cp_{_slugify(entity)}",
                entity=entity,
                category="checkpoint_error",
                description=f"{count} checkpoint errors involving {entity}",
                occurrences=count,
                confidence=min(0.4 + count * 0.1, 0.9),
            )
            for entity, count in error_entities.items()
        ]

    def _cluster(self, patterns: list[FailurePattern]) -> list[FailurePattern]:
        """Cluster patterns by entity, merging duplicates."""
        by_entity: dict[str, list[FailurePattern]] = collections.defaultdict(list)
        for p in patterns:
            by_entity[p.entity].append(p)

        merged = []
        for entity, group in by_entity.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            total_occ = sum(p.occurrences for p in group)
            categories = list(set(p.category for p in group))
            examples = []
            for p in group:
                examples.extend(p.examples)

            merged.append(FailurePattern(
                pattern_id=f"cluster_{_slugify(entity)}",
                entity=entity,
                category="+".join(categories),
                description=f"{total_occ} failures across {len(group)} sources for {entity}",
                occurrences=total_occ,
                examples=examples[:10],
                confidence=max(p.confidence for p in group),
            ))

        merged.sort(key=lambda p: (-p.occurrences, -p.confidence))
        return merged


# ---------------------------------------------------------------------------
# Correction generation
# ---------------------------------------------------------------------------

AGENT_CONFIG_FILES = {
    "claude": "CLAUDE.md",
    "cursor": ".cursorrules",
    "copilot": ".github/copilot-instructions.md",
    "codex": "AGENTS.md",
    "aider": ".aider.conf.yml",
}


def generate_corrections(
    patterns: list[FailurePattern],
    agents: list[str] | None = None,
) -> list[Correction]:
    """
    Generate corrections from failure patterns for agent config files.

    Args:
        patterns: Mined failure patterns
        agents: Which agent configs to target (default: all detected)

    Returns:
        List of corrections ready to be applied
    """
    if agents is None:
        agents = [name for name, path in AGENT_CONFIG_FILES.items() if Path(path).exists()]
        if not agents:
            agents = ["claude"]

    corrections = []
    for pattern in patterns:
        if pattern.confidence < 0.5:
            continue

        correction_text = _pattern_to_correction(pattern)
        if not correction_text:
            continue

        for agent in agents:
            target = AGENT_CONFIG_FILES.get(agent, f".{agent}rules")
            corrections.append(Correction(
                target_file=target,
                section="## Learned Corrections",
                content=correction_text,
                pattern=pattern,
            ))

    return corrections


def apply_corrections(corrections: list[Correction], dry_run: bool = True) -> list[Correction]:
    """
    Apply corrections to agent config files.

    Args:
        dry_run: If True, only report what would be changed

    Returns:
        List of corrections that were (or would be) applied
    """
    applied = []
    for corr in corrections:
        path = Path(corr.target_file)

        if dry_run:
            corr.applied = False
            applied.append(corr)
            continue

        try:
            if path.exists():
                content = path.read_text(encoding="utf-8")
            else:
                content = f"# {path.name}\n\n"

            section_header = f"\n\n{corr.section}\n\n"
            if corr.section not in content:
                content += section_header

            # Append correction under section
            idx = content.find(corr.section)
            if idx >= 0:
                insert_point = content.find("\n##", idx + len(corr.section))
                if insert_point < 0:
                    insert_point = len(content)
                content = (
                    content[:insert_point]
                    + f"\n- {corr.content}\n"
                    + content[insert_point:]
                )

            path.write_text(content, encoding="utf-8")
            corr.applied = True
            applied.append(corr)
        except OSError as exc:
            logger.error("Failed to apply correction to %s: %s", path, exc)

    return applied


def _pattern_to_correction(pattern: FailurePattern) -> str:
    """Convert a failure pattern into a human-readable correction."""
    if pattern.category == "prism_negative":
        return (
            f"**{pattern.entity}**: PRISM recorded {pattern.occurrences} negative "
            f"outcomes. Review this entity carefully before modifying — past changes "
            f"have caused issues."
        )
    if pattern.category == "belief_invalidated":
        return (
            f"**{pattern.entity}**: A prior belief about this entity was invalidated. "
            f"Verify current state before relying on cached understanding."
        )
    if pattern.category == "skill_failure":
        return (
            f"**{pattern.entity}**: Automated skill synthesis failed for this entity "
            f"(fitness below threshold). Manual attention needed."
        )
    if pattern.category == "checkpoint_error":
        return (
            f"**{pattern.entity}**: {pattern.occurrences} session errors involving "
            f"this entity. Common failure point — proceed with extra validation."
        )
    return f"**{pattern.entity}**: {pattern.description}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def learning_report(
    patterns: list[FailurePattern],
    corrections: list[Correction],
) -> str:
    """Generate a human-readable learning report."""
    lines = ["# Entroly Learning Report", ""]

    if not patterns:
        lines.append("No recurring failure patterns found.")
        return "\n".join(lines)

    lines.append(f"## {len(patterns)} Failure Patterns Found")
    lines.append("")

    for p in patterns[:20]:
        lines.append(
            f"- **{p.entity}** ({p.category}): {p.occurrences} occurrences, "
            f"confidence {p.confidence:.0%}"
        )
        if p.examples:
            for ex in p.examples[:2]:
                lines.append(f"  - {ex[:100]}")

    lines.append("")
    lines.append(f"## {len(corrections)} Corrections Generated")
    lines.append("")

    for c in corrections:
        status = "applied" if c.applied else "dry-run"
        lines.append(f"- [{status}] {c.target_file}: {c.content[:100]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", text)[:40].strip("_").lower()


def _extract_yaml_field(frontmatter: str, field: str) -> str | None:
    match = re.search(rf"^{field}:\s*(.+)$", frontmatter, re.MULTILINE)
    return match.group(1).strip() if match else None
