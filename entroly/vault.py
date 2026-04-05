"""
Obsidian Vault Manager
======================

Manages the persistent Knowledge Surface (Obsidian Vault) for CogOps.

Directory contract:
  vault/
    beliefs/        # Durable system understanding
    verification/   # Challenges to understanding
    actions/        # Task outputs and reports
    evolution/      # Skill specs, trials, promotions
      skills/
        skill-id/
          SKILL.md
          metrics.json
          tests/
          tool.py
      registry.md
    media/          # Shared render assets only

Every belief artifact carries machine-auditable frontmatter:
  claim_id, entity, status, confidence, sources, last_checked, derived_from
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Vault Configuration
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

VAULT_DIRS = ("beliefs", "verification", "actions", "evolution", "media")


@dataclass
class VaultConfig:
    """Configuration for the Obsidian vault."""
    base_path: str = ""
    auto_create: bool = True

    @property
    def path(self) -> Path:
        if not self.base_path:
            return Path(os.getcwd()) / ".entroly" / "vault"
        return Path(self.base_path)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Belief Artifact Schema
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@dataclass
class BeliefArtifact:
    """A machine-auditable belief written to the vault."""
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity: str = ""
    status: str = "inferred"  # observed | inferred | verified | stale | hypothesis
    confidence: float = 0.5
    sources: List[str] = field(default_factory=list)
    last_checked: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    derived_from: List[str] = field(default_factory=list)
    title: str = ""
    body: str = ""

    def to_markdown(self) -> str:
        """Render as markdown with YAML frontmatter."""
        sources_yaml = "\n".join(f"  - {s}" for s in self.sources) if self.sources else "  - unknown"
        derived_yaml = "\n".join(f"  - {d}" for d in self.derived_from) if self.derived_from else "  - system"

        return (
            f"---\n"
            f"claim_id: {self.claim_id}\n"
            f"entity: {self.entity}\n"
            f"status: {self.status}\n"
            f"confidence: {self.confidence}\n"
            f"sources:\n{sources_yaml}\n"
            f"last_checked: {self.last_checked}\n"
            f"derived_from:\n{derived_yaml}\n"
            f"---\n\n"
            f"# {self.title or self.entity}\n\n"
            f"{self.body}\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "entity": self.entity,
            "status": self.status,
            "confidence": self.confidence,
            "sources": self.sources,
            "last_checked": self.last_checked,
            "derived_from": self.derived_from,
            "title": self.title,
        }


@dataclass
class VerificationArtifact:
    """A verification challenge against a belief."""
    challenges: str = ""  # claim_id being challenged
    result: str = "pending"  # confirmed | contradicted | inconclusive | pending
    confidence_delta: float = 0.0
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    method: str = ""
    title: str = ""
    body: str = ""

    def to_markdown(self) -> str:
        return (
            f"---\n"
            f"challenges: {self.challenges}\n"
            f"result: {self.result}\n"
            f"confidence_delta: {self.confidence_delta:+.2f}\n"
            f"checked_at: {self.checked_at}\n"
            f"method: {self.method}\n"
            f"---\n\n"
            f"# {self.title}\n\n"
            f"{self.body}\n"
        )


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# The Vault Manager
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class VaultManager:
    """
    Manages the Obsidian vault directory structure and artifact I/O.

    This is the persistence layer for the Living Exocortex. All belief,
    verification, action, and evolution artifacts pass through here.
    """

    def __init__(self, config: Optional[VaultConfig] = None):
        self.config = config or VaultConfig()
        self._base = self.config.path
        self._initialized = False

    def ensure_structure(self) -> Dict[str, Any]:
        """Create the vault directory structure if it doesn't exist."""
        if self._initialized:
            return {"status": "already_initialized", "path": str(self._base)}

        created = []
        for d in VAULT_DIRS:
            dir_path = self._base / d
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(d)

        # Ensure evolution/skills/ exists
        skills_dir = self._base / "evolution" / "skills"
        if not skills_dir.exists():
            skills_dir.mkdir(parents=True, exist_ok=True)
            created.append("evolution/skills")

        # Create registry.md if missing
        registry = self._base / "evolution" / "registry.md"
        if not registry.exists():
            registry.write_text(
                "# Skill Registry\n\n"
                "Index of all dynamically generated skills.\n\n"
                "| Skill ID | Status | Created | Description |\n"
                "|---|---|---|---|\n",
                encoding="utf-8",
            )
            created.append("evolution/registry.md")

        self._initialized = True
        logger.info(f"Vault initialized at {self._base} (created: {created})")

        return {
            "status": "initialized",
            "path": str(self._base),
            "created": created,
        }

    # â”€â”€ Belief Operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def write_belief(self, artifact: BeliefArtifact) -> Dict[str, Any]:
        """Write a belief artifact to the vault."""
        self.ensure_structure()

        # Sanitize entity for filename
        safe_name = _safe_filename(artifact.entity or artifact.claim_id)
        file_path = self._base / "beliefs" / f"{safe_name}.md"

        file_path.write_text(artifact.to_markdown(), encoding="utf-8")

        logger.info(f"Vault: wrote belief '{artifact.entity}' -> {file_path}")
        return {
            "status": "written",
            "directory": "beliefs",
            "path": str(file_path),
            "claim_id": artifact.claim_id,
            "entity": artifact.entity,
        }

    def read_belief(self, entity: str) -> Optional[Dict[str, Any]]:
        """Read a belief artifact by entity name."""
        self.ensure_structure()
        beliefs_dir = self._base / "beliefs"

        safe_name = _safe_filename(entity)
        file_path = beliefs_dir / f"{safe_name}.md"

        if not file_path.exists():
            # Try fuzzy match
            for md in beliefs_dir.rglob("*.md"):
                if entity.lower() in md.stem.lower():
                    file_path = md
                    break
            else:
                return None

        content = file_path.read_text(encoding="utf-8", errors="replace")
        frontmatter = _parse_frontmatter(content)
        body = _extract_body(content)

        return {
            "path": str(file_path),
            "frontmatter": frontmatter or {},
            "body": body,
        }

    def list_beliefs(self) -> List[Dict[str, Any]]:
        """List all belief artifacts with their frontmatter."""
        self.ensure_structure()
        beliefs_dir = self._base / "beliefs"
        results = []

        for md in sorted(beliefs_dir.rglob("*.md")):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                fm = _parse_frontmatter(content)
                results.append({
                    "file": str(md.relative_to(beliefs_dir)),
                    "entity": fm.get("entity", md.stem) if fm else md.stem,
                    "status": fm.get("status", "unknown") if fm else "unknown",
                    "confidence": float(fm.get("confidence", 0)) if fm else 0,
                    "last_checked": fm.get("last_checked", "") if fm else "",
                })
            except Exception as e:
                logger.debug(f"Vault: failed to read {md}: {e}")

        return results

    # â”€â”€ Verification Operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def write_verification(self, artifact: VerificationArtifact) -> Dict[str, Any]:
        """Write a verification artifact to the vault."""
        self.ensure_structure()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_title = _safe_filename(artifact.title or artifact.challenges)
        file_path = self._base / "verification" / f"{timestamp}_{safe_title}.md"

        file_path.write_text(artifact.to_markdown(), encoding="utf-8")

        # If verification confirmed, update the belief's confidence
        if artifact.result == "confirmed" and artifact.challenges:
            self._update_belief_confidence(
                artifact.challenges,
                artifact.confidence_delta,
            )

        logger.info(f"Vault: wrote verification -> {file_path}")
        return {
            "status": "written",
            "directory": "verification",
            "path": str(file_path),
            "challenges": artifact.challenges,
            "result": artifact.result,
        }

    # â”€â”€ Action Operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def write_action(
        self,
        title: str,
        content: str,
        action_type: str = "report",
    ) -> Dict[str, Any]:
        """Write an action output (report, PR brief, etc.) to the vault."""
        self.ensure_structure()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_title = _safe_filename(title)
        file_path = self._base / "actions" / f"{timestamp}_{safe_title}.md"

        file_path.write_text(
            f"---\ntype: {action_type}\ntimestamp: {timestamp}\n---\n\n"
            f"# {title}\n\n{content}\n",
            encoding="utf-8",
        )

        logger.info(f"Vault: wrote action '{title}' -> {file_path}")
        return {
            "status": "written",
            "directory": "actions",
            "path": str(file_path),
            "type": action_type,
        }

    # â”€â”€ Coverage Index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def coverage_index(self) -> Dict[str, Any]:
        """Build a coverage index of all beliefs for the router."""
        beliefs = self.list_beliefs()

        total = len(beliefs)
        verified = sum(1 for b in beliefs if b["status"] == "verified")
        stale = sum(1 for b in beliefs if b["status"] == "stale")
        avg_confidence = (
            sum(b["confidence"] for b in beliefs) / total if total else 0.0
        )

        return {
            "total_beliefs": total,
            "verified": verified,
            "stale": stale,
            "inferred": total - verified - stale,
            "average_confidence": round(avg_confidence, 3),
            "entities": [b["entity"] for b in beliefs],
        }

    def mark_beliefs_stale_for_files(self, changed_files: List[str]) -> Dict[str, Any]:
        """Mark beliefs stale when their sources overlap the changed files."""
        self.ensure_structure()

        changed_paths = {
            Path(p).as_posix().lower()
            for p in changed_files
            if p
        }
        changed_stems = {
            Path(p).stem.lower()
            for p in changed_files
            if p
        }

        updated_entities: List[str] = []
        updated_files: List[str] = []
        already_stale: List[str] = []

        beliefs_dir = self._base / "beliefs"
        for md in beliefs_dir.rglob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                fm = _parse_frontmatter(content)
                if not fm:
                    continue

                entity = fm.get("entity", md.stem)
                entity_lc = entity.lower()
                sources = _extract_sources(content)

                matched = False
                for src in sources:
                    src_path = Path(src.split(":", 1)[0]).as_posix().lower()
                    src_stem = Path(src_path).stem.lower()
                    if src_path in changed_paths or src_stem in changed_stems:
                        matched = True
                        break

                if not matched:
                    matched = any(stem in entity_lc for stem in changed_stems)

                if not matched:
                    continue

                status = fm.get("status", "")
                if status == "stale":
                    already_stale.append(entity)
                    continue

                updated = content
                if "status:" in updated:
                    import re
                    updated = re.sub(r"^status:\s+.+$", "status: stale", updated, count=1, flags=re.M)
                md.write_text(updated, encoding="utf-8")
                updated_entities.append(entity)
                updated_files.append(str(md))
            except Exception as e:
                logger.debug(f"Vault: failed to mark stale for {md}: {e}")

        return {
            "status": "updated",
            "changed_files": len(changed_files),
            "updated_entities": updated_entities,
            "updated_files": updated_files,
            "already_stale": already_stale,
        }

    # â”€â”€ Private Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _update_belief_confidence(
        self, claim_id: str, delta: float
    ) -> None:
        """Update a belief's confidence after verification."""
        beliefs_dir = self._base / "beliefs"
        for md in beliefs_dir.rglob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                fm = _parse_frontmatter(content)
                if fm and fm.get("claim_id") == claim_id:
                    old_conf = float(fm.get("confidence", 0.5))
                    new_conf = max(0.0, min(1.0, old_conf + delta))
                    # Rewrite the confidence line
                    updated = content.replace(
                        f"confidence: {fm['confidence']}",
                        f"confidence: {new_conf}",
                    )
                    # Also update status to verified if delta is positive
                    if delta > 0 and "status: inferred" in updated:
                        updated = updated.replace(
                            "status: inferred", "status: verified"
                        )
                    # Update last_checked
                    now = datetime.now(timezone.utc).isoformat()
                    if "last_checked:" in updated:
                        import re
                        updated = re.sub(
                            r"last_checked: .+",
                            f"last_checked: {now}",
                            updated,
                        )
                    md.write_text(updated, encoding="utf-8")
                    logger.info(
                        f"Vault: updated belief {claim_id} confidence "
                        f"{old_conf:.2f} â†' {new_conf:.2f}"
                    )
                    break
            except Exception as e:
                logger.debug(f"Vault: failed to update {md}: {e}")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Utility Functions
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _safe_filename(s: str) -> str:
    """Convert a string to a safe filename."""
    import re
    safe = re.sub(r'[^\w\-.]', '_', s.strip().lower())
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe[:80] or "untitled"


def _parse_frontmatter(content: str) -> Optional[Dict[str, str]]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return None
    end = content.find("---", 3)
    if end < 0:
        return None

    fm_text = content[3:end].strip()
    result: Dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line and not line.strip().startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key and value:
                result[key] = value
    return result if result else None


def _extract_sources(content: str) -> List[str]:
    """Extract sources list from frontmatter."""
    if not content.startswith("---"):
        return []
    end = content.find("---", 3)
    if end < 0:
        return []

    fm_text = content[3:end].strip().splitlines()
    sources: List[str] = []
    in_sources = False
    for line in fm_text:
        stripped = line.strip()
        if stripped.startswith("sources:"):
            in_sources = True
            continue
        if in_sources:
            if stripped.startswith("- "):
                sources.append(stripped[2:].strip())
                continue
            if stripped and not stripped.startswith("-"):
                break
    return sources


def _extract_body(content: str) -> str:
    """Extract body content after frontmatter."""
    if not content.startswith("---"):
        return content
    end = content.find("---", 3)
    if end < 0:
        return content
    return content[end + 3:].strip()


