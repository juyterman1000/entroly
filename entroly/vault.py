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

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .path_safety import resolve_file_within, resolve_output_within

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

def _atomic_write_text(path: Path, text: str) -> None:
    """Write a vault artifact so a reader never sees a partial one.

    ``Path.write_text`` truncates the target and then writes, so an
    interruption leaves a belief whose frontmatter no longer parses. Such a
    file reads as "not a belief" while the append-only ledger still records
    it, which is the same vault-disagrees-with-its-audit-trail failure a
    filename collision produced. Writing a sibling temp file and renaming it
    makes the replacement atomic on POSIX and Windows alike.
    """

    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    except OSError:
        try:
            temp_path.unlink()
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        raise


def _yaml_scalar(value: Any) -> str:
    """Flatten a value to a single frontmatter line.

    The frontmatter parser is line-based, so any newline in a value begins a
    new key and lets that value forge the ones around it. Collapsing all
    whitespace is enough to close that, and is lossless for the identifiers,
    paths and timestamps these fields actually hold.
    """

    return " ".join(str(value).split())


@dataclass
class BeliefArtifact:
    """A machine-auditable belief written to the vault."""
    claim_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity: str = ""
    status: str = "inferred"  # observed | inferred | verified | stale | hypothesis
    confidence: float = 0.5
    sources: list[str] = field(default_factory=list)
    last_checked: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    derived_from: list[str] = field(default_factory=list)
    title: str = ""
    body: str = ""
    # Directory the sources are relative to, itself relative to the project
    # root. `entroly compile scripts` records `helper.py` as a source and
    # `scripts` here, so the pair resolves to exactly one file. Without it a
    # source is only a suffix and can be matched against the wrong file, or
    # against none. Empty on beliefs written before this field existed, and
    # omitted from the frontmatter in that case so their bytes do not change.
    source_root: str = ""

    def to_markdown(self) -> str:
        """Render as markdown with YAML frontmatter."""
        # Frontmatter is parsed line by line, so a newline inside any value
        # starts a new key. Entity names come from indexed source code, which
        # is untrusted input: an entity of "x\nclaim_id: FORGED" rewrote the
        # claim_id -- the identifier the append-only ledger is cross-referenced
        # by. Every scalar is flattened to one line before it is written.
        sources_yaml = (
            "\n".join(f"  - {_yaml_scalar(s)}" for s in self.sources)
            if self.sources
            else "  - unknown"
        )
        derived_yaml = (
            "\n".join(f"  - {_yaml_scalar(d)}" for d in self.derived_from)
            if self.derived_from
            else "  - system"
        )
        source_root = _yaml_scalar(self.source_root)
        source_root_yaml = f"source_root: {source_root}\n" if source_root else ""

        return (
            f"---\n"
            f"claim_id: {_yaml_scalar(self.claim_id)}\n"
            f"entity: {_yaml_scalar(self.entity)}\n"
            f"status: {_yaml_scalar(self.status)}\n"
            f"confidence: {self.confidence}\n"
            f"sources:\n{sources_yaml}\n"
            f"{source_root_yaml}"
            f"last_checked: {_yaml_scalar(self.last_checked)}\n"
            f"derived_from:\n{derived_yaml}\n"
            f"---\n\n"
            f"# {self.title or self.entity}\n\n"
            f"{self.body}\n"
        )

    def to_dict(self) -> dict[str, Any]:
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
            # Flattened for the same reason as BeliefArtifact: `challenges`
            # holds the claim_id this verification is about, and a newline in
            # any value would let it rewrite the others.
            f"---\n"
            f"challenges: {_yaml_scalar(self.challenges)}\n"
            f"result: {_yaml_scalar(self.result)}\n"
            f"confidence_delta: {self.confidence_delta:+.2f}\n"
            f"checked_at: {_yaml_scalar(self.checked_at)}\n"
            f"method: {_yaml_scalar(self.method)}\n"
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

    def __init__(self, config: VaultConfig | None = None):
        self.config = config or VaultConfig()
        self._base = self.config.path
        self._initialized = False

    def ensure_structure(self) -> dict[str, Any]:
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

    def write_belief(self, artifact: BeliefArtifact) -> dict[str, Any]:
        """Write a belief artifact to the vault."""
        self.ensure_structure()

        # Sanitize entity for filename. The mapping is many-to-one, so if the
        # preferred name is already held by a *different* entity, fall back to
        # a digest-qualified one. Without this the second belief overwrote the
        # first and the vault silently held one belief while the append-only
        # ledger recorded two -- the audit trail asserting something the vault
        # had destroyed.
        entity = artifact.entity or artifact.claim_id
        preferred, disambiguated = _belief_filenames(entity)
        file_path = self._base / "beliefs" / f"{preferred}.md"
        if file_path.exists():
            owner = _entity_of(file_path)
            if owner is not None and owner != entity:
                file_path = self._base / "beliefs" / f"{disambiguated}.md"

        safe_path = resolve_output_within(self._base, file_path)
        if safe_path is None:
            raise ValueError(f"Refusing to write outside vault: {file_path}")
        _atomic_write_text(safe_path, artifact.to_markdown())

        # The per-entity file is overwrite-in-place; the append-only ledger
        # preserves every version for as-of/diff/timeline queries. A ledger
        # failure must be visible in the result, but never lose the write.
        try:
            from .vault_time import BeliefLedger
            ledger = BeliefLedger(self._base).record(artifact)
        except Exception as exc:
            logger.error(f"Vault: belief ledger append failed: {exc}")
            ledger = {"status": "failed", "error": str(exc)}

        logger.info(f"Vault: wrote belief '{artifact.entity}' -> {file_path}")
        return {
            "status": "written",
            "directory": "beliefs",
            "path": str(safe_path),
            "claim_id": artifact.claim_id,
            "entity": artifact.entity,
            "ledger": ledger,
        }

    def read_belief(self, entity: str) -> dict[str, Any] | None:
        """Read a belief artifact by entity name."""
        self.ensure_structure()
        beliefs_dir = self._base / "beliefs"

        preferred, disambiguated = _belief_filenames(entity)
        file_path = beliefs_dir / f"{preferred}.md"

        # The preferred name may be held by a different entity after a
        # sanitisation collision, in which case this entity lives under its
        # digest-qualified name.
        if file_path.exists() and _entity_of(file_path) not in (None, entity):
            alternative = beliefs_dir / f"{disambiguated}.md"
            if alternative.exists():
                file_path = alternative

        if not file_path.exists():
            file_path = beliefs_dir / f"{disambiguated}.md"

        if not file_path.exists():
            # Last resort: find the file that *claims* this entity. Matching on
            # the recorded entity rather than on the filename is what keeps the
            # answer correct -- the previous substring match returned the first
            # file whose stem merely contained the query, so asking for `cache`
            # answered with `cache_aligner`'s belief under `cache_aligner`'s
            # name, which in an auditable store is a wrong answer, not a
            # near miss.
            for md in sorted(beliefs_dir.rglob("*.md")):
                candidate = resolve_file_within(beliefs_dir, md)
                if candidate is None:
                    continue
                if _entity_of(candidate) == entity:
                    file_path = md
                    break
            else:
                return None

        safe_path = resolve_file_within(beliefs_dir, file_path)
        if safe_path is None:
            return None
        content = safe_path.read_text(encoding="utf-8", errors="replace")
        frontmatter = _parse_frontmatter(content)
        body = _extract_body(content)

        return {
            "path": str(safe_path),
            "frontmatter": frontmatter or {},
            "body": body,
        }

    def list_beliefs(self) -> list[dict[str, Any]]:
        """List all belief artifacts with their frontmatter."""
        self.ensure_structure()
        beliefs_dir = self._base / "beliefs"
        results = []

        for md in sorted(beliefs_dir.rglob("*.md")):
            try:
                safe_path = resolve_file_within(beliefs_dir, md)
                if safe_path is None:
                    continue
                content = safe_path.read_text(encoding="utf-8", errors="replace")
                fm = _parse_frontmatter(content)
                results.append({
                    "file": str(safe_path.relative_to(beliefs_dir.resolve())),
                    "entity": fm.get("entity", md.stem) if fm else md.stem,
                    "status": fm.get("status", "unknown") if fm else "unknown",
                    "confidence": float(fm.get("confidence", 0)) if fm else 0,
                    "last_checked": fm.get("last_checked", "") if fm else "",
                })
            except Exception as e:
                logger.debug(f"Vault: failed to read {md}: {e}")

        return results

    # â”€â”€ Verification Operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def write_verification(self, artifact: VerificationArtifact) -> dict[str, Any]:
        """Write a verification artifact to the vault."""
        self.ensure_structure()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_title = _safe_filename(artifact.title or artifact.challenges)
        file_path = self._base / "verification" / f"{timestamp}_{safe_title}.md"

        safe_path = resolve_output_within(self._base, file_path)
        if safe_path is None:
            raise ValueError(f"Refusing to write outside vault: {file_path}")
        _atomic_write_text(safe_path, artifact.to_markdown())

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
            "path": str(safe_path),
            "challenges": artifact.challenges,
            "result": artifact.result,
        }

    # â”€â”€ Action Operations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def write_action(
        self,
        title: str,
        content: str,
        action_type: str = "report",
    ) -> dict[str, Any]:
        """Write an action output (report, PR brief, etc.) to the vault."""
        self.ensure_structure()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_title = _safe_filename(title)
        file_path = self._base / "actions" / f"{timestamp}_{safe_title}.md"

        safe_path = resolve_output_within(self._base, file_path)
        if safe_path is None:
            raise ValueError(f"Refusing to write outside vault: {file_path}")
        _atomic_write_text(
            safe_path,
            f"---\ntype: {_yaml_scalar(action_type)}\ntimestamp: {_yaml_scalar(timestamp)}\n---\n\n"
            f"# {title}\n\n{content}\n",
        )

        logger.info(f"Vault: wrote action '{title}' -> {file_path}")
        return {
            "status": "written",
            "directory": "actions",
            "path": str(safe_path),
            "type": action_type,
        }

    # â”€â”€ Coverage Index â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def coverage_index(self) -> dict[str, Any]:
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

    def mark_beliefs_stale_for_files(self, changed_files: list[str]) -> dict[str, Any]:
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

        updated_entities: list[str] = []
        updated_files: list[str] = []
        already_stale: list[str] = []

        beliefs_dir = self._base / "beliefs"
        for md in beliefs_dir.rglob("*.md"):
            try:
                safe_path = resolve_file_within(beliefs_dir, md)
                if safe_path is None:
                    continue
                content = safe_path.read_text(encoding="utf-8", errors="replace")
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
                _atomic_write_text(safe_path, updated)
                updated_entities.append(entity)
                updated_files.append(str(safe_path))
            except Exception as e:
                logger.debug(f"Vault: failed to mark stale for {md}: {e}")

        return {
            "status": "updated",
            "changed_files": len(changed_files),
            "updated_entities": updated_entities,
            "updated_files": updated_files,
            "already_stale": already_stale,
        }

    def mark_beliefs_ungrounded(
        self, roots: list[str] | None = None
    ) -> dict[str, Any]:
        """Retract beliefs whose every cited source file is gone from disk.

        Compilation is additive: it writes a belief when it sees a file and
        never revisits one whose file was deleted or moved. The belief then
        keeps its original confidence forever, and retrieval returns it beside
        live beliefs with nothing to tell them apart -- a confident claim about
        code that no longer exists, which is the failure mode the fail-closed
        rule exists to prevent.

        Sources are recorded relative to whichever directory compilation was
        given, and a belief does not record which directory that was, so each
        source is resolved against the project root and against every supplied
        root before it is called missing. A belief is retracted only when no
        candidate resolves; the bias is deliberately toward leaving a belief
        alone, because wrongly retracting a live belief loses knowledge while
        wrongly keeping a dead one is merely noise this scan will catch again.

        The belief is marked, never deleted: it stays auditable, and the next
        compilation that sees the file again restores it.
        """

        # Function-local, matching mark_beliefs_stale_for_files: `vault` sits in
        # the import cycle CLAUDE.md flags as load-bearing.
        import re

        self.ensure_structure()

        search_roots = [self._base.resolve().parent.parent]
        for root in roots or []:
            try:
                search_roots.append(Path(root).resolve())
            except (OSError, ValueError):
                continue

        project_root = search_roots[0]

        # Legacy beliefs carry no `source_root`, so their sources can only be
        # matched by path *suffix*: `entroly compile scripts` recorded
        # `_release_artifacts.py` and `entroly compile entroly` recorded
        # `adaptive_budget.py`, with nothing saying which directory either came
        # from. Joining those against the project root alone retracted 275 of
        # 715 real beliefs here. The index is built lazily because a vault
        # written entirely by current code never needs it.
        suffix_index: set[str] | None = None

        def _grounded(source: str, source_root: str) -> bool:
            nonlocal suffix_index
            raw = source.split(":", 1)[0].replace("\\", "/").strip().lstrip("./")
            if not _is_path_like(raw):
                # `write_belief` substitutes the sentinel `unknown` when a
                # belief carries no provenance, and callers pass bare labels
                # too. "Provenance was never recorded" is the opposite of
                # "the file is gone" and must never retract anything.
                return True
            if source_root:
                # Exact: the belief recorded the directory its sources are
                # relative to, so there is exactly one file to look for.
                return (project_root / source_root / raw).exists()
            if suffix_index is None:
                suffix_index = _path_suffix_index(search_roots)
            # Deliberately permissive: a bare `helper.py` matches any file of
            # that name anywhere in the tree, so a belief about a deleted
            # `scripts/helper.py` survives while a `tests/helper.py` exists.
            # That is the safe direction -- wrongly retracting loses knowledge,
            # wrongly keeping is noise the next scan catches -- and it only
            # affects beliefs written before `source_root` existed.
            return raw in suffix_index

        retracted: list[str] = []
        already: list[str] = []
        beliefs_dir = self._base / "beliefs"
        for md in beliefs_dir.rglob("*.md"):
            try:
                safe_path = resolve_file_within(beliefs_dir, md)
                if safe_path is None:
                    continue
                content = safe_path.read_text(encoding="utf-8", errors="replace")
                fm = _parse_frontmatter(content)
                if not fm:
                    continue

                sources = _extract_sources(content)
                if not sources:
                    continue  # nothing to verify against; leave it alone
                source_root = str(fm.get("source_root", "") or "").strip()
                # One surviving source is enough. A belief citing ten entities
                # in a file that still exists is still about real code; only a
                # belief with nothing left standing has lost its evidence.
                if any(_grounded(src, source_root) for src in sources):
                    continue

                entity = fm.get("entity", md.stem)
                if fm.get("status", "") == "ungrounded":
                    already.append(entity)
                    continue

                updated = content
                if "status:" in updated:
                    updated = re.sub(
                        r"^status:\s+.+$", "status: ungrounded", updated, count=1, flags=re.M
                    )
                # Confidence is the number retrieval ranks on. A belief with no
                # surviving source has no evidence left to justify one.
                if "confidence:" in updated:
                    updated = re.sub(
                        r"^confidence:\s+.+$", "confidence: 0.0", updated, count=1, flags=re.M
                    )
                _atomic_write_text(safe_path, updated)
                retracted.append(entity)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Vault: failed to check groundedness for {md}: {e}")

        return {
            "status": "updated",
            "retracted_entities": retracted,
            "already_ungrounded": already,
        }

    # â”€â”€ Private Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _update_belief_confidence(
        self, claim_id: str, delta: float
    ) -> None:
        """Update a belief's confidence after verification."""
        beliefs_dir = self._base / "beliefs"
        for md in beliefs_dir.rglob("*.md"):
            try:
                safe_path = resolve_file_within(beliefs_dir, md)
                if safe_path is None:
                    continue
                content = safe_path.read_text(encoding="utf-8", errors="replace")
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
                    _atomic_write_text(safe_path, updated)
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


def _entity_of(path: Path) -> str | None:
    """The entity a belief file claims, or None if it is not a readable belief."""

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    frontmatter = _parse_frontmatter(content)
    if frontmatter is None:
        return None
    return frontmatter.get("entity")


def _disambiguated_filename(entity: str) -> str:
    """A filename that survives a sanitisation collision.

    ``_safe_filename`` is many-to-one: it maps every character outside
    ``[\\w.-]`` to ``_`` and truncates at 80, so ``foo::bar`` and ``foo_bar``
    both become ``foo_bar``. Appending a digest of the full entity restores
    injectivity. Used only when a name is already taken by a different entity,
    so every existing belief keeps the filename it was written under.
    """

    import hashlib

    digest = hashlib.sha256(entity.encode("utf-8")).hexdigest()[:10]
    return f"{_safe_filename(entity)[:69]}-{digest}"


def _belief_filenames(entity: str) -> tuple[str, str]:
    """The preferred filename for an entity and its collision-safe alternative."""

    return _safe_filename(entity or "untitled"), _disambiguated_filename(entity)


def _parse_frontmatter(content: str) -> dict[str, str] | None:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return None
    end = content.find("---", 3)
    if end < 0:
        return None

    fm_text = content[3:end].strip()
    result: dict[str, str] = {}
    for line in fm_text.splitlines():
        if ":" in line and not line.strip().startswith("-"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key and value:
                result[key] = value
    return result if result else None


_GROUNDEDNESS_SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "target", "__pycache__",
    ".entroly", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".tmp", "site-packages",
})


def _is_path_like(value: str) -> bool:
    """Whether a recorded source names a file rather than a bare label.

    Beliefs cite `src/mod.py:12`, but also carry sentinels: `write_belief`
    stores `unknown` when no provenance was given. A token with no directory
    separator and no extension is a label, and treating one as a missing file
    would retract a belief for never having recorded where it came from.
    """

    if not value:
        return False
    return "/" in value or bool(Path(value).suffix)


def _path_suffix_index(roots: list[Path]) -> set[str]:
    """Every trailing path fragment of every real file under ``roots``.

    Indexing suffixes rather than full paths is what makes a belief's source
    resolvable without knowing which directory was compiled: `x.py`,
    `pkg/x.py` and `a/pkg/x.py` all hit the same file. Heavy generated trees
    are pruned -- on this repository they are three orders of magnitude larger
    than the source and contain nothing a belief can cite.
    """

    index: set[str] = set()
    seen_roots: set[Path] = set()
    for root in roots:
        if root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(root)
        stack = [root]
        while stack:
            directory = stack.pop()
            try:
                entries = list(directory.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.name in _GROUNDEDNESS_SKIP_DIRS:
                    continue
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file():
                    try:
                        parts = entry.relative_to(root).as_posix().split("/")
                    except ValueError:  # pragma: no cover - defensive
                        continue
                    for start in range(len(parts)):
                        index.add("/".join(parts[start:]))
    return index


def _extract_sources(content: str) -> list[str]:
    """Extract sources list from frontmatter."""
    if not content.startswith("---"):
        return []
    end = content.find("---", 3)
    if end < 0:
        return []

    fm_text = content[3:end].strip().splitlines()
    sources: list[str] = []
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


