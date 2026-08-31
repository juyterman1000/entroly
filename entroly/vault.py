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
import random
import time
import uuid
from dataclasses import dataclass, field, replace
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

_ATOMIC_REPLACE_ATTEMPTS = 18


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a vault artifact so a reader never sees a partial one.

    ``Path.write_text`` truncates the target and then writes, so an
    interruption leaves a belief whose frontmatter no longer parses. Such a
    file reads as "not a belief" while the append-only ledger still records
    it, which is the same vault-disagrees-with-its-audit-trail failure a
    filename collision produced. Writing a sibling temp file and renaming it
    makes the replacement atomic on POSIX and Windows alike.
    """

    # The token must be unique per *write*, not per process: threads in one
    # process share a pid, so a pid-only name made concurrent writers collide
    # on the same temp file and fail with EACCES on Windows. Atomicity held --
    # readers never saw a torn file -- but the writes themselves crashed.
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        # Windows refuses to rename over a file another handle has open, so a
        # concurrent reader makes `os.replace` raise EACCES. Retrying is what
        # keeps the swap both atomic and available: readers hold the file for
        # microseconds, and without this the durability fix would have traded
        # torn reads for lost writes. POSIX takes the first attempt every time.
        delay = 0.002
        for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                if attempt == _ATOMIC_REPLACE_ATTEMPTS - 1:
                    raise
                # Jittered, or concurrent writers retry in lockstep and keep
                # colliding with the same reader. Measured on Windows with four
                # writers against two readers: 61 of 240 writes failed with no
                # retry, 2 with a fixed backoff, 0 with jitter over this window.
                time.sleep(delay * (0.5 + random.random()))
                delay = min(delay * 2, 0.05)
    except OSError:
        try:
            temp_path.unlink()
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        raise


def _humanize_seconds(seconds: float) -> str:
    """A coarse duration. `0.0 day(s)` tells a reader nothing useful."""

    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.0f}h"
    return f"{seconds / 86400:.0f}d"


def vault_readiness(vault_base: str | Path, project_root: str | Path | None = None) -> dict[str, Any]:
    """Whether the belief vault can answer questions about the current tree.

    `entroly ingest` and `entroly search` use different stores: ingest fills
    `.entroly/receipts/index.json` from documents, search reads
    `.entroly/vault/beliefs/` built by `entroly compile`. Ingest reports a loud
    success either way, so running it and then searching returns whatever the
    vault last held, with nothing at the point of use saying the two are
    unrelated. Returning the reason lets a caller say so.

    Reports rather than decides: a stale answer is still an answer, and the
    caller may legitimately want it.
    """

    base = Path(vault_base)
    root = Path(project_root) if project_root is not None else base.resolve().parent.parent
    beliefs = sorted((base / "beliefs").glob("*.md")) if (base / "beliefs").exists() else []

    newest_belief = 0.0
    for path in beliefs:
        try:
            newest_belief = max(newest_belief, path.stat().st_mtime)
        except OSError:
            continue

    receipts_index = base.parent / "receipts" / "index.json"
    has_receipts = receipts_index.exists()

    reasons: list[str] = []
    if not beliefs:
        if has_receipts:
            reasons.append(
                "the belief vault is empty but a document index exists -- "
                "`entroly ingest` fills the document index, which `entroly search` "
                "does not read; run `entroly compile <dir>` to build beliefs"
            )
        else:
            reasons.append("the belief vault is empty; run `entroly compile <dir>`")
    else:
        newest_source = 0.0
        newest_source_name = ""
        for path in _iter_source_files(root):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > newest_source:
                newest_source, newest_source_name = mtime, path.name
        if newest_source > newest_belief:
            reasons.append(
                f"source is newer than the newest belief by "
                f"{_humanize_seconds(newest_source - newest_belief)} "
                f"(most recently {newest_source_name}); re-run `entroly compile <dir>` "
                "or results will describe code as it used to be"
            )

    return {
        "belief_count": len(beliefs),
        "has_document_index": has_receipts,
        "ready": not reasons,
        "reasons": reasons,
    }


def _iter_source_files(root: Path, limit: int = 4000):
    """Source files under ``root``, pruning generated trees. Bounded."""

    seen = 0
    stack = [root]
    while stack and seen < limit:
        directory = stack.pop()
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in _GROUNDEDNESS_SKIP_DIRS or entry.name.startswith("."):
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file() and entry.suffix in _SOURCE_SUFFIXES:
                seen += 1
                yield entry
                if seen >= limit:
                    return


def _split_frontmatter(content: str) -> tuple[str, str] | None:
    """Split a belief into its frontmatter block and the rest.

    The closing delimiter is a line that is exactly ``---``. Locating it with
    a substring search instead matched the first ``---`` *anywhere*, so an
    entity of ``x --- y`` ended the block mid-value: the entity parsed as
    ``x`` and every field after it -- status, confidence, sources, the
    claim_id the ledger is keyed by -- spilled into the body. Entity names
    come from indexed source code, so that input is not hypothetical.

    Returns ``(frontmatter_text, body_text)`` or None when there is no
    complete block.
    """

    if not content.startswith("---"):
        return None
    lines = content.splitlines(keepends=True)
    if not lines:
        return None
    consumed = len(lines[0])
    for line in lines[1:]:
        if line.strip() == "---":
            return content[len(lines[0]) : consumed], content[consumed + len(line) :]
        consumed += len(line)
    return None


def _set_frontmatter_field(content: str, key: str, value: Any) -> str:
    """Rewrite one frontmatter key, leaving the body untouched.

    The body is prose compiled from source docstrings, so it can legitimately
    contain the text ``confidence: 0.5`` or ``status: inferred``. An unanchored
    ``str.replace`` over the whole document rewrote those too, silently
    changing what a belief claims about the code while updating its metadata.
    Only the region above the closing ``---`` is eligible, and only the first
    match of an anchored key.
    """

    import re

    if not content.startswith("---"):
        return content
    end = content.find("\n---", 3)
    if end < 0:
        return content

    head, tail = content[:end], content[end:]
    pattern = re.compile(rf"^{re.escape(key)}:[^\r\n]*$", re.MULTILINE)
    if not pattern.search(head):
        return content
    return pattern.sub(f"{key}: {_yaml_scalar(value)}", head, count=1) + tail


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
        # An unsourced belief must not be rendered as though it cited a source
        # named "unknown". A reader -- human or machine -- cannot tell that
        # apart from a real citation, so a belief carrying `status: verified`
        # and `confidence: 0.99` was stored looking sourced when nothing backed
        # it. CLAUDE.md requires every write to carry sources; inventing one is
        # the fail-open direction.
        #
        # An explicit empty list says the same thing truthfully. Frontmatter
        # parsing is unaffected: `_parse_frontmatter` skips list-item lines, so
        # these entries were never read back into the parsed mapping anyway.
        sources_yaml = (
            "\n".join(f"  - {_yaml_scalar(s)}" for s in self.sources)
            if self.sources
            else "  []"
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

    #: How much evidence a status asserts. Only used to compare two claims
    #: about the same entity; an unranked status compares as neutral so a new
    #: vocabulary term can never silently outrank or be outranked.
    _STATUS_EVIDENCE_RANK = {
        "hypothesis": 0,
        "unsupported": 0,
        "stale": 1,
        "inferred": 2,
        "observed": 3,
        "verified": 4,
    }

    #: Statuses that report on a belief's freshness rather than assert a
    #: competing description of the code. Marking something stale is how the
    #: vault stays honest as the tree moves under it, so it is never refused
    #: for carrying less evidence -- it is not making an evidence claim at all.
    #: (The live staleness paths rewrite the field in place rather than going
    #: through `write_belief`; this keeps the two from disagreeing if one ever
    #: does.)
    _LIFECYCLE_STATUSES = frozenset({"stale"})

    def _weaker_than_current(self, artifact: BeliefArtifact) -> dict[str, Any] | None:
        """Describe the current belief if ``artifact`` would weaken it, else None.

        Deliberately narrow: it takes a drop in status rank *and* in confidence
        *and* citing nothing the current belief does not already cite. A
        genuine correction -- new evidence, a re-verification, a downgrade that
        cites why -- carries at least one of those and overwrites normally.
        What this refuses is the update that is weaker on every axis at once,
        which is not a correction but a regression.
        """
        try:
            record = self.read_belief(artifact.entity)
        except Exception:  # noqa: BLE001 - a guard must not break the write
            return None
        if not record:
            return None
        # read_belief returns {path, frontmatter, body}; the fields this
        # compares live in the frontmatter, not at the top level.
        current = record.get("frontmatter") or {}
        if not isinstance(current, dict):
            return None

        if artifact.status in self._LIFECYCLE_STATUSES:
            return None

        ranks = self._STATUS_EVIDENCE_RANK
        current_status = str(current.get("status") or "")
        incoming_rank = ranks.get(artifact.status)
        current_rank = ranks.get(current_status)
        if incoming_rank is None or current_rank is None:
            return None
        if incoming_rank >= current_rank:
            return None

        try:
            current_confidence = float(current.get("confidence") or 0.0)
        except (TypeError, ValueError):
            return None
        if float(artifact.confidence or 0.0) >= current_confidence:
            return None

        current_sources = {str(s) for s in (current.get("sources") or [])}
        if not current_sources:
            # Nothing to protect: the current claim cites no evidence either.
            return None
        if {str(s) for s in (artifact.sources or [])} - current_sources:
            # It brings a source the current belief does not have. That is new
            # evidence, so it is a correction and must be allowed through.
            return None

        return {
            "status": current_status,
            "confidence": current_confidence,
            "sources": len(current_sources),
            "claim_id": current.get("claim_id"),
        }

    def write_belief(self, artifact: BeliefArtifact) -> dict[str, Any]:
        """Write a belief artifact to the vault.

        A belief citing nothing cannot be ``verified``. Verified is a claim
        about evidence, and an empty ``sources`` list means there was none to
        check, so the status is downgraded here rather than taken on trust.

        The belief is kept, not rejected. Losing a claim because its provenance
        was never attached would be worse than holding it at a lower status.
        What must not happen is ``coverage_index`` counting it toward
        ``verified`` -- that is how a claim with no evidence ended up inside a
        number the vault asserts about its own trustworthiness.

        Retraction cannot cover this case. ``mark_beliefs_ungrounded`` retracts
        a belief whose every cited source is gone and deliberately skips one
        that cites nothing, because there is nothing to resolve against. So an
        unsourced ``verified`` belief was written as verified, never retracted,
        and counted as verified.
        """
        self.ensure_structure()

        if not artifact.sources and artifact.status == "verified":
            artifact = replace(artifact, status="unsupported")

        # Recency must not outrank evidence. Measured on this vault: a
        # `verified` belief at confidence 0.95 citing src/auth.py:42 was
        # replaced by a `hypothesis` at 0.1 citing nothing, and read_belief --
        # what vault_query and every agent actually reads -- returned the
        # hypothesis. The ledger kept both versions, so provenance survived,
        # but nothing reads the ledger by default, so the operative knowledge
        # was simply wrong.
        #
        # This is the dominant state-update failure reported for long-horizon
        # agents (SKILL.state, arXiv 2608.26263, whose error taxonomy puts
        # premature state overwrite/deletion at 68% of state errors), and the
        # same answer applies here -- a weaker patch must not corrupt current
        # state. The claim is still kept and still recorded in the ledger, in
        # keeping with the rule above that losing a claim is worse than holding
        # it at a lower status; it simply does not become what agents read.
        superseded = self._weaker_than_current(artifact)
        if superseded is not None:
            try:
                from .vault_time import BeliefLedger

                BeliefLedger(self._base).record(artifact)
            except Exception as exc:  # noqa: BLE001 - the refusal is the point
                logger.error(f"Vault: competing-claim ledger append failed: {exc}")
            logger.info(
                "Vault: kept stronger belief for '%s' (%s@%.2f citing %d source(s)) "
                "over incoming %s@%.2f citing %d",
                artifact.entity, superseded["status"], superseded["confidence"],
                superseded["sources"], artifact.status, artifact.confidence,
                len(artifact.sources or []),
            )
            return {
                "status": "kept_stronger_claim",
                "directory": "beliefs",
                "entity": artifact.entity,
                "claim_id": artifact.claim_id,
                "current": superseded,
                "reason": (
                    "the incoming belief is weaker in status and confidence and "
                    "cites no source the current one does not already carry; it "
                    "is recorded in the ledger as a competing claim rather than "
                    "replacing what agents read"
                ),
            }

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
                    # Fallback for beliefs whose sources are missing: match the
                    # entity against a changed file's stem *exactly*. This was a
                    # substring test, so editing `auth.py` also marked
                    # `authentication_service` and `oauth_client` stale --
                    # beliefs it has nothing to do with. Staleness is what gates
                    # trust in a belief, so marking fresh ones stale erodes the
                    # signal rather than erring safely.
                    matched = entity_lc in changed_stems

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

    def backfill_source_roots(self, roots: list[str] | None = None) -> dict[str, Any]:
        """Give legacy beliefs the `source_root` they were written without.

        Beliefs written before the field existed record a bare `helper.py`,
        so groundedness has to fall back to matching any file with that
        suffix -- permissive enough that a belief about a deleted
        `scripts/helper.py` survives while an unrelated `tests/helper.py`
        exists. Recovering the root converts those to exact resolution.

        The root is recovered only when the belief's sources resolve to
        exactly one directory. Two files sharing a basename leave it
        ambiguous, and a guess would be worse than the fallback: it would
        look authoritative while pointing at the wrong file.
        """

        import re

        self.ensure_structure()
        search_roots = [self._base.resolve().parent.parent]
        for root in roots or []:
            try:
                search_roots.append(Path(root).resolve())
            except (OSError, ValueError):
                continue
        project_root = search_roots[0]
        candidates = _path_owner_index(search_roots)

        filled: list[str] = []
        ambiguous: list[str] = []
        for md in (self._base / "beliefs").rglob("*.md"):
            try:
                safe_path = resolve_file_within(self._base / "beliefs", md)
                if safe_path is None:
                    continue
                content = safe_path.read_text(encoding="utf-8", errors="replace")
                frontmatter = _parse_frontmatter(content)
                if not frontmatter or frontmatter.get("source_root"):
                    continue
                sources = [
                    s.split(":", 1)[0].replace("\\", "/").strip().lstrip("./")
                    for s in _extract_sources(content)
                ]
                sources = [s for s in sources if _is_path_like(s)]
                if not sources:
                    continue

                entity = frontmatter.get("entity", md.stem)
                owners = {
                    tuple(sorted(candidates.get(source, ()))) for source in sources
                }
                resolved = {o for o in owners if len(o) == 1}
                if len(resolved) != 1 or len(owners) != 1:
                    ambiguous.append(entity)
                    continue

                # An empty prefix means the source is already project-relative
                # (`bench/accuracy.py`). Recording that as `source_root:` with
                # no value writes a key the frontmatter parser then drops, so
                # the belief never counted as filled and was rewritten on every
                # compile. `.` says the same thing and survives a round trip.
                root = next(iter(resolved))[0] or "."
                if not (project_root / root / sources[0]).exists():
                    ambiguous.append(entity)
                    continue

                updated = re.sub(
                    r"^(sources:(?:\r?\n[ \t]+-[^\r\n]*)+)",
                    lambda m: f"{m.group(1)}\nsource_root: {root}",
                    content,
                    count=1,
                    flags=re.MULTILINE,
                )
                if updated == content:
                    ambiguous.append(entity)
                    continue
                _atomic_write_text(safe_path, updated)
                filled.append(entity)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Vault: source_root backfill failed for {md}: {exc}")

        return {
            "status": "updated",
            "backfilled_entities": filled,
            "ambiguous_entities": ambiguous,
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
                    try:
                        old_conf = float(fm.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        # A malformed confidence is not a reason to skip the
                        # update; treat it as the schema default and repair it.
                        old_conf = 0.5
                    new_conf = max(0.0, min(1.0, old_conf + delta))

                    updated = _set_frontmatter_field(content, "confidence", new_conf)
                    if delta > 0 and fm.get("status") == "inferred":
                        updated = _set_frontmatter_field(updated, "status", "verified")
                    updated = _set_frontmatter_field(
                        updated, "last_checked", datetime.now(timezone.utc).isoformat()
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


def _parse_frontmatter(content: str) -> dict[str, Any] | None:
    """Parse YAML frontmatter, including the block sequences beliefs cite.

    Provenance is written as a block sequence -- `sources:` followed by
    indented `- path` lines -- and the previous parser kept only `key: value`
    lines and skipped anything beginning with `-`. So `sources` and
    `derived_from` were written to disk correctly and then dropped on the way
    back in. `read_belief` is returned verbatim by the `vault_query` MCP tool,
    so an agent asking what the vault knows about an entity was told
    `status: verified, confidence: 0.95` with no citations attached and no way
    to check what the claim had been verified against.

    CLAUDE.md requires vault beliefs to be machine-auditable and omitted
    evidence to be inspectable. Evidence that survives the write and not the
    read is neither. Every caller that actually needed provenance had worked
    around this by re-scanning the raw text with `_extract_sources` next to its
    parse; fixing the shared parser is what lets those two agree, rather than
    leaving a second extractor free to drift from the first.
    """
    split = _split_frontmatter(content)
    if split is None:
        return None

    result: dict[str, Any] = {}
    pending_key: str | None = None
    for line in split[0].strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if pending_key is not None:
            if stripped.startswith("- "):
                result.setdefault(pending_key, []).append(stripped[2:].strip())
                continue
            if stripped == "[]":
                # `to_markdown` writes an explicit empty list for a belief that
                # cites nothing. That is a recorded value -- "asked, cited
                # nothing" -- not a missing one, and it must read back as such.
                result[pending_key] = []
                pending_key = None
                continue

        pending_key = None
        if stripped.startswith("-") or ":" not in stripped:
            continue

        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if value:
            result[key] = value
        else:
            # A bare `key:` opens a block sequence. It only becomes a list if
            # items follow, so a key with a genuinely empty scalar still parses
            # as absent, exactly as it did before.
            pending_key = key
    return result if result else None


_SOURCE_SUFFIXES = frozenset({
    ".py", ".rs", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".rb", ".c",
    ".h", ".cpp", ".hpp", ".cs", ".swift", ".kt", ".php", ".scala",
})


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


def _path_owner_index(roots: list[Path]) -> dict[str, set[str]]:
    """Map each path suffix to the root-relative directories that can supply it.

    A suffix owned by exactly one directory identifies a belief's
    `source_root`; one owned by several is ambiguous and must stay that way.
    """

    owners: dict[str, set[str]] = {}
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
                        suffix = "/".join(parts[start:])
                        owner = "/".join(parts[:start])
                        owners.setdefault(suffix, set()).add(owner)
    return owners


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
    """The sources a belief cites, or an empty list.

    Kept as a named helper because the groundedness scan reads it on its own
    terms, but it no longer walks the frontmatter itself: two extractors that
    can disagree about what a belief cites is the failure the audit trail
    exists to prevent.
    """
    fm = _parse_frontmatter(content) or {}
    value = fm.get("sources")
    return [str(v) for v in value] if isinstance(value, list) else []


def _extract_body(content: str) -> str:
    """Extract body content after frontmatter."""
    split = _split_frontmatter(content)
    if split is None:
        return content
    return split[1].strip()


