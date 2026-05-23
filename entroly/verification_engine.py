"""
Verification Engine
====================

Automated Verification layer — challenges the system's understanding.

Components:
  1. Contradiction Detector:   Cross-belief consistency checking
  2. Staleness Scanner:        Freshness checks across all beliefs
  3. Blast Radius Analyzer:    Impact analysis for changes
  4. Confidence Scorer:        Evidence-weighted confidence updates
  5. Gaps Detector:            Find unknowns in belief coverage
  6. Test Linker:              Link beliefs to test outcomes
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .vault import VaultManager, VerificationArtifact, _extract_body, _parse_frontmatter

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Contradiction:
    """Two beliefs that disagree."""
    belief_a: str       # claim_id
    belief_b: str       # claim_id
    entity: str
    conflict_type: str  # semantic, dependency, invariant, stale_vs_fresh
    description: str
    severity: str = "medium"  # low, medium, high


@dataclass
class StaleReport:
    """A belief past its freshness window."""
    claim_id: str
    entity: str
    status: str
    last_checked: str
    hours_since: float
    confidence: float


@dataclass
class CoverageGap:
    """A file or module with no corresponding belief."""
    file_path: str
    reason: str
    suggested_entity: str


@dataclass
class BlastRadius:
    """Impact analysis result for a change."""
    changed_entity: str
    affected_beliefs: list[str] = field(default_factory=list)
    affected_entities: list[str] = field(default_factory=list)
    risk_level: str = "low"
    description: str = ""


@dataclass
class VerificationReport:
    """Complete verification pass result."""
    contradictions: list[Contradiction] = field(default_factory=list)
    stale_beliefs: list[StaleReport] = field(default_factory=list)
    coverage_gaps: list[CoverageGap] = field(default_factory=list)
    total_beliefs_checked: int = 0
    verified_count: int = 0
    stale_count: int = 0
    low_confidence_count: int = 0
    mean_confidence: float = 0.0
    artifacts_written: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_beliefs_checked": self.total_beliefs_checked,
            "verified_count": self.verified_count,
            "stale_count": self.stale_count,
            "low_confidence_count": self.low_confidence_count,
            "mean_confidence": round(self.mean_confidence, 3),
            "contradictions": len(self.contradictions),
            "coverage_gaps": len(self.coverage_gaps),
            "artifacts_written": self.artifacts_written,
            "contradiction_details": [
                {"entity": c.entity, "type": c.conflict_type,
                 "description": c.description, "severity": c.severity}
                for c in self.contradictions
            ],
            "stale_details": [
                {"entity": s.entity, "claim_id": s.claim_id,
                 "hours_since": round(s.hours_since, 1)}
                for s in self.stale_beliefs
            ],
            "gap_details": [
                {"file": g.file_path, "reason": g.reason}
                for g in self.coverage_gaps
            ],
        }


# ══════════════════════════════════════════════════════════════════════
# The Verification Engine
# ══════════════════════════════════════════════════════════════════════

class VerificationEngine:
    """
    Runs automated verification passes against the belief vault.

    This is the Belief CI engine — it challenges every belief in the
    vault and produces verification artifacts.
    """

    def __init__(
        self,
        vault: VaultManager,
        freshness_hours: float = 24.0,
        min_confidence: float = 0.5,
    ):
        self._vault = vault
        self._freshness_hours = freshness_hours
        self._min_confidence = min_confidence

    def full_verification_pass(self) -> VerificationReport:
        """Run a complete verification pass on all beliefs."""
        report = VerificationReport()
        beliefs = self._load_all_beliefs()
        report.total_beliefs_checked = len(beliefs)

        if not beliefs:
            return report

        # 1. Staleness check
        report.stale_beliefs = self._check_staleness(beliefs)
        report.stale_count = len(report.stale_beliefs)

        # 2. Contradiction detection
        report.contradictions = self._detect_contradictions(beliefs)

        # 3. Confidence analysis
        confidences = [b.get("confidence", 0) for b in beliefs]
        report.mean_confidence = sum(confidences) / len(confidences) if confidences else 0
        report.verified_count = sum(1 for b in beliefs if b.get("status") == "verified")
        report.low_confidence_count = sum(
            1 for c in confidences if c < self._min_confidence
        )

        # 4. Write verification artifacts
        if report.contradictions:
            self._write_contradiction_report(report.contradictions)
            report.artifacts_written += 1

        if report.stale_beliefs:
            self._write_staleness_report(report.stale_beliefs)
            report.artifacts_written += 1

        # 5. Write summary verification artifact
        self._write_summary_artifact(report)
        report.artifacts_written += 1

        logger.info(
            f"VerificationEngine: checked {report.total_beliefs_checked} beliefs | "
            f"stale={report.stale_count} contradictions={len(report.contradictions)} "
            f"gaps={len(report.coverage_gaps)} mean_conf={report.mean_confidence:.2f}"
        )
        return report

    def check_belief(self, claim_id: str) -> dict[str, Any]:
        """Verify a single belief by claim_id."""
        beliefs = self._load_all_beliefs()
        target = None
        for b in beliefs:
            if b.get("claim_id") == claim_id:
                target = b
                break

        if not target:
            return {"status": "not_found", "claim_id": claim_id}

        issues = []

        # Staleness
        stale = self._check_staleness([target])
        if stale:
            issues.append(f"stale: {stale[0].hours_since:.1f}h since last check")

        # Low confidence
        conf = float(target.get("confidence", 0))
        if conf < self._min_confidence:
            issues.append(f"low_confidence: {conf:.2f} < {self._min_confidence}")

        # Check for contradictions with other beliefs
        contras = self._detect_contradictions(beliefs)
        related = [c for c in contras if claim_id in (c.belief_a, c.belief_b)]
        if related:
            issues.extend(f"contradiction: {c.description}" for c in related)

        status = "verified" if not issues else "needs_attention"
        return {
            "status": status,
            "claim_id": claim_id,
            "entity": target.get("entity", ""),
            "confidence": conf,
            "issues": issues,
        }

    def blast_radius(
        self,
        changed_files: list[str],
    ) -> BlastRadius:
        """Analyze the blast radius of file changes."""
        beliefs = self._load_all_beliefs()
        affected_beliefs: list[str] = []
        affected_entities: list[str] = []

        for belief in beliefs:
            sources = belief.get("sources_list", [])
            entity = belief.get("entity", "")
            body = belief.get("body", "")

            for changed in changed_files:
                changed_stem = Path(changed).stem.lower()
                # Check if any source references the changed file
                source_match = any(changed_stem in s.lower() for s in sources)
                # Check if the body mentions the changed file
                body_match = changed_stem in body.lower()
                # Check entity name overlap
                entity_match = changed_stem in entity.lower()

                if source_match or body_match or entity_match:
                    cid = belief.get("claim_id", "")
                    if cid and cid not in affected_beliefs:
                        affected_beliefs.append(cid)
                    if entity and entity not in affected_entities:
                        affected_entities.append(entity)

        n = len(affected_beliefs)
        risk = "low" if n <= 2 else "medium" if n <= 5 else "high"

        return BlastRadius(
            changed_entity=", ".join(changed_files),
            affected_beliefs=affected_beliefs,
            affected_entities=affected_entities,
            risk_level=risk,
            description=f"{n} beliefs affected by changes to {', '.join(changed_files)}",
        )

    def coverage_gaps(self, source_dir: str) -> list[CoverageGap]:
        """Find source files with no corresponding belief."""
        gaps = []
        beliefs = self._load_all_beliefs()
        known = set()
        for b in beliefs:
            entity = b.get("entity", "").lower()
            known.add(entity)
            for src in b.get("sources_list", []):
                known.add(Path(src.split(":")[0]).stem.lower())

        skip = {"__pycache__", "node_modules", ".git", "target", "dist", "build", "venv", ".venv"}
        root = Path(source_dir)
        for fpath in root.rglob("*"):
            if fpath.is_file() and fpath.suffix.lower() in (".py", ".rs", ".ts", ".js"):
                if any(p in str(fpath) for p in skip):
                    continue
                stem = fpath.stem.lower()
                if stem not in known and stem not in ("__init__", "mod", "lib", "main", "index"):
                    gaps.append(CoverageGap(
                        file_path=str(fpath.relative_to(root)),
                        reason="no_belief_artifact",
                        suggested_entity=stem,
                    ))

        return gaps

    # ── Private Methods ────────────────────────────────────────────

    def _load_all_beliefs(self) -> list[dict[str, Any]]:
        """Load all beliefs with parsed frontmatter."""
        self._vault.ensure_structure()
        beliefs_dir = self._vault.config.path / "beliefs"
        results = []

        for md in beliefs_dir.rglob("*.md"):
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                fm = _parse_frontmatter(content)
                if not fm:
                    continue
                body = _extract_body(content)
                # Parse sources list from frontmatter area
                sources_list = []
                in_sources = False
                for line in content.split("---")[1].splitlines() if "---" in content else []:
                    ls = line.strip()
                    if ls.startswith("sources:"):
                        in_sources = True
                        continue
                    if in_sources:
                        if ls.startswith("- "):
                            sources_list.append(ls[2:].strip())
                        elif ls and not ls.startswith("-"):
                            in_sources = False

                results.append({
                    **fm,
                    "confidence": float(fm.get("confidence", 0)),
                    "body": body,
                    "file": str(md),
                    "sources_list": sources_list,
                })
            except Exception as e:
                logger.debug(f"VerificationEngine: failed to read {md}: {e}")

        return results

    def _check_staleness(self, beliefs: list[dict]) -> list[StaleReport]:
        """Check beliefs for staleness."""
        stale = []
        now = datetime.now(timezone.utc)

        for b in beliefs:
            # Status-driven staleness: the change listener or sync pipeline
            # sets status to "stale" when source files change. Respect that
            # regardless of the timestamp — the code reality has drifted.
            if b.get("status") == "stale":
                stale.append(StaleReport(
                    claim_id=b.get("claim_id", ""),
                    entity=b.get("entity", ""),
                    status="stale",
                    last_checked=b.get("last_checked", "unknown"),
                    hours_since=0,  # status-driven, not time-driven
                    confidence=b.get("confidence", 0),
                ))
                continue

            last = b.get("last_checked", "")
            if not last:
                # No last_checked → treat as stale
                stale.append(StaleReport(
                    claim_id=b.get("claim_id", ""),
                    entity=b.get("entity", ""),
                    status=b.get("status", ""),
                    last_checked="never",
                    hours_since=float('inf'),
                    confidence=b.get("confidence", 0),
                ))
                continue

            try:
                checked = datetime.fromisoformat(last.replace("Z", "+00:00"))
                hours = (now - checked).total_seconds() / 3600
                if hours > self._freshness_hours:
                    stale.append(StaleReport(
                        claim_id=b.get("claim_id", ""),
                        entity=b.get("entity", ""),
                        status=b.get("status", ""),
                        last_checked=last,
                        hours_since=hours,
                        confidence=b.get("confidence", 0),
                    ))
            except (ValueError, TypeError):
                pass

        return stale

    def _detect_contradictions(self, beliefs: list[dict]) -> list[Contradiction]:
        """Detect contradictions between beliefs."""
        contradictions = []

        # Group beliefs by entity for comparison
        by_entity: dict[str, list[dict]] = {}
        for b in beliefs:
            entity = b.get("entity", "")
            if entity:
                by_entity.setdefault(entity, []).append(b)

        # Check for same-entity contradictions
        for entity, group in by_entity.items():
            if len(group) < 2:
                continue
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    # Status contradiction: one verified, one stale
                    if (a.get("status") == "verified" and b.get("status") == "stale") or \
                       (a.get("status") == "stale" and b.get("status") == "verified"):
                        contradictions.append(Contradiction(
                            belief_a=a.get("claim_id", ""),
                            belief_b=b.get("claim_id", ""),
                            entity=entity,
                            conflict_type="stale_vs_fresh",
                            description="Same entity has both verified and stale beliefs",
                            severity="medium",
                        ))

                    # Confidence divergence
                    ca = float(a.get("confidence", 0))
                    cb = float(b.get("confidence", 0))
                    if abs(ca - cb) > 0.4:
                        contradictions.append(Contradiction(
                            belief_a=a.get("claim_id", ""),
                            belief_b=b.get("claim_id", ""),
                            entity=entity,
                            conflict_type="confidence_divergence",
                            description=f"Confidence diverges: {ca:.2f} vs {cb:.2f}",
                            severity="low",
                        ))

        # Check for dependency contradictions
        # If A says it depends on B, but B doesn't exist as a belief
        known_entities = {b.get("entity", "").lower() for b in beliefs}
        # Also index types/functions documented inside each belief body, so a
        # wiki-link like [[ContextFragment]] resolves to the belief that
        # defines `pub struct ContextFragment`, not just to a file named
        # ContextFragment.md. Covers Python (class/def), Rust (struct/enum/
        # fn/trait/type), and TS/JS (class/function/interface/type).
        symbol_pattern = re.compile(
            r'`(?:pub\s+)?(?:class|def|fn|struct|enum|interface|function|type|trait)\s+(\w+)'
        )
        for b in beliefs:
            for sym_match in symbol_pattern.finditer(b.get("body", "")):
                known_entities.add(sym_match.group(1).lower())
        for b in beliefs:
            body = b.get("body", "")
            # Find [[wiki links]]
            for match in re.finditer(r'\[\[(\w+(?:\.\w+)?)\]\]', body):
                ref = match.group(1).lower()
                if ref not in known_entities:
                    contradictions.append(Contradiction(
                        belief_a=b.get("claim_id", ""),
                        belief_b="",
                        entity=b.get("entity", ""),
                        conflict_type="broken_reference",
                        description=f"References [[{match.group(1)}]] but no such belief exists",
                        severity="medium",
                    ))

        return contradictions

    def _write_contradiction_report(self, contradictions: list[Contradiction]) -> None:
        """Write contradiction report to vault."""
        body_parts = [f"Found {len(contradictions)} contradiction(s).\n"]
        for i, c in enumerate(contradictions, 1):
            body_parts.append(
                f"### {i}. {c.conflict_type} ({c.severity})\n"
                f"- Entity: `{c.entity}`\n"
                f"- {c.description}\n"
            )

        artifact = VerificationArtifact(
            challenges="multiple",
            result="contradictions_found",
            confidence_delta=-0.05,
            method="automated_contradiction_scan",
            title="Contradiction Report",
            body="\n".join(body_parts),
        )
        self._vault.write_verification(artifact)

    def _write_staleness_report(self, stale: list[StaleReport]) -> None:
        """Write staleness report to vault."""
        body_parts = [f"Found {len(stale)} stale belief(s).\n"]
        for s in stale:
            body_parts.append(
                f"- **{s.entity}** — {s.hours_since:.0f}h since last check "
                f"(conf={s.confidence:.2f})"
            )

        artifact = VerificationArtifact(
            challenges="multiple",
            result="staleness_detected",
            confidence_delta=-0.02,
            method="automated_staleness_scan",
            title="Staleness Report",
            body="\n".join(body_parts),
        )
        self._vault.write_verification(artifact)

    def _write_summary_artifact(self, report: VerificationReport) -> None:
        """Write a summary verification artifact."""
        body = (
            f"## Verification Summary\n\n"
            f"- **Beliefs checked:** {report.total_beliefs_checked}\n"
            f"- **Verified:** {report.verified_count}\n"
            f"- **Stale:** {report.stale_count}\n"
            f"- **Low confidence:** {report.low_confidence_count}\n"
            f"- **Mean confidence:** {report.mean_confidence:.3f}\n"
            f"- **Contradictions:** {len(report.contradictions)}\n"
            f"- **Coverage gaps:** {len(report.coverage_gaps)}\n"
        )
        artifact = VerificationArtifact(
            challenges="vault",
            result="pass" if not report.contradictions and not report.stale_beliefs else "issues_found",
            confidence_delta=0.0,
            method="full_verification_pass",
            title="Verification Pass Summary",
            body=body,
        )
        self._vault.write_verification(artifact)
