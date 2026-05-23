"""
Change Pipeline — Change-Driven Flow (④)
==========================================

Handles code changes, PRs, diffs, and scheduled events.
Triggers the Truth → Belief → Verification → Action pipeline.

Components:
  1. Diff Analyzer:       Parse diffs into structured change sets
  2. Belief Diff:         What beliefs are affected by the change
  3. PR Brief Generator:  Auto-generate PR summaries with blast radius
  4. Review Assistant:    Detect architecture violations, duplicates
  5. Doc Refresh:         Trigger belief refresh after changes
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .vault import VaultManager, VerificationArtifact
from .verification_engine import VerificationEngine

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ChangeSet:
    """A structured representation of a code change."""
    files_added: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    files_deleted: list[str] = field(default_factory=list)
    functions_changed: list[str] = field(default_factory=list)
    classes_changed: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    intent: str = "unknown"  # feature, bugfix, refactor, test, docs, security, performance
    commit_message: str = ""


@dataclass
class BeliefDiff:
    """How a code change affects existing beliefs."""
    stale_beliefs: list[str] = field(default_factory=list)    # become stale
    invalidated_beliefs: list[str] = field(default_factory=list)  # contradicted
    new_coverage_needed: list[str] = field(default_factory=list)  # new files
    unchanged_beliefs: list[str] = field(default_factory=list)
    blast_radius: str = "low"  # low, medium, high


@dataclass
class ReviewFinding:
    """A code review finding."""
    severity: str  # info, warning, error
    category: str  # architecture, duplication, naming, safety, test_gap
    file: str
    line: int = 0
    message: str = ""
    suggestion: str = ""


@dataclass
class PRBrief:
    """An auto-generated PR brief."""
    title: str
    summary: str
    changeset: ChangeSet
    belief_diff: BeliefDiff
    findings: list[ReviewFinding] = field(default_factory=list)
    risk_level: str = "low"
    action_path: str = ""
    verification_path: str = ""

    def to_markdown(self) -> str:
        lines = [f"# {self.title}\n"]

        # Summary
        lines.append(f"## Summary\n{self.summary}\n")

        # Change stats
        cs = self.changeset
        lines.append("## Changes")
        lines.append(f"- **Intent:** {cs.intent}")
        lines.append(f"- **Lines:** +{cs.lines_added} / -{cs.lines_removed}")
        if cs.files_added:
            lines.append(f"- **Added:** {', '.join(cs.files_added)}")
        if cs.files_modified:
            lines.append(f"- **Modified:** {', '.join(cs.files_modified)}")
        if cs.files_deleted:
            lines.append(f"- **Deleted:** {', '.join(cs.files_deleted)}")
        if cs.functions_changed:
            lines.append(f"- **Functions changed:** {', '.join(cs.functions_changed[:10])}")

        # Belief impact
        bd = self.belief_diff
        lines.append(f"\n## Belief Impact (blast radius: {bd.blast_radius})")
        if bd.stale_beliefs:
            lines.append(f"- **Stale:** {', '.join(bd.stale_beliefs)}")
        if bd.invalidated_beliefs:
            lines.append(f"- **Invalidated:** {', '.join(bd.invalidated_beliefs)}")
        if bd.new_coverage_needed:
            lines.append(f"- **New coverage needed:** {', '.join(bd.new_coverage_needed)}")

        # Review findings
        if self.findings:
            lines.append(f"\n## Review Findings ({len(self.findings)})")
            for f in self.findings:
                icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(f.severity, "⚪")
                lines.append(f"- {icon} **{f.category}** [{f.file}]: {f.message}")
                if f.suggestion:
                    lines.append(f"  - Suggestion: {f.suggestion}")

        lines.append(f"\n**Risk Level:** {self.risk_level}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Diff Analyzer
# ══════════════════════════════════════════════════════════════════════

def parse_diff(diff_text: str, commit_message: str = "") -> ChangeSet:
    """Parse a unified diff into a structured ChangeSet."""
    cs = ChangeSet(commit_message=commit_message)

    # Parse file headers
    for m in re.finditer(r'^diff --git a/(.+?) b/(.+?)$', diff_text, re.M):
        file_a, file_b = m.group(1), m.group(2)
        if file_a == "/dev/null":
            cs.files_added.append(file_b)
        elif file_b == "/dev/null":
            cs.files_deleted.append(file_a)
        else:
            cs.files_modified.append(file_b)

    # Fallback: parse --- / +++ headers
    if not cs.files_added and not cs.files_modified and not cs.files_deleted:
        for m in re.finditer(r'^\+\+\+ b/(.+)$', diff_text, re.M):
            f = m.group(1)
            if f != "/dev/null" and f not in cs.files_modified:
                cs.files_modified.append(f)

    # Count lines
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            cs.lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            cs.lines_removed += 1

    # Extract changed functions/classes from @@  headers
    for m in re.finditer(r'^@@.*@@\s*(?:(?:def|fn|func|function)\s+(\w+)|class\s+(\w+))', diff_text, re.M):
        fn_name = m.group(1) or m.group(2)
        if fn_name and fn_name not in cs.functions_changed:
            cs.functions_changed.append(fn_name)

    # Classify intent
    cs.intent = _classify_change_intent(diff_text, commit_message)

    return cs


def _classify_change_intent(diff: str, message: str) -> str:
    """Classify the intent of a change."""
    text = f"{message} {diff[:2000]}".lower()
    if re.search(r'\b(fix|bug|patch|hotfix|issue|error)\b', text):
        return "bugfix"
    if re.search(r'\b(test|spec|assert|expect|mock)\b', text):
        return "test"
    if re.search(r'\b(security|cve|vulnerability|auth|permission)\b', text):
        return "security"
    if re.search(r'\b(perf|optim|speed|fast|cache|latency)\b', text):
        return "performance"
    if re.search(r'\b(refactor|clean|rename|extract|move)\b', text):
        return "refactor"
    if re.search(r'\b(doc|readme|comment|explain)\b', text):
        return "docs"
    return "feature"


# ══════════════════════════════════════════════════════════════════════
# Review Assistant
# ══════════════════════════════════════════════════════════════════════

# Architecture violation patterns
_VIOLATIONS = [
    ("hardcoded_secret", re.compile(
        r'(?:password|secret|api_key|token)\s*=\s*["\'][^"\']{8,}["\']', re.I),
     "error", "safety", "Possible hardcoded secret"),
    ("todo_fixme", re.compile(r'\b(TODO|FIXME|HACK|XXX|TEMP)\b'),
     "warning", "maintenance", "Contains TODO/FIXME marker"),
    ("broad_except", re.compile(r'except\s*(?:Exception|BaseException|\s*:)'),
     "warning", "safety", "Broad exception catch — may swallow errors"),
    ("unsafe_unwrap", re.compile(r'\.unwrap\(\)'),
     "warning", "safety", "Rust .unwrap() may panic — consider .expect() or ?"),
    ("magic_number", re.compile(r'(?:if|while|for).*\b(?:42|100|1000|9999|1024)\b'),
     "info", "naming", "Possible magic number — consider named constant"),
    ("long_function", None, "info", "architecture", "Function exceeds 50 lines"),
]


def review_diff(diff_text: str) -> list[ReviewFinding]:
    """Review a diff for common issues."""
    findings: list[ReviewFinding] = []

    # Pattern-based review on added lines
    current_file = ""
    current_line = 0
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@"):
            m = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)', line)
            if m:
                current_line = int(m.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            current_line += 1
            content = line[1:]
            for name, pattern, severity, category, message in _VIOLATIONS:
                if pattern and pattern.search(content):
                    findings.append(ReviewFinding(
                        severity=severity,
                        category=category,
                        file=current_file,
                        line=current_line,
                        message=message,
                    ))

    # Duplicate detection: same code added to multiple files
    added_blocks: dict[str, list[str]] = {}
    current_file = ""
    block: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            if block and current_file:
                key = "\n".join(block[-5:])
                added_blocks.setdefault(key, []).append(current_file)
            current_file = line[6:]
            block = []
        elif line.startswith("+") and not line.startswith("+++"):
            block.append(line[1:].strip())
    # Final flush
    if block and current_file:
        key = "\n".join(block[-5:])
        added_blocks.setdefault(key, []).append(current_file)

    for block_key, files in added_blocks.items():
        if len(files) > 1 and len(block_key) > 50:
            findings.append(ReviewFinding(
                severity="warning",
                category="duplication",
                file=files[0],
                message=f"Similar code added to {len(files)} files: {', '.join(files[:3])}",
                suggestion="Consider extracting to a shared utility",
            ))

    return findings


# ══════════════════════════════════════════════════════════════════════
# The Change Pipeline
# ══════════════════════════════════════════════════════════════════════

class ChangePipeline:
    """
    Processes code changes through the Change-Driven flow (④).

    Pipeline: Diff → ChangeSet → BeliefDiff → PR Brief → Vault
    """

    def __init__(self, vault: VaultManager, verification: VerificationEngine):
        self._vault = vault
        self._verification = verification

    def process_diff(
        self,
        diff_text: str,
        commit_message: str = "",
        pr_title: str = "",
    ) -> PRBrief:
        """Process a diff through the full change pipeline."""

        # 1. Parse diff
        changeset = parse_diff(diff_text, commit_message)

        # 2. Compute belief diff
        changeset.files_added + changeset.files_modified
        belief_diff = self._compute_belief_diff(changeset)

        # 3. Review
        findings = review_diff(diff_text)

        # 4. Risk assessment
        risk = self._assess_risk(changeset, belief_diff, findings)

        # 5. Generate brief
        title = pr_title or self._generate_title(changeset)
        summary = self._generate_summary(changeset, belief_diff)

        brief = PRBrief(
            title=title,
            summary=summary,
            changeset=changeset,
            belief_diff=belief_diff,
            findings=findings,
            risk_level=risk,
        )

        # 6. Write to vault
        action_result = self._vault.write_action(
            title=title,
            content=brief.to_markdown(),
            action_type="pr_brief",
        )
        brief.action_path = action_result.get("path", "")

        # 7. Write verification if beliefs were affected
        if belief_diff.stale_beliefs or belief_diff.invalidated_beliefs:
            verification_result = self._vault.write_verification(VerificationArtifact(
                challenges=", ".join(
                    belief_diff.stale_beliefs[:5] + belief_diff.invalidated_beliefs[:5]
                ),
                result="change_detected",
                confidence_delta=-0.1,
                method="change_pipeline",
                title=f"Change Impact: {title}",
                body=(
                    f"Code change detected affecting {len(belief_diff.stale_beliefs)} beliefs.\n\n"
                    f"Stale: {', '.join(belief_diff.stale_beliefs)}\n"
                    f"Invalidated: {', '.join(belief_diff.invalidated_beliefs)}\n"
                    f"New coverage needed: {', '.join(belief_diff.new_coverage_needed)}"
                ),
            ))
            brief.verification_path = verification_result.get("path", "")

        logger.info(
            f"ChangePipeline: {title} | "
            f"risk={risk} findings={len(findings)} "
            f"stale_beliefs={len(belief_diff.stale_beliefs)}"
        )
        return brief

    def refresh_docs(self, changed_files: list[str]) -> dict[str, Any]:
        """Trigger belief refresh for changed files."""
        refreshed = self._vault.mark_beliefs_stale_for_files(changed_files)
        refreshed["status"] = "refreshed"
        refreshed["total"] = len(refreshed.get("updated_entities", []))
        refreshed["stale_marked"] = refreshed.get("updated_entities", [])
        return refreshed

    # ── Private ──────────────────────────

    def _compute_belief_diff(self, cs: ChangeSet) -> BeliefDiff:
        """Determine which beliefs are affected by a change."""
        bd = BeliefDiff()
        beliefs = self._vault.list_beliefs()

        known_entities = {b["entity"].lower(): b for b in beliefs}

        for f in cs.files_modified:
            stem = Path(f).stem.lower()
            if stem in known_entities:
                bd.stale_beliefs.append(known_entities[stem]["entity"])
            else:
                # Check partial matches
                for entity, b in known_entities.items():
                    if stem in entity:
                        bd.stale_beliefs.append(b["entity"])

        for f in cs.files_deleted:
            stem = Path(f).stem.lower()
            if stem in known_entities:
                bd.invalidated_beliefs.append(known_entities[stem]["entity"])

        for f in cs.files_added:
            stem = Path(f).stem.lower()
            if stem not in known_entities:
                bd.new_coverage_needed.append(f)

        # Blast radius
        total = len(bd.stale_beliefs) + len(bd.invalidated_beliefs)
        bd.blast_radius = "low" if total <= 2 else "medium" if total <= 5 else "high"

        return bd

    def _assess_risk(self, cs: ChangeSet, bd: BeliefDiff, findings: list[ReviewFinding]) -> str:
        errors = sum(1 for f in findings if f.severity == "error")
        if errors > 0 or cs.intent == "security":
            return "high"
        if bd.blast_radius == "high" or len(bd.invalidated_beliefs) > 0:
            return "high"
        if bd.blast_radius == "medium" or cs.lines_added + cs.lines_removed > 500:
            return "medium"
        return "low"

    def _generate_title(self, cs: ChangeSet) -> str:
        intent_map = {
            "bugfix": "Bug Fix", "feature": "Feature", "refactor": "Refactor",
            "test": "Tests", "docs": "Documentation", "security": "Security",
            "performance": "Performance",
        }
        prefix = intent_map.get(cs.intent, "Change")
        files = cs.files_modified + cs.files_added
        if files:
            primary = Path(files[0]).stem
            return f"{prefix}: {primary}" + (f" (+{len(files)-1} more)" if len(files) > 1 else "")
        return f"{prefix}: {cs.commit_message[:50]}" if cs.commit_message else prefix

    def _generate_summary(self, cs: ChangeSet, bd: BeliefDiff) -> str:
        parts = []
        total_files = len(cs.files_added) + len(cs.files_modified) + len(cs.files_deleted)
        parts.append(f"This {cs.intent} touches {total_files} file(s) "
                     f"(+{cs.lines_added}/-{cs.lines_removed} lines).")
        if bd.stale_beliefs:
            parts.append(f"It affects {len(bd.stale_beliefs)} existing belief(s) "
                        f"which should be re-verified.")
        if bd.new_coverage_needed:
            parts.append(f"{len(bd.new_coverage_needed)} new file(s) need belief coverage.")
        return " ".join(parts)
