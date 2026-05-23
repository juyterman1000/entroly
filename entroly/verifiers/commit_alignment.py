"""
TRIAD — Triangulated Diff-Message-PR Alignment Verifier
=========================================================

Detects hallucinated commit messages and PR descriptions — cases where
the text claims something that the actual diff doesn't do.

The problem TRIAD solves:
    LLMs generate commit messages and PR descriptions that sound
    plausible but don't match the actual code changes. Example:

        Diff:    Adds a new logging statement to process_order()
        Message: "Refactored payment processing to use async/await"
                 ← HALLUCINATED. No refactoring, no async, no payment.

    This is dangerous because:
    1. Future developers trust commit messages for archaeology
    2. Code reviewers may rubber-stamp based on the description
    3. CI/CD pipelines may use message parsing for release notes

Mathematical Foundation
-----------------------
Three independent signals triangulated via Bayesian combination:

1. **Token Alignment** (Signal A):
   TA = J(tokens(diff), tokens(message))
   Jaccard similarity between diff content tokens and message tokens.
   Low TA = message talks about things not in the diff.

2. **Structural Alignment** (Signal S):
   Extract *structural events* from the diff:
     - added_functions: set of new def/class names
     - removed_functions: set of deleted def/class names
     - modified_files: set of file paths
     - changed_imports: set of added/removed imports

   Extract *claimed events* from the message:
     - action verbs: "add", "remove", "refactor", "fix", "update"
     - subjects: nouns/identifiers following the verbs

   SA = |matched_events| / max(1, |claimed_events|)

3. **Scope Consistency** (Signal C):
   Does the message's scope match the diff's scope?
     - Message says "refactor X" but diff only touches Y → inconsistent
     - Message says "fix bug in auth" but diff touches 20 files → suspicious
     - Message says "minor fix" but diff is 500 lines → inconsistent

   SC ∈ {1.0, 0.7, 0.3} based on scope heuristics.

Final score:
   TRIAD = 0.40 × TA + 0.35 × SA + 0.25 × SC

References
----------
- Conventional Commits spec (conventionalcommits.org)
- Tian et al. (2022): "What Makes a Good Commit Message?"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Diff parsing ─────────────────────────────────────────────────────


@dataclass
class DiffAnalysis:
    """Structural analysis of a unified diff."""
    files_changed: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    added_functions: set[str] = field(default_factory=set)
    removed_functions: set[str] = field(default_factory=set)
    modified_functions: set[str] = field(default_factory=set)
    added_imports: set[str] = field(default_factory=set)
    removed_imports: set[str] = field(default_factory=set)
    added_classes: set[str] = field(default_factory=set)
    removed_classes: set[str] = field(default_factory=set)
    content_tokens: set[str] = field(default_factory=set)

    @property
    def total_lines(self) -> int:
        return self.lines_added + self.lines_removed

    @property
    def structural_events(self) -> set[str]:
        """All structural changes as normalized event strings."""
        events: set[str] = set()
        for f in self.added_functions:
            events.add(f"add_function:{f}")
        for f in self.removed_functions:
            events.add(f"remove_function:{f}")
        for f in self.modified_functions:
            events.add(f"modify_function:{f}")
        for i in self.added_imports:
            events.add(f"add_import:{i}")
        for i in self.removed_imports:
            events.add(f"remove_import:{i}")
        for c in self.added_classes:
            events.add(f"add_class:{c}")
        for c in self.removed_classes:
            events.add(f"remove_class:{c}")
        for fp in self.files_changed:
            events.add(f"modify_file:{fp}")
        return events


# Patterns for extracting structure from diff lines
_DIFF_FILE = re.compile(r"^(?:---|\+\+\+)\s+[ab]/(.+)$", re.MULTILINE)
_DIFF_HUNK = re.compile(r"^@@\s+.*\s+@@", re.MULTILINE)
_FUNC_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*\(")
_CLASS_DEF = re.compile(r"^\s*class\s+(\w+)")
_IMPORT_LINE = re.compile(r"^\s*(?:from\s+(\S+)\s+)?import\s+(.+)")


def _parse_diff(diff_text: str) -> DiffAnalysis:
    """Parse a unified diff into structural events."""
    analysis = DiffAnalysis()

    # Extract file paths
    for m in _DIFF_FILE.finditer(diff_text):
        fp = m.group(1)
        if fp and fp not in analysis.files_changed:
            analysis.files_changed.append(fp)

    for line in diff_text.split("\n"):
        stripped = line.lstrip()

        if stripped.startswith("+") and not stripped.startswith("+++"):
            # Added line
            content = stripped[1:]
            analysis.lines_added += 1

            # Function definitions
            fm = _FUNC_DEF.match(content)
            if fm:
                analysis.added_functions.add(fm.group(1))

            # Class definitions
            cm = _CLASS_DEF.match(content)
            if cm:
                analysis.added_classes.add(cm.group(1))

            # Imports
            im = _IMPORT_LINE.match(content)
            if im:
                mod = im.group(1) or im.group(2).split(",")[0].strip()
                analysis.added_imports.add(mod.split(".")[0])

            # Content tokens
            words = re.findall(r"[a-z][a-z0-9_]+", content.lower())
            analysis.content_tokens.update(w for w in words if len(w) >= 3)

        elif stripped.startswith("-") and not stripped.startswith("---"):
            # Removed line
            content = stripped[1:]
            analysis.lines_removed += 1

            fm = _FUNC_DEF.match(content)
            if fm:
                analysis.removed_functions.add(fm.group(1))

            cm = _CLASS_DEF.match(content)
            if cm:
                analysis.removed_classes.add(cm.group(1))

            im = _IMPORT_LINE.match(content)
            if im:
                mod = im.group(1) or im.group(2).split(",")[0].strip()
                analysis.removed_imports.add(mod.split(".")[0])

            words = re.findall(r"[a-z][a-z0-9_]+", content.lower())
            analysis.content_tokens.update(w for w in words if len(w) >= 3)

    # Modified = added ∩ removed (same name appears in both)
    analysis.modified_functions = analysis.added_functions & analysis.removed_functions
    analysis.added_functions -= analysis.modified_functions
    analysis.removed_functions -= analysis.modified_functions

    return analysis


# ── Message parsing ──────────────────────────────────────────────────


@dataclass
class MessageAnalysis:
    """Analysis of a commit message or PR description."""
    raw: str
    subject: str = ""
    body: str = ""
    action_verbs: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    claimed_events: set[str] = field(default_factory=set)
    content_tokens: set[str] = field(default_factory=set)
    scope_claim: str = ""     # "minor", "moderate", "major", "refactor"


# Action verb → canonical event type mapping
_ACTION_VERBS: dict[str, str] = {}
_ACTION_GROUPS = [
    (["add", "adds", "added", "adding", "introduce", "introduces", "create", "creates",
      "implement", "implements", "implemented"], "add"),
    (["remove", "removes", "removed", "removing", "delete", "deletes", "drop", "drops"], "remove"),
    (["fix", "fixes", "fixed", "fixing", "repair", "patch", "patches", "resolve", "resolves"], "fix"),
    (["refactor", "refactors", "refactored", "refactoring", "restructure", "reorganize",
      "clean", "cleans", "cleanup"], "refactor"),
    (["update", "updates", "updated", "updating", "modify", "modifies", "change", "changes",
      "adjust", "adjusts", "tweak", "tweaks"], "update"),
    (["rename", "renames", "renamed", "move", "moves", "moved"], "rename"),
    (["optimize", "optimizes", "optimized", "improve", "improves", "speed", "performance"], "optimize"),
    (["document", "documents", "documented", "docs"], "document"),
    (["test", "tests", "tested", "testing"], "test"),
    (["migrate", "migrates", "upgrade", "upgrades", "bump", "bumps"], "migrate"),
]
for _verbs, _canonical in _ACTION_GROUPS:
    for _v in _verbs:
        _ACTION_VERBS[_v] = _canonical


_SCOPE_MINOR = re.compile(
    r"\b(minor|small|tiny|trivial|typo|whitespace|formatting|lint|nit)\b",
    re.IGNORECASE,
)
_SCOPE_MAJOR = re.compile(
    r"\b(major|breaking|rewrite|overhaul|redesign|rearchitect)\b",
    re.IGNORECASE,
)
_SCOPE_REFACTOR = re.compile(
    r"\b(refactor|restructure|reorganize|clean\s*up)\b",
    re.IGNORECASE,
)

_VERB_SUBJECT_RE = re.compile(
    r"\b("
    + "|".join(sorted(_ACTION_VERBS.keys(), key=len, reverse=True))
    + r")\s+(?:the\s+|a\s+)?"
    r"(\w+(?:[_./]\w+)*)",
    re.IGNORECASE,
)


def _parse_message(message: str) -> MessageAnalysis:
    """Parse a commit message into claimed events."""
    analysis = MessageAnalysis(raw=message)

    # Split subject and body
    parts = message.strip().split("\n", 1)
    analysis.subject = parts[0].strip()
    analysis.body = parts[1].strip() if len(parts) > 1 else ""

    full_text = message.lower()

    # Extract verb-subject pairs
    for m in _VERB_SUBJECT_RE.finditer(full_text):
        verb_raw = m.group(1).lower()
        subj = m.group(2).lower()
        canonical = _ACTION_VERBS.get(verb_raw, verb_raw)
        analysis.action_verbs.append(canonical)
        analysis.subjects.append(subj)
        analysis.claimed_events.add(f"{canonical}:{subj}")

    # Content tokens
    words = re.findall(r"[a-z][a-z0-9_]+", full_text)
    analysis.content_tokens = {w for w in words if len(w) >= 3}

    # Scope classification
    if _SCOPE_MINOR.search(message):
        analysis.scope_claim = "minor"
    elif _SCOPE_MAJOR.search(message):
        analysis.scope_claim = "major"
    elif _SCOPE_REFACTOR.search(message):
        analysis.scope_claim = "refactor"
    else:
        analysis.scope_claim = "moderate"

    return analysis


# ── TRIAD signals ────────────────────────────────────────────────────


def _signal_token_alignment(diff: DiffAnalysis, msg: MessageAnalysis) -> float:
    """Signal A: Token-level Jaccard between diff content and message."""
    # Filter out trivially common programming words
    _COMMON = frozenset({
        "self", "return", "import", "from", "class", "def", "none",
        "true", "false", "pass", "init", "str", "int", "float",
        "list", "dict", "set", "type", "name", "value", "data",
    })
    d_tokens = diff.content_tokens - _COMMON
    m_tokens = msg.content_tokens - _COMMON

    if not d_tokens and not m_tokens:
        return 1.0  # Both empty → trivially aligned
    if not d_tokens or not m_tokens:
        return 0.0

    union = d_tokens | m_tokens
    intersection = d_tokens & m_tokens
    return len(intersection) / len(union) if union else 0.0


def _signal_structural_alignment(diff: DiffAnalysis, msg: MessageAnalysis) -> float:
    """Signal S: Do the claimed events match the structural events?

    Each claimed event (action:subject) must match BOTH:
      1. The action type has a compatible event in the diff
      2. The subject token appears somewhere in the diff content

    Cross-matches (e.g., 'fix' ≈ 'modify') are allowed only when
    the subject is also grounded in the diff content tokens.
    """
    if not msg.claimed_events:
        return 1.0  # No claims → can't be wrong

    diff_events = diff.structural_events

    matched = 0
    for claim in msg.claimed_events:
        action, subject = claim.split(":", 1) if ":" in claim else (claim, "")

        # Ground truth: does the subject appear in the diff at all?
        subject_grounded = (
            subject.lower() in diff.content_tokens
            or any(subject.lower() in event.lower() for event in diff_events)
        )

        # Check if the action type matches a diff event
        action_match = False
        for event in diff_events:
            event_type = event.split(":")[0] if ":" in event else ""
            if action in event_type or event_type.startswith(action):
                action_match = True
                break

        # Cross-match: 'fix' ≈ 'modify', 'refactor' ≈ 'modify'
        # BUT only if the subject is also grounded in the diff.
        if not action_match and subject_grounded:
            for event in diff_events:
                event_type = event.split(":")[0] if ":" in event else ""
                if action == "fix" and event_type in ("modify_function", "modify_file"):
                    action_match = True
                    break
                if action == "refactor" and event_type.startswith("modify"):
                    action_match = True
                    break

        # Both action and subject must be grounded
        if subject_grounded and action_match:
            matched += 1
        elif subject_grounded:
            matched += 0.5  # Subject exists but action doesn't match
        elif action_match:
            matched += 0.3  # Action matches but subject is invented

    return matched / len(msg.claimed_events) if msg.claimed_events else 1.0


def _signal_scope_consistency(diff: DiffAnalysis, msg: MessageAnalysis) -> float:
    """Signal C: Does the message's scope claim match the diff's actual scope?

    Scope classification:
      minor: ≤30 lines, ≤2 files
      moderate: ≤200 lines, ≤10 files
      major: anything larger
      refactor: balanced adds/removes (ratio > 0.3)
    """
    total = diff.total_lines
    n_files = len(diff.files_changed)
    scope = msg.scope_claim

    if scope == "minor":
        # "minor" claim: consistent only if diff is genuinely small
        if total > 50 or n_files > 3:
            return 0.3  # Scope mismatch: says minor, actually substantial
        if total > 20 or n_files > 2:
            return 0.6  # Borderline
        return 1.0
    elif scope == "major":
        if total < 20 and n_files <= 2:
            return 0.5  # Says major but diff is tiny
        return 1.0
    elif scope == "refactor":
        # Refactors should have roughly balanced adds/removes
        if diff.lines_added > 0 and diff.lines_removed > 0:
            balance = min(diff.lines_added, diff.lines_removed) / \
                      max(diff.lines_added, diff.lines_removed)
            if balance > 0.3:
                return 1.0  # Balanced changes → consistent refactor
            return 0.6  # Mostly adds or mostly removes → not really a refactor
        # Pure addition or pure deletion claimed as "refactor"
        return 0.4
    else:
        return 0.8  # "moderate" is always somewhat consistent


# ── Result ───────────────────────────────────────────────────────────


@dataclass
class TriadResult:
    """Full TRIAD verification result."""
    message: str
    diff_summary: dict[str, Any]
    token_alignment: float       # Signal A ∈ [0, 1]
    structural_alignment: float  # Signal S ∈ [0, 1]
    scope_consistency: float     # Signal C ∈ [0, 1]
    triad_score: float           # Weighted combination ∈ [0, 1]
    mismatches: list[str]        # Human-readable mismatch descriptions
    verdict: str                 # "aligned", "suspect", "misaligned"

    def explain(self) -> str:
        lines = [
            "=== TRIAD — Diff-Message Alignment ===",
            f"verdict: {self.verdict}  TRIAD={self.triad_score:.3f}",
            f"  token_alignment:      {self.token_alignment:.3f}",
            f"  structural_alignment: {self.structural_alignment:.3f}",
            f"  scope_consistency:    {self.scope_consistency:.3f}",
        ]
        if self.mismatches:
            lines.append("")
            lines.append("  Mismatches:")
            for m in self.mismatches:
                lines.append(f"    - {m}")
        return "\n".join(lines)


# ── Top-level API ────────────────────────────────────────────────────


def triad_verify(
    message: str,
    diff: str,
    fail_threshold: float = 0.30,
    warn_threshold: float = 0.55,
) -> TriadResult:
    """Verify a commit message / PR description against a diff.

    Args:
        message: Commit message or PR description text.
        diff: Unified diff text.
        fail_threshold: TRIAD score below this → verdict="misaligned".
        warn_threshold: TRIAD score below this → verdict="suspect".

    Returns:
        TriadResult with three-signal breakdown and overall verdict.
    """
    diff_analysis = _parse_diff(diff)
    msg_analysis = _parse_message(message)

    # Compute three signals
    ta = _signal_token_alignment(diff_analysis, msg_analysis)
    sa = _signal_structural_alignment(diff_analysis, msg_analysis)
    sc = _signal_scope_consistency(diff_analysis, msg_analysis)

    # Weighted combination
    triad = 0.40 * ta + 0.35 * sa + 0.25 * sc

    # Detect specific mismatches for explainability
    mismatches: list[str] = []

    if ta < 0.1:
        mismatches.append(
            "Message content shares almost no tokens with the diff"
        )

    # Check for claims about things not in the diff
    for claim in msg_analysis.claimed_events:
        action, subject = claim.split(":", 1) if ":" in claim else (claim, "")
        if subject and not any(
            subject.lower() in t for t in diff_analysis.content_tokens
        ):
            mismatches.append(
                f"Message claims '{action} {subject}' but '{subject}' "
                f"does not appear in the diff"
            )

    if sc < 0.5:
        mismatches.append(
            f"Scope mismatch: message says '{msg_analysis.scope_claim}' "
            f"but diff is {diff_analysis.total_lines} lines across "
            f"{len(diff_analysis.files_changed)} files"
        )

    # Verdict
    if triad < fail_threshold:
        verdict = "misaligned"
    elif triad < warn_threshold:
        verdict = "suspect"
    else:
        verdict = "aligned"

    return TriadResult(
        message=message,
        diff_summary={
            "files_changed": diff_analysis.files_changed,
            "lines_added": diff_analysis.lines_added,
            "lines_removed": diff_analysis.lines_removed,
            "added_functions": sorted(diff_analysis.added_functions),
            "removed_functions": sorted(diff_analysis.removed_functions),
            "modified_functions": sorted(diff_analysis.modified_functions),
        },
        token_alignment=round(ta, 4),
        structural_alignment=round(sa, 4),
        scope_consistency=round(sc, 4),
        triad_score=round(triad, 4),
        mismatches=mismatches,
        verdict=verdict,
    )
