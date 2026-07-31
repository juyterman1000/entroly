"""
50-pattern regex classifier for verifiable CLI outcomes.

Classifies shell commands + exit codes into deterministic outcome
categories. Each classification carries a source-strength weight
(hook=0.7, CI=1.0, proxy=0.4) for hierarchical Bayesian relabeling.

Design: ordered pattern list. First match wins. Patterns are grouped
by category so maintenance is obvious. The classifier is pure — no
I/O, no state — so it's trivially testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Outcome categories ───────────────────────────────────────────────

CATEGORY_TEST = "test"
CATEGORY_BUILD = "build"
CATEGORY_LINT = "lint"
CATEGORY_TYPECHECK = "typecheck"
CATEGORY_FORMAT = "format"
CATEGORY_OTHER = "other"

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Classification:
    """Result of classifying a shell command."""

    category: str        # test, build, lint, typecheck, format, other
    verdict: str         # pass, fail, unknown
    tool: str            # e.g. "pytest", "cargo_test", "eslint"
    confidence: float    # 0.0–1.0 match quality
    source_strength: float = 0.7  # hook default; CI overrides to 1.0


@dataclass(frozen=True)
class _Pattern:
    """A single classifier pattern."""

    regex: re.Pattern
    category: str
    tool: str
    # If True, exit_code==0 → pass, else → fail.
    # If False, verdict is always determined by stdout parsing.
    exit_code_determines_verdict: bool = True


# ── Pattern table (50 patterns) ──────────────────────────────────────
# Order matters: first match wins. More specific patterns go first.

_PATTERNS: list[_Pattern] = [
    # ── Test runners ─────────────────────────────────────────────────
    _Pattern(re.compile(r"\bpytest\b|python\s+-m\s+pytest\b|py\.test\b"), CATEGORY_TEST, "pytest"),
    _Pattern(re.compile(r"\bunittest\b|python\s+-m\s+unittest\b"), CATEGORY_TEST, "unittest"),
    _Pattern(re.compile(r"\bnose2?\b|python\s+-m\s+nose\b"), CATEGORY_TEST, "nose"),
    _Pattern(re.compile(r"\bjest\b|npx\s+jest\b"), CATEGORY_TEST, "jest"),
    _Pattern(re.compile(r"\bvitest\b|npx\s+vitest\b"), CATEGORY_TEST, "vitest"),
    _Pattern(re.compile(r"\bmocha\b|npx\s+mocha\b"), CATEGORY_TEST, "mocha"),
    _Pattern(re.compile(r"\bcypress\b|npx\s+cypress\b"), CATEGORY_TEST, "cypress"),
    _Pattern(re.compile(r"\bplaywright\b|npx\s+playwright\b"), CATEGORY_TEST, "playwright"),
    _Pattern(re.compile(r"\bgo\s+test\b"), CATEGORY_TEST, "go_test"),
    _Pattern(re.compile(r"\bcargo\s+test\b"), CATEGORY_TEST, "cargo_test"),
    _Pattern(re.compile(r"\bdotnet\s+test\b"), CATEGORY_TEST, "dotnet_test"),
    _Pattern(re.compile(r"\brspec\b|bundle\s+exec\s+rspec\b"), CATEGORY_TEST, "rspec"),
    _Pattern(re.compile(r"\bphpunit\b"), CATEGORY_TEST, "phpunit"),
    _Pattern(re.compile(r"\bmaven\s+test\b|mvn\s+test\b"), CATEGORY_TEST, "maven_test"),
    _Pattern(re.compile(r"\bgradle\s+test\b|gradlew\s+test\b"), CATEGORY_TEST, "gradle_test"),
    _Pattern(re.compile(r"\bnpm\s+(?:run\s+)?test\b|yarn\s+(?:run\s+)?test\b|pnpm\s+test\b"), CATEGORY_TEST, "npm_test"),
    _Pattern(re.compile(r"\bzig\s+test\b"), CATEGORY_TEST, "zig_test"),
    _Pattern(re.compile(r"\bswift\s+test\b"), CATEGORY_TEST, "swift_test"),
    _Pattern(re.compile(r"\belixir.*test\b|mix\s+test\b"), CATEGORY_TEST, "mix_test"),

    # ── Build tools ──────────────────────────────────────────────────
    _Pattern(re.compile(r"\bcargo\s+build\b"), CATEGORY_BUILD, "cargo_build"),
    _Pattern(re.compile(r"\bgo\s+build\b"), CATEGORY_BUILD, "go_build"),
    _Pattern(re.compile(r"\bnpm\s+run\s+build\b|yarn\s+build\b|pnpm\s+build\b"), CATEGORY_BUILD, "npm_build"),
    _Pattern(re.compile(r"\btsc\b(?!.*--noEmit)"), CATEGORY_BUILD, "tsc_build"),
    _Pattern(re.compile(r"\bwebpack\b"), CATEGORY_BUILD, "webpack"),
    _Pattern(re.compile(r"\bvite\s+build\b"), CATEGORY_BUILD, "vite_build"),
    _Pattern(re.compile(r"\bmake\b(?!\s+test)"), CATEGORY_BUILD, "make"),
    _Pattern(re.compile(r"\bcmake\b"), CATEGORY_BUILD, "cmake"),
    _Pattern(re.compile(r"\bgcc\b|g\+\+\b|clang\b|clang\+\+\b"), CATEGORY_BUILD, "cc"),
    _Pattern(re.compile(r"\bjavac\b"), CATEGORY_BUILD, "javac"),
    _Pattern(re.compile(r"\bdotnet\s+build\b"), CATEGORY_BUILD, "dotnet_build"),
    _Pattern(re.compile(r"\bzig\s+build\b"), CATEGORY_BUILD, "zig_build"),

    # ── Linters ──────────────────────────────────────────────────────
    _Pattern(re.compile(r"\bruff\b(?:\s+check)?"), CATEGORY_LINT, "ruff"),
    _Pattern(re.compile(r"\beslint\b|npx\s+eslint\b"), CATEGORY_LINT, "eslint"),
    _Pattern(re.compile(r"\bpylint\b"), CATEGORY_LINT, "pylint"),
    _Pattern(re.compile(r"\bflake8\b"), CATEGORY_LINT, "flake8"),
    _Pattern(re.compile(r"\bclippy\b|cargo\s+clippy\b"), CATEGORY_LINT, "clippy"),
    _Pattern(re.compile(r"\bgolangci-lint\b|golint\b"), CATEGORY_LINT, "golint"),
    _Pattern(re.compile(r"\brubocop\b"), CATEGORY_LINT, "rubocop"),
    _Pattern(re.compile(r"\bshellcheck\b"), CATEGORY_LINT, "shellcheck"),
    _Pattern(re.compile(r"\bhadolint\b"), CATEGORY_LINT, "hadolint"),

    # ── Type checkers ────────────────────────────────────────────────
    _Pattern(re.compile(r"\bmypy\b"), CATEGORY_TYPECHECK, "mypy"),
    _Pattern(re.compile(r"\bpyright\b|npx\s+pyright\b"), CATEGORY_TYPECHECK, "pyright"),
    _Pattern(re.compile(r"\btsc\s+.*--noEmit\b|tsc\b.*--noEmit"), CATEGORY_TYPECHECK, "tsc_check"),
    _Pattern(re.compile(r"\bflow\b"), CATEGORY_TYPECHECK, "flow"),
    _Pattern(re.compile(r"\bcargo\s+check\b"), CATEGORY_TYPECHECK, "cargo_check"),

    # ── Formatters ───────────────────────────────────────────────────
    _Pattern(re.compile(r"\bblack\b(?:\s+--check)?"), CATEGORY_FORMAT, "black"),
    _Pattern(re.compile(r"\bprettier\b|npx\s+prettier\b"), CATEGORY_FORMAT, "prettier"),
    _Pattern(re.compile(r"\bisort\b"), CATEGORY_FORMAT, "isort"),
    _Pattern(re.compile(r"\brustfmt\b|cargo\s+fmt\b"), CATEGORY_FORMAT, "rustfmt"),
    _Pattern(re.compile(r"\bgofmt\b|goimports\b"), CATEGORY_FORMAT, "gofmt"),
]

assert len(_PATTERNS) == 50, f"Expected 50 patterns, got {len(_PATTERNS)}"


# ── Stdout verdict refinement ────────────────────────────────────────
# When the exit code says pass but stdout says otherwise (or vice
# versa), these patterns let us refine the verdict.

_STDOUT_FAIL_SIGNALS = re.compile(
    r"FAIL(?:ED|URE)?|ERROR|"
    r"(?:\d+)\s+(?:failed|errors?|failures?)|"
    r"AssertionError|panic|SEGFAULT|"
    r"(?:Build|Compilation)\s+(?:failed|error)",
    re.IGNORECASE,
)

_STDOUT_PASS_SIGNALS = re.compile(
    r"(?:\d+)\s+passed|"
    r"(?:All\s+)?tests?\s+passed|"
    r"OK\s*$|"
    r"Build\s+succeeded|"
    r"0\s+(?:errors?|failures?|failed)",
    re.IGNORECASE,
)


def classify(
    command: str,
    exit_code: int,
    stdout_tail: str = "",
    source_strength: float = 0.7,
) -> Optional[Classification]:
    """Classify a shell command into a verifiable outcome.

    Returns None if the command doesn't match any known pattern
    (i.e. it's not a verifiable task — just regular shell usage).

    Args:
        command: The shell command string.
        exit_code: Process exit code (0 = success).
        stdout_tail: Last ~1KB of stdout for verdict refinement.
        source_strength: Signal strength (0.7 = hook, 1.0 = CI, 0.4 = proxy).
    """
    # Fast path: proxy implicit feedback events use [proxy:archetype] format.
    # These don't match any shell pattern — handle them directly.
    if command.startswith("[proxy:"):
        archetype = command.removeprefix("[proxy:").removesuffix("]")
        # Map archetype to a category for the Bayesian cells
        if "/" in archetype:
            category = archetype.split("/")[0]
        else:
            category = archetype
        # Map to known categories or use "other"
        cat_map = {
            "test": CATEGORY_TEST, "build": CATEGORY_BUILD,
            "lint": CATEGORY_LINT, "typecheck": CATEGORY_TYPECHECK,
            "format": CATEGORY_FORMAT,
        }
        category = cat_map.get(category, CATEGORY_OTHER)
        verdict = VERDICT_PASS if exit_code == 0 else VERDICT_FAIL
        return Classification(
            category=category,
            verdict=verdict,
            tool=archetype,
            confidence=0.4,  # lower confidence for implicit signals
            source_strength=source_strength,
        )

    for pat in _PATTERNS:
        if pat.regex.search(command):
            # Primary verdict from exit code
            if pat.exit_code_determines_verdict:
                if exit_code == 0:
                    verdict = VERDICT_PASS
                elif exit_code in (4, 5) and pat.category == CATEGORY_TEST:
                    # pytest exit 4 = usage error, 5 = no tests collected
                    # These are not model failures — skip them
                    verdict = VERDICT_UNKNOWN
                else:
                    verdict = VERDICT_FAIL
            else:
                verdict = VERDICT_UNKNOWN

            # Refine with stdout signals (can override in edge cases)
            confidence = 0.85  # base confidence for regex + exit code

            if stdout_tail:
                has_fail = bool(_STDOUT_FAIL_SIGNALS.search(stdout_tail))
                has_pass = bool(_STDOUT_PASS_SIGNALS.search(stdout_tail))

                if has_fail and not has_pass:
                    if verdict == VERDICT_PASS:
                        # Exit code says pass but stdout says fail — lower confidence
                        confidence = 0.4
                    else:
                        confidence = 0.95  # exit code + stdout agree on fail
                elif has_pass and not has_fail:
                    if verdict == VERDICT_FAIL:
                        confidence = 0.4
                    else:
                        confidence = 0.95  # exit code + stdout agree on pass
                # Both or neither: keep base confidence

            return Classification(
                category=pat.category,
                verdict=verdict,
                tool=pat.tool,
                confidence=confidence,
                source_strength=source_strength,
            )

    return None  # Not a verifiable command
