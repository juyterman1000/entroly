"""
Output token reduction — effort-based routing and verbosity steering.

Complements the input-side compression pipeline with output-side savings:
  1. EffortClassifier: maps query complexity → effort level
  2. VerbositySteerer: injects system-prompt directives to bound output length
  3. OutputBudget: enforces token budget on model responses via max_tokens

Works with the existing response_contract.py (instruction-based) and
proxy_transform.py distill_response (post-hoc trimming) as three layers:
  - Pre-generation:  VerbositySteerer injects effort-appropriate directives
  - Generation:      OutputBudget sets max_tokens proportional to effort
  - Post-generation: distill_response strips remaining filler

The proxy wires all three; each can be used standalone via MCP tools or SDK.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Effort levels
# ---------------------------------------------------------------------------

class Effort(IntEnum):
    """Output effort levels, from minimal to exhaustive."""
    MINIMAL = 1    # one-line answers: yes/no, file path, error message
    CONCISE = 2    # 2-5 sentences, no code blocks unless asked
    STANDARD = 3   # default behavior
    DETAILED = 4   # full explanations with examples
    EXHAUSTIVE = 5 # deep analysis with all context


EFFORT_DIRECTIVES: dict[Effort, str] = {
    Effort.MINIMAL: (
        "Answer in the shortest complete form. One line if possible. "
        "No pleasantries, no summaries, no explanations unless the answer "
        "is wrong without them. Preserve errors and uncertainty."
    ),
    Effort.CONCISE: (
        "Lead with the result in 2-5 sentences. Omit routine narration, "
        "filler, and repeated context. Include code only when directly asked. "
        "Preserve failures, evidence, and required next actions."
    ),
    Effort.STANDARD: "",  # no injection — default model behavior
    Effort.DETAILED: (
        "Provide a thorough explanation with examples. Walk through the "
        "reasoning step by step. Include relevant code snippets and "
        "edge cases. Cite sources and evidence."
    ),
    Effort.EXHAUSTIVE: (
        "Provide an exhaustive analysis. Cover every relevant angle, "
        "include complete code, test cases, edge cases, performance "
        "implications, and alternative approaches. Hold nothing back."
    ),
}

EFFORT_MAX_TOKENS: dict[Effort, int] = {
    Effort.MINIMAL: 150,
    Effort.CONCISE: 500,
    Effort.STANDARD: 4096,
    Effort.DETAILED: 8192,
    Effort.EXHAUSTIVE: 16384,
}


# ---------------------------------------------------------------------------
# Effort classifier
# ---------------------------------------------------------------------------

# Query patterns that suggest minimal effort
_MINIMAL_PATTERNS = [
    re.compile(r"^(?:yes|no|true|false)\??$", re.I),
    re.compile(r"^(?:what (?:is|are) the (?:path|file|name|type|version))", re.I),
    re.compile(r"^(?:where is|which file|what port|what url)", re.I),
    re.compile(r"^(?:does .+ exist|is .+ (?:running|installed|enabled))", re.I),
    re.compile(r"^(?:show me the (?:error|output|status|result))", re.I),
]

# Query patterns that suggest exhaustive effort
_EXHAUSTIVE_PATTERNS = [
    re.compile(r"(?:explain|analyze|review|audit|investigate) .{40,}", re.I),
    re.compile(r"(?:write|implement|build|create|design) .+ (?:with|including|that)", re.I),
    re.compile(r"(?:compare|contrast|difference between)", re.I),
    re.compile(r"(?:architecture|design|system) (?:review|analysis|overview)", re.I),
    re.compile(r"(?:step.by.step|thorough|comprehensive|detailed|in.depth)", re.I),
]

# Query patterns that suggest concise effort
_CONCISE_PATTERNS = [
    re.compile(r"^(?:fix|run|test|lint|build|deploy|push|pull|merge)\b", re.I),
    re.compile(r"^(?:update|change|rename|move|delete|remove|add)\b", re.I),
    re.compile(r"(?:quick|brief|short|tl;?dr|summarize)", re.I),
]


@dataclass
class EffortClassification:
    effort: Effort
    confidence: float  # 0.0-1.0
    reason: str
    query_complexity: float  # 0.0-1.0 information density of the query


def classify_effort(query: str) -> EffortClassification:
    """
    Classify the appropriate output effort level for a query.

    Uses a cascade of pattern matching, query length, and information
    density to determine how much output the query warrants.
    """
    query_stripped = query.strip()

    # Empty or trivial
    if len(query_stripped) < 5:
        return EffortClassification(Effort.MINIMAL, 0.95, "trivial query", 0.0)

    # Pattern matching (highest confidence)
    for pat in _MINIMAL_PATTERNS:
        if pat.search(query_stripped):
            return EffortClassification(
                Effort.MINIMAL, 0.9, "matches minimal pattern", 0.1,
            )

    for pat in _EXHAUSTIVE_PATTERNS:
        if pat.search(query_stripped):
            return EffortClassification(
                Effort.EXHAUSTIVE, 0.8, "matches exhaustive pattern", 0.9,
            )

    for pat in _CONCISE_PATTERNS:
        if pat.search(query_stripped):
            return EffortClassification(
                Effort.CONCISE, 0.8, "matches concise pattern", 0.3,
            )

    # Length-based heuristic
    word_count = len(query_stripped.split())
    question_marks = query_stripped.count("?")
    code_blocks = query_stripped.count("```")

    complexity = _query_complexity(query_stripped)

    _GENERATIVE_VERBS = {"explain", "describe", "review", "analyze", "summarize", "refactor", "implement", "design"}
    first_word = query_stripped.split()[0].lower() if query_stripped else ""
    if word_count <= 3 and first_word not in _GENERATIVE_VERBS:
        effort = Effort.MINIMAL
        reason = "very short query"
    elif word_count <= 3:
        effort = Effort.CONCISE
        reason = "short generative query"
    elif word_count <= 10 and question_marks <= 1 and code_blocks == 0:
        effort = Effort.CONCISE
        reason = "short single-question query"
    elif code_blocks > 0 or word_count > 50:
        effort = Effort.DETAILED
        reason = "complex query with code or multiple parts"
    else:
        effort = Effort.STANDARD
        reason = "standard query"

    return EffortClassification(effort, 0.6, reason, complexity)


def _query_complexity(text: str) -> float:
    """Estimate query complexity as 0-1 information density."""
    words = text.split()
    n = len(words)
    if n == 0:
        return 0.0

    unique_ratio = len(set(w.lower() for w in words)) / n
    avg_word_len = sum(len(w) for w in words) / n
    has_code = "```" in text or "`" in text
    question_count = text.count("?")
    conjunction_count = sum(
        1 for w in words if w.lower() in {"and", "or", "but", "also", "plus", "with"}
    )

    score = (
        0.3 * min(unique_ratio, 1.0)
        + 0.2 * min(avg_word_len / 8.0, 1.0)
        + 0.15 * (1.0 if has_code else 0.0)
        + 0.15 * min(question_count / 3.0, 1.0)
        + 0.1 * min(conjunction_count / 4.0, 1.0)
        + 0.1 * min(n / 50.0, 1.0)
    )
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Verbosity steerer
# ---------------------------------------------------------------------------

class VerbositySteerer:
    """
    Injects effort-appropriate directives into the system prompt to steer
    LLM output verbosity before generation begins.
    """

    def __init__(self, default_effort: Effort = Effort.STANDARD):
        self._default = default_effort
        self._override: Effort | None = None

    def set_effort(self, effort: Effort | int | str) -> None:
        """Override the auto-classified effort level for this session."""
        if isinstance(effort, str):
            effort = Effort[effort.upper()]
        elif isinstance(effort, int):
            effort = Effort(effort)
        self._override = effort
        logger.info("VerbositySteerer: effort overridden to %s", effort.name)

    def clear_override(self) -> None:
        self._override = None

    def steer(
        self,
        messages: list[dict[str, Any]],
        effort: Effort | None = None,
    ) -> tuple[list[dict[str, Any]], Effort, int]:
        """
        Inject verbosity directive into messages and compute max_tokens.

        Returns (modified_messages, effective_effort, recommended_max_tokens).
        """
        if effort is None:
            effort = self._override or self._default

        if effort == Effort.STANDARD:
            return messages, effort, EFFORT_MAX_TOKENS[effort]

        directive = EFFORT_DIRECTIVES[effort]
        if not directive:
            return messages, effort, EFFORT_MAX_TOKENS[effort]

        modified = list(messages)
        if modified and modified[0].get("role") == "system":
            modified[0] = {
                **modified[0],
                "content": modified[0]["content"] + "\n\n" + directive,
            }
        else:
            modified.insert(0, {"role": "system", "content": directive})

        return modified, effort, EFFORT_MAX_TOKENS[effort]

    def auto_steer(
        self, messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], EffortClassification, int]:
        """
        Classify the last user message and steer accordingly.

        Returns (modified_messages, classification, recommended_max_tokens).
        """
        if self._override:
            classification = EffortClassification(
                self._override, 1.0, "manual override", 0.5,
            )
        else:
            last_user_msg = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        last_user_msg = content
                    elif isinstance(content, list):
                        last_user_msg = " ".join(
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") == "text"
                        )
                    break
            classification = classify_effort(last_user_msg)

        modified, _, max_tokens = self.steer(messages, classification.effort)
        return modified, classification, max_tokens


# ---------------------------------------------------------------------------
# Output budget
# ---------------------------------------------------------------------------

@dataclass
class OutputBudget:
    """
    Enforces token budget on model responses.

    This sets max_tokens in the API request, not post-hoc truncation.
    Combine with VerbositySteerer (pre-gen) and distill_response (post-gen)
    for three-layer output optimization.
    """

    effort: Effort = Effort.STANDARD
    custom_max_tokens: int | None = None

    @property
    def max_tokens(self) -> int:
        if self.custom_max_tokens is not None:
            return self.custom_max_tokens
        return EFFORT_MAX_TOKENS[self.effort]

    def apply_to_request(self, request_body: dict[str, Any]) -> dict[str, Any]:
        """Apply the output budget to an API request body."""
        modified = dict(request_body)
        if self.effort != Effort.STANDARD or self.custom_max_tokens:
            modified["max_tokens"] = self.max_tokens
        return modified


# ---------------------------------------------------------------------------
# Savings estimator
# ---------------------------------------------------------------------------

def estimate_output_savings(
    original_tokens: int,
    effort: Effort,
    distill_mode: str = "full",
) -> dict[str, Any]:
    """
    Estimate output token savings from the three-layer pipeline.

    Layer 1 (VerbositySteerer): reduces generation via prompt directives
    Layer 2 (OutputBudget): hard cap via max_tokens
    Layer 3 (distill_response): post-hoc filler removal
    """
    # Empirical reduction ratios per effort level (from Headroom benchmarks)
    steering_ratios = {
        Effort.MINIMAL: 0.15,
        Effort.CONCISE: 0.35,
        Effort.STANDARD: 1.0,
        Effort.DETAILED: 1.0,
        Effort.EXHAUSTIVE: 1.0,
    }

    distill_ratios = {
        "lite": 0.92,
        "full": 0.78,
        "ultra": 0.62,
    }

    after_steering = int(original_tokens * steering_ratios.get(effort, 1.0))
    budget_cap = EFFORT_MAX_TOKENS[effort]
    after_budget = min(after_steering, budget_cap)
    after_distill = int(after_budget * distill_ratios.get(distill_mode, 0.78))

    return {
        "original_tokens": original_tokens,
        "after_steering": after_steering,
        "after_budget": after_budget,
        "after_distill": after_distill,
        "total_reduction": 1.0 - (after_distill / original_tokens) if original_tokens > 0 else 0.0,
        "effort": effort.name,
        "distill_mode": distill_mode,
    }
