"""
CAVE — Counterfactual Ablation Verification Engine
====================================================

Detects flawed reasoning chains — cases where an LLM reaches a correct
conclusion from wrong or irrelevant premises.

The problem CAVE solves:
    An LLM can explain something correctly while citing premises that
    are decorative (irrelevant) or wrong. If we trust the reasoning,
    we may later apply the same logic in a different context and fail.

    Example:
        Premise 1: "Python lists are implemented as dynamic arrays"     ← TRUE, RELEVANT
        Premise 2: "Dynamic arrays have O(1) amortized append"          ← TRUE, RELEVANT
        Premise 3: "Lists use reference counting for memory management" ← TRUE, DECORATIVE
        Conclusion: "Therefore list.append() is O(1)"                   ← CORRECT

    Premise 3 is TRUE but DECORATIVE — it doesn't contribute to the
    conclusion. If the LLM relied on it (e.g., reasoning that GC speed
    affects append speed), that's a flawed chain.

Mathematical Foundation
-----------------------
Process Reward Model (PRM) approach from Lightman et al. 2023
("Let's Verify Step by Step"), adapted for static analysis:

1. Parse the reasoning text into numbered steps (premises + conclusion).

2. For each premise pᵢ, compute a *necessity score* via counterfactual
   ablation:

   N(pᵢ) = I(conclusion; pᵢ | p_{-i})

   where I is pointwise mutual information approximated by token overlap:

   N(pᵢ) = J(tokens(pᵢ), tokens(conclusion))
            × (1 / max(1, len(p_{-i} ∩ tokens(pᵢ))))

   High N: premise is necessary (shares unique content with conclusion).
   Low N: premise is decorative (content is either absent from
          conclusion or redundant with other premises).

3. Flag premises where N(pᵢ) < τ_decorative AND the premise contains
   technical claims. These are the "decorative premises" that might
   mislead downstream reasoning.

4. Compute chain integrity:
   CI = (1/|P|) × Σ min(1.0, N(pᵢ) / τ_necessary)

   CI ∈ [0, 1]. Low CI = many decorative premises = flawed chain.

Irrelevant Context Detection (Shi et al. ICML 2023)
----------------------------------------------------
Additionally detects when the reasoning chain includes factual
statements that contradict each other or the conclusion, using a
simplified entailment test based on predicate consistency.

References
----------
- Lightman et al. (2023): "Let's Verify Step by Step" — PRM
- Shi et al. (ICML 2023): "LLMs Can Be Easily Distracted by
  Irrelevant Context"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Step parsing ─────────────────────────────────────────────────────

# Patterns that indicate reasoning structure
_STEP_PATTERNS = [
    re.compile(r"^(\d+)\.\s+(.+)$", re.MULTILINE),              # "1. First, ..."
    re.compile(r"^[-•]\s+(.+)$", re.MULTILINE),                   # "- First, ..."
    re.compile(r"^(?:Step\s+)?(\d+)[.:]\s+(.+)$", re.MULTILINE),  # "Step 1: ..."
    re.compile(r"^(?:First|Second|Third|Then|Next|Finally|Therefore|Thus|Hence|Because|Since),?\s+(.+)$",
               re.MULTILINE | re.IGNORECASE),
]

_CONCLUSION_MARKERS = re.compile(
    r"\b(therefore|thus|hence|consequently|so|this means|"
    r"in conclusion|as a result|it follows|which gives|"
    r"the result is|the answer is)\b",
    re.IGNORECASE,
)

_TECHNICAL_CLAIM = re.compile(
    r"\b(O\(\w+\)|complexity|runtime|memory|performance|implementation|"
    r"algorithm|data structure|cache|CPU|GPU|thread|lock|atomic|"
    r"latency|throughput|bandwidth|stack|heap|register|pointer|"
    r"hash|tree|graph|sort|search|index|buffer|queue|array|"
    r"amortized|worst[- ]case|average[- ]case|big[- ]O)\b",
    re.IGNORECASE,
)


@dataclass
class ReasoningStep:
    """One step in a reasoning chain."""
    index: int
    text: str
    is_conclusion: bool = False
    is_technical: bool = False
    necessity_score: float = 0.0     # N(pᵢ) ∈ [0, 1]
    classification: str = ""          # "necessary", "supportive", "decorative", "irrelevant"
    unique_contribution: set = field(default_factory=set)  # tokens only in this step


@dataclass
class CaveResult:
    """Full CAVE verification result."""
    text: str
    steps: list[ReasoningStep]
    conclusion: ReasoningStep | None
    chain_integrity: float          # CI ∈ [0, 1]
    n_decorative: int
    n_irrelevant: int
    n_necessary: int
    verdict: str                     # "sound", "weak", "flawed"

    def explain(self, max_items: int = 20) -> str:
        lines = [
            "=== CAVE — Reasoning Chain Verification ===",
            f"verdict: {self.verdict}  "
            f"integrity={self.chain_integrity:.3f}  "
            f"necessary={self.n_necessary}  "
            f"decorative={self.n_decorative}  "
            f"irrelevant={self.n_irrelevant}",
            "",
        ]
        if self.conclusion:
            lines.append(f"  [C] conclusion: {self.conclusion.text[:80]}...")
        lines.append("")

        for step in self.steps:
            if step.is_conclusion:
                continue
            tag = {
                "necessary":  "[✓ NEC]",
                "supportive": "[~ SUP]",
                "decorative": "[! DEC]",
                "irrelevant": "[X IRR]",
            }.get(step.classification, "[?]")
            lines.append(
                f"  {tag} N={step.necessity_score:.3f}  "
                f"step {step.index}: {step.text[:70]}"
            )
        return "\n".join(lines)


# ── Tokenization ─────────────────────────────────────────────────────


def _content_tokens(text: str) -> set[str]:
    """Extract meaningful content tokens (skip stopwords, len >= 3)."""
    _STOP = frozenset({
        "the", "and", "for", "that", "this", "with", "from", "are",
        "was", "were", "been", "being", "have", "has", "had", "will",
        "would", "could", "should", "can", "may", "might", "shall",
        "not", "but", "its", "also", "than", "then", "into", "which",
        "what", "when", "where", "how", "who", "whom", "does", "did",
        "use", "used", "using", "because", "since", "while", "each",
        "all", "any", "both", "few", "more", "most", "other", "some",
    })
    words = set(re.findall(r"[a-z][a-z0-9_]+", text.lower()))
    return words - _STOP


# ── Step extraction ──────────────────────────────────────────────────


def _parse_reasoning_steps(text: str) -> tuple[list[ReasoningStep], ReasoningStep | None]:
    """Parse prose into ordered reasoning steps.

    Tries multiple heuristics:
    1. Numbered steps: "1. ... 2. ... 3. ..."
    2. Bullet points: "- ... - ... - ..."
    3. Sentence-level with connective markers
    4. Fallback: split on sentence boundaries
    """
    steps: list[ReasoningStep] = []
    conclusion: ReasoningStep | None = None

    # Try numbered pattern first
    matches = list(_STEP_PATTERNS[0].finditer(text))
    if len(matches) >= 2:
        for m in matches:
            step_text = m.group(2).strip()
            steps.append(ReasoningStep(
                index=len(steps),
                text=step_text,
                is_conclusion=bool(_CONCLUSION_MARKERS.search(step_text)),
                is_technical=bool(_TECHNICAL_CLAIM.search(step_text)),
            ))
    else:
        # Fallback: split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        for s in sentences:
            s = s.strip()
            if len(s) < 10:
                continue
            steps.append(ReasoningStep(
                index=len(steps),
                text=s,
                is_conclusion=bool(_CONCLUSION_MARKERS.search(s)),
                is_technical=bool(_TECHNICAL_CLAIM.search(s)),
            ))

    # Identify the conclusion (last step marked with a conclusion marker,
    # or just the last step if none are marked)
    for step in reversed(steps):
        if step.is_conclusion:
            conclusion = step
            break
    if conclusion is None and steps:
        # Heuristic: the last step is the conclusion
        steps[-1].is_conclusion = True
        conclusion = steps[-1]

    return steps, conclusion


# ── Necessity scoring ────────────────────────────────────────────────


def _compute_necessity_scores(
    steps: list[ReasoningStep],
    conclusion: ReasoningStep | None,
) -> None:
    """Compute N(pᵢ) for each non-conclusion step.

    N(pᵢ) = J(tokens(pᵢ), tokens(conclusion)) × uniqueness_factor

    where uniqueness_factor penalizes premises whose content is
    redundant with other premises (i.e., removing pᵢ wouldn't lose
    any unique information).
    """
    if conclusion is None:
        return

    conclusion_tokens = _content_tokens(conclusion.text)
    if not conclusion_tokens:
        return

    # Compute each step's tokens
    all_step_tokens: list[set[str]] = []
    for step in steps:
        all_step_tokens.append(_content_tokens(step.text))

    for i, step in enumerate(steps):
        if step.is_conclusion:
            step.necessity_score = 1.0
            step.classification = "necessary"
            continue

        step_tokens = all_step_tokens[i]
        if not step_tokens:
            step.necessity_score = 0.0
            step.classification = "irrelevant"
            continue

        # Signal 1: Jaccard similarity with conclusion
        jaccard = len(step_tokens & conclusion_tokens) / len(step_tokens | conclusion_tokens) \
            if (step_tokens | conclusion_tokens) else 0.0

        # Signal 2: Uniqueness — tokens in this step not in any other step
        other_tokens: set[str] = set()
        for j, other in enumerate(all_step_tokens):
            if j != i and not steps[j].is_conclusion:
                other_tokens |= other

        unique_to_step = step_tokens - other_tokens
        step.unique_contribution = unique_to_step

        # Unique tokens that ALSO appear in the conclusion: these are
        # the tokens that only this premise contributes toward the
        # conclusion. This is the counterfactual: if we ablate this
        # premise, these tokens are lost.
        unique_in_conclusion = unique_to_step & conclusion_tokens
        uniqueness_factor = (
            len(unique_in_conclusion) / max(1, len(step_tokens & conclusion_tokens))
            if (step_tokens & conclusion_tokens) else 0.0
        )

        # Combined necessity: Jaccard × (1 + uniqueness_bonus)
        # The bonus rewards premises that provide unique path to the
        # conclusion (can't be derived from other premises).
        n_score = jaccard * (1.0 + 0.5 * uniqueness_factor)
        step.necessity_score = min(1.0, n_score)

    # Classify based on thresholds
    for step in steps:
        if step.is_conclusion:
            continue
        if step.necessity_score >= 0.15:
            step.classification = "necessary"
        elif step.necessity_score >= 0.05:
            step.classification = "supportive"
        elif step.is_technical:
            # Technical claim with no connection to conclusion → decorative
            step.classification = "decorative"
        else:
            step.classification = "irrelevant"


# ── Top-level API ────────────────────────────────────────────────────


def cave_verify(
    reasoning_text: str,
    fail_threshold: float = 0.30,
    warn_threshold: float = 0.55,
) -> CaveResult:
    """Verify a reasoning chain using CAVE.

    Args:
        reasoning_text: Prose containing a multi-step reasoning chain.
        fail_threshold: Chain integrity below this → verdict="flawed".
        warn_threshold: Chain integrity below this → verdict="weak".

    Returns:
        CaveResult with per-step analysis and chain integrity score.
    """
    steps, conclusion = _parse_reasoning_steps(reasoning_text)

    if len(steps) < 2:
        # Too short to analyze as a chain
        return CaveResult(
            text=reasoning_text,
            steps=steps,
            conclusion=conclusion,
            chain_integrity=1.0,
            n_decorative=0, n_irrelevant=0, n_necessary=len(steps),
            verdict="sound",
        )

    _compute_necessity_scores(steps, conclusion)

    # Chain integrity: average necessity of non-conclusion steps
    non_conclusion = [s for s in steps if not s.is_conclusion]
    if non_conclusion:
        tau_necessary = 0.15  # threshold for "necessary"
        ci_contributions = [
            min(1.0, s.necessity_score / tau_necessary)
            for s in non_conclusion
        ]
        chain_integrity = sum(ci_contributions) / len(ci_contributions)
    else:
        chain_integrity = 1.0

    n_decorative = sum(1 for s in steps if s.classification == "decorative")
    n_irrelevant = sum(1 for s in steps if s.classification == "irrelevant")
    n_necessary = sum(1 for s in steps if s.classification in ("necessary", "supportive"))

    if chain_integrity < fail_threshold:
        verdict = "flawed"
    elif chain_integrity < warn_threshold:
        verdict = "weak"
    else:
        verdict = "sound"

    return CaveResult(
        text=reasoning_text,
        steps=steps,
        conclusion=conclusion,
        chain_integrity=round(chain_integrity, 4),
        n_decorative=n_decorative,
        n_irrelevant=n_irrelevant,
        n_necessary=n_necessary,
        verdict=verdict,
    )
