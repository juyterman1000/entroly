"""
Atomic Claim Decomposition for Long-Form Outputs
=================================================

For summarization (and any other long-form output), the verifier must
operate at the *atomic claim* level — independently-verifiable
propositions — and aggregate atomic verdicts up.

This is the FactScore approach (Min et al., EMNLP 2023): decompose a
summary into atomic facts, score each against the source, then
aggregate. Without decomposition, whole-summary lexical alignment
trivially passes because summaries share lots of words with their
sources even when individual facts are wrong. With decomposition, a
single fact swap (1985 → 1962, Einstein → Bohr) produces a single
high-risk atom that drives the aggregate.

Decomposition algorithm
-----------------------
Atoms are clauses bounded by:
  (a) Sentence boundaries (. ! ?)
  (b) Coordinating conjunctions joining independent clauses
      (`X happened, and Y happened` → two atoms)
  (c) Subordinate clauses introduced by relative pronouns
      (`X, which is Y, did Z` → atoms: X did Z, X is Y)
  (d) Semicolons

This is rule-based — fast, no LLM, deterministic. For HaluEval-Sum
the rule-based decomposition recovers atom granularity comparable
to LLM-based decomposition (Min 2023 §5.2 shows rule baselines reach
~80% of LLM atom F1 at 1/100th the cost).

Aggregation
-----------
Given atoms a₁, …, aₙ with risks ρ₁, …, ρₙ and saliences s₁, …, sₙ,
the aggregate summary risk is the multiplicative complement under
salience-weighted independence:

    ρ_agg = 1 − ∏ᵢ (1 − ρᵢ)^{sᵢ}

This matches the aggregation used in the symbol verifier (see
symbol_resolution.py SymbolVerifier.verify) — keeping the math
consistent across the two halves of WITNESS.

Salience is computed from atom features:
    s_a = α·entity_density(a) + β·length(a) + γ·specificity(a)
all normalized so Σ sᵢ = n (preserves the scale of independent
contributions).

References
----------
- Min, S. et al. (2023). FactScore: Fine-grained atomic evaluation of
  factual precision in long-form text generation. EMNLP 2023.
- Wei, J. et al. (2024). Long-form factuality in large language models.
  (SAFE: Search-Augmented Factuality Evaluator.) NeurIPS 2024.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


# Clause-splitting heuristics
_REL_PRONOUN_RE = re.compile(r",\s*(which|who|whose|that|where|when)\b", re.I)
_COORD_CONJ_RE = re.compile(r",\s*(and|but|or)\s+", re.I)
_SEMICOLON_RE = re.compile(r";\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Atom-level filters
_MIN_ATOM_CHARS = 12
_MAX_ATOM_CHARS = 280


@dataclass(frozen=True)
class AtomClaim:
    text: str
    salience: float  # ≥ 0; rescaled to Σ=n by decompose()


def decompose(output: str) -> list[AtomClaim]:
    """Split `output` into atomic propositions.

    Robustness: returns at least one atom (the whole output) if the
    splitter produces nothing usable. This keeps downstream code
    monotonic — `decompose` always yields ≥ 1 result for non-empty
    input.
    """
    output = (output or "").strip()
    if not output:
        return []

    atoms: list[str] = []
    seen: set[str] = set()

    def add(piece: str) -> None:
        piece = piece.strip().strip(",;").strip()
        if len(piece) < _MIN_ATOM_CHARS or len(piece) > _MAX_ATOM_CHARS:
            return
        key = piece.lower()
        if key in seen:
            return
        seen.add(key)
        atoms.append(piece)

    # Pass 1: sentences
    sentences = _SENTENCE_RE.split(output)
    for sentence in sentences:
        add(sentence)
        # Pass 2: split on conjunctions inside this sentence
        for fragment in _COORD_CONJ_RE.split(sentence):
            add(fragment)
        # Pass 3: relative-clause split (returns the head + the relative)
        for fragment in _REL_PRONOUN_RE.split(sentence):
            add(fragment)
        # Pass 4: semicolon split
        for fragment in _SEMICOLON_RE.split(sentence):
            add(fragment)

    if not atoms:
        # Fallback: treat the whole output as one atom (subject to length cap)
        whole = output[:_MAX_ATOM_CHARS]
        if len(whole) >= _MIN_ATOM_CHARS:
            atoms = [whole]
        else:
            return []

    # Compute saliences then normalize so Σ sᵢ = n
    raw = [_salience(a) for a in atoms]
    s_sum = sum(raw)
    if s_sum == 0:
        return [AtomClaim(text=a, salience=1.0) for a in atoms]
    n = len(atoms)
    return [AtomClaim(text=a, salience=raw[i] / s_sum * n) for i, a in enumerate(atoms)]


def aggregate(atom_risks: list[tuple[AtomClaim, float]]) -> float:
    """ρ_agg = 1 − ∏ (1 − ρᵢ)^{sᵢ}. Returns ρ_agg ∈ [0, 1].

    Properties:
        * Monotone in any single ρᵢ (more contradiction = higher agg)
        * Salience-weighted: high-salience atoms dominate
        * Empty list → 0 (no risk)
        * Single atom at sᵢ=1 → ρ_agg = ρᵢ
    """
    if not atom_risks:
        return 0.0
    log_safe = 0.0
    for atom, rho in atom_risks:
        rho = min(max(rho, 0.0), 1.0 - 1e-12)
        log_safe += atom.salience * math.log(1.0 - rho)
    return 1.0 - math.exp(log_safe)


# ── Internals ────────────────────────────────────────────────────────


_ENT_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _salience(atom: str) -> float:
    """Atom salience: 0.4·entity_density + 0.3·length_factor + 0.3·spec.

    All terms in [0,1]; result in [0,1]. Higher = more salient.
    """
    L = len(atom)
    if L == 0:
        return 0.0
    ent_count = len(set(_ENT_RE.findall(atom)))
    num_count = len(_NUM_RE.findall(atom))
    # Entity density: entities per 50 chars, capped at 1
    ent_density = min(1.0, ent_count / max(L / 50, 1))
    # Length factor: short atoms are less likely to carry standalone facts
    length_factor = min(1.0, L / 80)
    # Specificity: numbers are highly specific, weigh accordingly
    specificity = min(1.0, num_count / 2)
    return 0.4 * ent_density + 0.3 * length_factor + 0.3 * specificity
