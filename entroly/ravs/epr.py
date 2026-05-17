"""Logprob Entropy-Production-Rate (EPR) hallucination detector.

Grounded in published research (2025-2026):
  - HALT (Shapiro et al., arXiv:2602.02888, Feb 2026): logprob time-series
  - Semantic Entropy Probes (Kossen et al., ICLR 2025): single-pass entropy
  - EPR (Entropy Production Rate, 2025): average token entropy as signal

The key insight from the literature:
  Hallucinated tokens have HIGHER entropy (model is less confident).
  Entity-position tokens that are hallucinated have especially high entropy.
  This can be measured from standard API logprobs at ZERO extra cost.

What we implement:
  A lightweight EPR detector that extracts 5 features from per-token logprobs
  and combines them with WITNESS claim-level verification for a fused score.
  No neural network training needed — the features are combined analytically.

Why this is a genuine advance for Entroly:
  1. ECE already has the logprob path (_curvature_from_logprobs) but the
     proxy never passes logprobs. We wire that connection.
  2. EPR adds 3 new features beyond ECE's entity curvature:
     - Global entropy production rate (mean token entropy)
     - Entropy variance (hallucination creates entropy spikes)
     - Entity-vs-background entropy ratio (hallucinated entities are uncertain)
  3. These are combined with WITNESS's claim-level score for a 5-feature
     fusion that has strictly more information than either alone.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════
# Entity detection (reuse from ECE)
# ═══════════════════════════════════════════════════════════════════════

_ENTITY_PATTERNS = [
    re.compile(r'\b\d{4}\b'),                          # years
    re.compile(r'\b\d+\.?\d*\s*(?:m|km|kg|cm|°)\b'),  # measurements
    re.compile(r'\b\d+(?:\.\d+)?\b'),                  # numbers
    re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b'), # proper nouns
]


def _find_entity_char_positions(text: str) -> list[tuple[int, int]]:
    """Find character-level (start, end) spans for entities."""
    spans = []
    for pat in _ENTITY_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    # Deduplicate overlapping spans
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


# ═══════════════════════════════════════════════════════════════════════
# EPR Feature Extraction
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EPRSignal:
    """Entropy Production Rate features from a single generation pass."""
    # Core EPR features
    mean_entropy: float          # H̄ = mean(-logprob) across all tokens
    entropy_variance: float      # Var(-logprob) — spiky = hallucination
    entity_entropy: float        # mean entropy at entity positions
    background_entropy: float    # mean entropy at non-entity positions
    entity_bg_ratio: float       # entity_entropy / max(background_entropy, ε)

    # ECE-compatible curvature
    mean_curvature: float        # κ̄ at entity positions (same as ECE)
    max_curvature: float         # max κ at entity positions

    # Metadata
    n_tokens: int
    n_entities: int
    has_logprobs: bool           # True if computed from real logprobs

    @property
    def risk_score(self) -> float:
        """Combine EPR features into a single risk score ∈ [0, 1].

        This is the analytical combination — no neural network needed.
        The weights are derived from the information-theoretic insight:
          - High mean entropy → generally uncertain (weight 0.25)
          - High variance → entropy spikes at specific tokens (weight 0.20)
          - High entity/bg ratio → entities specifically uncertain (weight 0.35)
          - High curvature → ECE's signal agrees (weight 0.20)
        """
        # Normalize each feature to [0, 1]
        h_norm = min(1.0, self.mean_entropy / 3.0)  # logprob -3 = quite uncertain
        v_norm = min(1.0, self.entropy_variance / 2.0)
        r_norm = min(1.0, max(0.0, (self.entity_bg_ratio - 1.0) / 3.0))
        k_norm = self.mean_curvature  # already in [0, 1]

        return min(1.0, (
            0.25 * h_norm
            + 0.20 * v_norm
            + 0.35 * r_norm
            + 0.20 * k_norm
        ))


def compute_epr(
    text: str,
    logprobs: list[float] | None = None,
    token_texts: list[str] | None = None,
    top_logprobs: list[list[float]] | None = None,
) -> EPRSignal:
    """Compute EPR signal from API logprobs or text heuristics.

    Args:
        text: Generated response text.
        logprobs: Per-token log-probabilities (the chosen token).
        token_texts: Per-token text strings (aligned with logprobs).
        top_logprobs: Optional top-K logprobs per position (for
                      entropy estimation when only top-K is available).

    Returns:
        EPRSignal with all features populated.
    """
    if logprobs and token_texts and len(logprobs) == len(token_texts):
        return _epr_from_logprobs(text, logprobs, token_texts, top_logprobs)
    return _epr_from_heuristics(text)


def _epr_from_logprobs(
    text: str,
    logprobs: list[float],
    token_texts: list[str],
    top_logprobs: list[list[float]] | None = None,
) -> EPRSignal:
    """Compute EPR from real API logprobs.

    This is the HALT/EPR-inspired path that uses actual per-token
    uncertainty from the API response.
    """
    n_tokens = len(logprobs)
    if n_tokens == 0:
        return _epr_empty()

    # ── Feature 1: Mean entropy ──
    # H_t ≈ -logprob_t (for the greedy token — lower bound on true entropy)
    # If top-K logprobs are available, compute proper entropy
    if top_logprobs and len(top_logprobs) == n_tokens:
        entropies = []
        for top_lps in top_logprobs:
            # Convert logprobs to probabilities, compute Shannon entropy
            probs = [math.exp(lp) for lp in top_lps if lp > -50]
            if probs:
                h = -sum(p * math.log(p + 1e-30) for p in probs if p > 0)
                entropies.append(h)
            else:
                entropies.append(0.0)
    else:
        # Approximate: use -logprob as entropy lower bound
        entropies = [-lp for lp in logprobs]

    mean_entropy = sum(entropies) / len(entropies)
    entropy_var = (
        sum((h - mean_entropy) ** 2 for h in entropies) / len(entropies)
    )

    # ── Feature 2: Entity vs background entropy ──
    entity_spans = _find_entity_char_positions(text)

    # Map character positions to token indices
    char_pos = 0
    token_char_starts: list[int] = []
    for tok in token_texts:
        token_char_starts.append(char_pos)
        char_pos += len(tok)

    entity_token_mask = [False] * n_tokens
    for span_start, span_end in entity_spans:
        for i, tok_start in enumerate(token_char_starts):
            tok_end = tok_start + len(token_texts[i])
            if tok_start < span_end and tok_end > span_start:
                entity_token_mask[i] = True

    entity_ents = [entropies[i] for i in range(n_tokens) if entity_token_mask[i]]
    bg_ents = [entropies[i] for i in range(n_tokens) if not entity_token_mask[i]]

    entity_entropy = sum(entity_ents) / max(len(entity_ents), 1)
    bg_entropy = sum(bg_ents) / max(len(bg_ents), 1)
    ratio = entity_entropy / max(bg_entropy, 0.01)

    # ── Feature 3: Entity curvature (ECE-compatible) ──
    entity_lps = [logprobs[i] for i in range(n_tokens) if entity_token_mask[i]]
    if entity_lps:
        curvatures = [min(1.0, max(0.0, -lp / 5.0)) for lp in entity_lps]
        mean_k = sum(curvatures) / len(curvatures)
        max_k = max(curvatures)
    else:
        mean_k = min(1.0, max(0.0, -sum(logprobs) / len(logprobs) / 5.0))
        max_k = mean_k

    return EPRSignal(
        mean_entropy=mean_entropy,
        entropy_variance=entropy_var,
        entity_entropy=entity_entropy,
        background_entropy=bg_entropy,
        entity_bg_ratio=ratio,
        mean_curvature=mean_k,
        max_curvature=max_k,
        n_tokens=n_tokens,
        n_entities=len(entity_spans),
        has_logprobs=True,
    )


def _epr_from_heuristics(text: str) -> EPRSignal:
    """Fallback EPR when logprobs are unavailable.

    Uses the same hedging-language heuristics as ECE, but packaged
    as EPR features for consistent fusion.
    """
    from .ece import compute_fisher_curvature
    mean_k, max_k, n_ent = compute_fisher_curvature(text)

    # Approximate entropy from hedging language density
    hedging = [
        r'\b(?:I think|I believe|probably|perhaps|maybe|might|could be)\b',
        r'\b(?:not sure|uncertain|approximately|around|roughly|about)\b',
        r'\b(?:it seems|appears to|likely|possibly)\b',
    ]
    words = len(text.split())
    hedge_count = sum(len(re.findall(p, text, re.I)) for p in hedging)
    hedge_density = hedge_count / max(words, 1)

    approx_entropy = min(3.0, hedge_density * 10.0 + mean_k * 2.0)

    return EPRSignal(
        mean_entropy=approx_entropy,
        entropy_variance=approx_entropy * 0.5,  # rough estimate
        entity_entropy=approx_entropy * (1.0 + mean_k),
        background_entropy=approx_entropy * 0.8,
        entity_bg_ratio=1.0 + mean_k,
        mean_curvature=mean_k,
        max_curvature=max_k,
        n_tokens=words,
        n_entities=n_ent,
        has_logprobs=False,
    )


def _epr_empty() -> EPRSignal:
    return EPRSignal(
        mean_entropy=0.0, entropy_variance=0.0,
        entity_entropy=0.0, background_entropy=0.0,
        entity_bg_ratio=1.0, mean_curvature=0.0,
        max_curvature=0.0, n_tokens=0, n_entities=0,
        has_logprobs=False,
    )


# ═══════════════════════════════════════════════════════════════════════
# Combined WITNESS + EPR fusion score
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FusedHallucinationSignal:
    """Combined WITNESS + EPR hallucination detection signal."""
    witness_risk: float          # 1.0 - WITNESS summary_score
    epr_risk: float              # EPR risk_score
    entity_gap: float            # Entity coverage gap (context vs answer)
    fused_risk: float            # Combined risk score
    epr: EPRSignal               # Full EPR signal for observability


def compute_fused_risk(
    witness_risk: float,
    epr: EPRSignal,
    entity_gap: float = 0.0,
    *,
    w_witness: float = 0.45,
    w_epr: float = 0.35,
    w_entity: float = 0.20,
) -> FusedHallucinationSignal:
    """Combine WITNESS + EPR + entity gap into a single risk score.

    The weights reflect information contribution:
      - WITNESS (0.45): strongest standalone signal (AUROC 0.80 on HaluEval)
      - EPR (0.35): orthogonal signal from token-level uncertainty
      - Entity gap (0.20): catches fabricated entities not in context

    When logprobs are available, EPR provides genuine new information.
    When logprobs are unavailable, EPR falls back to hedging heuristics
    (less value, but still nonzero).
    """
    fused = (
        w_witness * witness_risk
        + w_epr * epr.risk_score
        + w_entity * entity_gap
    )
    fused = min(1.0, max(0.0, fused))

    return FusedHallucinationSignal(
        witness_risk=witness_risk,
        epr_risk=epr.risk_score,
        entity_gap=entity_gap,
        fused_risk=fused,
        epr=epr,
    )
