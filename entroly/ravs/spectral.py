"""Spectral Hallucination Detector — EigenScore for Black-Box APIs.

Research grounding:
  - INSIDE/EigenScore (Chen et al., ICLR 2024): eigenvalues of embedding
    covariance as semantic consistency measure
  - LapEigvals (EMNLP 2025): Laplacian eigenvalues of attention graphs
  - GLSim (NeurIPS 2025): embedding similarity as hallucination signal

The key insight:
  We CAN'T access internal model weights/attention maps through a black-box
  API. But we CAN compute a LOCAL version of the spectral signal using
  token-level embeddings that WE compute (TF-IDF, character n-grams, or
  sentence embeddings).

What we implement:
  A "spectral consistency" score that measures whether the RESPONSE's
  entity embeddings are consistent with the CONTEXT's entity embeddings
  using the eigenvalue spectrum of their cross-similarity matrix.

Mathematical formulation:
  Let C = {c_1, ..., c_m} be entity embeddings from the context
  Let R = {r_1, ..., r_n} be entity embeddings from the response
  
  Build the cross-similarity matrix K ∈ R^{n×m} where K_ij = sim(r_i, c_j)
  
  Compute SVD: K = U Σ V^T
  
  The spectral consistency score is:
    σ_score = (σ_1 / Σ σ_i) · (σ_1 / max_possible)
  
  Interpretation:
    - If σ_1 dominates → response entities align with a single context
      direction → FAITHFUL
    - If spectrum is flat → response entities scatter across unrelated
      dimensions → HALLUCINATED
    - If σ_1 ≈ 0 → no similarity at all → HALLUCINATED

  This is a NOVEL application of the EigenScore idea to black-box
  verification (original requires model internals).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


# ── Entity extraction ──────────────────────────────────────────────────

_ENTITY_PATTERNS = [
    re.compile(r'\b\d{4}\b'),                           # years
    re.compile(r'\b\d+\.?\d*\s*(?:m|km|kg|cm|ft|°)\b'), # measurements
    re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b'),        # numbers
    re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){0,3}\b'),  # proper nouns
]


def _extract_entities(text: str) -> list[str]:
    """Extract entity strings from text."""
    entities = []
    seen = set()
    for pat in _ENTITY_PATTERNS:
        for m in pat.finditer(text):
            ent = m.group().strip()
            if ent.lower() not in seen and len(ent) > 1:
                seen.add(ent.lower())
                entities.append(ent)
    return entities


# ── Character n-gram embedding ─────────────────────────────────────────

def _char_ngram_vector(text: str, n: int = 3) -> dict[str, float]:
    """Build a character n-gram frequency vector (sparse).
    
    This is our "embedding" — it works for any text without requiring
    a neural model, tokenizer, or external service.
    """
    text = text.lower().strip()
    if len(text) < n:
        return {text: 1.0}
    counts: Counter = Counter()
    for i in range(len(text) - n + 1):
        counts[text[i:i + n]] += 1
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Spectral consistency score ─────────────────────────────────────────

@dataclass(frozen=True)
class SpectralSignal:
    """Spectral consistency analysis result."""
    score: float              # spectral consistency ∈ [0, 1], higher = more faithful
    sigma_ratio: float        # σ₁ / Σσ_i — spectral concentration
    max_sim: float            # max cross-similarity
    mean_sim: float           # mean cross-similarity
    n_ctx_entities: int
    n_resp_entities: int
    spectral_gap: float       # σ₁ - σ₂ — gap between top two singular values


def _svd_2x2(a: float, b: float, c: float, d: float) -> list[float]:
    """Singular values of a 2x2 matrix [[a,b],[c,d]]."""
    s1 = a * a + b * b + c * c + d * d
    s2 = math.sqrt(max(0, (a * a + b * b - c * c - d * d) ** 2 + 4 * (a * c + b * d) ** 2))
    return [math.sqrt(max(0, (s1 + s2) / 2)), math.sqrt(max(0, (s1 - s2) / 2))]


def _compute_singular_values(K: list[list[float]]) -> list[float]:
    """Compute singular values of matrix K via K^T K eigenvalues.
    
    For small matrices (typical: 2-20 entities), this is fast and exact
    enough. We compute eigenvalues of K^T K using power iteration.
    """
    n = len(K)
    if n == 0 or not K[0]:
        return []
    m = len(K[0])
    
    # K^T K is m x m
    KtK = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            s = 0.0
            for k in range(n):
                s += K[k][i] * K[k][j]
            KtK[i][j] = s
    
    # Power iteration for top eigenvalues
    import random
    rng = random.Random(42)
    
    singular_values = []
    residual = [row[:] for row in KtK]
    
    for _ in range(min(n, m, 5)):  # at most 5 singular values
        # Power iteration
        v = [rng.gauss(0, 1) for _ in range(m)]
        norm = math.sqrt(sum(x * x for x in v))
        if norm < 1e-10:
            break
        v = [x / norm for x in v]
        
        for _iter in range(50):  # usually converges in <20
            # w = residual @ v
            w = [sum(residual[i][j] * v[j] for j in range(m)) for i in range(m)]
            eigenval = sum(w[i] * v[i] for i in range(m))
            norm = math.sqrt(sum(x * x for x in w))
            if norm < 1e-12:
                break
            v_new = [x / norm for x in w]
            # Check convergence
            diff = sum((v_new[i] - v[i]) ** 2 for i in range(m))
            v = v_new
            if diff < 1e-10:
                break
        
        sv = math.sqrt(max(0, eigenval))
        if sv < 1e-8:
            break
        singular_values.append(sv)
        
        # Deflate: remove this component
        for i in range(m):
            for j in range(m):
                residual[i][j] -= eigenval * v[i] * v[j]
    
    return sorted(singular_values, reverse=True)


def compute_spectral_consistency(
    context: str,
    response: str,
) -> SpectralSignal:
    """Compute spectral consistency between context and response entities.
    
    Uses character n-gram embeddings to build a cross-similarity matrix
    and analyzes its singular value spectrum.
    """
    ctx_entities = _extract_entities(context)
    resp_entities = _extract_entities(response)
    
    if not resp_entities:
        # No entities in response → can't hallucinate entities → neutral
        return SpectralSignal(
            score=0.5, sigma_ratio=0.0, max_sim=0.0, mean_sim=0.0,
            n_ctx_entities=len(ctx_entities), n_resp_entities=0,
            spectral_gap=0.0,
        )
    
    if not ctx_entities:
        # No context entities but response has entities → suspicious
        return SpectralSignal(
            score=0.2, sigma_ratio=0.0, max_sim=0.0, mean_sim=0.0,
            n_ctx_entities=0, n_resp_entities=len(resp_entities),
            spectral_gap=0.0,
        )
    
    # Build embeddings
    ctx_vecs = [_char_ngram_vector(e) for e in ctx_entities]
    resp_vecs = [_char_ngram_vector(e) for e in resp_entities]
    
    # Cross-similarity matrix K[i][j] = sim(resp_i, ctx_j)
    n, m = len(resp_vecs), len(ctx_vecs)
    K = [[_cosine_sim(resp_vecs[i], ctx_vecs[j]) for j in range(m)] for i in range(n)]
    
    # Flatten for stats
    all_sims = [K[i][j] for i in range(n) for j in range(m)]
    max_sim = max(all_sims) if all_sims else 0.0
    mean_sim = sum(all_sims) / len(all_sims) if all_sims else 0.0
    
    # SVD
    sigmas = _compute_singular_values(K)
    
    if not sigmas or sigmas[0] < 1e-8:
        return SpectralSignal(
            score=0.1, sigma_ratio=0.0, max_sim=max_sim, mean_sim=mean_sim,
            n_ctx_entities=len(ctx_entities), n_resp_entities=len(resp_entities),
            spectral_gap=0.0,
        )
    
    sigma_sum = sum(sigmas)
    sigma_ratio = sigmas[0] / sigma_sum if sigma_sum > 0 else 0.0
    spectral_gap = (sigmas[0] - sigmas[1]) / sigmas[0] if len(sigmas) > 1 else 1.0
    
    # Score: combine spectral concentration with absolute similarity
    # Higher σ_ratio = more concentrated = more consistent
    # Higher max_sim = at least one entity matches well
    score = 0.5 * sigma_ratio + 0.3 * max_sim + 0.2 * mean_sim
    score = min(1.0, max(0.0, score))
    
    return SpectralSignal(
        score=score,
        sigma_ratio=sigma_ratio,
        max_sim=max_sim,
        mean_sim=mean_sim,
        n_ctx_entities=len(ctx_entities),
        n_resp_entities=len(resp_entities),
        spectral_gap=spectral_gap,
    )
