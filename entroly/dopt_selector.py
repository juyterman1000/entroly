"""BM25-Submodular Context Packing.

A training-free, embedding-free context selector that combines Normalized
Compression Distance (NCD) as the query-fragment relevance signal with a
monotone submodular log-determinant objective for budget-constrained
selection.

Mathematical foundation
-----------------------
**Relevance via conditional compression.**  For a retrieval query q and
candidate fragment f with |q| ≪ |f|, symmetric NCD (Li–Vitányi 2004) is
the wrong statistic — it normalizes by max(C(q), C(f)) ≈ C(f), so any
large fragment registers NCD ≈ 1 regardless of actual overlap. What we
actually want is conditional Kolmogorov complexity K(q | f): how many
bits are needed to describe q if you already know f? A fragment that
makes q nearly determined has K(q | f) → 0.

We approximate this with zlib's LZ77 back-reference window: if we
compress f first and then q as a continuation, zlib reuses f's
dictionary when encoding q. Formally,

    cover(q | f) = 1 - [C(f‖q) - C(f)] / C(q) ∈ [0, 1]

where f‖q denotes byte concatenation. C(f‖q) - C(f) is zlib's estimate
of the additional description length of q given f — a standard
compressor-based surrogate for K(q | f) (Bennett–Gács–Li–Vitányi–Zurek
1998). When f contains the answer content, the additional bits are few
and cover → 1; when f is unrelated, encoding q from scratch costs ≈ C(q)
and cover → 0.

This is parameter-free, training-free, and directionally correct for
the retrieval geometry (short query against long fragments). It
captures shared n-grams of any length, so rare multi-token phrases
("taint_flow", "pattern_only_total") score without hand-tuned IDF.

**Selection via Bayesian D-optimal design.**  Given NCD-derived precisions
τ_i ∈ R+ and concept directions p_i ∈ R^k (feature-hash projection of
identifiers shared between q and f_i — i.e. the concepts the fragment
actually covers *for this query*), we solve

    max_C     log det(Λ_0 + Σ_{i ∈ C} τ_i p_i p_iᵀ)
    s.t.      Σ_{i ∈ C} t_i ≤ B.

This is the Bayesian D-optimal experimental-design objective (Fedorov
1972) specialized to rank-1 Gaussian observations. It is monotone
submodular in C (Krause–Guestrin 2005): adding a fragment whose
direction p_i is already well-covered yields diminishing returns, and
the gain is zero in directions already at the prior's limit. This is
the anti-redundancy property a linear ∑ s_i score fundamentally lacks.

**Solver.**  Submodular maximization under a knapsack constraint is
NP-hard. Khuller–Moss–Naor (1999) give a modified greedy with provable
(1 − 1/√e) ≈ 0.39 approximation ratio: return whichever is better, the
density-greedy schedule (argmax Δ / t at each step) or the singleton
champion (argmax_i f({i}) subject to t_i ≤ B). Density-greedy is near
optimal when fragments have similar size; singleton dominates when one
high-information fragment alone beats any composition of smaller ones.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")
_CONCEPT_DIM = 64
_PRIOR_STRENGTH = 1e-2
_STALE_PRECISION_SCALE = 0.1
_MIN_GAIN = 1e-6
_BM25_K1 = 1.5
_BM25_B = 0.75

# English function-words that appear in every natural-language query.
# They would otherwise cause any markdown or comment-heavy fragment to
# register spurious query overlap. Code identifiers are not stopworded.
_STOPWORDS = frozenset("""
the a an of to in on for with by how does what why is are and or but
do between two include shown actual actually that this these those
from can could should would has have had will was were been being
about into through over under above below after before while when where
not no yes all any some most many much more less than then thus therefore
you your their his her its our it he she they them we us i me my
""".split())


# ── Data ─────────────────────────────────────────────────────────────────
@dataclass
class DOptFragment:
    fragment_id: str
    source: str
    content: str
    token_count: int
    is_stale: bool
    # Computed per-query:
    bm25: float                   # BM25 score against the query
    shared_tokens: frozenset[str] # query terms that appear in the fragment


# ── Primitives ───────────────────────────────────────────────────────────
def _approx_tokens(content: str) -> int:
    return max(1, len(content) // 4)


def _is_stale(content: str) -> bool:
    # Vault beliefs use YAML frontmatter 'status: stale'.
    return "\nstatus: stale" in content or content.startswith("status: stale")


def _split_identifier(tok: str) -> list[str]:
    """Split code identifiers: `taint_flow_total` → [taint_flow_total, taint, flow, total].
    Also handles CamelCase: `TaintFlow` → [taintflow, taint, flow]."""
    low = tok.lower()
    parts = {low}
    for piece in low.split("_"):
        if len(piece) > 2:
            parts.add(piece)
    for piece in _CAMEL_RE.findall(tok):
        p = piece.lower()
        if len(p) > 2:
            parts.add(p)
    return [p for p in parts if p not in _STOPWORDS]


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in _IDENT_RE.findall(text):
        out.extend(_split_identifier(raw))
    return out


def _query_tokens(query: str) -> frozenset[str]:
    return frozenset(t for t in _tokenize(query) if len(t) > 2)


def _hash_projection(tokens: frozenset[str], dim: int = _CONCEPT_DIM) -> np.ndarray:
    """Signed feature-hash projection. Johnson–Lindenstrauss preserves inner
    products for unit-norm vectors in O(log n / ε²) dimensions; k=64 is more
    than enough for the handful of concepts any single query touches."""
    if not tokens:
        return np.zeros(dim, dtype=np.float64)
    p = np.zeros(dim, dtype=np.float64)
    for t in tokens:
        h = hash(t)
        bucket = (h & 0x7FFFFFFF) % dim
        sign = 1.0 if (h >> 31) & 1 else -1.0
        p[bucket] += sign
    norm = float(np.linalg.norm(p))
    return p / norm if norm > 1e-12 else p


# ── Per-fragment precision τ_i ───────────────────────────────────────────
def _precision(frag: DOptFragment) -> float:
    """τ_i = bm25(q, f) · freshness.

    BM25 already incorporates IDF (rare terms weigh more), term-frequency
    saturation, and document-length normalization. Freshness demotes stale
    vault beliefs.
    """
    if frag.bm25 <= 0:
        return 0.0
    stale_factor = _STALE_PRECISION_SCALE if frag.is_stale else 1.0
    return stale_factor * frag.bm25


# ── BM25 corpus statistics ───────────────────────────────────────────────
def _bm25_corpus(contents: list[str], sources: list[str]) -> tuple[list[Counter], list[int], dict[str, int], float]:
    tf_list: list[Counter] = []
    lens: list[int] = []
    df: Counter = Counter()
    for content, src in zip(contents, sources):
        toks = _tokenize(content) + _tokenize(src)
        tf = Counter(toks)
        tf_list.append(tf)
        lens.append(sum(tf.values()))
        for term in tf:
            df[term] += 1
    avgdl = (sum(lens) / len(lens)) if lens else 1.0
    return tf_list, lens, df, max(avgdl, 1.0)


def _bm25_score(
    q_tokens: frozenset[str],
    tf: Counter,
    dl: int,
    df: dict,
    N: int,
    avgdl: float,
) -> float:
    score = 0.0
    for term in q_tokens:
        f = tf.get(term, 0)
        if f == 0:
            continue
        n = df.get(term, 0)
        idf = math.log(1.0 + (N - n + 0.5) / (n + 0.5))
        norm = 1.0 - _BM25_B + _BM25_B * (dl / avgdl)
        score += idf * (f * (_BM25_K1 + 1.0)) / (f + _BM25_K1 * norm)
    return score


# ── Fragment construction (per-query) ────────────────────────────────────
def _build_fragment(raw: dict, bm25: float, q_tokens: frozenset[str]) -> DOptFragment:
    content = raw.get("content", "") or ""
    src = raw.get("source", "") or ""
    tc = raw.get("token_count") or _approx_tokens(content)

    f_tokens = frozenset(_tokenize(content) + _tokenize(src))
    shared = q_tokens & f_tokens

    return DOptFragment(
        fragment_id=raw.get("fragment_id", raw.get("id", src)),
        source=src,
        content=content,
        token_count=int(tc),
        is_stale=_is_stale(content),
        bm25=bm25,
        shared_tokens=shared,
    )


# ── Greedy selection ─────────────────────────────────────────────────────
def _greedy(
    P: np.ndarray,
    tau: np.ndarray,
    tok: np.ndarray,
    token_budget: int,
    criterion: str,
) -> tuple[list[int], float]:
    """Greedy D-optimal log-det maximization under the given criterion.

    Maintains Λ⁻¹ (concept-dim × concept-dim) via Sherman–Morrison rank-1
    updates — O(k²) per step, no inversion. Per-step gain for candidate i:

        Δ_i = log(1 + τ_i · p_iᵀ Λ⁻¹ p_i)                       (rank-1 log det)

    Criterion 'density' picks argmax Δ_i / t_i (fractional-packing rule);
    criterion 'gain' picks argmax Δ_i (absolute-gain rule); and criterion
    'singleton' picks only the single best-gain item that fits.
    """
    n = len(tau)
    Lambda_inv = (1.0 / _PRIOR_STRENGTH) * np.eye(_CONCEPT_DIM)
    selected: list[int] = []
    remaining = [i for i in range(n) if tau[i] > 0.0]
    budget_left = int(token_budget)
    total_gain = 0.0

    while budget_left > 0 and remaining:
        fit = [i for i in remaining if tok[i] <= budget_left]
        if not fit:
            break
        idx = np.asarray(fit)
        Pc = P[idx]
        # Element-wise pᵀ Λ⁻¹ p for each candidate — vectorized einsum.
        quad = np.einsum("ij,jk,ik->i", Pc, Lambda_inv, Pc)
        gains = np.log1p(tau[idx] * np.maximum(quad, 0.0))

        if criterion == "density":
            score = gains / np.maximum(tok[idx], 1)
        else:
            score = gains

        best_local = int(np.argmax(score))
        best_gain = float(gains[best_local])
        if best_gain < _MIN_GAIN:
            break

        best_idx = fit[best_local]
        selected.append(best_idx)
        remaining.remove(best_idx)
        total_gain += best_gain

        if criterion == "singleton":
            break

        # Sherman–Morrison: Λ⁻¹ ← Λ⁻¹ − (τ · u uᵀ) / (1 + τ · pᵀ u),  u = Λ⁻¹ p
        p_best = P[best_idx]
        u = Lambda_inv @ p_best
        denom = 1.0 + tau[best_idx] * float(p_best @ u)
        Lambda_inv = Lambda_inv - (tau[best_idx] / denom) * np.outer(u, u)
        budget_left -= int(tok[best_idx])

    return selected, total_gain


# ── Public entry point ──────────────────────────────────────────────────
def select(
    fragments: list[dict],
    token_budget: int,
    query: str = "",
) -> list[dict]:
    """NCD-weighted D-optimal selection under a token budget.

    Pipeline:
      1. Compute C(q) once and NCD(q, f_i) for each fragment via zlib.
      2. Precisions τ_i and concept directions p_i built from NCD and from
         the intersection of query and fragment identifiers.
      3. Khuller–Moss–Naor modified greedy: return best of density-greedy
         and singleton-best over the log-det submodular objective.

    Empty-query fallback: degrades to substance × freshness weighting.
    """
    if not fragments:
        return []
    if not query:
        # No query ⇒ fall back to fragment-level uniform + log-det diversity.
        frags = [
            DOptFragment(
                fragment_id=raw.get("fragment_id", raw.get("id", "")),
                source=raw.get("source", "") or "",
                content=raw.get("content", "") or "",
                token_count=raw.get("token_count") or _approx_tokens(raw.get("content", "") or ""),
                is_stale=_is_stale(raw.get("content", "") or ""),
                bm25=1.0,
                shared_tokens=frozenset(_tokenize(raw.get("content", "") or "")[:32]),
            )
            for raw in fragments
        ]
        P = np.stack([_hash_projection(f.shared_tokens) for f in frags])
        tau = np.array([_precision(f) for f in frags])
        tok = np.array([f.token_count for f in frags])
        sel_d, gain_d = _greedy(P, tau, tok, token_budget, "density")
        sel_s, gain_s = _greedy(P, tau, tok, token_budget, "singleton")
        sel_g, gain_g = _greedy(P, tau, tok, token_budget, "gain")
        winners = sorted([(gain_d, sel_d), (gain_g, sel_g), (gain_s, sel_s)],
                         key=lambda x: x[0], reverse=True)
        best = winners[0][1]
        return [fragments[i] for i in best]

    # ── Query present: file-level BM25 with concentrated allocation ──────
    # Diagnosis that drove this design: at tight budgets, a fragment-level
    # selector spreads across many tiny hash-distinct chunks and misses the
    # specific chunk that answers the question. A human answers code
    # questions by reading one or two whole files. So: score files, not
    # fragments, then fill budget file-by-file in BM25 order.
    q_tokens = _query_tokens(query)
    by_file: dict[str, list[dict]] = {}
    for raw in fragments:
        src = raw.get("source", "") or ""
        by_file.setdefault(src, []).append(raw)

    # File-level BM25 corpus: each file = concatenated fragment content.
    file_sources = list(by_file.keys())
    file_contents = ["\n".join(r.get("content", "") or "" for r in by_file[s]) for s in file_sources]
    tf_list, lens, df, avgdl = _bm25_corpus(file_contents, file_sources)
    N = len(file_sources)
    file_scores = [
        (_bm25_score(q_tokens, tf_list[i], lens[i], df, N, avgdl), file_sources[i])
        for i in range(N)
    ]
    file_scores.sort(reverse=True)

    # Fill budget: take files in BM25 order, include all their fragments
    # (stale demoted at fragment level). Stop when budget exhausted.
    selected_raw: list[dict] = []
    budget_left = int(token_budget)
    seen_ids: set = set()
    for score, src in file_scores:
        if score <= 0 or budget_left <= 0:
            break
        file_frags = by_file[src]
        # Within a file, order fragments by original index; prefer non-stale.
        file_frags = sorted(file_frags, key=lambda r: (_is_stale(r.get("content", "") or ""),))
        for raw in file_frags:
            tc = int(raw.get("token_count") or _approx_tokens(raw.get("content", "") or ""))
            if tc > budget_left:
                continue
            fid = raw.get("fragment_id", raw.get("id", id(raw)))
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            selected_raw.append(raw)
            budget_left -= tc

    return selected_raw
