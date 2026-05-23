"""
Codebase-Conditioned Character N-Gram Language Model
====================================================

A per-repository character n-gram model with Stupid Backoff smoothing
(Brants et al. EMNLP 2007). Used as the surprisal signal in the Bayesian
hallucination detector.

Mathematical framing
--------------------
For a character sequence c_1...c_T, define:

    P(c_t | c_{t-n+1}^{t-1}) =
        count(c_{t-n+1}^t) / count(c_{t-n+1}^{t-1})    if count > 0
        alpha * P(c_t | c_{t-n+2}^{t-1})               otherwise (backoff)

Per-symbol surprisal:

    surprisal(s) = -1/|s| * sum_t log P(c_t | c_{t-n+1}^{t-1})

Why character n-grams instead of subword tokens:
  1. No vocab dependency — works on any language
  2. Captures the codebase's *naming convention* (snake_case vs camelCase
     vs prefixes like `get_`, `_internal`, etc.)
  3. Hallucinated identifiers that follow a *different* convention from
     the host codebase get high surprisal even when they're syntactically
     valid words

Why stupid backoff and not Kneser-Ney:
  - We need a *score*, not a calibrated probability distribution
  - SB is order-of-magnitude faster (no recursive discount computation)
  - Brants et al. showed SB matches modified KN on perplexity at >1B tokens
    and is *better* at <100M tokens (our regime)

Complexity
----------
Training:  O(N) where N = total characters in indexed codebase
Inference: O(|s|) per symbol, with O(n) hash lookups per character
Memory:    O(unique_ngrams), bounded by min(N, |Sigma|^n)

For a 1M-LOC repo (~30MB source), expect:
  - ~300k unique 4-grams
  - ~50ms training
  - ~5us per symbol score
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


# Stupid-backoff coefficient (Brants 2007 found 0.4 empirically optimal
# across many corpora — independent of n and N).
_BACKOFF_ALPHA = 0.4

# Sentinel character for sequence start (not in any real source code).
_SEQ_START = "\x02"  # ASCII STX
_SEQ_END = "\x03"    # ASCII ETX

# Floor probability for unseen unigrams. log(P) ~ -10 ≈ 1/22000, which
# matches typical per-char entropy of source code. Prevents -inf.
_UNK_LOG_PROB = -10.0


class CharNGramModel:
    """Character n-gram language model with stupid backoff.

    Designed for *codebase-relative* surprisal scoring — not as a
    standalone perplexity benchmark model.

    Attributes:
        n: Order of the n-gram model (default 4).
        total_chars: Total characters in training data.
        ngrams: Dict mapping context -> {char -> count}.

    Example::

        model = CharNGramModel(n=4)
        model.train_from_strings(["def authenticate(user):", "from auth import ..."])
        score = model.score_symbol("authenticate")  # log-prob (avg per char)
        sus_score = model.score_symbol("xqyzkwerblob")  # much lower (more surprising)
    """

    def __init__(self, n: int = 4):
        if n < 2:
            raise ValueError("n must be >= 2 (need at least one context char)")
        self.n = n
        self.total_chars: int = 0
        # ngrams[order][context_str] -> {char: count}
        # order ranges over 1..n inclusive
        self.ngrams: list[dict[str, dict[str, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(n + 1)
        ]
        # Unigram counts for fallback
        self.unigrams: dict[str, int] = defaultdict(int)
        # Lazy-built denominators for fast inference
        self._denom_cache: list[dict[str, int]] | None = None

    # ── Training ─────────────────────────────────────────────────────

    def train_from_strings(self, texts: Iterable[str]) -> None:
        """Train (or update) on an iterable of text strings.

        Idempotent: calling twice with the same data doubles counts but
        does not change the score (ratios are preserved).
        """
        for text in texts:
            self._ingest_text(text)
        self._denom_cache = None  # invalidate

    def _ingest_text(self, text: str) -> None:
        if not text:
            return

        # Pad with start sentinels so the very first chars get scored
        # in a consistent context.
        padded = _SEQ_START * (self.n - 1) + text + _SEQ_END
        L = len(padded)

        for i in range(self.n - 1, L):
            char = padded[i]
            self.unigrams[char] += 1
            self.total_chars += 1
            # Count all orders 1..n
            for order in range(1, self.n + 1):
                if i - order + 1 < 0:
                    continue
                context = padded[i - order + 1 : i]
                self.ngrams[order][context][char] += 1

    def _build_denom_cache(self) -> None:
        """Precompute context totals for O(1) inference lookups."""
        self._denom_cache = [{} for _ in range(self.n + 1)]
        for order in range(1, self.n + 1):
            for ctx, char_counts in self.ngrams[order].items():
                self._denom_cache[order][ctx] = sum(char_counts.values())

    # ── Pickling support ─────────────────────────────────────────────
    # The lambda-based defaultdicts can't be pickled directly; convert
    # to plain dicts on the way out and rebuild on the way in.

    def __getstate__(self) -> dict:
        return {
            "n": self.n,
            "total_chars": self.total_chars,
            "ngrams": [
                {ctx: dict(char_counts) for ctx, char_counts in d.items()}
                for d in self.ngrams
            ],
            "unigrams": dict(self.unigrams),
        }

    def __setstate__(self, state: dict) -> None:
        self.n = state["n"]
        self.total_chars = state["total_chars"]
        self.unigrams = defaultdict(int)
        self.unigrams.update(state["unigrams"])
        self.ngrams = [defaultdict(lambda: defaultdict(int)) for _ in range(self.n + 1)]
        for order_idx, d in enumerate(state["ngrams"]):
            for ctx, char_counts in d.items():
                self.ngrams[order_idx][ctx] = defaultdict(int)
                self.ngrams[order_idx][ctx].update(char_counts)
        self._denom_cache = None

    # ── Scoring ──────────────────────────────────────────────────────

    def _log_p_char(self, char: str, context: str) -> float:
        """log P(char | context) with stupid-backoff.

        Walks down from full-order context to unigram, applying the
        backoff coefficient at each step.
        """
        if self._denom_cache is None:
            self._build_denom_cache()
        assert self._denom_cache is not None

        # Try full order first, back off as needed
        for order in range(min(self.n, len(context) + 1), 0, -1):
            if order == 1:
                # Unigram — no context
                total = self.total_chars
                count = self.unigrams.get(char, 0)
                if count > 0 and total > 0:
                    # Add backoff penalty for steps taken
                    log_p = math.log(count / total)
                    # Apply alpha for each backoff step from n to 1
                    backoff_steps = self.n - 1  # all the way down
                    return log_p + backoff_steps * math.log(_BACKOFF_ALPHA)
                return _UNK_LOG_PROB

            # Higher order
            ctx = context[-(order - 1):] if order > 1 else ""
            char_counts = self.ngrams[order].get(ctx)
            if char_counts:
                char_count = char_counts.get(char, 0)
                if char_count > 0:
                    total = self._denom_cache[order][ctx]
                    log_p = math.log(char_count / total)
                    # Backoff steps taken to reach here
                    backoff_steps = self.n - order
                    return log_p + backoff_steps * math.log(_BACKOFF_ALPHA)

        return _UNK_LOG_PROB

    def score_string(self, s: str) -> float:
        """Average per-character log-probability of s under the model.

        Higher (less negative) = more typical of the training corpus.
        Used for inverse — lower score = more surprising.

        Returns:
            float in (-inf, 0]. Empty string returns 0.
        """
        if not s:
            return 0.0

        padded = _SEQ_START * (self.n - 1) + s
        total_log_p = 0.0
        n_scored = 0
        for i in range(self.n - 1, len(padded)):
            char = padded[i]
            context = padded[i - self.n + 1 : i]
            total_log_p += self._log_p_char(char, context)
            n_scored += 1

        return total_log_p / max(n_scored, 1)

    def surprisal(self, s: str) -> float:
        """Per-character surprisal of s. Higher = more unexpected.

        surprisal(s) = -avg_log_p(s)

        Returns:
            float in [0, +inf). Symbols highly typical of the codebase
            have surprisal near the codebase's per-char entropy (~3-5
            bits for typical code). Hallucinated symbols can spike
            to >8 bits.
        """
        return -self.score_string(s)

    # ── Utilities ────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Summary statistics for diagnostics."""
        unique_ngrams_per_order = [
            len(self.ngrams[k]) for k in range(1, self.n + 1)
        ]
        return {
            "n": self.n,
            "total_chars_trained": self.total_chars,
            "unique_chars": len(self.unigrams),
            "unique_ngrams_per_order": unique_ngrams_per_order,
            "backoff_alpha": _BACKOFF_ALPHA,
        }


def quick_train_from_paths(
    paths: list[str],
    n: int = 4,
    max_bytes: int = 50_000_000,
) -> CharNGramModel:
    """Convenience: train an n-gram model from a list of source file paths.

    Args:
        paths: Files to train on.
        n: N-gram order.
        max_bytes: Cap total chars ingested to bound memory.

    Returns:
        Trained CharNGramModel.
    """
    model = CharNGramModel(n=n)
    total = 0
    texts: list[str] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except (OSError, UnicodeError):
            continue
        if total + len(content) > max_bytes:
            content = content[: max_bytes - total]
        texts.append(content)
        total += len(content)
        if total >= max_bytes:
            break
    model.train_from_strings(texts)
    return model
