#!/usr/bin/env python3
"""
Falsification-first benchmark for belief-conditioned compression.
=================================================================

Claim under test:
    Conditioning information density on the agent's belief state
    (H(X | beliefs)) improves the budget/answer-recall frontier by evicting
    redundant restatements of already-known beliefs in favour of novel,
    answer-bearing content -- WITHOUT wrongly discarding fragments that merely
    *resemble* a belief but carry a critical new detail.

This script does NOT try to confirm the claim. It tries to break it:

  1. FRONTIER TEST   -- at tight budgets, does belief-conditioning retain more
                        ANSWER fragments than the baseline information_score?
  2. FALSE-KNOWN TEST-- the corpus includes TRICKY fragments that share most of
                        their text with a high-confidence belief but add a new,
                        load-bearing fact. If conditioning drops these, the
                        method is dangerous and the claim is falsified.

Both scorers feed the SAME greedy value/token selector (a fair knapsack proxy),
so the only variable is the scoring function.

Run:  python bench/belief_conditioned_bench.py
"""

from __future__ import annotations

import entroly_core as ec

# ── Beliefs the agent already holds (text, confidence) ────────────────────────
BELIEFS: list[tuple[str, float]] = [
    ("the rate limiter uses a sliding window of request timestamps per key", 0.95),
    ("the payment processor charges an amount in a currency via the stripe gateway", 0.92),
    ("authentication validates a bearer token with hmac compare_digest against a secret", 0.90),
    ("the user repository fetches a user by email from the session query", 0.88),
]

# ── Candidate fragments. label ∈ {ANSWER, KNOWN_NOISE, TRICKY, UNRELATED} ──────
# ANSWER     : novel, answer-bearing for the query -> MUST be retained
# KNOWN_NOISE: restates a held belief almost verbatim -> SHOULD be evicted first
# TRICKY     : resembles a belief but adds a critical NEW fact -> MUST be retained
# UNRELATED  : filler, low value either way
CORPUS: list[tuple[str, str]] = [
    ("ANSWER", "the websocket reconnect handler applies exponential backoff with a jitter cap of thirty seconds and aborts after five retries"),
    ("ANSWER", "the cache invalidation hook fires on write and purges the dependent keys computed from the dependency graph edges"),
    ("ANSWER", "the migration runner acquires an advisory lock so concurrent deploys cannot apply the same schema change twice"),
    ("KNOWN_NOISE", "the rate limiter uses a sliding window of request timestamps per key"),
    ("KNOWN_NOISE", "the payment processor charges an amount in a currency via the stripe gateway"),
    ("KNOWN_NOISE", "authentication validates a bearer token with hmac compare_digest against a secret"),
    ("TRICKY", "authentication validates a bearer token with hmac compare_digest against a secret that now rotates every twenty four hours via the kms key schedule"),
    ("TRICKY", "the rate limiter uses a sliding window of request timestamps per key and additionally sheds load by returning 429 once redis memory exceeds eighty percent"),
    ("UNRELATED", "the readme documents the local development setup and how to run the formatter"),
    ("UNRELATED", "the changelog lists the release notes for the previous minor version"),
]


def tok(text: str) -> int:
    return max(1, len(text) // 4)


def greedy_select(scored: list[tuple[str, str, float, int]], budget: int) -> list[str]:
    """Greedy value/token knapsack proxy. Returns selected fragment texts."""
    order = sorted(scored, key=lambda r: r[2] / r[3], reverse=True)
    out, used = [], 0
    for _label, text, _val, t in order:
        if used + t <= budget:
            out.append(text)
            used += t
    return out


def score_corpus(use_beliefs: bool) -> list[tuple[str, str, float, int]]:
    others = [c for _, c in CORPUS]
    rows = []
    for label, text in CORPUS:
        ref = [o for o in others if o != text]
        if use_beliefs:
            val = ec.py_conditional_information_score(text, ref, BELIEFS)
        else:
            val = ec.py_information_score(text, ref)
        rows.append((label, text, val, tok(text)))
    return rows


def count(selected: list[str], label: str) -> int:
    by_label = {t: lab for lab, t in CORPUS}
    return sum(1 for s in selected if by_label.get(s) == label)


def main() -> None:
    base_rows = score_corpus(use_beliefs=False)
    cond_rows = score_corpus(use_beliefs=True)

    n_answer = sum(1 for lab, _ in CORPUS if lab == "ANSWER")
    n_tricky = sum(1 for lab, _ in CORPUS if lab == "TRICKY")
    total_tokens = sum(tok(c) for _, c in CORPUS)

    print("=" * 78)
    print("BELIEF-CONDITIONED COMPRESSION — FALSIFICATION BENCHMARK")
    print("=" * 78)
    print(f"corpus: {len(CORPUS)} fragments, {total_tokens} tokens total | "
          f"{n_answer} ANSWER, {n_tricky} TRICKY, "
          f"{sum(1 for lab,_ in CORPUS if lab=='KNOWN_NOISE')} KNOWN_NOISE\n")

    print(f"{'budget':>7} | {'baseline answer/noise/tricky':^32} | {'conditioned answer/noise/tricky':^33}")
    print("-" * 78)

    falsified = False
    frontier_wins = 0
    budgets = [b for b in range(40, total_tokens, 30)]
    for budget in budgets:
        bsel = greedy_select(base_rows, budget)
        csel = greedy_select(cond_rows, budget)
        ba, bn, bt = count(bsel, "ANSWER"), count(bsel, "KNOWN_NOISE"), count(bsel, "TRICKY")
        ca, cn, ct = count(csel, "ANSWER"), count(csel, "KNOWN_NOISE"), count(csel, "TRICKY")
        flag = ""
        # FALSIFICATION 1: conditioning drops a TRICKY (false-known) that baseline kept
        if ct < bt:
            flag = "  <-- FALSIFIED: dropped a TRICKY novel-detail fragment"
            falsified = True
        # FRONTIER: conditioning keeps >= answers and <= known-noise
        if ca >= ba and cn <= bn and (ca > ba or cn < bn):
            frontier_wins += 1
        print(f"{budget:>7} | {ba:^10}{bn:^11}{bt:^11} | {ca:^10}{cn:^11}{ct:^12}{flag}")

    print("-" * 78)
    print("\nVERDICT")
    if falsified:
        print("  FALSIFIED: belief-conditioning evicted a fragment that resembled a")
        print("  belief but carried a new load-bearing fact. The method is unsafe as-is.")
    else:
        print("  NOT falsified on the false-known failure mode: every TRICKY fragment")
        print("  retained by the baseline was also retained when conditioning was on.")
        print(f"  Frontier improvement (more answers / less known-noise): "
              f"{frontier_wins}/{len(budgets)} budget points.")
        if frontier_wins == 0:
            print("  ...but NO budget point showed an improvement -> claim unsupported (no benefit).")


if __name__ == "__main__":
    main()
