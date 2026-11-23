"""Redundancy and estimator-fidelity benchmark for the IOS selection lane.

Why this exists
---------------
``bench/evaluate.py`` already exercises this lane -- it calls
``EntrolyEngine.optimize()`` directly on the in-memory Rust engine, so it is
properly isolated and does measure real selection. What it does *not* measure
is redundancy. Its four metrics are recall, precision,
``context_efficiency = sum(entropy * tokens) / total_tokens``, and latency, and
none of them penalise selecting the same content twice: two near-identical
high-entropy fragments both score well on every one of them.

That blind spot matters because of a specific assumption the dedup design now
rests on. ``DedupIndex::insert`` deliberately no longer merges on Hamming
distance at all -- it hard-drops only on an exact normalized-content-hash match
among band candidates -- because the Hamming arm was measured to contribute 3
merges in 2.25M pairs while carrying the entire false-merge risk, and a false
merge discards a fragment permanently. See the rationale at ``dedup.rs:377``.

That decision is well evidenced, and it explicitly defers the cost: a missed
merge "costs tokens and is then penalised downstream anyway by the IOS
diversity term". **That clause is an empirical claim about a different
subsystem, and nothing measures it.** If the diversity term does not in fact
suppress near-duplicates, the tokens are simply lost, and no existing metric
would show it -- which is what this harness is for.

Note what this means for the fingerprint width, because the intuition points
the wrong way. Widening does *not* fix merging: ``d_H`` is a sufficient
statistic for the angle between trigram-TF vectors, so a one-token edit moves
~3 of n trigrams whether the token is semantic or cosmetic, and `limit < 42`
vs `limit > 42` sits at the same distance as `tmp` vs `temp`. Widening 64 ->
1024 at matched d/B was measured to leave false merges flat and make capture
worse. Width still helps the *soft* diversity penalty, whose detection power
scales with ``1/sqrt(B)``; it does not help a destructive drop. The estimator
metrics below therefore describe the penalty path, not the merge path.

A second, smaller effect the estimator metrics expose: ``dedup::simhash`` sets
bit b iff its accumulator is ``> 0``, so an exactly tied accumulator always
resolves to 0. Because features are word *trigrams*, the tie rate depends on
the parity of the trigram count -- 3 words make 1 trigram and cannot tie, 4
words make 2 and tie on 37.5% of bits. The symptom is a fingerprint popcount
far from the balanced B/2, which ``popcount_imbalance`` reports.

What it reports
---------------
Estimator fidelity, against exact TF-cosine over the *same* trigram feature
space the fingerprint uses -- so the comparison isolates estimator error rather
than a tokenization difference:

* ``estimator_mae``         mean |simhash_cosine - true_cosine| over all pairs
* ``estimator_bias``        signed mean; positive means similarity is
                            systematically over-reported
* ``orthogonal_inflation``  mean reported cosine for pairs whose true cosine is
                            ~0. Should be ~0; the tie bias pushes it up
* ``popcount_imbalance``    mean |popcount - B/2| / (B/2). 0 is unbiased; a
                            large value is the tie bias directly

Product cost, over whatever cases are supplied:

* ``redundant_token_fraction``  share of selected tokens spent on a fragment
                                whose true cosine to an earlier-selected
                                fragment is >= --redundancy-tau. This is the
                                waste metric: lower is better
* ``duplicate_suppression``     of planted near-duplicate pairs, the share
                                where at most one member survived selection
* ``false_suppression``         share of genuinely distinct fragments that were
                                dropped. Guards the other direction, so a fix
                                cannot score well by suppressing everything

Non-vacuity guards
------------------
A benchmark that silently measures nothing is worse than none. Before reporting
it asserts that the corpus actually contains both near-duplicate and orthogonal
pairs, that selection ran and returned tokens, and that the planted similarity
spread is non-degenerate. Any of those failing is an error, not a zero score.

Usage
-----
    python -m bench.redundancy                  # synthetic graded corpus
    python -m bench.redundancy --cases bench/cases.json
    python -m bench.redundancy --json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

SIMHASH_BITS = 64

# Default thresholds. `REDUNDANCY_TAU` is the true-cosine level at which two
# fragments are considered to carry the same information, so the second one's
# tokens are waste. `ORTHOGONAL_TAU` is the level below which two fragments are
# considered genuinely distinct, so suppressing one is an error.
REDUNDANCY_TAU = 0.80
ORTHOGONAL_TAU = 0.30

# Mirrors `dedup::simhash`'s word trimming: strip leading/trailing characters
# that are neither alphanumeric nor underscore, but keep the raw token when
# trimming would erase it entirely (that carve-out is what makes standalone
# operators like `<` visible to the fingerprint -- see dedup.rs).
_EDGE_PUNCT = re.compile(r"^[^0-9A-Za-z_]+|[^0-9A-Za-z_]+$")


def dedup_words(text: str) -> list[str]:
    """Tokenize exactly as ``dedup::simhash`` does, so features line up."""
    out: list[str] = []
    for raw in text.split():
        trimmed = _EDGE_PUNCT.sub("", raw)
        token = trimmed if trimmed else raw
        if token:
            out.append(token)
    return out


def dedup_features(text: str) -> Counter[str]:
    """Trigram features, with the unigram fallback for short inputs."""
    words = dedup_words(text)
    if not words:
        return Counter()
    if len(words) >= 3:
        return Counter(
            f"{words[i]} {words[i + 1]} {words[i + 2]}".lower()
            for i in range(len(words) - 2)
        )
    return Counter(w.lower() for w in words)


def true_cosine(a: str, b: str) -> float:
    """Exact TF-cosine over the fingerprint's own feature space.

    This is the ground truth the fingerprint is *trying* to estimate, computed
    without hashing, so any gap is estimator error and nothing else.
    """
    fa, fb = dedup_features(a), dedup_features(b)
    if not fa or not fb:
        return 0.0
    shared = set(fa) & set(fb)
    dot = sum(fa[k] * fb[k] for k in shared)
    na = math.sqrt(sum(v * v for v in fa.values()))
    nb = math.sqrt(sum(v * v for v in fb.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def estimated_cosine(fp_a: int, fp_b: int, hamming) -> float:
    """The estimator the selector actually uses: cos(pi * d / B).

    Deliberately NOT clamped at 0. Clamping is correct when reporting a
    similarity and wrong when measuring the estimator, because it would hide
    exactly the negative excursions that reveal the noise floor.
    """
    return math.cos(math.pi * hamming(fp_a, fp_b) / SIMHASH_BITS)


# ── Synthetic graded corpus ──────────────────────────────────────────────
#
# Planted so true cosine is controlled rather than hoped for. Two fixture traps
# are deliberately avoided:
#
#   * An *exact* duplicate proves nothing about SimHash, because `insert`
#     short-circuits on a normalized content hash before distance is consulted.
#     Every near-duplicate here is edited, never copied.
#   * Tie rate depends on trigram-count parity, so a corpus built at a single
#     fragment length would measure the bias at one parity and miss it. Lengths
#     sweep both parities.

_VERBS = ("connect", "resolve", "encode", "flush", "retry", "validate")
_NOUNS = ("socket", "payload", "cursor", "buffer", "manifest", "token")


def _body(rng: random.Random, n_words: int) -> str:
    parts = []
    while len(" ".join(parts).split()) < n_words:
        parts.append(
            f"{rng.choice(_VERBS)}_{rng.choice(_NOUNS)}({rng.choice(_NOUNS)}, "
            f"{rng.randrange(1000)})"
        )
    return " ".join(parts)


def synthetic_corpus(seed: int = 20261122) -> dict[str, Any]:
    """A corpus with planted near-duplicate pairs and orthogonal controls."""
    rng = random.Random(seed)
    frags: list[dict[str, Any]] = []
    pairs: list[tuple[str, str]] = []

    # Word counts chosen to hit both trigram-count parities at several scales.
    for idx, n_words in enumerate((6, 7, 12, 13, 30, 31, 60, 61)):
        base = f"def base_{idx}():\n    return {_body(rng, n_words)}"
        base_src = f"planted/base_{idx}.py"
        frags.append({"content": base, "source": base_src,
                      "token_count": max(1, len(base) // 4),
                      "expected_selected": True})

        # Near-duplicate: edit a single token, so content hashes differ and the
        # distance arm is the only thing that can catch it.
        words = base.split()
        words[-1] = "sentinel_edit"
        edited = " ".join(words)
        edit_src = f"planted/edit_{idx}.py"
        frags.append({"content": edited, "source": edit_src,
                      "token_count": max(1, len(edited) // 4),
                      "expected_selected": False})
        pairs.append((base_src, edit_src))

        # Orthogonal control at the same length, so any suppression of it is
        # attributable to estimator error rather than to length.
        other = f"def other_{idx}():\n    return {_body(rng, n_words)}"
        frags.append({"content": other, "source": f"planted/other_{idx}.py",
                      "token_count": max(1, len(other) // 4),
                      "expected_selected": True})

    return {
        "id": "synthetic_graded_redundancy",
        "query": "return",
        "fragments": frags,
        # Budget deliberately tight enough that the selector must choose, but
        # loose enough that a correct selector can take every distinct
        # fragment. Redundancy is then a choice, not a budget artifact.
        "token_budget": sum(f["token_count"] for f in frags
                            if f["expected_selected"]) + 40,
        "planted_pairs": pairs,
    }


# ── Measurement ──────────────────────────────────────────────────────────

def estimator_report(fragments: Sequence[dict], simhash, hamming) -> dict:
    """Compare the 64-bit estimator against exact TF-cosine, pairwise."""
    contents = [f["content"] for f in fragments]
    fps = [simhash(c) for c in contents]

    imbalance = [
        abs(bin(fp).count("1") - SIMHASH_BITS / 2) / (SIMHASH_BITS / 2)
        for fp in fps
    ]

    errs: list[float] = []
    signed: list[float] = []
    orthogonal_reported: list[float] = []
    dup_reported: list[float] = []
    spread: list[float] = []

    for i in range(len(contents)):
        for j in range(i + 1, len(contents)):
            t = true_cosine(contents[i], contents[j])
            e = estimated_cosine(fps[i], fps[j], hamming)
            spread.append(t)
            errs.append(abs(e - t))
            signed.append(e - t)
            if t < ORTHOGONAL_TAU:
                orthogonal_reported.append(e)
            if t >= REDUNDANCY_TAU:
                dup_reported.append(e)

    # Non-vacuity: without both populations the headline numbers are undefined.
    if not orthogonal_reported:
        raise SystemExit(
            "corpus contains no orthogonal pairs (true cosine < "
            f"{ORTHOGONAL_TAU}); orthogonal_inflation would be vacuous"
        )
    if not dup_reported:
        raise SystemExit(
            "corpus contains no near-duplicate pairs (true cosine >= "
            f"{REDUNDANCY_TAU}); duplicate metrics would be vacuous"
        )
    if max(spread) - min(spread) < 0.5:
        raise SystemExit(
            f"planted similarity spread is degenerate "
            f"({min(spread):.3f}..{max(spread):.3f}); the corpus cannot "
            "discriminate an estimator change"
        )

    def mean(xs):
        return sum(xs) / len(xs)

    return {
        "pairs": len(errs),
        "estimator_mae": round(mean(errs), 4),
        "estimator_bias": round(mean(signed), 4),
        "orthogonal_pairs": len(orthogonal_reported),
        "orthogonal_inflation": round(mean(orthogonal_reported), 4),
        "duplicate_pairs": len(dup_reported),
        "duplicate_reported_mean": round(mean(dup_reported), 4),
        "duplicate_reported_min": round(min(dup_reported), 4),
        "popcount_imbalance": round(mean(imbalance), 4),
        "noise_floor": round(math.pi / (2 * math.sqrt(SIMHASH_BITS)), 4),
    }


def selection_report(case: dict, engine_factory) -> dict:
    """Run the real selection lane and price the redundancy it admits."""
    engine = engine_factory()
    by_source = {f["source"]: f["content"] for f in case["fragments"]}

    ingested = 0
    for frag in case["fragments"]:
        res = dict(engine.ingest(frag["content"], frag["source"],
                                 frag.get("token_count", 0), False))
        if res.get("status") == "ingested":
            ingested += 1

    result = dict(engine.optimize(case["token_budget"], case.get("query", "")))
    selected = [dict(s) for s in result.get("selected", [])]

    if not selected:
        raise SystemExit(
            f"case {case['id']!r}: selection returned nothing "
            f"({ingested} fragments ingested, budget "
            f"{case['token_budget']}); nothing to measure"
        )

    # Redundancy is order-dependent: the first occurrence is the one that
    # carries the information, every later near-copy is the waste.
    order = [s.get("source", "") for s in selected]
    tokens = {s.get("source", ""): s.get("token_count", 0) for s in selected}
    redundant_tokens = 0
    redundant_sources = []
    for pos, src in enumerate(order):
        content = by_source.get(src)
        if content is None:
            continue
        for earlier in order[:pos]:
            prior = by_source.get(earlier)
            if prior is None:
                continue
            if true_cosine(content, prior) >= REDUNDANCY_TAU:
                redundant_tokens += tokens.get(src, 0)
                redundant_sources.append(src)
                break

    total_tokens = sum(tokens.values()) or 1
    chosen = set(order)

    planted = case.get("planted_pairs") or []
    suppressed = sum(1 for a, b in planted
                     if not (a in chosen and b in chosen))

    # False suppression: a fragment distinct from everything else that was
    # nonetheless dropped while budget remained.
    distinct_expected = []
    for frag in case["fragments"]:
        if not frag.get("expected_selected", False):
            continue
        content = frag["content"]
        worst = max(
            (true_cosine(content, other["content"])
             for other in case["fragments"] if other["source"] != frag["source"]),
            default=0.0,
        )
        if worst < ORTHOGONAL_TAU:
            distinct_expected.append(frag["source"])
    dropped = [s for s in distinct_expected if s not in chosen]

    return {
        "case_id": case["id"],
        "method": result.get("method", "unknown"),
        "fragments_ingested": ingested,
        "selected": len(selected),
        "tokens_used": total_tokens,
        "redundant_token_fraction": round(redundant_tokens / total_tokens, 4),
        "redundant_sources": sorted(set(redundant_sources)),
        "planted_pairs": len(planted),
        "duplicate_suppression": (round(suppressed / len(planted), 4)
                                  if planted else None),
        "distinct_expected": len(distinct_expected),
        "false_suppression": (round(len(dropped) / len(distinct_expected), 4)
                              if distinct_expected else None),
        "falsely_suppressed": sorted(dropped),
    }


def main() -> int:
    global REDUNDANCY_TAU

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=None,
                    help="case file to price for redundancy "
                         "(default: synthetic graded corpus)")
    ap.add_argument("--redundancy-tau", type=float, default=REDUNDANCY_TAU)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    REDUNDANCY_TAU = args.redundancy_tau

    try:
        from entroly_core import py_hamming_distance, py_simhash
    except ImportError:
        print("entroly_core not available. Run `maturin develop --release` "
              "in entroly-core/.", file=sys.stderr)
        return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench.evaluate import create_engine_from_config, load_tuning_config

    # Same tuning config bench/evaluate.py uses, so the two harnesses describe
    # the same selector rather than two differently-weighted ones.
    #
    # load_tuning_config reports a missing config on stdout. Under --json that
    # would prepend prose to the document and make it unparseable, so stdout is
    # diverted to stderr for the duration: the message is still seen, but it
    # cannot corrupt a machine-read report.
    with contextlib.redirect_stdout(sys.stderr):
        config = load_tuning_config(None)

    synthetic = synthetic_corpus()
    cases = [synthetic]
    if args.cases:
        with open(args.cases, encoding="utf-8") as fh:
            cases = json.load(fh)

    estimator = estimator_report(synthetic["fragments"],
                                 py_simhash, py_hamming_distance)
    selections = [selection_report(c, lambda: create_engine_from_config(config))
                  for c in cases]

    def avg(key):
        vals = [s[key] for s in selections if s.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    report = {
        "simhash_bits": SIMHASH_BITS,
        "redundancy_tau": REDUNDANCY_TAU,
        "orthogonal_tau": ORTHOGONAL_TAU,
        "estimator": estimator,
        "cases": selections,
        "mean_redundant_token_fraction": avg("redundant_token_fraction"),
        "mean_duplicate_suppression": avg("duplicate_suppression"),
        "mean_false_suppression": avg("false_suppression"),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    e = estimator
    print(f"Estimator fidelity vs exact TF-cosine  ({SIMHASH_BITS}-bit, "
          f"{e['pairs']} pairs)")
    print(f"  noise floor pi/(2*sqrt(B))      {e['noise_floor']:.4f}")
    print(f"  MAE                             {e['estimator_mae']:.4f}")
    print(f"  signed bias                     {e['estimator_bias']:+.4f}")
    print(f"  popcount imbalance              {e['popcount_imbalance']:.4f}"
          "   (0 = unbiased; tie bias raises this)")
    print(f"  orthogonal pairs reported as    {e['orthogonal_inflation']:+.4f}"
          f"   (n={e['orthogonal_pairs']}, true ~0)")
    print(f"  near-dup pairs reported as      {e['duplicate_reported_mean']:.4f}"
          f"   (min {e['duplicate_reported_min']:.4f}, n={e['duplicate_pairs']})")
    print()
    print("Selection cost")
    for s in selections:
        print(f"  {s['case_id']}")
        print(f"    method={s['method']}  selected={s['selected']}  "
              f"tokens={s['tokens_used']}")
        print(f"    redundant_token_fraction  {s['redundant_token_fraction']:.4f}")
        if s["duplicate_suppression"] is not None:
            print(f"    duplicate_suppression     {s['duplicate_suppression']:.4f}"
                  f"  ({s['planted_pairs']} planted pairs)")
        if s["false_suppression"] is not None:
            print(f"    false_suppression         {s['false_suppression']:.4f}"
                  f"  ({s['distinct_expected']} distinct fragments)")
        if s["redundant_sources"]:
            print(f"    waste: {', '.join(s['redundant_sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
