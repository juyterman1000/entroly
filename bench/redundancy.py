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
    python -m bench.redundancy --sweep-floor    # trace the diversity_floor trade
    python -m bench.redundancy --json
"""
from __future__ import annotations

import argparse
import contextlib
import copy
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


# ── Hard negatives: similar but carrying distinct information ────────────
#
# The case that actually constrains the admission gate, and the one the two
# corpora above cannot exercise. `synthetic_graded_redundancy` contains planted
# duplicates, so a looser gate always looks better on it. `cases.json` contains
# no duplicates, so a looser gate costs nothing there. Neither contains a pair
# that is *lexically* near-identical and *semantically* different, which is the
# only pair a drop can get wrong.
#
# These four transformation classes are ported from Arm 9 of
# `entroly-core/src/simhash_probe.rs`, deliberately rather than invented here,
# so the Python gate measurement and the Rust distance probe argue about the
# same population. Each carries a load-bearing detail:
#
#   * operators are matched *with surrounding spaces*, or `<` rewrites a generic
#     parameter list such as `Vec<T>` and the pair stops being a predicate
#     inversion;
#   * the integer swap maps 7<->3 and everything else to 7, keeping the
#     replacement the same width, because equal width keeps the surrounding
#     trigrams aligned — a longer replacement would be an easy negative;
#   * the templated class is a long identical sentence around one differing
#     subject noun, the shape where overlap is highest and the distinction is
#     nonetheless total.

_SUBJECTS = ("user", "group", "project", "workspace", "team", "webhook",
             "channel", "invoice")
_CONFIRM_VERBS = ("delete", "archive", "duplicate", "restore", "export")


def _replace_first(text: str, old: str, new: str) -> str | None:
    at = text.find(old)
    return None if at < 0 else text[:at] + new + text[at + len(old):]


def flip_comparison(text: str) -> str | None:
    """Invert the first comparison — the opposite predicate, same tokens."""
    for old, new in ((" <= ", " >= "), (" >= ", " <= "), (" == ", " != "),
                     (" < ", " > "), (" > ", " < ")):
        out = _replace_first(text, old, new)
        if out is not None:
            return out
    return None


def invert_boolean(text: str) -> str | None:
    """Invert the first returned boolean — the opposite verdict."""
    for old, new in (("return true", "return false"), ("return false", "return true"),
                     ("return True", "return False"), ("return False", "return True")):
        out = _replace_first(text, old, new)
        if out is not None:
            return out
    return None


def change_integer_literal(text: str) -> str | None:
    """Change the first standalone integer, preserving its width."""
    for match in re.finditer(r"\d+", text):
        start, end = match.span()
        before_ok = start == 0 or not (text[start - 1].isalnum()
                                       or text[start - 1] in "_.")
        after_ok = end >= len(text) or not (text[end].isalnum()
                                            or text[end] in "_.")
        if not (before_ok and after_ok):
            continue
        original = match.group()
        swapped = "".join("3" if c == "7" else "7" for c in original)
        if swapped != original:
            return text[:start] + swapped + text[end:]
    return None


def templated_negative_pairs() -> list[tuple[str, str]]:
    """Catalogue entries differing in exactly one subject noun."""
    out: list[tuple[str, str]] = []
    for verb in _CONFIRM_VERBS:
        def line(subject: str, verb: str = verb) -> str:
            return (f'"confirm.{verb}": "Are you sure you want to {verb} this '
                    f'{subject}? This action cannot be undone and any members '
                    f'will immediately lose access to its contents."')
        for left, right in zip(_SUBJECTS, _SUBJECTS[1:]):
            out.append((line(left), line(right)))
    return out


_HARD_NEGATIVE_SEEDS = (
    "fn pick(limit: usize) -> bool {\n    if limit < 42 {\n        return true;\n    }\n    return false;\n}",
    "def retry(attempts, ceiling):\n    if attempts >= 7:\n        return False\n    return True",
    "fn admit(score: f64, floor: f64) -> bool {\n    score > floor\n}",
    "def is_stale(age_turns):\n    return age_turns == 15",
)


def hard_negative_corpus() -> dict:
    """Pairs distinct *by construction*, as lexically close as the edit allows.

    Every fragment here should survive selection. A gate that drops either
    member of any pair has destroyed information, so the correct score on this
    corpus is exactly 0.0 — unlike the other two, where the right answer is a
    trade-off.
    """
    frags: list[dict] = []
    pairs: list[tuple[str, str]] = []
    labels: dict[tuple[str, str], str] = {}
    source_of: dict[str, str] = {}

    def intern(content: str, label: str, idx: int, side: str) -> str:
        """One fragment per distinct content, shared across pairs.

        Interning is required, not tidiness. A seed feeds three transformations
        and a subject appears in more than one templated pair, so adding a
        fragment per pair member would enter the *same text* under several
        source names. Those are exact duplicates, dedup removes them correctly,
        and the corpus would then measure its own construction defect rather
        than the gate: before interning this reported 0.84 suppression with the
        gate fully disabled.
        """
        if content in source_of:
            return source_of[content]
        src = f"hardneg/{label}_{idx}_{side}"
        source_of[content] = src
        frags.append({"content": content, "source": src,
                      "token_count": max(1, len(content) // 4),
                      "expected_selected": True})
        return src

    def add_pair(label: str, idx: int, left: str, right: str) -> None:
        # Non-vacuity, per pair: an edit that returned its input is not a
        # negative at all, and a corpus of them would score a perfect 0.0 while
        # testing nothing. The Rust arm learned this the hard way — its
        # `every_hard_negative_generator_changes_the_text` guard exists because
        # a no-op generator once made a regression test worthless.
        assert left != right, f"{label}[{idx}] generator returned its input unchanged"
        la = intern(left, label, idx, "a")
        lb = intern(right, label, idx, "b")
        pairs.append((la, lb))
        labels[(la, lb)] = label

    for idx, seed in enumerate(_HARD_NEGATIVE_SEEDS):
        for label, fn in (("flipped_comparison", flip_comparison),
                          ("inverted_boolean", invert_boolean),
                          ("changed_integer", change_integer_literal)):
            edited = fn(seed)
            if edited is not None:
                add_pair(label, idx, seed, edited)

    for idx, (left, right) in enumerate(templated_negative_pairs()):
        add_pair("templated_subject", idx, left, right)

    # Every fragment must be textually unique, or dedup removes it and the
    # resulting drop is wrongly attributed to the gate.
    contents = [f["content"] for f in frags]
    assert len(contents) == len(set(contents)), (
        "hard-negative corpus contains duplicate content; dedup would remove it "
        "and the suppression measured would be the fixture's, not the gate's"
    )

    # Non-vacuity, corpus-wide: these must actually be *hard*. If no pair is
    # lexically close, no gate would ever touch them and a clean score would
    # mean nothing. Require that the population reaches into the range where
    # gates live, rather than hoping it does.
    close = sum(1 for a, b in pairs
                if true_cosine(
                    next(f["content"] for f in frags if f["source"] == a),
                    next(f["content"] for f in frags if f["source"] == b),
                ) >= 0.70)
    if close == 0:
        raise SystemExit(
            "hard-negative corpus contains no lexically close pair; a gate "
            "would never act on it and a score of 0.0 would be vacuous"
        )

    return {
        "id": "hard_negatives_similar_but_distinct",
        "query": "return limit score",
        "fragments": frags,
        # Generous: every fragment must fit, so a dropped one was dropped by
        # the gate and not by the budget.
        "token_budget": sum(f["token_count"] for f in frags) + 200,
        "planted_pairs": pairs,
        "pair_labels": labels,
        "close_pairs": close,
    }


def selected_sources(case: dict, engine_factory) -> set[str]:
    """Sources surviving selection for this case, under this configuration."""
    engine = engine_factory()
    for frag in case["fragments"]:
        engine.ingest(frag["content"], frag["source"],
                      frag.get("token_count", 0), False)
    result = dict(engine.optimize(case["token_budget"], case.get("query", "")))
    return {dict(s).get("source", "") for s in result.get("selected", [])}


def hard_negative_report(case: dict, engine_factory, baseline: set[str]) -> dict:
    """Pairs the gate breaks that survived *without* it.

    Differential on purpose. A fragment can go unselected for reasons that have
    nothing to do with the gate: measured here, three score exactly zero
    relevance to the query, and the greedy requires `density > 0` to admit
    anything, so they are never selected at any gate setting. Reporting raw
    survival would charge the gate 0.1136 before it does anything at all.

    `baseline` is the surviving set with the gate disabled. Only pairs intact
    there and broken here are attributable, and the correct value for that is
    exactly 0.0 — every pair is distinct by construction, so any attributable
    loss is the gate destroying information. Broken out by class to show *which*
    distinction it cannot see.
    """
    chosen = selected_sources(case, engine_factory)

    per_class: dict[str, list[int]] = {}
    casualties: list[str] = []
    for pair in case["planted_pairs"]:
        label = case["pair_labels"][pair]
        intact_before = pair[0] in baseline and pair[1] in baseline
        if not intact_before:
            continue  # not attributable to the gate
        lost = not (pair[0] in chosen and pair[1] in chosen)
        per_class.setdefault(label, []).append(1 if lost else 0)
        if lost:
            casualties.extend(s for s in pair if s not in chosen)

    total = [v for vs in per_class.values() for v in vs]
    if not total:
        raise SystemExit(
            "no hard-negative pair survived the gate-off baseline, so nothing "
            "is attributable and this measurement is vacuous"
        )
    return {
        "case_id": case["id"],
        "attributable_pairs": len(total),
        "close_pairs": case["close_pairs"],
        "gate_induced_suppression": round(sum(total) / len(total), 4),
        "by_class": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_class.items())},
        "casualties": sorted(set(casualties))[:8],
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


# Ladder spanning the shipped settings. The product defaults are 0.10
# (`EntrolyConfig.ios_diversity_floor`) with proxy profiles at 0.05 and 0.15, so
# the sweep has to straddle those rather than start at them. 0.0 is included as
# the reference point where the penalty is allowed to veto outright.
DEFAULT_FLOOR_LADDER = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.40)


def sweep_floor(cases: Sequence[dict], base_config: dict, floors: Sequence[float],
                engine_builder) -> list[dict]:
    """Trace the suppression trade-off `diversity_floor` controls.

    ``knapsack_sds.rs`` computes ``diversity_factor(..).max(diversity_floor)``
    and multiplies the candidate's base value by it, so the floor is a hard cap
    on suppression: at 0.10 a candidate estimated 99.9% redundant still keeps
    10% of its value. Because selection is by density, such a candidate wins
    whenever its density exceeds ~1/floor times a distinct competitor's. The
    floor therefore cannot veto redundancy, only discount it.

    Lowering it should trade ``redundant_token_fraction`` down against
    ``false_suppression`` up. This measures that curve instead of assuming it.
    """
    rows: list[dict] = []
    for floor in floors:
        config = copy.deepcopy(base_config)
        config.setdefault("ios", {})["diversity_floor"] = floor

        per_case = [selection_report(c, lambda: engine_builder(config))
                    for c in cases]

        def mean(key: str) -> float | None:
            vals = [r[key] for r in per_case if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        rows.append({
            "diversity_floor": floor,
            "selected": sum(r["selected"] for r in per_case),
            "tokens_used": sum(r["tokens_used"] for r in per_case),
            "redundant_token_fraction": mean("redundant_token_fraction"),
            "duplicate_suppression": mean("duplicate_suppression"),
            "false_suppression": mean("false_suppression"),
        })

    # Non-vacuity: if every floor yields an identical outcome the knob is not
    # reaching the selector, and a flat curve would otherwise read as "the floor
    # does not matter" rather than "the floor is not wired".
    varying = {
        (r["redundant_token_fraction"], r["duplicate_suppression"],
         r["false_suppression"], r["tokens_used"])
        for r in rows
    }
    if len(varying) == 1:
        raise SystemExit(
            "diversity_floor had no effect across "
            f"{floors[0]}..{floors[-1]} — the knob is not reaching the "
            "selector, so this sweep measures nothing. Check that "
            "ios_diversity_floor is plumbed from the tuning config through to "
            "knapsack_sds::ios_select."
        )
    return rows


def main() -> int:
    global REDUNDANCY_TAU

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=None,
                    help="case file to price for redundancy "
                         "(default: synthetic graded corpus)")
    ap.add_argument("--redundancy-tau", type=float, default=REDUNDANCY_TAU)
    ap.add_argument("--sweep-floor", nargs="?", const="default", default=None,
                    metavar="F1,F2,...",
                    help="sweep ios_diversity_floor and report the "
                         "redundancy-vs-false-suppression trade "
                         f"(default ladder: {','.join(str(f) for f in DEFAULT_FLOOR_LADDER)})")
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

    # Exploration must be off for this harness to measure anything.
    #
    # bench/evaluate.py's default tuning config sets `exploration_rate = 0.1`,
    # an epsilon-greedy term that promotes a lower-ranked fragment one time in
    # ten. That is a legitimate production knob -- `EntrolyConfig`'s own shipped
    # default is 0 precisely "so selection is reproducible" -- but against a
    # measurement harness it is fatal. Measured here across three trials at a
    # FIXED configuration on bench/cases.json, mean false suppression came out
    # 0.0208, 0.0417, 0.0347. Any two single runs can therefore differ by 2x
    # from noise alone, which is larger than the effects this harness exists to
    # detect, and a before/after comparison would attribute that spread to
    # whatever was changed in between. I made exactly that mistake before
    # pinning it here.
    config.setdefault("knapsack", {})["exploration_rate"] = 0.0

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

    if args.sweep_floor is not None:
        floors = (DEFAULT_FLOOR_LADDER if args.sweep_floor == "default"
                  else tuple(float(f) for f in args.sweep_floor.split(",")))
        report["floor_sweep"] = sweep_floor(cases, config, floors,
                                            create_engine_from_config)

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

    if "floor_sweep" in report:
        print()
        print("diversity_floor sweep  (floor caps suppression: a candidate "
              "keeps >= floor of its value)")
        print(f"  {'floor':>6} {'tokens':>7} {'redundant':>10} {'dup_supp':>9} "
              f"{'false_supp':>11}")
        print("  " + "-" * 48)
        for row in report["floor_sweep"]:
            def fmt(v):
                return "     n/a" if v is None else f"{v:>8.4f}"
            marker = "  <- shipped default" if row["diversity_floor"] == 0.10 else ""
            print(f"  {row['diversity_floor']:>6.2f} {row['tokens_used']:>7} "
                  f"{fmt(row['redundant_token_fraction'])[:10]:>10} "
                  f"{fmt(row['duplicate_suppression'])[:9]:>9} "
                  f"{fmt(row['false_suppression'])[:11]:>11}{marker}")

        fracs = [r["redundant_token_fraction"] for r in report["floor_sweep"]
                 if r["redundant_token_fraction"] is not None]
        if fracs and (max(fracs) - min(fracs)) < 0.05:
            print()
            print("  Reading: the curve is flat, including at floor=0.00 where the")
            print("  penalty may zero a candidate's value outright. That is")
            print("  structural, not a tuning artifact -- the greedy in")
            print("  knapsack_sds::ios_select starts from best_density = 0.0, admits")
            print("  any candidate with positive density that fits, and breaks only")
            print("  when nothing fits the budget. There is no absolute admission")
            print("  threshold on the diversity-penalised value, and")
            print("  min_candidate_value gates only lower-resolution variants.")
            print("  So the diversity term REORDERS candidates and cannot EXCLUDE")
            print("  one. Whenever the budget exceeds the distinct content, every")
            print("  redundant fragment that fits is still admitted at the tail.")
            print()
            print("  The penalty is not weak, which is what makes this conclusive.")
            print("  simhash_cosine_lcb reports 0.86-0.99 for the planted pairs at")
            print("  Hamming 0-2, so diversity_factor lands at 0.01-0.14 -- a 7x to")
            print("  100x cut to those candidates' value. It is computed, it is")
            print("  large, and the admitted set does not change. So neither a")
            print("  floor change nor a wider fingerprint (which would raise LCB")
            print("  detection power) can reduce this waste: no value of the")
            print("  penalty excludes a candidate that fits.")
            print()
            print("  Consequence for dedup.rs:377, which drops the Hamming merge arm")
            print("  on the grounds that a missed merge is 'penalised downstream")
            print("  anyway by the IOS diversity term': the penalty discounts, it")
            print("  does not exclude, so the tokens are spent either way whenever")
            print("  the budget has room. Closing this needs an admission gate, not")
            print("  a better estimator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
